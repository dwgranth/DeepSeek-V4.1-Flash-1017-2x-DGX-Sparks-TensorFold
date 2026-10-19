# DeepSeek-V4.1-Flash-1017 (flash17) on 2× DGX Spark — field-proven deployment recipe

A companion to [jayleaton/deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark).
Everything here was **executed end-to-end on a real 2× DGX Spark (GB10) pair on 2026-10-18**, and the
repo ships the exact scripts that worked — including **four fixes the upstream flow needs as of today**
(the upstream docs describe a flow that fails at `aot-needs` with `NoKernel` if followed literally).

The upstream recipe remains the source of truth for the engine itself. This repo adds:

- the correct Python reference source for the AOT fill (TensorFold **1.0.2**, not the vendored 0.6.0),
- the `_gemm` kernel graft (upstream deleted it; the engine still requires it),
- the `rope.bin` row count that boots (CONTEXT+2048),
- the missing asset-toolchain pip packages,
- a graceful start/stop pair that survives the nested-ssh launch trap,
- a 7-test verification battery, and
- measured numbers so you can compare your build against ours.

## Verified results (this build)

| Metric | Measured | Upstream reference |
| --- | --- | --- |
| Cold boot (first boot, writes ~95 GB prepared/node + calib) | 427 s | "several minutes" |
| Warm restart (prepared + calib exist) | ~50 s | — |
| Smoke gate: `What is 17 * 23?` | **391** | 391 |
| Speculative drafting (smoke request) | drafted 20 / accepted 13 | — |
| Prefill @ 30,093 tokens | **2,182 tok/s** | ~2,198–2,349 |
| Decode after 30K-token prompt | **85.6 tok/s** | ~64.1 |
| Burst decode (counting, 92 % draft acceptance) | 107.7 tok/s | 87.9 short decode |
| Verification battery | **7/7** (`scripts/model-test.py`) | — |

Serving profile: TP=2 over the 200 Gb/s CX7 link, 4 slots × 1,048,576 context, 2.1 M-token pool,
vision + tool calling + structured output on, model ID `DeepSeek-V4.1-Flash-TF`, port 8888.

## Layout used

```
~/deepseek-v41-tensorfold-spark/     upstream clone (--recurse-submodules), on BOTH nodes
~/dsv41-tf-zig/
  flash17/                           EXL3 pack (190 GB, SHA256SUMS-verified, byte-identical both nodes)
  engram-src-1017/                   DeepSeek-V4.1-Flash-1017 shards 00047+00048 (~190 GB, head only)
  engram-r0/  engram-r1/             packed per-rank Engram bins (engram-r1 on the worker)
  assets-r0/  assets-r1/             sigs.json + needs.json + aot/ + rope.bin + engram-host.bin
  prepared-r0/r1  state-r0/r1  cache-r0/r1
  config -> the repo's config/prod-zig.env (written on both nodes)
```

## Step 1 — clone, weights, Engram  → `scripts/pack-engram.sh`

```bash
git clone --recurse-submodules https://github.com/jayleaton/deepseek-v41-tensorfold-spark.git
cd deepseek-v41-tensorfold-spark
WORKER=dwgranth@10.100.128.2 ../dsv41-flash17-2x-spark-recipe/scripts/pack-engram.sh
```

Downloads the flash17 pack (SHA256SUMS-verified, 47/47) and the **1017** Engram source shards
(the September shards do NOT match this pack — `pack_engram.py --list` names 00047/00048), packs
`engram-l{1,14}-r{rank}of2.bin` per rank, `--check`s both, and rsyncs the pack + rank-1 bins to the worker.

## Step 2 — build the Zig image  → `scripts/build-zig-image.sh`

Builds on an ARM64 node (the GB10 head works), runs the upstream `check-zig-image.sh` library gate,
saves/rsyncs/loads the tar on the worker, and prints both imageIDs (must match).

## Step 3 — Python reference source (FIX #1, #2)  → `scripts/prepare-python-reference.sh`

**Do not** build sigs from the vendored `vendor/TensorFold` (0.6.0) as the upstream docs imply.

1. The engine is exported from a **TensorFold 1.0.2** kit snapshot (`engine/zig/SOURCE-EXPORT.md`,
   rev `f8fe17d`). The recipe's `patches/0001-0003` were largely **upstreamed into 1.0.2** — applied
   there with `--forward`, they contribute exactly what is still missing: the `deepseek_v41` family
   new files + patch 0003. Against 0.6.0 the sigs mismatch kernels the 1.0.2-based engine launches
   and `aot-needs` dies with `NoKernel`.
2. 1.0.2 also **removed the EXL3 Triton prefill `_gemm`** (moved to CUDA), but the engine still
   enumerates `_gemm` variants (the captures' fallback path). `fixes/tf102-exl3-prefill/prefill.py`
   grafts the 0.6.0 kernel verbatim **at its original module path** — required, because the recipe's
   own `dsv41_triton_options.json` keys its options rule to `tensorfold.cuda.exl3.prefill._gemm`
   (num_warps 8 / num_stages 4). Grafting under a different module path fails the options lookup.

Found by instrumentating `aot_needs.zig`'s `resolve()` in a Debug build — if upstream ever ships a
newer engine export, re-run that trick before assuming the graft list.

## Step 4 — runtime assets  → `scripts/generate-assets.sh`

Runs the whole chain in a container from the image's builder stage, on the head:
toolchain (`torch`, `triton==3.7.1`, ptxas **13.3.73** via `nvidia-cuda-nvcc-cu13`, plus
**numpy** and **tokenizers** — both are real import deps of the generators), then per rank:
`sigs` → `aot-needs` → AOT compile (144/144 expected) → `rope_tables.py --rows 1050624`
(**FIX #3**: CONTEXT+2048; the script's default is refused at boot) → `engram_host.py`.
Finishes by syncing rank-1 assets to the worker (**FIX #4**: `--check-aot` validates each
node's *local* assets; the worker needs its own copy).

## Step 5 — config

Copy `config/prod-zig-flash17.env.example` to `config/prod-zig.env` on **each** node and fill the
placeholders. Two deliberate deviations from the upstream template, both marked in the file:

- `HOST=0.0.0.0` — the template ships `HOST=localhost`, which binds 127.0.0.1 only: nothing off-node
  can connect. (Symptom: instant *connection refused* from other hosts, while `curl localhost:8888/health` works.)
- `PORT=8888` — match your firewall rules (upstream default 8000).

Everything else is the upstream flash17 profile, verbatim.

## Step 6 — preflight

```bash
scripts/run-zig.sh 0 --dry-run && scripts/run-zig.sh 0 --check-aot   # head
ssh <worker> "cd ~/deepseek-v41-tensorfold-spark && scripts/run-zig.sh 1 --dry-run && scripts/run-zig.sh 1 --check-aot"
```

`--check-aot` enumerates ~11 min per rank; exit 1 lists missing variants (compile them with `--have`).

## Step 7 — launch  → `scripts/start.sh` / `scripts/stop.sh`

Rank **1 first**, then rank 0. The scripts encode a real trap: a bash script that ssh-es the worker
and *then* launches the local rank can die right after the worker ssh returns (nested-ssh
stdin/channel theft) — the local rank then never starts and rank 1 sits in its 1800 s plan timeout.
Launch each rank in its own top-level invocation.

Stop is graceful by design: SIGTERM rank 0 only (`TF_DSV41_DRAIN_S=20` for in-flight requests,
503 for new ones), rank 1 orphan-exits; the launcher's `--rm` cleans both containers.

## Step 8 — verify  → `scripts/model-test.py`

The smoke gate (`17 * 23` → **391**) plus math, generated-code-that-actually-runs, JSON mode,
knowledge, 30K-token needle recall, and the speed profile. `/health`'s `drafted_total > 0`
confirms drafting is live.

## Gotcha list (all encountered; all fixed)

| # | Symptom | Cause | Fix |
| --- | --- | --- | --- |
| 1 | `aot-needs` fails `NoKernel` after full enumeration | sigs built from TF 0.6.0; engine is 1.0.2-based | sigs from TF 1.0.2 `f8fe17d` + patches `--forward` (`scripts/prepare-python-reference.sh`) |
| 2 | `aot-needs` fails `NoKernel` on `_gemm` (instrumented resolve shows the name) | 1.0.2 removed the EXL3 Triton prefill; engine still enumerates it | graft `fixes/tf102-exl3-prefill/prefill.py` at the **original module path** (options rule is path-keyed) |
| 3 | Boot refuses: `rope.bin holds 6144 rows; needs 1050624` | rope generated without env → default rows | generate with `--rows $((CONTEXT+2048))` |
| 4 | `ModuleNotFoundError: No module named 'tokenizers'` in `engram_host.py` | asset venv incomplete | `pip install numpy tokenizers` in the asset venv |
| 5 | Worker `--check-aot` fails: needs `assets-r1/sigs.json` | assets synced only head→head | per-rank assets must exist **on each node** |
| 6 | Other hosts get instant *connection refused* | `HOST=localhost` binds 127.0.0.1 | `HOST=0.0.0.0` in `config/prod-zig.env` |
| 7 | Local rank never launches after a script ssh-es the worker | nested-ssh stdin/channel theft kills the calling script | launch ranks in separate top-level invocations (`scripts/start.sh`) |
| 8 | `pgrep -f <script>.sh` reports RUNNING after the script exits | pgrep matches its own ssh polling wrapper | match more specifically, or check container/log state instead |
| 9 | Boot stops on the first wrong asset, naming the exact regen command | engine validates rope/engram-host/aot up front | **read the boot log first** — it usually prints the fix |

## Operations notes

- **Warm restart ≈ 50 s** (prepared files + calib-zig persist per rank and per pack — never reuse
  another pack's prepared/state folders).
- **Clock cap** (recommended for thermals, needs root): `nvidia-smi -lgc 300,2100` on both nodes.
- **Firewall**: the serve has no auth; bind deliberately and allowlist deliberately. Keep the head's
  firewall rules explicit; expect internet scanners on any exposed port.
- **Mutual exclusion**: this serve owns port 8888 + the full UMA on both nodes — stop it before
  starting any other GPU workload (and vice versa).
- **Rollback**: the previous serving stack (e.g. the GLM-5.3 TensorFold recipe) stays intact in its
  own directory; stop this serve, then run that stack's `./start.sh`.

## Credits & license

Built on the upstream recipe and engine by **jayleaton** (Apache-2.0), TensorFold by **ashhart**
(Apache-2.0; the grafted `_gemm` is verbatim TF 0.6.0), DeepSeek-V4.1-Flash-1017 by DeepSeek, and the
MiaAI-Lab 2× DGX Spark kits as layout reference — see `NOTICE`. This repo is Apache-2.0 (`LICENSE`);
weights and upstream components keep their own terms.
