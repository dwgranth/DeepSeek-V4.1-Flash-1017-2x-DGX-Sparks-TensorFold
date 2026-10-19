#!/bin/bash
# Generate the per-rank runtime assets for the Zig engine on the HEAD node, in a
# container built from the builder stage of tensorfold-dsv41-zig:local.
#
# Prereqs: build-zig-image.sh ran (image exists); prepare-python-reference.sh ran
# (build/python-source-102 ready); the flash17 pack is downloaded.
#
# Fixes baked in (each caused a real failure without it):
#   F1 toolchain needs numpy AND tokenizers (engram-host imports tokenizers)
#   F2 rope tables MUST be generated with --rows $((CONTEXT+2048)) (1050624 at 1M);
#      the script's default (CONTEXT 4096 + 2048) is refused at boot
#   F3 per-rank assets must exist ON EACH NODE — run-zig.sh N --check-aot validates locally
#   F4 the AOT compile needs ptxas 13.3.73 (TRITON_PTXAS_PATH / TRITON_PTXAS_BLACKWELL_PATH)
#
# Env: WORKER, BASE, REPO, IMG, CONTEXT, RANKS
set -euo pipefail
REPO=${REPO:-$HOME/deepseek-v41-tensorfold-spark}
BASE=${BASE:-$HOME/dsv41-tf-zig}
IMG=${IMG:-tensorfold-dsv41-zig:local}
WORKER=${WORKER:-dwgranth@10.100.128.2}
CONTEXT=${CONTEXT:-1048576}
ROWS=$((CONTEXT + 2048))
RANKS=${RANKS:-"0 1"}

echo "== builder stage + container"
docker build -q -f $REPO/docker/zig/Dockerfile --target build -t $IMG:builder $REPO/build/zig
docker rm -f dsv41-assets >/dev/null 2>&1 || true
docker run -d --gpus all --name dsv41-assets \
  -v $REPO:/repo -v $BASE:/stage "$IMG:builder" sleep infinity

echo "== toolchain (torch, triton==3.7.1, ptxas 13.3.73, numpy, tokenizers)  [F1]"
docker exec dsv41-assets bash -c '
apt-get update -qq && apt-get install -y -qq python3-venv python3-pip >/dev/null
python3 -m venv /stage/.asset-venv
/stage/.asset-venv/bin/pip install -q --upgrade pip
/stage/.asset-venv/bin/pip install -q torch triton==3.7.1 numpy tokenizers
/stage/.asset-venv/bin/pip install -q "nvidia-cuda-nvcc-cu13==13.3.73" || echo "WARN: ptxas wheel missing -> falls back to image ptxas"
/stage/.asset-venv/bin/python -c "import torch,triton; print(\"torch\",torch.__version__,\"triton\",triton.__version__)"
'

echo "== extract the zig host tools from the builder image"
rm -rf $REPO/zig-out && mkdir -p $REPO/zig-out/bin
cid=$(docker create $IMG:builder)
docker cp "$cid:/out/bin/." $REPO/zig-out/bin/ && docker rm "$cid" >/dev/null

echo "== per-rank: sigs -> aot-needs -> AOT compile -> rope -> engram-host"
for RANK in $RANKS; do
  ASSETS=$BASE/assets-r$RANK
  mkdir -p "$ASSETS"
  docker exec -w /repo/engine/zig dsv41-assets \
    /stage/.asset-venv/bin/python tools/zig/triton_fill.py sigs \
    --py /repo/build/python-source-102/src --out /stage/assets-r$RANK/sigs.json

  docker exec -w /repo/engine/zig dsv41-assets bash -c "
    set -a; . /repo/config/prod-zig.env; set +a
    export TF_DSV41_ASSETS=/stage/assets-r$RANK TF_TP_WORLD=2 TF_TP_KERNEL_IMAGE=/tp_mailbox.fatbin PATH=/repo/zig-out/bin:\$PATH
    tf-dsv41-m1 aot-needs /stage/assets-r$RANK/sigs.json /stage/assets-r$RANK/needs.json"

  docker exec -e PY_SOURCE=/repo/build/python-source-102/src -w /repo/engine/zig dsv41-assets bash -c "
    set -e
    PTXAS=\$(ls /stage/.asset-venv/lib/python3*/site-packages/nvidia/cuda_nvcc/bin/ptxas 2>/dev/null | head -1)
    [ -z \"\$PTXAS\" ] && PTXAS=/usr/local/cuda/bin/ptxas
    echo \"using ptxas: \$PTXAS\"; \"\$PTXAS\" --version | tail -1
    export TRITON_PTXAS_PATH=\$PTXAS TRITON_PTXAS_BLACKWELL_PATH=\$PTXAS
    /stage/.asset-venv/bin/python tools/zig/triton_fill.py compile \
      --py /repo/build/python-source-102/src --needs /stage/assets-r$RANK/needs.json \
      --options tools/zig/dsv41_triton_options.json --arch 121 --out /stage/assets-r$RANK/aot
  " | tee /tmp/compile-r$RANK.log | tail -5
  grep -q '"failed": \[\]' /tmp/compile-r$RANK.log || { echo "compile had failures (rank $RANK)"; exit 1; }

  # F2: rows = CONTEXT + 2048
  docker exec -e PYTHONPATH=/repo/build/python-source-102/src -w /repo/engine/zig dsv41-assets \
    /stage/.asset-venv/bin/python tools/zig/dsv41_rope_tables.py \
    --config /stage/flash17/config.json --rows $ROWS --out /stage/assets-r$RANK/rope.bin

  docker exec -e PYTHONPATH=/repo/build/python-source-102/src -w /repo/engine/zig dsv41-assets \
    /stage/.asset-venv/bin/python tools/zig/dsv41_engram_host.py \
    --py-src /repo/build/python-source-102/src --config /stage/flash17/config.json \
    --tokenizer /stage/flash17/tokenizer.json --out /stage/assets-r$RANK/engram-host.bin

  echo "rank $RANK assets: $(ls $ASSETS/aot/cubins 2>/dev/null | wc -l) cubins, rope $(stat -c%s $ASSETS/rope.bin) bytes"
done

echo "== ship rank-1 assets to the worker  [F3]"
rsync -a $BASE/assets-r1/ "$WORKER:$BASE/assets-r1/"
cp $REPO/config/pfdense-table.json $BASE/cache-r0/ 2>/dev/null || true
ssh -n "$WORKER" "mkdir -p $BASE/cache-r1 && cp $REPO/config/pfdense-table.json $BASE/cache-r1/" 2>/dev/null || \
  rsync -a $REPO/config/pfdense-table.json "$WORKER:$BASE/cache-r1/"

echo "ASSETS COMPLETE — next: scripts/run-zig.sh N --check-aot on EACH node, then scripts/start.sh"
