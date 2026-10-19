#!/bin/bash
# Download the flash17 pack + the 1017 Engram source shards, verify, pack per-rank
# Engram bins, and ship the pack + rank-1 assets to the worker.
#
# Env overrides: WORKER (ssh dst), BASE (staging root), PACK_REPO, ENGRAM_REPO
# Assumes: head can ssh the worker (fabric or LAN); hf CLI on PATH; ~400GB disk head-side.
set -euo pipefail
WORKER=${WORKER:-dwgranth@10.100.128.2}          # worker from the HEAD (fabric unreachable from outside)
BASE=${BASE:-$HOME/dsv41-tf-zig}
PACK_REPO=${PACK_REPO:-jayleaton/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw}
ENGRAM_REPO=${ENGRAM_REPO:-deepseek-ai/DeepSeek-V4.1-Flash-1017}
REPO=${REPO:-$HOME/deepseek-v41-tensorfold-spark}

mkdir -p $BASE/{flash17,engram-src-1017,engram-r0,engram-r1,assets-r0,assets-r1,prepared-r0,prepared-r1,state-r0,state-r1,cache-r0,cache-r1,build}

echo "== 1. pack download + SHA verify"
hf download "$PACK_REPO" --local-dir $BASE/flash17
( cd $BASE/flash17 && sha256sum -c SHA256SUMS | grep -c ": OK$" )

echo "== 2. engram source (shards named by --list; the September shards do NOT match this pack)"
hf download "$ENGRAM_REPO" model.safetensors.index.json --local-dir $BASE/engram-src-1017
SHARDS=$(python3 $REPO/scripts/pack_engram.py --src $BASE/engram-src-1017 --list | tr '\n' ' ')
echo "   shards: $SHARDS"
hf download "$ENGRAM_REPO" $SHARDS --local-dir $BASE/engram-src-1017

echo "== 3. pack per-rank bins + check"
for RANK in 0 1; do
  python3 $REPO/scripts/pack_engram.py --src $BASE/engram-src-1017 --config $BASE/flash17/config.json \
    --out $BASE/engram-r$RANK --rank $RANK
  python3 $REPO/scripts/pack_engram.py --src $BASE/engram-src-1017 --config $BASE/flash17/config.json \
    --check $BASE/engram-r$RANK --rank $RANK
done

echo "== 4. ship pack + rank-1 engram to the worker (byte-identical pack required)"
rsync -a $BASE/flash17/ "$WORKER:$BASE/flash17/"
rsync -a $BASE/engram-r1/ "$WORKER:$BASE/engram-r1/"
echo OK
