#!/bin/bash
# Build the Python reference source the AOT fill needs, for the jayleaton
# deepseek-v41-tensorfold-spark Zig engine (exported from a TensorFold 1.0.2 kit snapshot).
#
# WHY NOT vendor/TensorFold (0.6.0) as upstream's docs say:
#   The engine snapshot is based on TensorFold 1.0.2 (f8fe17d). The recipe's
#   patches/0001-0003 were largely upstreamed INTO 1.0.2, so against 1.0.2 the
#   patches reduce to exactly what is still missing (the dsv41 family new files
#   + 0003). Against 0.6.0 the sigs end up missing/mismatching kernels the
#   1.0.2-based engine launches -> `tf-dsv41-m1 aot-needs` fails NoKernel.
# Additionally 1.0.2 removed the EXL3 Triton prefill `_gemm` (moved to CUDA),
# but the engine still enumerates `_gemm` fallback variants; fixes/ grafts it
# back at its ORIGINAL module path (tensorfold.cuda.exl3.prefill) so the
# recipe's existing options rule in tools/zig/dsv41_triton_options.json applies.
#
# Usage: scripts/prepare-python-reference.sh [REPO_DIR]
set -euo pipefail
REPO=${1:-$HOME/deepseek-v41-tensorfold-spark}
TF102=f8fe17d24629aedabf90bbf78279dd776e6d62e7   # TensorFold 1.0.2 (see engine/zig/SOURCE-EXPORT.md)
SRC=$REPO/build/python-source-102

cd "$REPO"
mkdir -p build
rm -rf "$SRC"
mkdir -p "$SRC"
git -C vendor/TensorFold fetch --tags --quiet 2>/dev/null || true
git -C vendor/TensorFold archive "$TF102" | tar -x -C "$SRC"

echo "== applying recipe patches with --forward (already-upstream hunks skip; new files land)"
for p in patches/*.patch; do
  patch -d "$SRC" -p1 --forward --silent < "$p" || echo "  (skipped already-upstream hunks in $p)"
done

echo "== grafting _gemm (TensorFold 0.6.0 exl3 prefill) at original module path"
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$SRC/src/tensorfold/cuda/exl3"
touch "$SRC/src/tensorfold/cuda/__init__.py" "$SRC/src/tensorfold/cuda/exl3/__init__.py"
cp "$HERE/fixes/tf102-exl3-prefill/prefill.py" "$SRC/src/tensorfold/cuda/exl3/prefill.py"

test -f "$SRC/src/tensorfold/families/deepseek_v41/cuda/config.py" || { echo "dsv41 family missing"; exit 1; }
grep -q "def _gemm" "$SRC/src/tensorfold/cuda/exl3/prefill.py"
echo "OK: reference source ready at $SRC"
echo "   use PY_SOURCE=$SRC/src for triton_fill.py / rope / engram-host"
