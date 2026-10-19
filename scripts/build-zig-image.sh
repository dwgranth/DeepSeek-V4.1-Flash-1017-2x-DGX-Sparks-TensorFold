#!/bin/bash
# Build the Zig engine image on an ARM64 node (the GB10 head works) and ship it to the worker.
set -euo pipefail
REPO=${REPO:-$HOME/deepseek-v41-tensorfold-spark}
WORKER=${WORKER:-dwgranth@10.100.128.2}
BASE=${BASE:-$HOME/dsv41-tf-zig}
IMG=${IMG:-tensorfold-dsv41-zig:local}

cd "$REPO"
mkdir -p build/zig/src build/zig/fatbins
git archive HEAD engine/zig | tar -x --strip-components=2 -C build/zig/src/
touch build/zig/fatbins/.keep
docker build -f docker/zig/Dockerfile --build-arg ZIG_COMMIT="$(git rev-parse HEAD)" -t "$IMG" build/zig
scripts/check-zig-image.sh "$IMG"
echo "imageID: $(docker image inspect "$IMG" --format '{{.Id}}')"
mkdir -p "$BASE/build"
docker save "$IMG" -o "$BASE/build/zig-image.tar"
rsync -a "$BASE/build/zig-image.tar" "$WORKER:$BASE/build/"
ssh -n "$WORKER" "docker load -i $BASE/build/zig-image.tar && docker image inspect $IMG --format '{{.Id}}'"
rm -f "$BASE/build/zig-image.tar"
ssh -n "$WORKER" "rm -f $BASE/build/zig-image.tar"
echo "verify the two imageIDs match."
