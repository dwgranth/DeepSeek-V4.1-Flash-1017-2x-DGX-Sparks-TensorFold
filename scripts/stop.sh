#!/bin/bash
# Graceful stop: signal rank 0 only (DRAIN_S=20 for in-flight requests); rank 1
# exits when orphaned. The launcher uses --rm, so containers clean themselves up.
set -euo pipefail
WORKER=${WORKER:-dwgranth@10.100.128.2}
docker stop -t 80 tensorfold-zig-r0
for i in $(seq 1 24); do
  R1=$(ssh -n "$WORKER" "docker ps -q --filter name=tensorfold-zig-r1 | wc -l")
  [ "$R1" = "0" ] && { echo "rank 1 exited"; exit 0; }
  sleep 5
done
echo "rank 1 still up after 2 min; check it on the worker"
exit 1
