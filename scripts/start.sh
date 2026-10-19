#!/bin/bash
# Launch the 2-rank Zig serve. Rank 1 (worker) FIRST, then rank 0 (head).
#
# GOTCHA learned the hard way: launch each rank in its own top-level ssh
# invocation. A bash script that ssh-es the worker and then launches locally
# can die right after the worker ssh returns (nested-ssh stdin/channel theft).
set -euo pipefail
REPO=${REPO:-$HOME/deepseek-v41-tensorfold-spark}
WORKER=${WORKER:-dwgranth@10.100.128.2}
BASE=${BASE:-$HOME/dsv41-tf-zig}

ssh -n "$WORKER" "cd $REPO && setsid nohup scripts/run-zig.sh 1 > $BASE/zig-r1.log 2>&1 < /dev/null & disown; echo rank1-launched"
sleep 5
cd "$REPO"
setsid nohup scripts/run-zig.sh 0 > "$BASE/zig-r0.log" 2>&1 < /dev/null &
disown -a
echo "rank0-launched; first boot writes prepared files (~95GB/node) + calib-zig; later boots ~1 min"
echo "watch: tail -f $BASE/zig-r0.log   until: [tensorfold] serving DeepSeek-V4.1-Flash-TF"
