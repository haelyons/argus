#!/bin/bash
# run.sh <script> [args]: one process per GPU, each taking shard i/n. Fails if any shard fails.
cd "$(dirname "$0")" && source ~/venv/bin/activate && mkdir -p logs out
n=${GPUS:-$(nvidia-smi -L | wc -l)}
t=$(date -u +%Y%m%dT%H%M%S)
for i in $(seq 0 $((n - 1))); do
  CUDA_VISIBLE_DEVICES=$i python -m scripts.$1 --shard $i/$n "${@:2}" > logs/$1-$t-$i.out 2>&1 &
  pids="$pids $!"
done
for p in $pids; do wait $p || fail=1; done
tail -n 3 logs/$1-$t-*.out
exit ${fail:-0}
