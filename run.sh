#!/bin/bash
# run.sh <stage> [args]: one process per GPU, each taking shard i/n. Output kept under logs/.
cd "$(dirname "$0")" && source ~/venv/bin/activate && mkdir -p logs out
n=${GPUS:-$(nvidia-smi -L | wc -l)}
t=$(date -u +%Y%m%dT%H%M%S)
for i in $(seq 0 $((n - 1))); do
  CUDA_VISIBLE_DEVICES=$i python -m argus.$1 --shard $i/$n "${@:2}" > logs/$1-$t-$i.out 2>&1 &
done
wait
tail -n 3 logs/$1-$t-*.out
