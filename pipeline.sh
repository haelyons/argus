#!/bin/bash
# pipeline.sh <input>: every operation on one input, in dependency order. Stops at the first failure.
set -e
cd "$(dirname "$0")" && source ~/venv/bin/activate
i="--input $1"
GPUS=1 ./run.sh s_checks $i
GPUS=1 ./run.sh s_weights
./run.sh s_collect $i && python -m scripts.s_collect $i --reduce
python -m argus.pick $1
./run.sh s_direct_shares $i && python -m scripts.s_direct_shares $i --reduce
python -m scripts.s_compare_peaks $i
for s in s_gradients s_group_edits s_readers s_feature_edits; do ./run.sh $s $i; done
python -m argus.runlog
