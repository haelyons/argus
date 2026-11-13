#!/bin/bash
# bg.sh <stage> [args]: start run.sh detached; bg.sh -r <module> [args]: one detached CPU process.
cd "$(dirname "$0")" && source ~/venv/bin/activate && mkdir -p logs
if [ "$1" = -r ]; then
  setsid nohup python -m argus.$2 "${@:3}" > logs/$2-cpu-$(date -u +%Y%m%dT%H%M%S).out 2>&1 < /dev/null &
else
  setsid nohup ./run.sh "$@" > /dev/null 2>&1 < /dev/null &
fi
