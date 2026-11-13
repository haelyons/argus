#!/bin/bash
# chain.sh <running-stage> <next> [next...]: when the running stage exits, run the next ones in order.
cd "$(dirname "$0")"
while pgrep -f "argus\.$1 " > /dev/null; do sleep 10; done
for s in "${@:2}"; do ./run.sh $s > /dev/null 2>&1; done
