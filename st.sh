#!/bin/bash
# st.sh <stage>: progress of the newest run of a stage on the cluster, from its run records.
ssh argus "cd sky_workdir; pgrep -fc '[p]ython -m argus'; tail -qn 1 logs/$1-*-0.out | tail -n 1 | cut -c1-240; python3 - $1" <<'PY' 2>&1 | grep -v Warning
import json, glob, sys
fs = sorted(glob.glob(f"logs/{sys.argv[1]}-*of*.json"))
for f in [g for g in fs if g.split("-")[1][:13] >= fs[-1].split("-")[1][:11]][-8:]:
    r = json.load(open(f))
    print(f[5:], r["notes"].get("tokens"), round(r["seconds"]), {k: round(v.get("fwd_equiv", 0), 2) for k, v in r["ops"].items()},
          {k: round(v) for k, v in r["stages"].items()}, round(r["peak_gpu_gb"], 1))
PY
