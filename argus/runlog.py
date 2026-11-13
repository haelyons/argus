"""One standard record per run, and where a run's data lives.

A run is one operation on one input at one commit: out/<input>/<operation>/<date>-<commit>/.
Each process writes run-<shard>.json there:
  question     the script's own first lines
  upstream     the run directories it read
  tok_s        plain forward tokens/second on this GPU
  ops          per operation: seconds, tokens, and cost in plain-forward-pass equivalents.
               An op entered k times per pass is given tokens / k each time.
  stages       seconds by named span
  cheaper      before/after pairs, when an operation was replaced by a cheaper one
python -m argus.runlog          writes out/runs.jsonl, one row per run, keeping rows already there
"""

import argparse
import glob
import json
import os
import resource
import sys
import time
from contextlib import contextmanager

import torch

from argus import net

RUN = open("COMMIT").read().strip() if os.path.exists("COMMIT") else "uncommitted"


def latest(input, op):
    return sorted(glob.glob(f"out/{input}/{op}/*"))[-1]


def start(op, doc, input=None):
    """Parses --input, --shard i/n and --reduce, and opens the run's log."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="swissprot-random")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--reduce", action="store_true")
    a = ap.parse_args()
    return Log(op, input or a.input, tuple(map(int, a.shard.split("/"))), doc, a.reduce)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Log:
    def __init__(self, op, input, shard, doc, reduce=False):
        self.t0, self.input, self.shard, self.reduce = time.time(), input, shard, reduce
        self.dir = f"out/{input}/{op}/{RUN}"
        os.makedirs(self.dir, exist_ok=True)
        self.path = f"{self.dir}/run-{'reduce' if reduce else f'{shard[0]}of{shard[1]}'}.json"
        gpu = torch.cuda.is_available()
        self.rec = dict(
            op=op, input=input, run=RUN, question=" ".join(doc.split()), shard=shard, argv=sys.argv, upstream=[],
            tok_s=None, ops={}, stages={}, cheaper=[], notes={},
            gpu_type=torch.cuda.get_device_name(0) if gpu else None,
            gpu_count=shard[1],
        )

    def read(self, op, input=None):
        """The newest run directory of an upstream operation; recorded as read."""
        d = latest(input or self.input, op)
        if d not in self.rec["upstream"]:
            self.rec["upstream"].append(d)
        return d

    def plain(self, m, ids):
        """Times a plain forward pass; every op cost is expressed against it."""
        with torch.no_grad():
            net.logits(m, net.run(m, ids))
            _sync()
            t = time.time()
            net.logits(m, net.run(m, ids))
            _sync()
        self.rec["tok_s"] = ids.numel() / (time.time() - t)

    @contextmanager
    def op(self, name, tokens):
        _sync()
        t = time.time()
        yield
        _sync()
        o = self.rec["ops"].setdefault(name, dict(seconds=0.0, tokens=0))
        o["seconds"] += time.time() - t
        o["tokens"] += tokens

    @contextmanager
    def stage(self, name):
        t = time.time()
        yield
        self.rec["stages"][name] = self.rec["stages"].get(name, 0.0) + time.time() - t

    def cheaper(self, what, before, after):
        self.rec["cheaper"].append(dict(what=what, before=before, after=after))

    def note(self, **kw):
        self.rec["notes"].update(kw)
        self.write()

    def write(self):
        r = self.rec
        for o in r["ops"].values():
            if r["tok_s"] and o["tokens"]:
                o["fwd_equiv"] = o["seconds"] / o["tokens"] * r["tok_s"]
        r["seconds"] = time.time() - self.t0
        r["peak_host_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        r["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
        with open(self.path, "w") as f:
            json.dump(r, f, indent=1, default=str)


def index():
    rows = []
    for d in sorted(glob.glob("out/*/*/*/")):
        rs = [json.load(open(f)) for f in sorted(glob.glob(d + "run-*.json"))]
        if not rs:
            continue
        sh = [r for r in rs if r.get("tok_s")]
        ops = {k for r in sh for k in r["ops"]}
        mean = lambda xs: sum(xs) / len(xs) if xs else None
        rows.append(dict(
            input=rs[0]["input"], op=rs[0]["op"], run=rs[0]["run"], path=d.rstrip("/")[4:], question=rs[0]["question"],
            upstream=sorted({u[4:] for r in rs for u in r["upstream"]}),
            gpu_type=rs[0]["gpu_type"], gpu_count=max(r["gpu_count"] for r in rs),
            seconds=max(r.get("seconds", 0) for r in rs), tok_s=mean([r["tok_s"] for r in sh]),
            fwd_equiv={k: mean([r["ops"][k]["fwd_equiv"] for r in sh if "fwd_equiv" in r["ops"].get(k, {})]) for k in sorted(ops)},
            peak_gpu_gb=max(r.get("peak_gpu_gb", 0) for r in rs), peak_host_gb=max(r.get("peak_host_gb", 0) for r in rs),
            notes={k: v for r in rs for k, v in r["notes"].items()}, cheaper=[c for r in rs for c in r["cheaper"]],
        ))
    if os.path.exists("out/runs.jsonl"):
        new = {r["path"] for r in rows}
        rows = [r for r in map(json.loads, open("out/runs.jsonl")) if r["path"] not in new] + rows
    with open("out/runs.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


if __name__ == "__main__":
    index()
