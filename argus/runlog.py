"""One standard record per run, written to logs/<stage>-<utc>-<shard>.json.

  tok_s        plain forward tokens/second on this GPU
  ops          per operation: seconds, tokens, and cost in plain-forward-pass equivalents
  stages       seconds by named span
  cheaper      before/after pairs, when an operation was replaced by a cheaper one
"""

import json
import resource
import sys
import time
from contextlib import contextmanager

import torch

from argus import net


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Log:
    def __init__(self, stage, shard=(0, 1)):
        self.t0 = time.time()
        self.path = f"logs/{stage}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{shard[0]}of{shard[1]}.json"
        gpu = torch.cuda.is_available()
        self.rec = dict(
            stage=stage, shard=shard, argv=sys.argv, tok_s=None, ops={}, stages={}, cheaper=[], notes={},
            gpu_type=torch.cuda.get_device_name(0) if gpu else None,
            gpu_count=shard[1],
        )

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
