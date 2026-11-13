"""Targets for the per-target scripts, and each target's on- and matched off-examples from s_collect.

inputs/<name>/targets.json   replace freely: {"features": [j, ...], "units": [[block, unit], ...]}
               default rule: 32 features evenly spaced in rank of firing frequency;
               one random unit in every third block.
inputs/<name>/examples.json  per target and split: [{"on": [protein, pos], "off": [protein, pos]}, ...]
               on  = strongest s_collect examples, one per protein
               off = an s_collect sample token with the same residue where the target is off
                     (feature not firing; unit at or below its sample median), nearest in position
python -m argus.pick <name>
"""

import json
import os
import sys

import numpy as np
import torch

from argus import data, runlog

E = 4


def name(kind, t):
    return f"f{t}" if kind == "features" else f"u{t[0]}_{t[1]}"


def write_targets(input, collect):
    freq = torch.load(f"{collect}/train/features.pt")["stat"][0]
    feats = freq.argsort()[torch.linspace(0, len(freq) - 1, 32).long()].tolist()
    U = torch.load(f"{collect}/train/units.pt")["hi"][0].shape[-1]
    rng = np.random.default_rng(0)
    units = [[l, int(rng.integers(U))] for l in range(0, 80, 3)]
    json.dump(dict(features=feats, units=units), open(data.path(input, "targets.json"), "w"))


def pairs(on_tok, off_ok, samp, seqs):
    out, seen = [], set()
    for tok in on_tok.tolist():
        p, pos = divmod(tok, data.STRIDE)
        if p in seen or not 1 <= pos <= len(seqs[p]):
            continue
        seen.add(p)
        res = data.TOK[ord(seqs[p][pos - 1])]
        ok = off_ok & (samp["res"] == res) & (samp["tok"] // data.STRIDE != p)
        if not ok.any():
            continue
        cand = samp["tok"][ok]
        off = cand[(cand % data.STRIDE - pos).abs().argmin()].item()
        out.append(dict(on=[p, pos], off=list(divmod(off, data.STRIDE))))
        if len(out) == E:
            break
    return out


def main(input):
    collect = runlog.latest(input, "collect")
    if not os.path.exists(data.path(input, "targets.json")):
        write_targets(input, collect)
    tg = json.load(open(data.path(input, "targets.json")))
    seqs = data.sample(input).Sequence.tolist()
    ex = {}
    for split in ("train", "holdout"):
        samp = torch.load(f"{collect}/{split}/sample.pt")
        fhi = torch.load(f"{collect}/{split}/features.pt")["hi"][1]
        uhi = torch.load(f"{collect}/{split}/units.pt")["hi"][1]
        for j in tg["features"]:
            off_ok = ~((samp["fidx"] == j) & (samp["fval"] > 0)).any(1)
            ex.setdefault(name("features", j), {})[split] = pairs(fhi[:, j], off_ok, samp, seqs)
        for l, u in tg["units"]:
            a = samp["acts"][:, l, u].float()
            ex.setdefault(name("units", (l, u)), {})[split] = pairs(uhi[l, :, u], a <= a.median(), samp, seqs)
    json.dump(ex, open(data.path(input, "examples.json"), "w"))
    print({k: {s: len(v) for s, v in d.items()} for k, d in ex.items()})


def targets(input, shard=(0, 1)):
    """[(name, kind, target, examples)] for this shard."""
    tg, ex = (json.load(open(data.path(input, f))) for f in ("targets.json", "examples.json"))
    all_ = [(name(k, t), k, t, ex[name(k, t)]) for k in ("features", "units") for t in tg[k]]
    return all_[shard[0] :: shard[1]]


def ids_of(df, protein):
    return torch.from_numpy(data.tokens(df.Sequence[protein]))[None].cuda()


if __name__ == "__main__":
    main(sys.argv[1])
