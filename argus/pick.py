"""Targets for stages 6-9, and each target's on- and matched off-examples from stage 3.

targets.json   replace freely: {"features": [j, ...], "units": [[block, unit], ...]}
               default rule: 32 features evenly spaced in rank of firing frequency;
               one random unit in every third block.
examples.json  per target and split: [{"on": [protein, pos], "off": [protein, pos]}, ...]
               on  = strongest stage 3 examples, one per protein
               off = a stage 3 sample token with the same residue where the target is off
                     (feature not firing; unit at or below its sample median), nearest in position
python -m argus.pick
"""

import json
import os

import numpy as np
import torch

from argus import data

TARGETS, EXAMPLES = "targets.json", "examples.json"
E = 4


def name(kind, t):
    return f"f{t}" if kind == "features" else f"u{t[0]}_{t[1]}"


def write_targets():
    freq = torch.load("out/s3/train/features.pt")["stat"][0]
    feats = freq.argsort()[torch.linspace(0, len(freq) - 1, 32).long()].tolist()
    U = torch.load("out/s3/train/units.pt")["hi"][0].shape[-1]
    rng = np.random.default_rng(0)
    units = [[l, int(rng.integers(U))] for l in range(0, 80, 3)]
    json.dump(dict(features=feats, units=units), open(TARGETS, "w"))


def pairs(on_tok, off_ok, samp, seqs):
    out, seen = [], set()
    for tok in on_tok.tolist():
        p, pos = divmod(tok, 1024)
        if p in seen or not 1 <= pos <= len(seqs[p]):
            continue
        seen.add(p)
        res = data.TOK[ord(seqs[p][pos - 1])]
        ok = off_ok & (samp["res"] == res) & (samp["tok"] // 1024 != p)
        if not ok.any():
            continue
        cand = samp["tok"][ok]
        off = cand[(cand % 1024 - pos).abs().argmin()].item()
        out.append(dict(on=[p, pos], off=list(divmod(off, 1024))))
        if len(out) == E:
            break
    return out


def main():
    if not os.path.exists(TARGETS):
        write_targets()
    tg = json.load(open(TARGETS))
    seqs = data.sample().Sequence.tolist()
    ex = {}
    for split in ("train", "holdout"):
        samp = torch.load(f"out/s3/{split}/sample.pt")
        fhi = torch.load(f"out/s3/{split}/features.pt")["hi"][1]
        uhi = torch.load(f"out/s3/{split}/units.pt")["hi"][1]
        for j in tg["features"]:
            off_ok = ~((samp["fidx"] == j) & (samp["fval"] > 0)).any(1)
            ex.setdefault(name("features", j), {})[split] = pairs(fhi[:, j], off_ok, samp, seqs)
        for l, u in tg["units"]:
            a = samp["acts"][:, l, u].float()
            ex.setdefault(name("units", (l, u)), {})[split] = pairs(uhi[l, :, u], a <= a.median(), samp, seqs)
    json.dump(ex, open(EXAMPLES, "w"))
    print({k: {s: len(v) for s, v in d.items()} for k, d in ex.items()})


def targets(shard=(0, 1)):
    """[(name, kind, target, examples)] for this shard."""
    tg, ex = json.load(open(TARGETS)), json.load(open(EXAMPLES))
    all_ = [(name(k, t), k, t, ex[name(k, t)]) for k in ("features", "units") for t in tg[k]]
    return all_[shard[0] :: shard[1]]


def ids_of(df, protein):
    return torch.from_numpy(data.tokens(df.Sequence[protein]))[None].cuda()


def describe():
    """targets_features.csv: Biohub's description columns for the target features only."""
    import pandas as pd

    d = pd.read_parquet("data/features.parquet", columns=["feature_id", "category", "summary", "activation_pattern", "threshold", "uniref90_frequency", "uniref90_max_activation", "uniref90_idf"])
    d[d.feature_id.isin(json.load(open(TARGETS))["features"])].to_csv("targets_features.csv", index=False)


if __name__ == "__main__":
    main()
