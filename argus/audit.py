"""The labels every residue carries, and whether the residues where a detector is strongest share one.

Families: residue (amino acid), annotation (Swiss-Prot feature columns), pfam, feature (SAE features
firing there, from s_audit's codes). A share is scored above chance as (share - base) / (1 - base),
base being the share of all residues that carry the label.
"""

import re

import numpy as np
import pandas as pd
import torch

from argus import data

PF, CHUNK = 8, 8192
ANN = ["Active site", "Binding site", "Site", "Domain [FT]", "Region", "Motif", "Repeat", "Zinc finger", "Transmembrane",
       "Signal peptide", "Transit peptide", "Disulfide bond", "Modified residue", "Helix", "Beta strand", "Turn"]
FAMS = ["residue", "annotation", "pfam", "feature"]


def spans(cell):
    """1-based (start, end) of each feature in a UniProt feature cell with a plain location."""
    for loc in re.findall(r"(?:^|; )[A-Z_]+ ([^;]+)", cell):
        m = re.fullmatch(r"(\d+)(?:\.\.(\d+))?", loc.strip())
        if m:
            yield int(m[1]), int(m[2] or m[1])


def labels(input, dev):
    """Per token, in protein order: every label it carries, by family, padded with -1. Token (p, pos) is row off[p] + pos."""
    df = pd.read_parquet(data.path(input, "sample.parquet"))
    n = df.Sequence.str.len().values + 2
    off = np.concatenate([[0], np.cumsum(n)])
    bits = np.zeros(off[-1], np.int64)
    for b, col in enumerate(ANN):
        for p, cell in enumerate(df[col]):
            if not isinstance(cell, str):
                continue
            for a, e in spans(cell):
                if col == "Disulfide bond":
                    bits[off[p] + [a, e]] |= 1 << b
                else:
                    bits[off[p] + a : off[p] + e + 1] |= 1 << b
    bits = torch.from_numpy(bits)
    ann = torch.where((bits[:, None] >> torch.arange(len(ANN))) & 1 > 0, torch.arange(len(ANN)), -1)
    fams = [sorted(set(x.split(";")) - {""}) if isinstance(x, str) else [] for x in df.Pfam]
    names = sorted({f for fs in fams for f in fs})
    code = {f: i for i, f in enumerate(names)}
    pf = torch.full((len(df), PF), -1)
    for p, fs in enumerate(fams):
        pf[p, : len(fs[:PF])] = torch.tensor([code[f] for f in fs[:PF]], dtype=torch.long)
    res = torch.from_numpy(np.concatenate([data.tokens(s) for s in df.Sequence]))[:, None]
    lab = dict(residue=res, annotation=ann, pfam=pf.repeat_interleave(torch.from_numpy(n), 0))
    return {k: v.to(dev) for k, v in lab.items()}, torch.from_numpy(off).to(dev), dict(annotation=ANN, pfam=names)


def flat(tok, off):
    return off[tok.long() // data.STRIDE] + tok.long() % data.STRIDE


def features(lab, off, files):
    """Adds the feature family from s_audit's per-shard codes."""
    parts = [torch.load(f) for f in files]
    dev = off.device
    feat = torch.full((len(lab["residue"]), parts[0]["fidx"].shape[1]), -1, dtype=torch.int16, device=dev)
    for c in parts:
        feat[flat(c["tok"].to(dev), off)] = c["fidx"].to(dev)
    lab["feature"] = feat


def top_label(ids, expect):
    """ids (m, n), -1 = none. expect (labels,): the count chance gives. The label each row has most above chance, and its count."""
    cnt = torch.zeros(len(ids), len(expect) + 1, device=ids.device).scatter_add_(1, ids + 1, torch.ones_like(ids, dtype=torch.float))[:, 1:]
    lb = (cnt - expect).argmax(1)
    c = cnt.gather(1, lb[:, None])[:, 0]
    return torch.where(c > 0, lb, -1), c


def share(ids, lab):
    """ids (m, n, W), lab (m,). Fraction of each row's n tokens that carry its label."""
    return ((ids == lab[:, None, None]) & (lab[:, None, None] >= 0)).any(-1).float().mean(1)


def table(tok, lab, off):
    """tok: train, holdout, mid -> (rows, n) token ids of each row's strongest residues. Per family: the label
    of the train residues, the share of each set that carries it, and its base rate."""
    T, K = len(lab["residue"]), tok["train"].shape[1]
    base = {k: torch.bincount(v[v >= 0].long().flatten()).float() / T for k, v in lab.items()}
    cols = {f"{f}_{c}": [] for f in FAMS for c in ("label", "train", "holdout", "mid", "base")}
    for i in range(0, len(tok["train"]), CHUNK):
        fl = {k: flat(v[i : i + CHUNK].to(off.device), off) for k, v in tok.items()}
        for f in FAMS:
            lb, c = top_label(lab[f][fl["train"]].flatten(1).long(), K * base[f])
            cols[f"{f}_label"].append(lb.cpu())
            cols[f"{f}_train"].append((c / K).cpu())
            cols[f"{f}_holdout"].append(share(lab[f][fl["holdout"]], lb).cpu())
            cols[f"{f}_mid"].append(share(lab[f][fl["mid"]], lb).cpu())
            cols[f"{f}_base"].append(torch.where(lb >= 0, base[f][lb.clamp(min=0)], 0).cpu())
    return {k: torch.cat(v) for k, v in cols.items()}


def above(cols, f, split):
    return (cols[f"{f}_{split}"] - cols[f"{f}_base"]) / (1 - cols[f"{f}_base"])


def best(cols, split):
    """Each row's share above chance in its best family."""
    return torch.stack([above(cols, f, split) for f in FAMS]).max(0).values


def at(x):
    return {t: round((x >= t).float().mean().item(), 4) for t in (0.5, 0.75, 0.9)}


def clean(cols):
    """Share of rows at or above each threshold, on holdout and on the random-sample set."""
    return dict(above_chance="(share - base) / (1 - base)", any=at(best(cols, "holdout")), any_mid=at(best(cols, "mid")),
                **{f: dict(holdout=at(above(cols, f, "holdout")), mid=at(above(cols, f, "mid"))) for f in FAMS})
