"""Do the residues where a unit is most positive, or most negative, share one thing: an amino acid, a
Swiss-Prot annotation, a Pfam family or an SAE feature? Does the thing its strongest train residues
share also hold at its strongest holdout residues, and at its top 64 of a random holdout sample?
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch

from argus import data, net, runlog, sae

BUDGET, MID, PF, CHUNK = 8192, 64, 8, 8192
ANN = ["Active site", "Binding site", "Site", "Domain [FT]", "Region", "Motif", "Repeat", "Zinc finger", "Transmembrane",
       "Signal peptide", "Transit peptide", "Disulfide bond", "Modified residue", "Helix", "Beta strand", "Turn"]
FAMS = ["residue", "annotation", "pfam", "feature"]


def codes(m, s, df, log):
    """Which SAE features fire at every token of this shard: tok (T,), fidx (T, k) with -1 where off."""
    tok, fidx = [], []
    for idx, ids in data.batches(df, BUDGET, log.shard):
        ids = ids.cuda()
        if log.rec["tok_s"] is None:
            log.plain(m, ids)
        keep = ids != net.PAD
        with log.op("pass_to_60", int(keep.sum())), torch.no_grad():
            _, val, ind, _ = sae.encode(s, net.run(m, ids, stop=sae.LAYER)[keep])
        t = torch.from_numpy(idx).cuda()[:, None] * data.STRIDE + torch.arange(ids.shape[1], device="cuda")
        tok.append(t[keep].int().cpu())
        fidx.append(torch.where(val > 0, ind, -1).short().cpu())
        log.note(tokens=sum(map(len, tok)))
    os.makedirs(f"{log.dir}/shard{log.shard[0]}", exist_ok=True)
    torch.save(dict(tok=torch.cat(tok), fidx=torch.cat(fidx)), f"{log.dir}/shard{log.shard[0]}/codes.pt")


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


def top_label(ids):
    """ids (m, n), -1 = none. The most common label of each row and how many times it occurs."""
    s = ids.sort(1).values
    new = torch.ones_like(s, dtype=torch.bool)
    new[:, 1:] = s[:, 1:] != s[:, :-1]
    run = new.cumsum(1) - 1
    cnt = torch.zeros_like(s).scatter_add_(1, run, (s >= 0).long())
    c, r = cnt.max(1)
    first = (run == r[:, None]).float().argmax(1)
    return s.gather(1, first[:, None])[:, 0], c


def share(ids, lab):
    """ids (m, n, W), lab (m,). Fraction of each row's n tokens that carry its label."""
    return ((ids == lab[:, None, None]) & (lab[:, None, None] >= 0)).any(-1).float().mean(1)


def reduce(log):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    collect = log.read("collect")
    lab, off, names = labels(log.input, dev)
    flat = lambda tok: off[tok.long() // data.STRIDE] + tok.long() % data.STRIDE
    parts = [torch.load(f) for f in sorted(glob.glob(f"{log.dir}/shard*/codes.pt"))]
    feat = torch.full((len(lab["residue"]), parts[0]["fidx"].shape[1]), -1, dtype=torch.int16, device=dev)
    for c in parts:
        feat[flat(c["tok"].to(dev))] = c["fidx"].to(dev)
    lab["feature"] = feat
    T = len(feat)
    base = {k: torch.bincount(v[v >= 0].long().flatten()).float() / T for k, v in lab.items()}

    tr, ho = (torch.load(f"{collect}/{s}/units.pt") for s in ("train", "holdout"))
    N, K, U = tr["hi"][1].shape
    rows = lambda x: x.permute(0, 2, 1).reshape(N * U, -1)
    pole = lambda u, k: torch.cat([rows(u["hi"][k]), rows(u["lo"][k])])
    tok = dict(train=pole(tr, 1), holdout=pole(ho, 1))
    samp = torch.load(f"{collect}/holdout/sample.pt")
    mid = [[], []]
    for l in range(N):
        a = samp["acts"][:, l].to(dev).float()
        for i, sign in enumerate((1, -1)):
            v, j = (sign * a).topk(MID, 0)
            mid[i].append((samp["tok"][j.T.cpu()], sign * v[-1].cpu()))
    tok["mid"] = torch.cat([torch.cat([t for t, _ in m]) for m in mid])
    out = dict(block=torch.arange(N).repeat_interleave(U).repeat(2), unit=torch.arange(U).repeat(2 * N),
               pole=torch.tensor([1, -1]).repeat_interleave(N * U), peak=pole(tr, 0)[:, 0], edge_train=pole(tr, 0)[:, -1],
               edge_holdout=pole(ho, 0)[:, -1], edge_mid=torch.cat([torch.cat([v for _, v in m]) for m in mid]),
               proteins_train=(tok["train"] // data.STRIDE).sort(1).values.diff(dim=1).ne(0).sum(1) + 1)
    cols = {f"{f}_{c}": [] for f in FAMS for c in ("label", "train", "holdout", "mid", "base")}
    for i in range(0, 2 * N * U, CHUNK):
        fl = {k: flat(v[i : i + CHUNK].to(dev)) for k, v in tok.items()}
        for f in FAMS:
            lb, c = top_label(lab[f][fl["train"]].flatten(1).long())
            cols[f"{f}_label"].append(lb.cpu())
            cols[f"{f}_train"].append((c / K).cpu())
            cols[f"{f}_holdout"].append(share(lab[f][fl["holdout"]], lb).cpu())
            cols[f"{f}_mid"].append(share(lab[f][fl["mid"]], lb).cpu())
            cols[f"{f}_base"].append(torch.where(lb >= 0, base[f][lb.clamp(min=0)], 0).cpu())
    out |= {k: torch.cat(v) for k, v in cols.items()}
    pd.DataFrame({k: v.numpy() for k, v in out.items()}).to_parquet(f"{log.dir}/audit.parquet")
    json.dump(names, open(f"{log.dir}/labels.json", "w"))
    best = {b: torch.stack([out[f"{f}_{b}"] for f in FAMS]).max(0).values for b in ("holdout", "mid")}
    at = lambda x: {t: round((x >= t).float().mean().item(), 4) for t in (0.5, 0.75, 0.9)}
    log.note(poles=2 * N * U, tokens=T, any=at(best["holdout"]), any_mid=at(best["mid"]),
             **{f: dict(holdout=at(out[f"{f}_holdout"]), mid=at(out[f"{f}_mid"])) for f in FAMS})
    log.write()
    print(open(log.path).read())


def main():
    log = runlog.start("audit", __doc__)
    if log.reduce:
        return reduce(log)
    codes(net.load(), sae.load(), data.sample(log.input), log)
    log.write()


if __name__ == "__main__":
    main()
