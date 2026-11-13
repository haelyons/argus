"""Stage 3: one pass over the sample, per split, everything at once.

out/s3/<split>/
  units.pt     n tokens; stat (N, 14, U) sums over tokens, rows = STAT;
               hi, lo (N, K, U) strongest examples at each end as values and token ids
  features.pt  stat (4, F) rows = FSTAT; hi (K, F) values and token ids
  sample.pt    a random set of tokens captured densely: tok (S,), acts (S, N, U) f16,
               feature values and indices (S, 64), residue token (S,)
  cond_LL.npy  (F, U) f16: mean activation of each unit of block LL over tokens where the feature fires
Token id = protein index * 1024 + position (position 0 is <cls>).
python -m argus.s3 --shard i/n     collect one shard
python -m argus.s3 --reduce        merge shards
"""

import argparse
import glob
import os

import numpy as np
import torch

from argus import data, net, sae
from argus.collect import extremes
from argus.runlog import Log
from argus.s1 import checks

OUT = "out/s3"
K, S, BUDGET = 32, 8192, 8192
STAT = ["a", "a2", "g", "g2", "v", "v2", "n_g+v+", "n_g+v-", "n_g-v+", "a_g+v+", "a_g+v-", "a_g-v+", "a_g-v-", "n_a>0"]
FSTAT = ["tokens", "proteins", "f", "f2"]


def collect(m, s, df, shard, out, log):
    bl = net.blocks(m)
    N, U, F, dev = len(bl), bl[0].ffn.ffn_hidden_size, s.W_enc.shape[1], "cuda"
    p = S / (df.Length.sum() + 2 * len(df))
    stat = torch.zeros(N, len(STAT), U, dtype=torch.float64, device=dev)
    fstat = torch.zeros(len(FSTAT), F, dtype=torch.float64, device=dev)
    tab = torch.zeros(N, F, U, device=dev)
    hi, lo, fhi, n = [None] * N, [None] * N, None, 0
    samp = dict(tok=[], acts=[], fval=[], fidx=[], res=[])

    for idx, ids in data.batches(df, BUDGET, shard):
        ids = ids.to(dev)
        keep = ids != net.PAD
        T = int(keep.sum())
        tok = (torch.from_numpy(idx).to(dev)[:, None] * 1024 + torch.arange(ids.shape[1], device=dev))[keep]
        pick = torch.rand(T, device=dev, generator=torch.Generator(dev).manual_seed(int(idx[0]))) < p
        held, acts, fs = [], [], None

        def tap(l, r):
            nonlocal fs, fhi
            g, v, a = (r[k][keep] for k in "gva")
            with log.op("unit_stats", T):
                gp, vp = g > 0, v > 0
                pp, pn, np_ = gp & vp, gp & ~vp, ~gp & vp
                rows = [a, a * a, g, g * g, v, v * v, pp, pn, np_, a * pp, a * pn, a * np_, a * ~(gp | vp), a > 0]
                stat[l] += torch.stack([x.sum(0, dtype=torch.float64) for x in rows])
            with log.op("unit_extremes", T):
                hi[l] = extremes(hi[l], a, tok, K)
                lo[l] = extremes(lo[l], -a, tok, K)
            acts.append(a[pick].half().cpu())
            if l < sae.LAYER:
                held.append(a.half())
            if l == sae.LAYER - 1:
                with log.op("features", T):
                    _, val, ind, _ = sae.encode(s, r["x"])
                    dense = torch.zeros(*ids.shape, F, device=dev).scatter(-1, ind, val * keep[..., None])
                    flat = dense[keep]
                    fstat[:] += torch.stack([(flat > 0).sum(0), (dense.amax(1) > 0).sum(0), flat.sum(0), (flat * flat).sum(0)])
                    fhi = extremes(fhi, flat, tok, K)
                    for k, x in zip(("tok", "fval", "fidx", "res"), (tok, val[keep], ind[keep], ids[keep])):
                        samp[k].append(x[pick].cpu())
                    fs = torch.sparse_coo_tensor(flat.nonzero().T.flip(0), torch.ones(int((flat > 0).sum()), device=dev), (F, T)).coalesce()
                with log.op("feature_by_unit_held", T):
                    for j, h in enumerate(held):
                        tab[j] += torch.sparse.mm(fs, h.float())
                held.clear()
            if l >= sae.LAYER:
                with log.op("feature_by_unit", T):
                    tab[l] += torch.sparse.mm(fs, a)

        with log.op("pass", T), torch.no_grad():
            net.run(m, ids, tap=tap)
        samp["acts"].append(torch.stack(acts, 1))
        n += T
        log.note(tokens=n)

    os.makedirs(out, exist_ok=True)
    with log.stage("save"):
        cut = lambda xs: (torch.stack([x[0] for x in xs]).cpu(), torch.stack([x[1] for x in xs]).int().cpu())
        lo = [(-v, t) for v, t in lo]
        torch.save(dict(n=n, stat=stat.cpu(), hi=cut(hi), lo=cut(lo), rows=STAT), f"{out}/units.pt")
        torch.save(dict(stat=fstat.cpu(), hi=(fhi[0].cpu(), fhi[1].int().cpu()), rows=FSTAT), f"{out}/features.pt")
        torch.save({k: torch.cat(v) for k, v in samp.items()}, f"{out}/sample.pt")
        for l in range(N):
            np.save(f"{out}/tab_{l:02d}.npy", tab[l].cpu().numpy())
    del tab, stat
    torch.cuda.empty_cache()


def merge_ext(parts, k):
    v, t = torch.cat([p[0] for p in parts], -2), torch.cat([p[1] for p in parts], -2)
    v, j = v.topk(k, -2)
    return v, t.gather(-2, j)


def merge_ext_lo(parts, k):
    v, t = merge_ext([(-p[0], p[1]) for p in parts], k)
    return -v, t


def reduce():
    log = Log("s3-reduce")
    for split in ("train", "holdout"):
        dirs = sorted(glob.glob(f"{OUT}/{split}/shard*"))
        out = f"{OUT}/{split}"
        us = [torch.load(f"{d}/units.pt") for d in dirs]
        torch.save(dict(n=sum(u["n"] for u in us), stat=sum(u["stat"] for u in us), rows=STAT,
                        hi=merge_ext([u["hi"] for u in us], K), lo=merge_ext_lo([u["lo"] for u in us], K)), f"{out}/units.pt")
        fs = [torch.load(f"{d}/features.pt") for d in dirs]
        fstat = sum(f["stat"] for f in fs)
        torch.save(dict(stat=fstat, hi=merge_ext([f["hi"] for f in fs], K), rows=FSTAT), f"{out}/features.pt")
        ss = [torch.load(f"{d}/sample.pt") for d in dirs]
        torch.save({k: torch.cat([s[k] for s in ss]) for k in ss[0]}, f"{out}/sample.pt")
        count = fstat[0].clamp(min=1).numpy()[:, None]
        for l in range(len(glob.glob(f"{dirs[0]}/tab_*.npy"))):
            tab = sum(np.load(f"{d}/tab_{l:02d}.npy").astype(np.float64) for d in dirs)
            np.save(f"{out}/cond_{l:02d}.npy", (tab / count).astype(np.float16))
        log.note(**{split: dict(shards=len(dirs), tokens=sum(u["n"] for u in us))})
    log.write()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--reduce", action="store_true")
    a = ap.parse_args()
    if a.reduce:
        return reduce()
    shard = tuple(map(int, a.shard.split("/")))
    log = Log("s3", shard)
    m, s = net.load(), sae.load()
    df = data.sample()
    ids = next(data.batches(df[df.split == "train"], BUDGET, shard))[1].cuda()
    log.plain(m, ids)
    with torch.no_grad():
        log.note(checks=checks(m, s, ids))
    torch.cuda.empty_cache()
    for split in ("train", "holdout"):
        with log.stage(split):
            collect(m, s, df[df.split == split], shard, f"{OUT}/{split}/shard{shard[0]}", log)
    log.cheaper("feature-by-unit table for blocks before the SAE",
                before="separate pass to block 60 to get features first: 0.75 forward equivalents (s1 forward_to_60)",
                after="hold blocks 0-59 activations as f16 on the GPU until features exist; see ops.feature_by_unit_held")
    log.write()


if __name__ == "__main__":
    main()
