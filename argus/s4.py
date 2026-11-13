"""Stage 4: direct shares of each feature's input, averaged over the tokens where it fires.

pre = (state[60] @ W) / sd - c, and state[60] = embedding + sum of every head and unit write
before block 60, so pre splits exactly into one share per unit, per head and the embedding.
Shares see only writers at the same position; what attention moved shows up as head shares.

out/s4/<split>/
  unit_LL.npy   (F, U) f16   mean share of each unit of block LL
  parts.pt      head (F, 60, H), embed (F,), const (F,), pre (F,) mean input when firing,
                tokens (F,), base (60, U) mean of activation/sd over all tokens
                check: head.sum + unit.sum + embed - const == pre
python -m argus.s4 --shard i/n | --reduce
"""

import argparse
import glob
import os

import numpy as np
import torch

from argus import data, net, sae
from argus.runlog import Log

OUT = "out/s4"
BUDGET = 8192


def collect(m, s, df, shard, out, log):
    bl = net.blocks(m)
    L, U, H, F, dev = sae.LAYER, bl[0].ffn.ffn_hidden_size, bl[0].attn.n_heads, s.W_enc.shape[1], "cuda"
    W = sae.read_weight(s)
    unit = torch.zeros(L, F, U, device=dev)
    head = torch.zeros(L, F, W.shape[0], dtype=torch.float64, device=dev)
    emb = torch.zeros(F, W.shape[0], dtype=torch.float64, device=dev)
    base = torch.zeros(L, U, dtype=torch.float64, device=dev)
    pre_sum, count, n = torch.zeros(F, dtype=torch.float64, device=dev), torch.zeros(F, dtype=torch.float64, device=dev), 0

    for idx, ids in data.batches(df, BUDGET, shard):
        ids = ids.to(dev)
        keep = ids != net.PAD
        T = int(keep.sum())
        acts, zs = [], []

        def tap(l, r):
            acts.append(r["a"][keep].half())
            zs.append(r["z"][keep].flatten(1).half())

        with log.op("pass_to_60", T), torch.no_grad():
            x0 = m.esmc.embed(ids)
            x = net.run(m, ids, x=x0, stop=L, tap=tap)
            pre, val, ind, sd = sae.encode(s, x[keep])
            fire = torch.zeros_like(pre).scatter(1, ind, val) > 0
            f, t = fire.T.nonzero().T
            fs = torch.sparse_coo_tensor(torch.stack([f, t]), (1 / sd[t, 0]), (F, T)).coalesce()
            pre_sum += (pre * fire).sum(0)
            count += fire.sum(0)
            emb += torch.sparse.mm(fs, x0[keep])
        with log.op("shares", T), torch.no_grad():
            for l in range(L):
                a = acts[l].float()
                unit[l] += torch.sparse.mm(fs, a)
                head[l] += torch.sparse.mm(fs, zs[l].float())
                base[l] += (a / sd).sum(0)
        n += T
        log.note(tokens=n)

    os.makedirs(out, exist_ok=True)
    with log.stage("save"), torch.no_grad():
        hs = torch.stack([torch.einsum("fhd,hdf->fh", head[l].view(F, H, -1).float(), net.head_weight(m, l) @ W) for l in range(L)], 1)
        torch.save(dict(head=hs.cpu(), embed=torch.einsum("fD,Df->f", emb.float(), W).cpu(), const=(s.b_dec @ s.W_enc).cpu(),
                        pre=pre_sum.cpu(), tokens=count.cpu(), base=base.cpu(), n=n), f"{out}/parts.pt")
        for l in range(L):
            np.save(f"{out}/unit_{l:02d}.npy", (unit[l] * (net.out_weight(m, l) @ W).T).cpu().numpy())
    del unit
    torch.cuda.empty_cache()


def reduce():
    log = Log("s4-reduce")
    for split in ("train", "holdout"):
        dirs = sorted(glob.glob(f"{OUT}/{split}/shard*"))
        out = f"{OUT}/{split}"
        ps = [torch.load(f"{d}/parts.pt") for d in dirs]
        tokens = sum(p["tokens"] for p in ps)
        c = tokens.clamp(min=1)
        n = sum(p["n"] for p in ps)
        head, embed, pre = (sum(p[k].double() for p in ps) for k in ("head", "embed", "pre"))
        head, embed, pre = head / c[:, None, None], embed / c, pre / c
        usum = torch.zeros_like(pre)
        for l in range(len(glob.glob(f"{dirs[0]}/unit_*.npy"))):
            u = sum(np.load(f"{d}/unit_{l:02d}.npy").astype(np.float64) for d in dirs) / c.numpy()[:, None]
            usum += torch.from_numpy(u.sum(1))
            np.save(f"{out}/unit_{l:02d}.npy", u.astype(np.float16))
        const = ps[0]["const"].double()
        resid = (head.sum((1, 2)) + usum + embed - const - pre)[tokens > 0]
        torch.save(dict(head=head.float(), embed=embed.float(), const=const.float(), pre=pre.float(), tokens=tokens,
                        unit_total=usum.float(), base=sum(p["base"] for p in ps) / n, n=n), f"{out}/parts.pt")
        log.note(**{split: dict(shards=len(dirs), tokens=n, share_sum_residual_max_abs=resid.abs().max().item(),
                                pre_mean_abs=pre.abs().mean().item())})
    log.write()
    print(open(log.path).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--reduce", action="store_true")
    a = ap.parse_args()
    if a.reduce:
        return reduce()
    shard = tuple(map(int, a.shard.split("/")))
    log = Log("s4", shard)
    m, s = net.load(), sae.load()
    df = data.sample()
    log.plain(m, next(data.batches(df, BUDGET, shard))[1].cuda())
    for split in ("train", "holdout"):
        with log.stage(split):
            collect(m, s, df[df.split == split], shard, f"{OUT}/{split}/shard{shard[0]}", log)
    log.write()


if __name__ == "__main__":
    main()
