"""The collector: activations in, examples out, plain torch; swap the body freely. And readers of what s_collect saved."""

import numpy as np
import torch


def extremes(k, *parts):
    """The k largest per column over parts, with their token ids. For the smallest, pass -vals and negate on read.

    A part is None (skipped), or (vals (..., n, C), toks alike) from an earlier call or another shard,
    or (activations (T, C), toks (T,)) of T new tokens.
    """
    vs, ts = [], []
    for v, t in filter(None, parts):
        if t.dim() == 1:
            v, i = v.topk(min(k, len(v)), 0)
            t = t[i]
        vs.append(v), ts.append(t)
    v, t = torch.cat(vs, -2), torch.cat(ts, -2)
    v, j = v.topk(min(k, v.shape[-2]), -2)
    return v, t.gather(-2, j)


def stats(collect, split):
    """Per unit mean and standard deviation (N, U), and per feature the share of tokens where it fires (F,)."""
    u, f = torch.load(f"{collect}/{split}/units.pt"), torch.load(f"{collect}/{split}/features.pt")
    mean = (u["stat"][:, 0] / u["n"]).float().cuda()
    sd = ((u["stat"][:, 1] / u["n"]).float().cuda() - mean**2).clamp(min=1e-12).sqrt()
    return collect, split, mean, sd, (f["stat"][0] / u["n"]).float().cuda()


def tracking(st, l, j=slice(None)):
    """r (F, U): correlation over tokens between block l's unit activations and each feature firing. (U,) for one feature j."""
    collect, split, mean, sd, p = st
    cond = torch.from_numpy(np.load(f"{collect}/{split}/cond_{l:02d}.npy", mmap_mode="r")[j].astype(np.float32)).cuda()
    return (cond - mean[l]) / sd[l] * ((p / (1 - p)).sqrt() * (p > 0))[j, None]


def top_units(st, N, k, each=None):
    """Per feature, the k units over all N blocks with the largest |r|, strongest first: unit (F, k) as
    block * U + unit, and r (F, k). each(l, r of block l) sees every block's full table on the way."""
    r = unit = None
    for l in range(N):
        t = tracking(st, l)
        if each:
            each(l, t)
        ids = (l * t.shape[1] + torch.arange(t.shape[1], device="cuda")).expand_as(t)
        t, ids = (t if r is None else torch.cat([r, t], 1)), (ids if r is None else torch.cat([unit, ids], 1))
        i = t.abs().topk(k, 1).indices
        r, unit = t.gather(1, i), ids.gather(1, i)
    return unit, r
