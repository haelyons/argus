"""The collector: activations in, examples out, plain torch; swap the body freely. And readers of what s_collect saved."""

import numpy as np
import torch


def extremes(run, a, tok, k):
    """Running k largest per column.

    a: (T, C) activations of T tokens; tok: (T,) their global token ids.
    run: None or (vals (k, C), toks (k, C)) from earlier calls. Returns the same pair, updated.
    For the smallest, pass -a and negate the values on read.
    """
    v, i = a.topk(min(k, len(a)), 0)
    t = tok[i]
    if run is not None:
        v, j = torch.cat([run[0], v]).topk(min(k, len(v) + len(run[0])), 0)
        t = torch.cat([run[1], t]).gather(0, j)
    return v, t


def merge(parts, k):
    """The k largest per column over several (vals, toks) pairs."""
    v, t = torch.cat([p[0] for p in parts], -2), torch.cat([p[1] for p in parts], -2)
    v, j = v.topk(k, -2)
    return v, t.gather(-2, j)


def stats(collect, split):
    """Per unit mean and standard deviation (N, U), and per feature the share of tokens where it fires (F,)."""
    u, f = torch.load(f"{collect}/{split}/units.pt"), torch.load(f"{collect}/{split}/features.pt")
    mean = (u["stat"][:, 0] / u["n"]).float().cuda()
    sd = ((u["stat"][:, 1] / u["n"]).float().cuda() - mean**2).clamp(min=1e-12).sqrt()
    return collect, split, mean, sd, (f["stat"][0] / u["n"]).float().cuda()


def tracking(st, l):
    """r (F, U): correlation over tokens between block l's unit activations and each feature firing."""
    collect, split, mean, sd, p = st
    cond = torch.from_numpy(np.load(f"{collect}/{split}/cond_{l:02d}.npy").astype(np.float32)).cuda()
    return (cond - mean[l]) / sd[l] * ((p / (1 - p)).sqrt() * (p > 0))[:, None]
