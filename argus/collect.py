"""The collector: activations in, examples out. One function, plain torch; swap the body freely."""

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
