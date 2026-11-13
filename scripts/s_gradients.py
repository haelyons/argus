"""Gradients. For each target and example, activation * d(target)/d(activation)
for every upstream unit and head at every position: the first-order effect of removing it.

Target value: a feature's input (pre) at the on-position, or a unit's activation there.
out/gradients/<target>.pt
  rank           (blocks, U) mean over train examples of the position-summed unit score
  per split, per example:
    value        the target's value
    unit_sum     (blocks, U) score summed over positions
    unit_pos     (blocks, L) sum of |score| over units, per position
    top          the 4096 largest |score| entries as flat indices into (blocks, L, U), and scores
    head         (blocks, L, H) f16 head scores
"""

import argparse
import os

import torch

from argus import data, net, pick, sae
from argus.runlog import Log

OUT, TOP = "out/gradients", 4096


def upstream(kind, t):
    """(number of blocks whose units are upstream, number of blocks to run)."""
    if kind == "features":
        return sae.LAYER, sae.LAYER
    return t[0], t[0] + 1


def attribute(m, s, ids, kind, t, pos):
    nu, stop = upstream(kind, t)
    acts, zs = [], []

    def tap(l, r):
        r["a"].retain_grad(), r["z"].retain_grad()
        acts.append(r["a"]), zs.append(r["z"])

    x = net.run(m, ids, x=m.esmc.embed(ids).requires_grad_(), stop=stop, tap=tap)
    value = sae.encode(s, x)[0][0, pos, t] if kind == "features" else acts[-1][0, pos, t[1]]
    value.backward()
    with torch.no_grad():
        L = ids.shape[1]
        us = torch.stack([a.grad[0] * a[0] for a in acts[:nu]]) if nu else torch.zeros(0, L, acts[0].shape[-1], device=ids.device)
        hs = torch.stack([(z.grad[0] * z[0]).sum(-1) for z in zs])
        k = min(TOP, us.numel())
        top = us.flatten().abs().topk(k).indices
        return dict(value=value.item(), unit_sum=us.sum(1).cpu(), unit_pos=us.abs().sum(-1).cpu(),
                    top=(top.cpu(), us.flatten()[top].cpu()), head=hs.half().cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    shard = tuple(map(int, ap.parse_args().shard.split("/")))
    log = Log("gradients", shard)
    m, s = net.load(), sae.load()
    df = data.sample()
    os.makedirs(OUT, exist_ok=True)
    for name, kind, t, ex in pick.targets(shard):
        rec = {}
        for split, pairs in ex.items():
            rec[split] = []
            for e in pairs:
                ids = pick.ids_of(df, e["on"][0])
                if log.rec["tok_s"] is None:
                    log.plain(m, ids)
                with log.op("forward_backward", ids.numel()):
                    rec[split].append(attribute(m, s, ids, kind, t, e["on"][1]) | e)
        rec["rank"] = torch.stack([r["unit_sum"] for r in rec["train"]]).mean(0)
        torch.save(rec, f"{OUT}/{name}.pt")
        log.note(done=name)
    log.write()


if __name__ == "__main__":
    main()
