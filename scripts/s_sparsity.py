"""How many unit activations can be set to zero, smallest first or by the signs of gate and value,
before ESM C 6B predicts masked residues worse?
"""

import glob

import numpy as np
import torch

from argus import data, net, runlog

FRACS = [0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999]
SIGNS = {"g+v+": (1, 1), "g+v-": (1, 0), "g-v+": (0, 1), "g-v-": (0, 0), "g-": (0, None), "v-": (None, 0)}
CONDS = ["none"] + [f"weakest_{f}" for f in FRACS] + list(SIGNS)
PROTEINS, SEED, BUDGET, THR_TOKENS = 1024, 0, 8192, 512


def thresholds(norm, collect):
    acts = torch.load(f"{collect}/train/sample.pt")["acts"][:THR_TOKENS].cuda().float()
    size = (acts.abs() * norm).flatten().sort().values
    return [size[int(f * len(size))].item() for f in FRACS]


def removal(cond, thr, wn, tally):
    """Edit for one block: zero the activations the condition selects, and count them."""

    def f(a, g, v):
        if cond in SIGNS:
            gs, vs = SIGNS[cond]
            drop = torch.ones_like(a, dtype=torch.bool)
            if gs is not None:
                drop &= (g > 0) == bool(gs)
            if vs is not None:
                drop &= (v > 0) == bool(vs)
        else:
            drop = a.abs() * wn < thr
        size = a.abs() * wn
        tally.add_(torch.stack([drop.float().mean(), (size * drop).sum() / size.sum()]))
        return a * ~drop

    return f


def main():
    log = runlog.start("sparsity", __doc__)
    if log.reduce:
        return reduce(log)
    shard, OUT = log.shard, log.dir
    m = net.load()
    N = len(net.blocks(m))
    norm = torch.from_numpy(np.load(log.read("weights", "none") + "/norm.npy")).cuda()
    thr = thresholds(norm, log.read("collect"))
    log.note(thr=thr)
    df = data.sample(log.input)
    for split in ("train", "holdout"):
        part = df[df.split == split].sample(PROTEINS, random_state=SEED)
        rows, tally, nb = [], torch.zeros(len(CONDS), N, 2, device="cuda"), 0
        for idx, ids in data.batches(part, BUDGET, shard):
            if log.rec["tok_s"] is None:
                log.plain(m, ids.cuda())
            ids, labels = (t.cuda() for t in data.mask(ids, idx))
            loss = []
            for c, cond in enumerate(CONDS):
                edits = {} if cond == "none" else {
                    ("act", l): removal(cond, thr[c - 1] if c <= len(FRACS) else None, norm[l], tally[c, l]) for l in range(N)}
                with log.op("masked_pass_edited" if edits else "masked_pass", ids.numel()), torch.no_grad():
                    loss.append(net.mlm_loss(m, net.run(m, ids, edits=edits), labels))
            rows.append((torch.from_numpy(idx), torch.stack(loss, 1).cpu(), (labels >= 0).sum(1).cpu()))
            nb += 1
            log.note(**{split: nb})
        idx, loss, masked = (torch.cat(x) for x in zip(*rows))
        torch.save(dict(conds=CONDS, idx=idx, loss=loss, masked=masked, tally=(tally / nb).cpu(), batches=nb, thr=thr),
                   f"{OUT}/{split}-shard{shard[0]}.pt")
    log.write()


def reduce(log):
    OUT = log.dir
    for split in ("train", "holdout"):
        ps = [torch.load(f) for f in sorted(glob.glob(f"{OUT}/{split}-shard*.pt"))]
        nb = sum(p["batches"] for p in ps)
        tally = sum(p["tally"] * p["batches"] for p in ps) / nb
        loss, masked = torch.cat([p["loss"] for p in ps]), torch.cat([p["masked"] for p in ps])
        torch.save(dict(conds=CONDS, idx=torch.cat([p["idx"] for p in ps]), loss=loss, masked=masked,
                        removed=tally[..., 0], mass=tally[..., 1], thr=ps[0]["thr"]), f"{OUT}/{split}.pt")
        per = (loss.sum(0) / masked.sum()).tolist()
        log.note(**{split: dict(proteins=len(masked), masked=int(masked.sum()),
                                loss={c: round(x, 4) for c, x in zip(CONDS, per)},
                                removed={c: round(x, 4) for c, x in zip(CONDS, tally[..., 0].mean(1).tolist())})})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
