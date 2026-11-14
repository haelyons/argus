"""Do later blocks need the SAE features that no unit copies? When features are taken out of the
residual state entering block 60, in quarters ordered by how well their best unit tracks them, how
much worse does ESM C 6B predict masked residues, and how much of the state did each quarter hold?
"""

import glob

import torch

from argus import data, net, runlog, sae

PROTEINS, SEED, BUDGET, Q = 1024, 0, 8192, 4
CONDS = ["none", "all"] + [f"quarter_{q}" for q in range(Q)]


def groups(copies):
    """(conditions, F) 1 where the feature is removed. quarter_0 holds the least tracked features."""
    r = torch.load(f"{copies}/copies.pt")["r"][:, 0].abs()
    q = r.argsort().argsort() * Q // len(r)
    return torch.stack([torch.zeros_like(r), torch.ones_like(r)] + [(q == i).float() for i in range(Q)]).cuda()


def main():
    log = runlog.start("copy_edits", __doc__)
    if log.reduce:
        return reduce(log)
    m, s = net.load(), sae.load()
    L = sae.LAYER
    sel = groups(log.read("copies"))
    df = data.sample(log.input)
    for split in ("train", "holdout"):
        part = df[df.split == split].sample(PROTEINS, random_state=SEED)
        rows, size, tokens = [], torch.zeros(len(CONDS), device="cuda"), 0
        for idx, ids in data.batches(part, BUDGET, log.shard):
            if log.rec["tok_s"] is None:
                log.plain(m, ids.cuda())
            ids, labels = (t.cuda() for t in data.mask(ids, idx))
            keep, loss = ids != net.PAD, []
            with torch.no_grad():
                with log.op("pass_to_60", ids.numel()):
                    x60 = net.run(m, ids, stop=L)
                    _, val, ind, sd = sae.encode(s, x60)
                for c in range(len(CONDS)):
                    with log.op("pass_from_60", ids.numel()):
                        dense = torch.zeros(*ids.shape, sel.shape[1], device="cuda").scatter(-1, ind, val * sel[c][ind])
                        step = dense @ s.W_dec
                        loss.append(net.mlm_loss(m, net.run(m, ids, x=x60 - step * sd, start=L), labels))
                    size[c] += (step.norm(dim=-1) / step.shape[-1] ** 0.5)[keep].sum()
            rows.append((torch.from_numpy(idx), torch.stack(loss, 1).cpu(), (labels >= 0).sum(1).cpu()))
            tokens += int(keep.sum())
            log.note(**{split: len(rows)})
        idx, loss, masked = (torch.cat(x) for x in zip(*rows))
        torch.save(dict(conds=CONDS, idx=idx, loss=loss, masked=masked, size=size.cpu(), tokens=tokens), f"{log.dir}/{split}-shard{log.shard[0]}.pt")
    log.write()


def reduce(log):
    for split in ("train", "holdout"):
        ps = [torch.load(f) for f in sorted(glob.glob(f"{log.dir}/{split}-shard*.pt"))]
        loss, masked = torch.cat([p["loss"] for p in ps]), torch.cat([p["masked"] for p in ps])
        size = sum(p["size"] for p in ps) / sum(p["tokens"] for p in ps)
        torch.save(dict(conds=CONDS, idx=torch.cat([p["idx"] for p in ps]), loss=loss, masked=masked, size=size), f"{log.dir}/{split}.pt")
        log.note(**{split: dict(proteins=len(masked), masked=int(masked.sum()),
                                loss={c: round(x, 4) for c, x in zip(CONDS, (loss.sum(0) / masked.sum()).tolist())},
                                size={c: round(x, 4) for c, x in zip(CONDS, size.tolist())})})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
