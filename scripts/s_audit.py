"""Do the residues where a unit is most positive, or most negative, share one thing more than chance
allows: an amino acid, a Swiss-Prot annotation, a Pfam family or an SAE feature? Does the thing its
strongest train residues share also hold at its strongest holdout residues, and at its top 64 of a
random holdout sample?
"""

import glob
import json
import os

import pandas as pd
import torch

from argus import audit, data, net, runlog, sae

BUDGET, MID = 8192, 64


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


def reduce(log):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    collect = log.read("collect")
    lab, off, names = audit.labels(log.input, dev)
    audit.features(lab, off, sorted(glob.glob(f"{log.dir}/shard*/codes.pt")))

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
    out |= audit.table(tok, lab, off)
    pd.DataFrame({k: v.numpy() for k, v in out.items()}).to_parquet(f"{log.dir}/audit.parquet")
    json.dump(names, open(f"{log.dir}/labels.json", "w"))
    log.note(poles=2 * N * U, tokens=len(lab["residue"]), **audit.clean(out))
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
