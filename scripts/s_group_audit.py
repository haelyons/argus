"""Do groups of co-firing unit poles have cleaner jobs than single poles? For each s_cofire group, do
the residues where its poles are jointly strongest share an amino acid, a Swiss-Prot annotation, a
Pfam family or an SAE feature more than chance allows, on holdout proteins? Is that more than its
own member poles manage alone, and more than a random group of the same size?
"""

import glob
import json

import pandas as pd
import torch

from argus import audit, data, net, runlog
from argus.collect import extremes, merge

BUDGET, TOP, MID = 8192, 32, 64


def scores(m, cl, df, samp, log):
    """Each group's mean pole activity in standard deviations, at every token: its TOP strongest tokens,
    and its MID strongest among the tokens in samp."""
    N, U = len(net.blocks(m)), net.blocks(m)[0].ffn.ffn_hidden_size
    K = len(cl["size"])
    blocks = lambda x: x.view(2, N, U).transpose(0, 1).reshape(N, 2 * U).cuda()
    ids_of = lambda c, shift: blocks(torch.where(c >= 0, c + shift, 2 * K).long())
    cid, ctl = ids_of(cl["cluster"], 0), ids_of(cl["control"], K)
    w = blocks(torch.where(cl["cluster"] >= 0, 1 / cl["sd"].clamp(min=1e-6), 0))
    size = cl["size"].clamp(min=1).repeat(2).cuda()
    hi = mid = None
    for idx, ids in data.batches(df, BUDGET, log.shard):
        ids = ids.cuda()
        if log.rec["tok_s"] is None:
            log.plain(m, ids)
        keep = ids != net.PAD
        T = int(keep.sum())
        tok = (torch.from_numpy(idx).cuda()[:, None] * data.STRIDE + torch.arange(ids.shape[1], device="cuda"))[keep]
        sc = torch.zeros(T, 2 * K + 1, device="cuda")

        def tap(l, r):
            a = r["a"][keep]
            x = torch.cat([a.relu(), (-a).relu()], 1) * w[l]
            sc.index_add_(1, cid[l], x)
            sc.index_add_(1, ctl[l], x)

        with log.op("pass_group_scores", T), torch.no_grad():
            net.run(m, ids, tap=tap)
        sc = sc[:, : 2 * K] / size
        hi = extremes(hi, sc, tok, TOP)
        if samp is not None:
            pick = torch.isin(tok, samp)
            mid = extremes(mid, sc[pick], tok[pick], MID)
    return dict(hi=tuple(x.cpu() for x in hi), mid=mid and tuple(x.cpu() for x in mid))


def reduce(log):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cl = torch.load(f"{log.read('cofire')}/clusters.pt")
    poles = pd.read_parquet(f"{log.read('audit')}/audit.parquet")
    lab, off, names = audit.labels(log.input, dev)
    audit.features(lab, off, sorted(glob.glob(f"{log.read('audit')}/shard*/codes.pt")))
    parts = {s: [torch.load(f) for f in sorted(glob.glob(f"{log.dir}/{s}-shard*.pt"))] for s in ("train", "holdout")}
    hi = {s: merge([p["hi"] for p in parts[s]], TOP) for s in parts}
    mid = merge([p["mid"] for p in parts["holdout"]], MID)
    tok = dict(train=hi["train"][1].T, holdout=hi["holdout"][1].T, mid=mid[1].T)
    K = len(cl["size"])
    out = dict(group=torch.arange(K).repeat(2), control=torch.arange(2).repeat_interleave(K), size=cl["size"].repeat(2),
               peak=hi["train"][0][0], edge_train=hi["train"][0][-1], edge_holdout=hi["holdout"][0][-1], edge_mid=mid[0][-1],
               proteins_train=(tok["train"] // data.STRIDE).sort(1).values.diff(dim=1).ne(0).sum(1) + 1)
    out |= audit.table(tok, lab, off)

    alone = audit.best({k: torch.from_numpy(poles[k].values) for k in poles if k.endswith(("_holdout", "_base"))}, "holdout")
    live = cl["cluster"] >= 0
    member = torch.cat([cl["cluster"][live], cl["control"][live] + K]).long()
    alone = alone[live].repeat(2)
    out["members_mean"] = torch.zeros(2 * K).index_add_(0, member, alone) / out["size"].clamp(min=1)
    out["members_max"] = torch.full((2 * K,), -1.0).scatter_reduce_(0, member, alone, "amax")
    pd.DataFrame({k: v.numpy() for k, v in out.items()}).to_parquet(f"{log.dir}/groups.parquet")
    json.dump(names, open(f"{log.dir}/labels.json", "w"))

    together = audit.best(out, "holdout")
    for name, rows in (("groups", out["control"] == 0), ("random_groups", out["control"] == 1)):
        rows = rows & (out["size"] > 0)
        log.note(**{name: dict(n=int(rows.sum()), **audit.clean({k: v[rows] for k, v in out.items()}),
                               members_mean=audit.at(out["members_mean"][rows]), members_max=audit.at(out["members_max"][rows]),
                               above_members_mean=round((together > out["members_mean"])[rows].float().mean().item(), 4),
                               above_members_max=round((together > out["members_max"])[rows].float().mean().item(), 4))})
    log.note(poles_alone=audit.at(alone[: int(live.sum())]))
    log.write()
    print(open(log.path).read())


def main():
    log = runlog.start("group_audit", __doc__)
    if log.reduce:
        return reduce(log)
    m, df = net.load(), data.sample(log.input)
    cl = torch.load(f"{log.read('cofire')}/clusters.pt")
    samp = torch.load(f"{log.read('collect')}/holdout/sample.pt")["tok"].cuda()
    for split in ("train", "holdout"):
        with log.stage(split):
            torch.save(scores(m, cl, df[df.split == split], samp if split == "holdout" else None, log), f"{log.dir}/{split}-shard{log.shard[0]}.pt")
    log.write()


if __name__ == "__main__":
    main()
