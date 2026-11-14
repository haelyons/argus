"""Is an SAE feature a copy of a small group of units? Take the 1, 2, 4 ... 64 units, in any block,
that best track whether a feature fires on train proteins, and add them weighted by that tracking.
How well does the sum track the feature, on train and on holdout proteins?
"""

import glob

import torch

from argus import data, net, runlog, sae
from argus.collect import stats, tracking

G, BUDGET = 64, 4096
SIZES = [1, 2, 4, 8, 16, 32, 64]


def pick(st, N, U):
    """Per feature, the G units with the largest |r| over all blocks, strongest first: unit (F, G), r (F, G)."""
    r = unit = None
    for l in range(N):
        t = tracking(st, l)
        ids = (l * U + torch.arange(U, device="cuda")).expand_as(t)
        t, ids = (t if r is None else torch.cat([r, t], 1)), (ids if r is None else torch.cat([unit, ids], 1))
        i = t.abs().topk(G, 1).indices
        r, unit = t.gather(1, i), ids.gather(1, i)
    return unit, r


def moments(m, s, df, unit, r, sd, log):
    """Sums over tokens for the correlation of each nested group's weighted sum with its feature firing."""
    N, U = sd.shape
    F, B = len(unit), len(SIZES)
    step = torch.tensor([j.bit_length() for j in range(G)], device="cuda")
    ent = []
    for l in range(N):
        f, j = (unit // U == l).nonzero().T
        u = unit[f, j] % U
        ent.append((u, f * B + step[j], r[f, j] / sd[l, u]))
    s1, s2, sf = (torch.zeros(F, B, dtype=torch.float64, device="cuda") for _ in range(3))
    nf, n = torch.zeros(F, dtype=torch.float64, device="cuda"), 0
    for idx, ids in data.batches(df, BUDGET, log.shard):
        ids = ids.cuda()
        if log.rec["tok_s"] is None:
            log.plain(m, ids)
        keep = ids != net.PAD
        T = int(keep.sum())
        sc, fire = torch.zeros(T, F * B, device="cuda"), []

        def tap(l, x):
            u, col, w = ent[l]
            sc.index_add_(1, col, x["a"][keep][:, u] * w)
            if l == sae.LAYER - 1:
                _, val, ind, _ = sae.encode(s, x["x"][keep])
                fire.extend([ind, val > 0])

        with log.op("pass_group_sums", T), torch.no_grad():
            net.run(m, ids, tap=tap)
        ind, on = fire
        sc = sc.view(T, F, B).cumsum_(-1)
        s1 += sc.sum(0, dtype=torch.float64)
        s2 += (sc * sc).sum(0, dtype=torch.float64)
        at = sc.gather(1, ind[..., None].expand(-1, -1, B)) * on[..., None]
        sf.index_add_(0, ind.flatten(), at.flatten(0, 1).double())
        nf += torch.bincount(ind[on], minlength=F)
        n += T
        log.note(tokens=n)
    return dict(n=n, s1=s1.cpu(), s2=s2.cpu(), sf=sf.cpu(), nf=nf.cpu())


def reduce(log):
    sel = torch.load(f"{log.dir}/units-shard0.pt")
    out = dict(unit=sel["unit"], r_unit=sel["r"], sizes=SIZES)
    for split in ("train", "holdout"):
        ps = [torch.load(f) for f in sorted(glob.glob(f"{log.dir}/{split}-shard*.pt"))]
        n = sum(p["n"] for p in ps)
        s1, s2, sf, nf = (sum(p[k] for p in ps) / n for k in ("s1", "s2", "sf", "nf"))
        p = nf[:, None]
        out[f"r_{split}"] = ((sf - s1 * p) / ((s2 - s1**2).clamp(min=1e-30) * p * (1 - p)).sqrt()).float()
        out[f"tokens_{split}"] = nf * n
        fired = nf > 0
        log.note(**{split: dict(features=int(fired.sum()), **{str(k): {t: round((out[f"r_{split}"][fired, i] >= t).float().mean().item(), 4)
                                                                      for t in (0.3, 0.5, 0.7, 0.9)} for i, k in enumerate(SIZES)})})
    fired = out["tokens_train"] > 0
    log.note(one_unit_vs_table_max_abs=round((out["r_train"][:, 0] - sel["r"][:, 0].abs())[fired].abs().max().item(), 4))
    torch.save(out, f"{log.dir}/group_copies.pt")
    log.write()
    print(open(log.path).read())


def main():
    log = runlog.start("group_copies", __doc__)
    if log.reduce:
        return reduce(log)
    m, s, df = net.load(), sae.load(), data.sample(log.input)
    st = stats(log.read("collect"), "train")
    N, U = st[3].shape
    with log.stage("pick"), torch.no_grad():
        unit, r = pick(st, N, U)
    torch.save(dict(unit=unit.cpu(), r=r.cpu()), f"{log.dir}/units-shard{log.shard[0]}.pt")
    for split in ("train", "holdout"):
        with log.stage(split):
            torch.save(moments(m, s, df[df.split == split], unit, r, st[3], log), f"{log.dir}/{split}-shard{log.shard[0]}.pt")
    log.write()


if __name__ == "__main__":
    main()
