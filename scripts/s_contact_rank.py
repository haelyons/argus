"""Which attention heads show a protein's contacts in their attention, and which heads and units carry
the coupling between residues in contact? On the train proteins of an input but KEPT of them, set aside
for s_contact_edits. Per head: of the L residue pairs, 24 or more apart, that its attention weighs most
(the pattern among the residues, made symmetric, less row mean times column mean over the mean), how
many are in contact in the protein's AlphaFold model, and what share of its attention among the residues
falls 24 or more residues away? Per head and per unit: SUBS residues of each
protein are replaced by another amino acid each. The gap is the change of the amino-acid logits at the
residues in contact with the replaced one less that at residues equally far along the chain and not in
contact. To first order, how much of the gap would go if the head's output or the unit's activation
were held at its value on the unchanged protein (the gradient of the gap times the change of the value,
summed over residues)?
"""

import glob

import torch

from argus import data, edit, net, pairs, pick, runlog

KEPT, SUBS, SEED, BUDGET, BACK = 15, 64, 0, 8192, 768
SHOW = dict(readable=[8, 64, 512], grad_heads=[8, 64, 512], grad_units=[64, 512, 4096])
AA = slice(4, 24)


def split(df):
    """Train proteins: the KEPT set aside, and the rest."""
    train = df[df.split == "train"]
    kept = set(train.sample(KEPT, random_state=SEED).idx)
    return sorted(kept), [p for p in train.idx if p not in kept]


def readable(m, ids, c):
    """(2, N, H): of every head's attention among the residues, the contact precision and the share that falls
    pairs.SEP or more residues away."""
    out, r = [], torch.arange(len(c), device=c.device)
    away = (r[:, None] - r).abs() >= pairs.SEP

    def tap(l, t):
        p = t["p"][0, :, 1:-1, 1:-1]
        out.append(torch.stack([pairs.precision(p, c)[0], (p * away).sum((1, 2)) / p.sum((1, 2))]))

    net.run(m, ids, tap=tap, pattern=True)
    return torch.stack(out, 1)


def carried(m, ids, c, gen, log):
    """Units (N, U) and heads (N, H): over SUBS replacements, the gradient of the gap times the change of the
    value from the unchanged protein, summed over residues."""
    T = ids.shape[1]
    with torch.no_grad():
        wt = edit.capture(m, ids)
        lw = net.logits(m, wt["x"][-1])[0, :, AA]
    w = pairs.weights(c)
    w = w[0] - w[1]
    pos = 1 + torch.randperm(T - 2, generator=gen)[:SUBS]
    aa = 4 + (ids[0, pos].cpu() - 3 + torch.randint(19, (len(pos),), generator=gen)) % 20
    su = sh = 0
    for k in range(0, len(pos), BACK // T):
        p = pos[k : k + BACK // T].cuda()
        mids = ids.repeat(len(p), 1)
        mids[torch.arange(len(p)), p] = aa[k : k + BACK // T].cuda()
        acts, zs = [], []

        def tap(l, r):
            r["a"].retain_grad(), r["z"].retain_grad()
            acts.append(r["a"]), zs.append(r["z"])

        with log.op("forward_backward", mids.numel()):
            x = net.run(m, mids, x=m.esmc.embed(mids).requires_grad_(), tap=tap)
            d = (net.logits(m, x)[..., AA] - lw).norm(dim=-1)[:, 1:-1]
            (d * w[p - 1]).sum().backward()
        with torch.no_grad():
            su = su + torch.stack([((a - a0) * a.grad).sum((0, 1)) for a, a0 in zip(acts, wt["a"])])
            sh = sh + torch.stack([((z - z0) * z.grad).sum((0, 1, 3)) for z, z0 in zip(zs, wt["z"])])
    return su, sh


def main():
    log = runlog.start("contact_rank", __doc__)
    if log.reduce:
        return reduce(log)
    m, df = net.load(), data.sample(log.input)
    kept, rest = split(df)
    mine = rest[log.shard[0] :: log.shard[1]]
    read, su, sh = [], 0, 0
    for p in mine:
        ids = pick.ids_of(df, p)
        c = pairs.contacts(df.Entry[p], ids.shape[1] - 2).cuda()
        if log.rec["tok_s"] is None:
            log.plain(m, ids.expand(BUDGET // ids.shape[1], -1))
        with log.op("forward_with_patterns", ids.numel()), torch.no_grad():
            read.append(readable(m, ids, c).half().cpu())
        u, h = carried(m, ids, c, torch.Generator().manual_seed(SEED + p), log)
        su, sh = su + u, sh + h
        log.note(done=len(read))
    torch.save(dict(proteins=mine, kept=kept, readable=torch.stack(read), units=su.cpu(), heads=sh.cpu()), f"{log.dir}/rank-shard{log.shard[0]}.pt")
    log.write()


def reduce(log):
    ps = [torch.load(f) for f in sorted(glob.glob(f"{log.dir}/rank-shard*.pt"))]
    read, far = torch.cat([p["readable"] for p in ps]).float().unbind(1)
    score = dict(readable=read.mean(0), far=far.mean(0), grad_heads=sum(p["heads"] for p in ps), grad_units=sum(p["units"] for p in ps))
    rank = {k: v.flatten().argsort(descending=True) for k, v in score.items()}
    proteins = sum((p["proteins"] for p in ps), [])
    torch.save(dict(proteins=proteins, kept=ps[0]["kept"], per_protein=read.half(), score=score, rank=rank), f"{log.dir}/rank.pt")
    where = lambda k, n: [int(i) // score[k].shape[1] for i in rank[k][:n]]
    log.note(proteins=len(proteins), kept=ps[0]["kept"],
             readable_precision={n: score["readable"].flatten()[rank["readable"][n - 1]].item() for n in [1, *SHOW["readable"]]},
             top8={k: [[int(i) // v.shape[1], int(i) % v.shape[1]] for i in rank[k][:8]] for k, v in score.items()},
             heads_in_both={n: len(set(rank["readable"][:n].tolist()) & set(rank["grad_heads"][:n].tolist())) for n in SHOW["readable"]},
             by_8_blocks={k: {n: torch.bincount(torch.tensor(where(k, n)) // 8, minlength=len(score[k]) // 8).tolist() for n in ns} for k, ns in SHOW.items()})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
