"""Does the coupling between residues in contact pass through the heads whose attention shows contacts?
On the holdout proteins of an input and the train proteins s_contact_rank set aside, every residue is
replaced by each of the 20 amino acids while a set of heads or units is held at its values on the
unchanged protein. The sets: the HEADS heads whose attention shows contacts best; as many heads at
random in the same blocks, drawn anew for each protein; the HEADS heads and the UNITS units that
s_contact_rank's gradient puts first; as many units at random in the same blocks. Per protein and set:
how much does a replacement at one residue change the amino-acid logits at another (norm over the 20
replacements and the 20 logits)? Of the L pairs, 24 or more apart, coupled most, how many are in contact
in the AlphaFold model? What is the mean coupling of the pairs in contact, and of pairs not in contact
at the same separations? What share of the logit change goes, at all residues and at those not replaced?
The primary comparison is the heads that show contacts against the random heads of the same blocks,
protein by protein, at each count, in the holdout proteins and in those set aside apart. Every other
comparison is secondary.
"""

import glob

import pandas as pd
import torch

from argus import data, edit, net, pairs, pick, runlog

HEADS, UNITS, SEED, BUDGET = [8, 64, 512], [64, 512, 4096], 0, 8192
AA = slice(4, 24)


def at_random(ids, per, seed):
    """As many ids as `ids` in each of its blocks (id = block * per + index), at random: among the block's
    others first, and among its own when those run out."""
    g, out = torch.Generator().manual_seed(seed), []
    for l, n in zip(*(ids // per).unique(return_counts=True)):
        block = l * per + torch.randperm(per, generator=g)
        own = torch.isin(block, ids)
        out.append(torch.cat([block[~own], block[own]])[:n])
    return torch.cat(out)


def sets(rank, p, U, H):
    """{name: (units, heads)} held in protein p."""
    e, s = torch.empty(0, dtype=torch.long), {}
    for k in HEADS:
        top = rank["readable"][:k]
        s[f"readable_{k}"], s[f"random_heads_{k}"], s[f"grad_heads_{k}"] = (e, top), (e, at_random(top, H, SEED + 1000 * p + k)), (e, rank["grad_heads"][:k])
    for n in UNITS:
        top = rank["grad_units"][:n]
        s[f"grad_units_{n}"], s[f"random_units_{n}"] = (top, e), (at_random(top, U, SEED + 1000 * p + n), e)
    return s


def scan(m, ids, held, U, H, log):
    """With nothing held, then per set: the coupling (S + 1, L, L) and the largest logit change under a
    replacement that leaves the protein as it is (S + 1). Per set: the share of the logit change that goes
    when it is held, at all residues and at those not replaced (S, 2)."""
    N, T, dev = len(net.blocks(m)), ids.shape[1], ids.device
    wt = edit.capture(m, ids)
    lw = net.logits(m, wt["x"][N])[0, :, AA]
    eds = [edit.remove_hold(torch.empty(0, dtype=torch.long, device=dev), u.to(dev), h.to(dev), wt, U, H) for u, h in held.values()]
    first = [min(l for _, l in ed) for ed in eds]
    sq, own = lw.new_zeros(len(eds) + 1, T - 2, T), lw.new_zeros(len(eds) + 1)
    dot, tot = lw.new_zeros(len(eds), 2), lw.new_zeros(2)
    for pos, mids in data.mutants(ids, BUDGET):
        if log.rec["tok_s"] is None:
            log.plain(m, mids)
        keep, same = {}, (mids == ids).all(1)
        part = torch.stack([torch.ones_like(mids, dtype=torch.bool), torch.arange(T, device=dev) != pos[:, None]])[:, :, 1:-1]
        with log.op("changed", mids.numel()):
            d0 = net.logits(m, net.run(m, mids, tap=lambda l, r: keep.update({l + 1: r["x"]}) if l + 1 in first else None))[..., AA] - lw
        ds = [d0]
        for f, ed in zip(first, eds):
            with log.op("held", mids.numel()):
                ds.append(net.logits(m, net.run(m, mids, x=keep.get(f), start=f, edits=ed))[..., AA] - lw)
        for i, d in enumerate(ds):
            sq[i].index_add_(0, pos - 1, d.square().sum(-1))
            own[i] = torch.maximum(own[i], (d.abs().amax((1, 2)) * same).max())
            if i:
                dot[i - 1] += ((d0 * (d0 - d)).sum(-1)[:, 1:-1] * part).sum((1, 2))
        tot += (d0.square().sum(-1)[:, 1:-1] * part).sum((1, 2))
    return sq.sqrt()[:, :, 1:-1].cpu(), (dot / tot).cpu(), own.cpu()


def main():
    log = runlog.start("contact_edits", __doc__)
    if log.reduce:
        return reduce(log)
    m, df = net.load(), data.sample(log.input)
    r = torch.load(log.read("contact_rank") + "/rank.pt")
    U, H = net.blocks(m)[0].ffn.ffn_hidden_size, net.blocks(m)[0].attn.n_heads
    part = df[(df.split == "holdout") | df.idx.isin(r["kept"])].sort_values("Length", ascending=False)
    for p in part.idx.tolist()[log.shard[0] :: log.shard[1]]:
        held = sets(r["rank"], p, U, H)
        with torch.no_grad():
            coupling, share, own = scan(m, pick.ids_of(df, p), held, U, H, log)
        torch.save(dict(protein=p, sets=["clean", *held], held=held, coupling=coupling, share=share, own=own), f"{log.dir}/p{p}.pt")
        log.note(**{f"p{p}": dict(residues=coupling.shape[1], own_residue_max_abs=own.max().item(), share_not_replaced=dict(zip(held, share[:, 1].tolist())))})
    log.write()


def reduce(log):
    df, rows = data.sample(log.input), []
    for f in sorted(glob.glob(f"{log.dir}/p*.pt")):
        d = torch.load(f)
        p, m = d["protein"], d["coupling"]
        c = pairs.contacts(df.Entry[p], m.shape[-1])
        (hits, chance), g, share = pairs.precision(m, c), pairs.gap(m, c), torch.cat([torch.zeros(1, 2), d["share"]])
        for i, s in enumerate(d["sets"]):
            rows.append(dict(protein=p, group="holdout" if df.split[p] == "holdout" else "set_aside", residues=m.shape[-1], set=s, precision=hits[i].item(), chance=chance,
                             contact=g[i, 0].item(), matched=g[i, 1].item(), gap=(g[i, 0] - g[i, 1]).item(), share=share[i, 0].item(), share_not_replaced=share[i, 1].item()))
    t = pd.DataFrame(rows)
    t.to_parquet(f"{log.dir}/proteins.parquet")

    def paired(g, a, b):
        """Per outcome: the means of sets a and b over the group's proteins, the mean of a less b, its standard
        error, and the number of proteins with a below b."""
        out = {}
        for y in ("precision", "gap"):
            v = t[t.group == g].pivot(index="protein", columns="set", values=y)
            out[y] = dict(a=v[a].mean(), b=v[b].mean(), diff=(v[a] - v[b]).mean(), se=(v[a] - v[b]).sem(), below=int((v[a] < v[b]).sum()), proteins=len(v))
        return out

    groups = ("holdout", "set_aside")
    cols = ["precision", "gap", "contact", "matched", "share", "share_not_replaced"]
    log.note(chance=t.chance.mean(),
             primary={g: {k: paired(g, f"readable_{k}", f"random_heads_{k}") for k in HEADS} for g in groups},
             secondary={g: dict(means=t[t.group == g].groupby("set", sort=False)[cols].mean().to_dict("index"),
                                grad_heads_less_readable={k: paired(g, f"grad_heads_{k}", f"readable_{k}") for k in HEADS},
                                grad_units_less_random_units={n: paired(g, f"grad_units_{n}", f"random_units_{n}") for n in UNITS}) for g in groups})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
