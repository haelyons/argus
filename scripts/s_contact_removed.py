"""Does the model still couple residues in contact without the heads whose attention shows contacts? On
the proteins of s_contact_edits, heads are set to zero, in the unchanged protein and under every
replacement of a residue by each of the 20 amino acids: the first 8 and the first TOP heads of
s_contact_rank's attention ranking, and as many heads as the TOP at random in the same blocks, drawn
anew for each protein. Beside them, the first TOP held at their values on the unchanged protein. Per
protein and set, as in s_contact_edits: the coupling, how many of its L strongest pairs 24 or more apart
are in contact, its mean at pairs in contact and at pairs not in contact at the same separations, and
the share of the logit change that goes. And what does each removal cost in masked loss per masked
residue?
"""

import glob

import torch

from argus import data, edit, net, pairs, pick, runlog

TOP, SEED, BUDGET = 64, 0, 8192


def main():
    log = runlog.start("contact_removed", __doc__)
    if log.reduce:
        return reduce(log)
    m, df = net.load(), data.sample(log.input)
    r = torch.load(log.read("contact_rank") + "/rank.pt")
    U, H = net.blocks(m)[0].ffn.ffn_hidden_size, net.blocks(m)[0].attn.n_heads
    order = df[(df.split == "holdout") | df.idx.isin(r["kept"])].sort_values("Length", ascending=False).idx.tolist()
    rank, e = r["rank"]["readable"], torch.empty(0, dtype=torch.long)
    i, n = log.shard
    for p in [p for k, p in enumerate(order) if min(k % (2 * n), 2 * n - 1 - k % (2 * n)) == i]:
        s = dict(removed_8=("removed", e, rank[:8]), removed_top=("removed", e, rank[:TOP]), removed_random=("removed", e, edit.at_random(rank[:TOP], H, SEED + 1000 * p + TOP)),
                 held_top=("held", e, rank[:TOP]))
        ids = pick.ids_of(df, p)
        mids, labels = (t.cuda() for t in data.mask(ids.cpu(), [p]))
        with torch.no_grad():
            coupling, share, own = edit.coupling(m, ids, s, U, H, log, BUDGET)
            loss = {k: (net.mlm_loss(m, net.run(m, mids, edits=edit.zero(e, h.cuda(), U, H) if how == "removed" else None), labels) / (labels >= 0).sum()).item()
                    for k, (how, _, h) in {"clean": ("", e, e), **s}.items() if how != "held"}
        torch.save(dict(protein=p, sets=["clean", *s], heads={k: v[2] for k, v in s.items()}, coupling=coupling, share=share, own=own, loss=loss), f"{log.dir}/p{p}.pt")
        log.note(**{f"p{p}": dict(residues=coupling.shape[1], own_residue_max_abs=own.max().item(), loss=loss, share_not_replaced=dict(zip(s, share[:, 1].tolist())))})
    log.write()


def reduce(log):
    files = sorted(glob.glob(f"{log.dir}/p*.pt"))
    t = pairs.outcomes(files, data.sample(log.input))
    loss = {(d["protein"], k): v for d in map(torch.load, files) for k, v in d["loss"].items()}
    t["loss"] = [loss.get((p, s), float("nan")) for p, s in zip(t.protein, t.set)]
    t.to_parquet(f"{log.dir}/proteins.parquet")
    cols = ["precision", "gap", "contact", "matched", "share", "share_not_replaced", "loss"]
    log.note(chance=t.chance.mean(),
             **{g: dict(means=t[t.group == g].groupby("set", sort=False)[cols].mean().to_dict("index"),
                        less={f"{a}_less_{b}": pairs.paired(t, g, a, b) for a, b in [("removed_top", "removed_random"), ("removed_top", "held_top"), ("removed_8", "clean")]})
                for g in ("holdout", "set_aside")})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
