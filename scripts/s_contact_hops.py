"""Do the heads whose attention shows contacts move the replaced residue's own changed values to its
partners in one step, or values that have already changed elsewhere? On the proteins of s_contact_edits,
every residue is replaced by each of the 20 amino acids. Of the first TOP heads of s_contact_rank's
attention ranking, the values their patterns mix are kept as on the unchanged protein: at the replaced
residue only; at every other token only; at every token; and their outputs are held. Per protein and
set, as in s_contact_edits: the coupling, how many of its L strongest pairs 24 or more apart are in
contact, its mean at pairs in contact and at pairs not in contact at the same separations, and the
share of the logit change that goes.
"""

import glob

import torch

from argus import data, edit, net, pairs, pick, runlog

TOP, BUDGET = 64, 8192
HOW = ["value_site", "value_rest", "value", "held"]


def main():
    log = runlog.start("contact_hops", __doc__)
    if log.reduce:
        return reduce(log)
    m, df = net.load(), data.sample(log.input)
    r = torch.load(log.read("contact_rank") + "/rank.pt")
    U, H = net.blocks(m)[0].ffn.ffn_hidden_size, net.blocks(m)[0].attn.n_heads
    order = df[(df.split == "holdout") | df.idx.isin(r["kept"])].sort_values("Length", ascending=False).idx.tolist()
    s = {how: (how, torch.empty(0, dtype=torch.long), r["rank"]["readable"][:TOP]) for how in HOW}
    i, n = log.shard
    for p in [p for k, p in enumerate(order) if min(k % (2 * n), 2 * n - 1 - k % (2 * n)) == i]:
        with torch.no_grad():
            coupling, share, own = edit.coupling(m, pick.ids_of(df, p), s, U, H, log, BUDGET)
        torch.save(dict(protein=p, sets=["clean", *s], coupling=coupling, share=share, own=own), f"{log.dir}/p{p}.pt")
        log.note(**{f"p{p}": dict(residues=coupling.shape[1], own_residue_max_abs=own.max().item(), share_not_replaced=dict(zip(s, share[:, 1].tolist())))})
    log.write()


def reduce(log):
    t = pairs.outcomes(sorted(glob.glob(f"{log.dir}/p*.pt")), data.sample(log.input))
    t.to_parquet(f"{log.dir}/proteins.parquet")
    cols = ["precision", "gap", "contact", "matched", "share", "share_not_replaced"]
    log.note(chance=t.chance.mean(),
             **{g: dict(means=t[t.group == g].groupby("set", sort=False)[cols].mean().to_dict("index"),
                        less={f"{a}_less_{b}": pairs.paired(t, g, a, b) for a, b in [("value_site", "value_rest"), ("value_site", "value"), ("value_rest", "clean"), ("value_site", "clean")]})
                for g in ("holdout", "set_aside")})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
