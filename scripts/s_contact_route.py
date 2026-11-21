"""Do the heads whose attention shows contacts carry the coupling between residues in contact by where
they attend or by what they move? On the proteins of s_contact_edits, every residue is replaced by each
of the 20 amino acids, for the first TOP heads of s_contact_rank's attention ranking and for as many
heads at random in the same blocks, drawn anew for each protein. Of each set, kept as on the unchanged
protein: only its attention patterns, so that the heads move what the replacement makes of the values
along their old patterns; only the values the patterns mix, so that the heads attend as the replacement
makes them and move the old values; and its outputs, which is both. Per protein and set, as in
s_contact_edits: the coupling, how many of its L strongest pairs 24 or more apart are in contact, its
mean at pairs in contact and at pairs not in contact at the same separations, and the share of the
logit change that goes.
"""

import glob

import torch

from argus import data, edit, net, pairs, pick, runlog

TOP, SEED, BUDGET = 64, 0, 8192
HOW = ["pattern", "value", "held"]


def sets(r, p, H):
    """{name: (how, units, heads)} for protein p."""
    top, e = r["rank"]["readable"][:TOP], torch.empty(0, dtype=torch.long)
    heads = dict(top=top, random=edit.at_random(top, H, SEED + 1000 * p + TOP))
    return {f"{how}_{k}": (how, e, v) for k, v in heads.items() for how in HOW}


def main():
    log = runlog.start("contact_route", __doc__)
    if log.reduce:
        return reduce(log)
    m, df = net.load(), data.sample(log.input)
    r = torch.load(log.read("contact_rank") + "/rank.pt")
    U, H = net.blocks(m)[0].ffn.ffn_hidden_size, net.blocks(m)[0].attn.n_heads
    order = df[(df.split == "holdout") | df.idx.isin(r["kept"])].sort_values("Length", ascending=False).idx.tolist()
    i, n = log.shard
    for p in [p for k, p in enumerate(order) if min(k % (2 * n), 2 * n - 1 - k % (2 * n)) == i]:
        s = sets(r, p, H)
        with torch.no_grad():
            coupling, share, own = edit.coupling(m, pick.ids_of(df, p), s, U, H, log, BUDGET)
        torch.save(dict(protein=p, sets=["clean", *s], heads={k: v[2] for k, v in s.items()}, coupling=coupling, share=share, own=own), f"{log.dir}/p{p}.pt")
        log.note(**{f"p{p}": dict(residues=coupling.shape[1], own_residue_max_abs=own.max().item(), share_not_replaced=dict(zip(s, share[:, 1].tolist())))})
    log.write()


def reduce(log):
    t = pairs.outcomes(sorted(glob.glob(f"{log.dir}/p*.pt")), data.sample(log.input))
    t.to_parquet(f"{log.dir}/proteins.parquet")
    cols = ["precision", "gap", "contact", "matched", "share", "share_not_replaced"]
    log.note(chance=t.chance.mean(),
             **{g: dict(means=t[t.group == g].groupby("set", sort=False)[cols].mean().to_dict("index"),
                        less={f"{a}_less_{b}": pairs.paired(t, g, a, b) for a, b in [("pattern_top", "value_top"), ("pattern_top", "clean"), ("value_top", "held_top"),
                                                                                    ("pattern_top", "pattern_random"), ("value_top", "value_random")]}) for g in ("holdout", "set_aside")})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
