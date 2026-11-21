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

import torch

from argus import data, edit, net, pairs, pick, runlog

HEADS, UNITS, SEED, BUDGET = [8, 64, 512], [64, 512, 4096], 0, 8192


def sets(rank, p, U, H):
    """{name: (units, heads)} held in protein p."""
    e, s = torch.empty(0, dtype=torch.long), {}
    for k in HEADS:
        top = rank["readable"][:k]
        s[f"readable_{k}"], s[f"random_heads_{k}"], s[f"grad_heads_{k}"] = (e, top), (e, edit.at_random(top, H, SEED + 1000 * p + k)), (e, rank["grad_heads"][:k])
    for n in UNITS:
        top = rank["grad_units"][:n]
        s[f"grad_units_{n}"], s[f"random_units_{n}"] = (top, e), (edit.at_random(top, U, SEED + 1000 * p + n), e)
    return s


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
            coupling, share, own = edit.coupling(m, pick.ids_of(df, p), {k: ("held", *v) for k, v in held.items()}, U, H, log, BUDGET)
        torch.save(dict(protein=p, sets=["clean", *held], held=held, coupling=coupling, share=share, own=own), f"{log.dir}/p{p}.pt")
        log.note(**{f"p{p}": dict(residues=coupling.shape[1], own_residue_max_abs=own.max().item(), share_not_replaced=dict(zip(held, share[:, 1].tolist())))})
    log.write()


def reduce(log):
    t = pairs.outcomes(sorted(glob.glob(f"{log.dir}/p*.pt")), data.sample(log.input))
    t.to_parquet(f"{log.dir}/proteins.parquet")
    groups = ("holdout", "set_aside")
    cols = ["precision", "gap", "contact", "matched", "share", "share_not_replaced"]
    log.note(chance=t.chance.mean(),
             primary={g: {k: pairs.paired(t, g, f"readable_{k}", f"random_heads_{k}") for k in HEADS} for g in groups},
             secondary={g: dict(means=t[t.group == g].groupby("set", sort=False)[cols].mean().to_dict("index"),
                                grad_heads_less_readable={k: pairs.paired(t, g, f"grad_heads_{k}", f"readable_{k}") for k in HEADS},
                                grad_units_less_random_units={n: pairs.paired(t, g, f"grad_units_{n}", f"random_units_{n}") for n in UNITS}) for g in groups})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
