"""Does what s_contact_edits, s_contact_parts and s_contact_route found hold on other proteins, scored
against experimental structures? On every protein of an input that names an experimental chain for each,
with the heads of s_contact_rank on swissprot-depth, every residue is replaced by each of the 20 amino
acids. The first TOP heads of the attention ranking: held at their values on the unchanged protein;
given alone to the unchanged protein; only their values kept; only their attention patterns kept. And
as many heads at random in the same blocks, drawn anew for each protein, held. Per protein and set: the
coupling, how many of its L strongest pairs 24 or more apart are in contact, its mean at pairs in
contact and at pairs not in contact at the same separations, and the share of the logit change that
goes or appears. Contacts are those of the experimental chain, among the residues it resolves; and
beside them those of the AlphaFold model.
"""

import glob

import pandas as pd
import torch

from argus import data, edit, net, pairs, pick, runlog

RANKED, TOP, SEED, BUDGET = "swissprot-depth", 64, 0, 8192
COMPARE = [("held_top", "held_random"), ("held_top", "clean"), ("given_top", "clean"), ("value_top", "pattern_top"), ("pattern_top", "clean")]


def main():
    log = runlog.start("contact_known", __doc__)
    if log.reduce:
        return reduce(log)
    m, df = net.load(), data.sample(log.input)
    top = torch.load(log.read("contact_rank", RANKED) + "/rank.pt")["rank"]["readable"][:TOP]
    U, H = net.blocks(m)[0].ffn.ffn_hidden_size, net.blocks(m)[0].attn.n_heads
    order, e = df.sort_values("Length", ascending=False).idx.tolist(), torch.empty(0, dtype=torch.long)
    i, n = log.shard
    for p in [p for k, p in enumerate(order) if min(k % (2 * n), 2 * n - 1 - k % (2 * n)) == i]:
        s = dict(held_top=("held", e, top), held_random=("held", e, edit.at_random(top, H, SEED + 1000 * p + TOP)), given_top=("given", e, top),
                 value_top=("value", e, top), pattern_top=("pattern", e, top))
        with torch.no_grad():
            coupling, share, own = edit.coupling(m, pick.ids_of(df, p), s, U, H, log, BUDGET)
        torch.save(dict(protein=p, sets=["clean", *s], coupling=coupling, share=share, own=own), f"{log.dir}/p{p}.pt")
        log.note(**{f"p{p}": dict(residues=coupling.shape[1], own_residue_max_abs=own.max().item(), share_not_replaced=dict(zip(s, share[:, 1].tolist())))})
    log.write()


def reduce(log):
    files, df = sorted(glob.glob(f"{log.dir}/p*.pt")), pd.read_parquet(data.path(log.input, "sample.parquet"))
    t = pd.concat([pairs.outcomes(files, df, structures=k == "structure").assign(group=k) for k in ("structure", "alphafold")])
    t.to_parquet(f"{log.dir}/proteins.parquet")
    cols = ["precision", "gap", "contact", "matched", "share", "share_not_replaced"]
    log.note(proteins=t.protein.nunique(),
             **{g: dict(chance=t[t.group == g].chance.mean(), means=t[t.group == g].groupby("set", sort=False)[cols].mean().to_dict("index"),
                        less={f"{a}_less_{b}": pairs.paired(t, g, a, b) for a, b in COMPARE}) for g in ("structure", "alphafold")})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
