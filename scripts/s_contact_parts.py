"""Which of the heads whose attention shows contacts does the coupling between residues in contact need,
and do they carry it alone? On the proteins of s_contact_edits, every residue is replaced by each of the
20 amino acids. Held at their values on the unchanged protein: the first 16, 32 and TOP heads of
s_contact_rank's attention ranking; the first TOP without the first 8; each half of a random split of
the first TOP, drawn anew for each protein; the TOP after them; and, in the blocks of the first TOP, as
many of the other heads, those whose attention reaches farthest along the chain. Given to the unchanged
protein, alone, at their values under the replacement: the first TOP; as many heads at random in the
same blocks, drawn anew for each protein; and the TOP that the gradient puts first. Per protein and set,
as in s_contact_edits: the coupling, how many of its L strongest pairs 24 or more apart are in contact,
its mean at pairs in contact and at pairs not in contact at the same separations, and the share of the
logit change that goes when the set is held or appears when it is given.
"""

import glob

import torch

from argus import data, edit, net, pairs, pick, runlog

TOP, SEED, BUDGET = 64, 0, 8192


def farthest(top, far, H):
    """As many heads as `top` in each of its blocks: of the block's others, those with the largest far (N, H)."""
    out = []
    for l, n in zip(*(top // H).unique(return_counts=True)):
        block = l * H + far[l].argsort(descending=True)
        out.append(block[~torch.isin(block, top)][:n])
    return torch.cat(out)


def sets(r, p, H):
    """{name: (how, units, heads)} for protein p."""
    rank, e = r["rank"]["readable"], torch.empty(0, dtype=torch.long)
    top = rank[:TOP]
    half = top[torch.randperm(TOP, generator=torch.Generator().manual_seed(SEED + 1000 * p))]
    held = dict(top_16=rank[:16], top_32=rank[:32], top=top, top_less_8=rank[8:TOP], half_a=half[: TOP // 2], half_b=half[TOP // 2 :],
                next=rank[TOP : 2 * TOP], far=farthest(top, r["score"]["far"], H))
    given = dict(top=top, random=edit.at_random(top, H, SEED + 1000 * p + TOP), grad=r["rank"]["grad_heads"][:TOP])
    return {f"held_{k}": ("held", e, v) for k, v in held.items()} | {f"given_{k}": ("given", e, v) for k, v in given.items()}


def main():
    log = runlog.start("contact_parts", __doc__)
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
    r = torch.load(log.read("contact_rank") + "/rank.pt")
    s = sets(r, 0, r["score"]["far"].shape[1])
    cols = ["precision", "gap", "contact", "matched", "share", "share_not_replaced"]
    against = dict(held_top=[k for k in s if k.startswith("held_") and k != "held_top"], given_top=["given_random", "given_grad"])
    log.note(chance=t.chance.mean(),
             attention={k: {y: r["score"][y].flatten()[s[k][2]].mean().item() for y in ("readable", "far")} for k in ("held_top", "held_next", "held_far")},
             **{g: dict(means=t[t.group == g].groupby("set", sort=False)[cols].mean().to_dict("index"),
                        less={a: {b: pairs.paired(t, g, b, a) for b in bs} for a, bs in against.items()}) for g in ("holdout", "set_aside")})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
