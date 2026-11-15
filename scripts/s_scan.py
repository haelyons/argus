"""Which residues of a protein decide a unit's activity where the unit is strong, and do the same
residues decide what ESM C 6B predicts? In one protein at a time, every residue is replaced by each of
the 20 amino acids. For the PAIRS poles of each block that write most into the residual state
(activation times the length of the unit's output vector), how do gate and value change at the
residue where the pole writes most in this protein? How far does the residual state after every block
move at every residue, and how do the amino-acid logits change at every residue?
"""

import torch

from argus import data, net, pick, runlog

PROTEINS, MAX_LEN, SEED, BUDGET, PAIRS = 100, 200, 0, 8192, 50
AA = slice(4, 24)


def strong(m, ids):
    """The clean pass: states after every block (N, T, D), and each block's PAIRS poles (pole, unit) with the
    largest write, each at the residue where it is largest, in block order."""
    x, best = [], []

    def tap(l, r):
        x.append(r["x"][0])
        best.append((torch.stack([r["a"][0], -r["a"][0]]) * net.out_weight(m, l).norm(dim=-1)).max(1))

    net.run(m, ids, tap=tap)
    size, pos = (torch.stack([b[k] for b in best]) for k in range(2))
    U = size.shape[-1]
    i = (torch.arange(len(x), device=size.device)[:, None] * 2 * U + size.flatten(1).topk(PAIRS, 1).indices).flatten()
    return torch.stack(x), dict(block=i // (2 * U), pole=1 - 2 * (i // U % 2), unit=i % U, pos=pos.flatten()[i], size=size.flatten()[i])


def read(m, batch, x, pairs):
    """Gate and value at the pairs (B, P), squared distance of every state from the clean one (N, B, T), logits (B, T, 20)."""
    g, v, d = [], [], []

    def tap(l, r):
        k = pairs["block"] == l
        g.append(r["g"][:, pairs["pos"][k], pairs["unit"][k]])
        v.append(r["v"][:, pairs["pos"][k], pairs["unit"][k]])
        d.append((r["x"] - x[l]).square().sum(-1))

    lg = net.logits(m, net.run(m, batch, tap=tap))[..., AA]
    return torch.cat(g, 1), torch.cat(v, 1), torch.stack(d), lg


def main():
    log = runlog.start("scan", __doc__)
    m, df = net.load(), data.sample(log.input)
    part = df[(df.split == "train") & (df.Length <= MAX_LEN)].sample(PROTEINS, random_state=SEED)
    for p in part.idx.tolist()[log.shard[0] :: log.shard[1]]:
        ids = pick.ids_of(df, p)
        T = ids.shape[1]
        with torch.no_grad():
            x, pairs = strong(m, ids)
            g0, v0, _, l0 = read(m, ids, x, pairs)
            state, rows = x.new_zeros(len(x), T - 2, T), []
            for pos, mids in data.mutants(ids, BUDGET):
                if log.rec["tok_s"] is None:
                    log.plain(m, mids)
                with log.op("changed", mids.numel()):
                    g, v, d, lg = read(m, mids, x, pairs)
                state.index_add_(1, pos - 1, d)
                rows.append([t.cpu() for t in (g - g0, v - v0, lg - l0, (mids == ids).all(1))])
        dg, dv, logits, own = (torch.cat(t) for t in zip(*rows))
        shape = lambda t: t.view(T - 2, 20, *t.shape[1:]).half()
        torch.save(dict({k: t.cpu() for k, t in pairs.items()}, protein=p, ids=ids[0].cpu(), g=g0[0].cpu(), v=v0[0].cpu(), logits_clean=l0[0].cpu(),
                        dg=shape(dg), dv=shape(dv), logits=shape(logits), state=state.sqrt().cpu()), f"{log.dir}/p{p}.pt")
        log.note(**{f"p{p}": dict(residues=T - 2, pairs=len(pairs["unit"]),
                                  own_residue_max_abs={k: t[own].abs().max().item() for k, t in dict(dg=dg, dv=dv, logits=logits).items()})})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
