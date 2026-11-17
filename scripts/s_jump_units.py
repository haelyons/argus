"""Which units, at which tokens, carry what switches the coupling of two contacting stretches on? For
the pairs of inputs/<input>/jump_units.json, as in s_jump_trace, each with a range of blocks and one
of its sets of tokens. Scored on every other residue of the stretches replaced by PICK, half of the
amino acids: for every unit of the blocks at every token of the set, how much its response to the
replacement differs between `on` and `off`, and how much its activation differs between them before
any replacement, each times the norm of what the unit writes. Then every residue of the stretches is
replaced by each of the 20 amino acids, under `off` while units are given their activations under
`on` with the same replacement, and under `on` while they are given those under `off`: the N highest
by either score and N at random, for N in NS, and every unit at every token of the set. How do the
amino-acid logits change at the residues of the two stretches, how far does the state after every
block move there, and what are the logits at every residue under the replacement that leaves the
input as it is?
"""

import glob
import json

import torch

from argus import data, edit, net, pick, runlog

HALF = 5
NS = [300, 3000, 30000]
PICK = torch.randperm(20, generator=torch.Generator().manual_seed(0))[:10].sort().values
RANK = ["response", "context", "random", "all"]
GIVEN = ["none", "on_to_off", "off_to_on"]
ROWS = [(0, j, 0) for j in (0, 1)] + [(d, k, n) for d in (1, 2) for k in (0, 1, 2) for n in range(len(NS))] + [(d, 3, 0) for d in (1, 2)]


def muts(x, at, k):
    """off and on with token at[k] replaced by each amino acid (20, T); the row that leaves it as it is."""
    return [next(data.mutants(i, 20 * i.shape[1], at[k : k + 1]))[1] for i in x], int(x[0][0, at[k]]) - 4


def score(m, x, at, a, b, new, s, blocks, log):
    """(2, L, T, U): for every unit of `blocks` at every token, summed over the scored replacements where the
    token is in set s: the difference between `on` and `off` of its response to the replacement, and of its
    activation without one; each times the norm of what the unit writes."""
    L = range(*blocks)
    w = torch.stack([net.out_weight(m, l).norm(dim=1) for l in L])
    out = w.new_zeros(2, len(L), x[0].shape[1], w.shape[1])
    for k in range(0, len(at), 2):
        mut, same = muts(x, at, k)
        M, g = data.sets(x[0], int(at[k]), a, b, new)[s], {}

        def tap(l, r):
            if l in L:
                d = g.pop(l) - r["a"][:, M]
                out[0, l - blocks[0], M] += (d - d[same])[PICK].abs().sum(0)
                out[1, l - blocks[0], M] += d[same].abs()

        with log.op("score", 2 * mut[0].numel()):
            net.run(m, mut[1], stop=blocks[1], tap=lambda l, r: g.update({l: r["a"][:, M]}) if l in L else None)
            net.run(m, mut[0], stop=blocks[1], tap=tap)
    return out * w[None, :, None]


def picks(sc, first, seed):
    """[rank][n] (NS[n], 3) of block, token, unit: the highest by each score, and as many of the scored at random."""
    T, U = sc.shape[2:]
    flat = sc.flatten(1)
    scored = (flat[1] > 0).nonzero()[:, 0]
    some = scored[torch.randperm(len(scored), generator=torch.Generator().manual_seed(seed))[: NS[-1]].to(scored.device)]
    return [[torch.stack([i[:N] // (T * U) + first, i[:N] // U % T, i[:N] % U], 1) for N in NS] for i in (*flat.topk(NS[-1]).indices, some)]


def scan(m, x, at, a, b, new, s, blocks, pk, rows, log):
    """As s_jump_trace's scan, for rows (given, rank, n): logits (R, E, 20, E, 20), state (R, N, E, E), clean (R, E, T, 20)."""
    out, L = [], range(*blocks)
    for k in range(len(at)):
        mut, same = muts(x, at, k)
        M, g = data.sets(x[0], int(at[k]), a, b, new)[s], [{}, {}]
        of = M.cumsum(0) - 1
        if log.rec["tok_s"] is None:
            log.plain(m, mut[0])
        with log.op("donor", 2 * mut[0].numel()):
            for j in (0, 1):
                net.run(m, mut[j], stop=blocks[1], tap=lambda l, r: g[j].update({l: r["a"][:, M]}) if l in L else None)
        for d, r, n in rows:
            ed, v = {}, g[2 - d] if d else None
            if d and r == 3:
                ed = {("act", l): net.put((slice(None), M), v[l]) for l in L}
            elif d:
                q = pk[r][n]
                q = q[M[q[:, 1]]]
                ed = {("act", l): net.put((slice(None), t, u), v[l][:, of[t], u]) for l in L for t, u in [q[q[:, 0] == l, 1:].T]}
            with log.op(GIVEN[d], mut[0].numel()):
                out.append(edit.moved(m, mut[d - 1 if d else r], at, same, ed))
    logits, state, clean = (torch.stack(t).view(len(at), len(rows), *t[0].shape).transpose(0, 1) for t in zip(*out))
    return logits, state.transpose(1, 2), clean


def reduce(log, targets):
    held = torch.tensor([i for i in range(20) if i not in PICK.tolist()])
    for name, t in targets.items():
        ps = [torch.load(f) for f in sorted(glob.glob(f"{log.dir}/{name}-shard*.pt"))]
        rows = torch.cat([p["rows"] for p in ps]).tolist()
        key = sorted(range(len(rows)), key=rows.__getitem__)
        d = {f: torch.cat([p[f] for p in ps])[key] for f in ("rows", "logits", "state", "clean")}
        torch.save(dict(t, ids=ps[0]["ids"], at=ps[0]["at"], inputs=ps[0]["inputs"], new=ps[0]["new"], picks=ps[0]["picks"], pick=PICK, ns=NS, rank=RANK, given=GIVEN, **d), f"{log.dir}/{name}.pt")
        c, h, cl = d["logits"][:, :, held].float().square().sum((2, 4)).sqrt(), len(ps[0]["at"]) // 2, d["clean"].float().flatten(1)
        size, step = torch.cat([c[:, :h, h:], c[:, h:, :h]], 1).flatten(1).norm(dim=1), (cl[1] - cl[0]).norm()
        log.note(**{name: dict(rows=d["rows"].tolist(), held_pattern_logits=data.pattern(c, c[1]), held_size_logits=(size / size[1]).tolist(),
                               clean_from_off=((cl - cl[0]).norm(dim=1) / step).tolist(), clean_from_on=((cl - cl[1]).norm(dim=1) / step).tolist())})
    log.write()
    print(open(log.path).read())


def main():
    log = runlog.start("jump_units", __doc__)
    targets = json.load(open(data.path(log.input, "jump_units.json")))
    if log.reduce:
        return reduce(log, targets)
    m, df = net.load(), data.sample(log.input)
    R = ROWS[len(ROWS) * log.shard[0] // log.shard[1] : len(ROWS) * (log.shard[0] + 1) // log.shard[1]]
    for name, t in targets.items():
        ids = pick.ids_of(df, t["protein"])
        a, b = (torch.arange(c - HALF, c + HALF + 1, device=ids.device) + 1 for c in t["centres"])
        x = [data.masked(ids, data.unmasked(ids, a, b, t[k], t["protein"] if t["random"] else None)) for k in ("off", "on")]
        at, new, s = torch.cat([a, b]), (x[0] != x[1])[0].nonzero()[:, 0], data.SETS.index(t["set"])
        with torch.no_grad():
            pk = picks(score(m, x, at, a, b, new, s, t["blocks"], log), t["blocks"][0], t["protein"])
            logits, state, clean = scan(m, x, at, a, b, new, s, t["blocks"], pk, R, log)
        torch.save(dict(ids=ids[0].cpu(), at=at.cpu(), inputs=torch.cat(x).cpu(), new=new.cpu(), picks=[[q.cpu() for q in r] for r in pk], rows=torch.tensor(R),
                        logits=logits, state=state, clean=clean), f"{log.dir}/{name}-shard{log.shard[0]}.pt")
    log.write()


if __name__ == "__main__":
    main()
