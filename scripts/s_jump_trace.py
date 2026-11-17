"""Where does ESM C 6B hold what switches the coupling of two contacting stretches on? For the pairs
of inputs/<input>/jump_edits.json: two maskings of one protein as in s_jump, `off` and `on`, a few
unmasked residues apart; under `on` the stretches couple as in the unmasked protein, under `off` they
do not. Every residue of the two stretches is replaced by each of the 20 amino acids: under `off`,
under `on`, and under `off` with one of the residues that `on` adds unmasked. Then under `off` while
a set of tokens is given what it holds under `on` with the same replacement: the state after one
block (STATES), or the activations of every unit of a range of blocks (RANGES). The sets: the
replaced residue (site), the rest of its stretch (own), the other stretch (far), the residues `on`
adds (new; for the state, each alone too), the other unmasked residues (seen), every other token
(rest). How do the amino-acid logits change at the residues of the two stretches, how far does the
state after every block move there, and what are the logits at every residue under the replacement
that leaves the input as it is?
"""

import glob
import json

import torch

from argus import data, net, pick, runlog

HALF = 5
STATES = [3, 15, 31, 47, 55, 63, 71, 75]
RANGES = [(0, 16), (16, 32), (32, 48), (48, 64), (64, 80)]
SETS = ["site", "own", "far", "new", "seen", "rest"]
EDIT = ["none", "state", "units"]
AA = slice(4, 24)


def rows(n):
    """(edit, which, where) for n new residues: no edit under masking `which` (off, on, off with new residue
    which - 2); the state after block STATES[where] or the units of blocks RANGES[where], at the set `which`
    (SETS, then each new residue)."""
    return ([(0, j, 0) for j in range(2 + n)] + [(1, s, i) for s in range(len(SETS) + n) for i in range(len(STATES))]
            + [(2, s, i) for i in range(len(RANGES)) for s in range(len(SETS))])


def sets(off, p, a, b, new):
    """Token masks (len(SETS) + len(new), T) for the replaced token p: SETS, then each new residue."""
    m = torch.zeros(len(SETS) + len(new), off.shape[1], dtype=torch.bool, device=off.device)
    own, far = (a, b) if p in a else (b, a)
    m[0, p] = m[1, own] = m[2, far] = m[3, new] = True
    m[1, p] = False
    m[4] = off[0] != data.MASK
    m[4, [0, -1]] = m[4, a] = m[4, b] = False
    m[5] = ~m[:5].any(0)
    m[len(SETS) + torch.arange(len(new), device=off.device), new] = True
    return m


def scan(m, x, at, a, b, new, rows, log):
    """For maskings x = [off, on, off with each new residue] and each row: the change of the logits at `at`
    under every replacement there (R, E, 20, E, 20); the distance every block's state moves at `at`, summed
    over the 20 amino acids (R, N, E, E); the logits at every token under the replacement that changes
    nothing (R, E, T, 20)."""
    out = []
    for k, p in enumerate(at.tolist()):
        mut = [next(data.mutants(i, 20 * i.shape[1], at[k : k + 1]))[1] for i in x]
        same, M, got = int(x[0][0, p]) - 4, sets(x[0], p, a, b, new), {}
        if log.rec["tok_s"] is None:
            log.plain(m, mut[0])

        def donor(e, i):
            """What `on` holds under the same replacements: the state after each block of STATES, or the
            activations of blocks RANGES[i]. One range is kept at a time."""
            if (e, i) not in got:
                for q in [q for q in got if q[0] == 2]:
                    del got[q]
                ls, g = STATES if e == 1 else range(*RANGES[i]), {}
                with log.op("donor", mut[1].numel()):
                    net.run(m, mut[1], stop=max(ls) + 1, tap=lambda l, r: g.update({l: r["xa"[e - 1]]}) if l in ls else None)
                got[e, i] = g
            return got[e, i]

        for e, s, i in rows:
            ids, ed = mut[s if e == 0 else 0], {}
            if e == 1:
                ed = {("state", STATES[i]): net.put((slice(None), M[s]), donor(1, 0)[STATES[i]][:, M[s]])}
            if e == 2:
                ed = {("act", l): net.put((slice(None), M[s]), v[:, M[s]]) for l, v in donor(2, i).items()}
            st = []
            with log.op(EDIT[e], ids.numel()):
                lg = net.logits(m, net.run(m, ids, edits=ed, tap=lambda l, r: st.append(r["x"][:, at])))[..., AA]
            st = torch.stack(st)
            out.append(((lg[:, at] - lg[same, at]).half().cpu(), (st - st[:, same : same + 1]).square().sum((1, 3)).sqrt().cpu(), lg[same].half().cpu()))
    logits, state, clean = (torch.stack(t).view(len(at), len(rows), *t[0].shape).transpose(0, 1) for t in zip(*out))
    return logits, state.transpose(1, 2), clean


def reduce(log, targets):
    for name, t in targets.items():
        ps = [torch.load(f) for f in sorted(glob.glob(f"{log.dir}/{name}-shard*.pt"))]
        rows = torch.cat([p["rows"] for p in ps]).tolist()
        key = sorted(range(len(rows)), key=rows.__getitem__)
        d = {f: torch.cat([p[f] for p in ps])[key] for f in ("rows", "logits", "state", "clean")}
        torch.save(dict(t, ids=ps[0]["ids"], at=ps[0]["at"], inputs=ps[0]["inputs"], new=ps[0]["new"], states=STATES, ranges=RANGES, sets=SETS, **d), f"{log.dir}/{name}.pt")
        c, h, cl = d["logits"].float().square().sum((2, 4)).sqrt(), len(ps[0]["at"]) // 2, d["clean"].float().flatten(1)
        size, step = torch.cat([c[:, :h, h:], c[:, h:, :h]], 1).flatten(1).norm(dim=1), (cl[1] - cl[0]).norm()
        log.note(**{name: dict(rows=d["rows"].tolist(), pattern_logits=data.pattern(c, c[1]), pattern_last_state=data.pattern(d["state"][:, -1], d["state"][1, -1]),
                               size_logits=(size / size[1]).tolist(), clean_from_off=((cl - cl[0]).norm(dim=1) / step).tolist(), clean_from_on=((cl - cl[1]).norm(dim=1) / step).tolist())})
    log.write()
    print(open(log.path).read())


def main():
    log = runlog.start("jump_trace", __doc__)
    targets = json.load(open(data.path(log.input, "jump_edits.json")))
    if log.reduce:
        return reduce(log, targets)
    m, df = net.load(), data.sample(log.input)
    for name, t in targets.items():
        ids = pick.ids_of(df, t["protein"])
        a, b = (torch.arange(c - HALF, c + HALF + 1, device=ids.device) + 1 for c in t["centres"])
        off, on = (data.masked(ids, data.unmasked(ids, a, b, t[k], t["protein"] if t["random"] else None)) for k in ("off", "on"))
        new = (off != on)[0].nonzero()[:, 0]
        x = [off, on] + [torch.where(torch.arange(ids.shape[1], device=ids.device) == j, ids, off) for j in new]
        R = rows(len(new))
        R = R[len(R) * log.shard[0] // log.shard[1] : len(R) * (log.shard[0] + 1) // log.shard[1]]
        with torch.no_grad():
            logits, state, clean = scan(m, x, torch.cat([a, b]), a, b, new, R, log)
        torch.save(dict(ids=ids[0].cpu(), at=torch.cat([a, b]).cpu(), inputs=torch.cat(x).cpu(), new=new.cpu(), rows=torch.tensor(R), logits=logits, state=state, clean=clean),
                   f"{log.dir}/{name}-shard{log.shard[0]}.pt")
    log.write()


if __name__ == "__main__":
    main()
