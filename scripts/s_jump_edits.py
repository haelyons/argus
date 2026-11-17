"""What does ESM C 6B hold outside two contacting stretches of a protein that switches their coupling
on? For the pairs of inputs/<input>/jump_edits.json: two maskings of one protein as in s_jump, `off`
and `on`, a few unmasked residues apart; under `on` the stretches couple as in the unmasked protein,
under `off` they do not. Under each masking every residue of the two stretches is replaced by each of
the 20 amino acids: with no edit; with the state after one block (STATES) set, at every token outside
the stretches, to what it is before any replacement under the other masking, or under the same one;
and so for the activation of every unit of a range of blocks (RANGES). How do the amino-acid logits
change at the residues of the two stretches, and how far does the state after every block move there?
"""

import glob
import json

import torch

from argus import data, net, pick, runlog

HALF, BUDGET = 5, 8192
STATES = [3, 7, 15, 23, 31, 39, 47, 55, 63, 71]
RANGES = [(0, 16), (16, 32), (32, 48), (48, 64), (64, 80), (0, 80)]
AA = slice(4, 24)
MASKING, SOURCE, EDIT = ["off", "on"], ["other", "same", "none"], ["state", "units", "none"]
ROWS = [(r, 2, 2, 0) for r in (0, 1)] + [(r, s, e, i) for r in (0, 1) for s in (0, 1) for e, n in ((0, len(STATES)), (1, len(RANGES))) for i in range(n)]


def clean(m, ids):
    """Before any replacement: the state after every block (T, D) and every unit's activation in it (T, U)."""
    x, a = [], []
    net.run(m, ids, tap=lambda l, r: (x.append(r["x"][0]), a.append(r["a"][0])))
    return x, a


def edits(e, i, rest, x, a):
    """Sets, at token positions `rest`, the state after block STATES[i] or the activations of blocks RANGES[i]."""
    if e == 0:
        return {("state", STATES[i]): net.put((slice(None), rest), x[STATES[i]][rest])}
    return {("act", l): net.put((slice(None), rest), a[l][rest]) for l in range(*RANGES[i])} if e == 1 else {}


def scan(m, ids, at, ed, log):
    """Under edits `ed`: the change of the logits at `at` under every replacement there (E, 20, E, 20); the
    distance every block's state moves at `at`, summed over the 20 amino acids (N, E, E); the largest logit
    change in a row that leaves the input as it is."""
    x = []
    l0 = net.logits(m, net.run(m, ids, edits=ed, tap=lambda l, r: x.append(r["x"][0, at])))[0, at, AA]
    state, rows = l0.new_zeros(len(x), len(at), len(at)), []
    for pos, mids in data.mutants(ids, BUDGET, at):
        if log.rec["tok_s"] is None:
            log.plain(m, mids)
        with log.op("edited" if ed else "changed", mids.numel()):
            d = []
            lg = net.logits(m, net.run(m, mids, edits=ed, tap=lambda l, r: d.append((r["x"][:, at] - x[l]).square().sum(-1))))[:, at, AA] - l0
        state.index_add_(1, torch.searchsorted(at, pos), torch.stack(d))
        rows.append((lg.cpu(), (mids == ids).all(1).cpu()))
    lg, own = (torch.cat(t) for t in zip(*rows))
    return lg.view(len(at), 20, len(at), 20).half(), state.sqrt().cpu(), lg[own].abs().max().item()


def reduce(log, targets):
    for name, t in targets.items():
        ps = [torch.load(f) for f in sorted(glob.glob(f"{log.dir}/{name}-shard*.pt"))]
        rows = torch.cat([p["rows"] for p in ps]).tolist()
        key = sorted(range(len(rows)), key=rows.__getitem__)
        d = {f: torch.cat([p[f] for p in ps])[key] for f in ("rows", "logits", "state", "own")}
        torch.save(dict(t, ids=ps[0]["ids"], at=ps[0]["at"], inputs=ps[0]["inputs"], states=STATES, ranges=RANGES, **d), f"{log.dir}/{name}.pt")
        on, note = sorted(rows).index([1, 2, 2, 0]), dict(own_residue_max_abs=d["own"].max().item())
        for f, c in (("logits", d["logits"].float().square().sum((2, 4)).sqrt()), ("last_state", d["state"][:, -1])):
            for (r, s, e, _), v in zip(sorted(rows), data.pattern(c, c[on])):
                note.setdefault(f"{f}_{MASKING[r]}_{SOURCE[s]}_{EDIT[e]}", []).append(v)
        log.note(**{name: note})
    log.write()
    print(open(log.path).read())


def main():
    log = runlog.start("jump_edits", __doc__)
    targets = json.load(open(data.path(log.input, "jump_edits.json")))
    if log.reduce:
        return reduce(log, targets)
    m, df = net.load(), data.sample(log.input)
    for name, t in targets.items():
        ids = pick.ids_of(df, t["protein"])
        a, b = (torch.arange(c - HALF, c + HALF + 1, device=ids.device) + 1 for c in t["centres"])
        at, rows = torch.cat([a, b]), ROWS[log.shard[0] :: log.shard[1]]
        rest = torch.tensor([i for i in range(ids.shape[1]) if i not in at.tolist()], device=ids.device)
        x = [data.masked(ids, data.unmasked(ids, a, b, t[k], t["protein"] if t["random"] else None)) for k in MASKING]
        with torch.no_grad():
            c = [clean(m, i) for i in x]
            logits, state, own = zip(*[scan(m, x[r], at, edits(e, i, rest, *c[r ^ (s == 0)]), log) for r, s, e, i in rows])
        torch.save(dict(ids=ids[0].cpu(), at=at.cpu(), inputs=torch.cat(x).cpu(), rows=torch.tensor(rows), logits=torch.stack(logits),
                        state=torch.stack(state), own=torch.tensor(own)), f"{log.dir}/{name}-shard{log.shard[0]}.pt")
    log.write()


if __name__ == "__main__":
    main()
