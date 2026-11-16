"""Is the effect of replacing two residues of a protein the sum of the effects of replacing each? For
the residue pairs of inputs/<input>/doubles.json, the two residues are replaced by each of the 20
amino acids, alone and together. How do the amino-acid logits change at every residue? How do the
gate and value of the units named with the pairs change at every residue?
"""

import json

import torch

from argus import data, net, pick, runlog

BUDGET = 8192
AA = slice(4, 24)


def scan(m, ids, pairs, units, log):
    by = {}
    for l, u in units:
        by.setdefault(l, []).append(u)

    def read(batch):
        """Logits (B, T, 20) and the units' gate and value (B, T, units)."""
        gv = {}

        def tap(l, r):
            if l in by:
                gv[l] = [r[k][..., by[l]] for k in "gv"]

        lg = net.logits(m, net.run(m, batch, tap=tap))[..., AA]
        return lg, [torch.cat([gv[l][k] for l in sorted(by)], -1) for k in range(2 * bool(by))]

    l0, w = read(ids)
    rows = []
    for _, mids in data.doubles(ids, pairs, BUDGET):
        if log.rec["tok_s"] is None:
            log.plain(m, mids)
        with log.op("changed", mids.numel()):
            lg, c = read(mids)
        rows.append([x.cpu() for x in (lg - l0, (mids == ids).all(1), *[x - y for x, y in zip(c, w)])])
    logits, own, *d = (torch.cat(x) for x in zip(*rows))
    shape = lambda x: x.view(len(pairs), 20, 20, *x.shape[1:]).half()
    out = dict(ids=ids[0].cpu(), logits=shape(logits), logits_clean=l0[0].cpu())
    if by:
        out |= dict(dg=shape(d[0]), dv=shape(d[1]), g=w[0][0].cpu(), v=w[1][0].cpu())
    return out, logits[own].abs().max().item()


def main():
    log = runlog.start("scan_doubles", __doc__)
    m, df = net.load(), data.sample(log.input)
    targets = json.load(open(data.path(log.input, "doubles.json")))
    for name, t in list(targets.items())[log.shard[0] :: log.shard[1]]:
        ids = pick.ids_of(df, t["protein"])
        t["units"] = sorted(tuple(u) for u in t.get("units", []))
        with torch.no_grad():
            out, own = scan(m, ids, torch.tensor(t["pairs"], device=ids.device), t["units"], log)
        torch.save(dict(t, **out), f"{log.dir}/{name}.pt")
        log.note(**{name: dict(pairs=len(t["pairs"]), own_residue_max_abs=own, logits=out["logits"].float().norm().item())})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
