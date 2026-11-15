"""Does a substitution reach the predicted residues through chosen units? In one protein, every
residue is replaced by each of the 20 amino acids. How do the chosen units' gate and value change at
every residue (kept for sets of at most KEEP units), and how do the amino-acid logits change at
every residue? How much of the logit change goes when the units are held at their values on the
unchanged protein, and how much appears when the unchanged protein is given only the units' values
from the changed one?
"""

import json

import torch

from argus import data, net, pick, runlog

BUDGET, KEEP = 8192, 64
AA = slice(4, 24)


def scan(m, ids, units, log):
    by, units = {}, sorted(units)
    for l, u in units:
        by.setdefault(l, []).append(u)
    first, T, small = min(by), ids.shape[1], len(units) <= KEEP

    def after(batch, x, edits=None):
        """Logits and the units' g, v, a (B, T, units of the block) per block, a alone for a large set, from the state entering `first`."""
        gva = {}

        def tap(l, r):
            if l in by:
                gva[l] = [r[k][..., by[l]] for k in ("gva" if small else "a")]

        return net.logits(m, net.run(m, batch, x=x, start=first, edits=edits, tap=tap))[..., AA], gva

    def put(gva):
        return {("act", l): net.put((slice(None), slice(None), by[l]), gva[l][-1]) for l in by}

    xw = net.run(m, ids, stop=first)
    lw, w = after(ids, xw)
    rows = []
    for pos, mids in data.mutants(ids, BUDGET):
        B, n = len(pos), mids.numel()
        if log.rec["tok_s"] is None:
            log.plain(m, mids)
        with log.op("changed", n):
            xm = net.run(m, mids, stop=first)
            lm, c = after(mids, xm)
        with log.op("held", n):
            lh, _ = after(mids, xm, put(w))
        with log.op("given", n):
            lg, _ = after(ids.expand(B, -1), xw.expand(B, -1, -1), put(c))
        own = (mids == ids).all(1)
        d = [torch.cat([c[l][k] - w[l][k] for l in sorted(by)], -1) for k in range(2 * small)]
        rows.append([x.cpu() for x in (lm - lw, lm - lh, lg - lw, own, *d)])
    logits, held, given, own, *d = (torch.cat(x) for x in zip(*rows))
    shape = lambda x: x.view(T - 2, 20, *x.shape[1:])
    out = dict(ids=ids[0].cpu(), logits=shape(logits), held=shape(held), given=shape(given), logits_clean=lw[0].cpu())
    if small:
        out |= dict(units=units, dg=shape(d[0]), dv=shape(d[1]), g=torch.cat([w[l][0] for l in sorted(by)], -1)[0].cpu(), v=torch.cat([w[l][1] for l in sorted(by)], -1)[0].cpu())
    return out, {k: x[own].abs().max().item() for k, x in dict(zip(("logits", "held", "given", "dg", "dv"), (logits, held, given, *d))).items()}


def main():
    log = runlog.start("scan_edits", __doc__)
    m, df = net.load(), data.sample(log.input)
    targets, U = json.load(open(data.path(log.input, "scan.json"))), net.blocks(m)[0].ffn.ffn_hidden_size
    for name, t in list(targets.items())[log.shard[0] :: log.shard[1]]:
        units = [tuple(u) for u in t["units"]] if "units" in t else [(l, u) for l in t["blocks"] for u in range(U)]
        with torch.no_grad():
            out, own = scan(m, pick.ids_of(df, t["protein"]), units, log)
        torch.save(dict(t, **out), f"{log.dir}/{name}.pt")
        log.note(**{name: dict(own_residue_max_abs=own, logits=out["logits"].norm().item(), held=out["held"].norm().item(), given=out["given"].norm().item())})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
