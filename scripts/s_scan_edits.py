"""Does a substitution reach the predicted residues through chosen units? In one protein, every
residue is replaced by each of the 20 amino acids. How do the chosen units' gate and value change at
every residue, and how do the amino-acid logits change at every residue? How much of the logit change
goes when the units are held at their values on the unchanged protein, and how much appears when the
unchanged protein is given only the units' values from the changed one?
"""

import json

import torch

from argus import data, net, pick, runlog

BUDGET = 8192
AA = slice(4, 24)


def scan(m, ids, units, log):
    by, units = {}, sorted(units)
    for l, u in units:
        by.setdefault(l, []).append(u)
    first, T = min(by), ids.shape[1]

    def after(batch, x, edits=None):
        """Logits and the units' g, v, a (B, T, units of the block) per block, from the state entering `first`."""
        gva = {}

        def tap(l, r):
            if l in by:
                gva[l] = [r[k][..., by[l]] for k in "gva"]

        return net.logits(m, net.run(m, batch, x=x, start=first, edits=edits, tap=tap))[..., AA], gva

    def put(gva):
        return {("act", l): net.put((slice(None), slice(None), by[l]), gva[l][2]) for l in by}

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
        d = [torch.cat([c[l][k] - w[l][k] for l in sorted(by)], -1) for k in range(2)]
        rows.append([x.cpu() for x in (lm - lw, lm - lh, lg - lw, d[0], d[1], own)])
    logits, held, given, dg, dv, own = (torch.cat(x) for x in zip(*rows))
    shape = lambda x: x.view(T - 2, 20, *x.shape[1:])
    out = dict(units=units, ids=ids[0].cpu(), logits=shape(logits), held=shape(held), given=shape(given), dg=shape(dg), dv=shape(dv),
               g=torch.cat([w[l][0] for l in sorted(by)], -1)[0].cpu(), v=torch.cat([w[l][1] for l in sorted(by)], -1)[0].cpu(), logits_clean=lw[0].cpu())
    return out, {k: x[own].abs().max().item() for k, x in dict(logits=logits, held=held, given=given, dg=dg, dv=dv).items()}


def main():
    log = runlog.start("scan_edits", __doc__)
    m, df = net.load(), data.sample(log.input)
    targets = json.load(open(data.path(log.input, "scan.json")))
    for name, t in list(targets.items())[log.shard[0] :: log.shard[1]]:
        with torch.no_grad():
            out, own = scan(m, pick.ids_of(df, t["protein"]), [tuple(u) for u in t["units"]], log)
        torch.save(dict(out, protein=t["protein"]), f"{log.dir}/{name}.pt")
        log.note(**{name: dict(own_residue_max_abs=own, logits=out["logits"].norm().item(), held=out["held"].norm().item(), given=out["given"].norm().item())})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
