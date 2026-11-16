"""Does ESM C 6B need the residues outside two contacting stretches of a protein to couple them? For
the proteins of inputs/<input>/jump.json, as in Zhang et al. 2024 and Nainani et al. 2025: every
residue is masked but two stretches of 2 * HALF + 1 residues and the k residues outside each (before
the first, after the second), for k = 0 to FLANKS, and last with nothing masked. Every residue of the
two stretches is replaced by each of the 20 amino acids. How do the amino-acid logits change at the
residues of the two stretches, and how far does the residual state after every block move there? What
are the logits at every residue before any replacement?
"""

import json

import torch

from argus import data, net, pick, runlog

HALF, FLANKS, BUDGET = 5, 96, 8192
AA = slice(4, 24)


def masked(ids, a, b, k):
    """ids (1, T) with every residue masked but token positions a and b, the k before a and the k after b."""
    keep = torch.zeros_like(ids, dtype=torch.bool)
    keep[:, max(1, int(a[0]) - k) : int(a[-1]) + 1] = keep[:, int(b[0]) : int(b[-1]) + 1 + k] = True
    keep[:, [0, -1]] = True
    return torch.where(keep, ids, data.MASK)


def scan(m, ids, at, log):
    """Logits before any replacement (T, 20); their change at `at` under every replacement there (E, 20, E, 20);
    the distance every block's state moves at `at`, summed over the 20 amino acids (N, E, E); the largest
    logit change in a row that leaves the input as it is."""
    x = []
    l0 = net.logits(m, net.run(m, ids, tap=lambda l, r: x.append(r["x"][0, at])))[0, :, AA]
    state, rows = l0.new_zeros(len(x), len(at), len(at)), []
    for pos, mids in data.mutants(ids, BUDGET, at):
        if log.rec["tok_s"] is None:
            log.plain(m, mids)
        with log.op("changed", mids.numel()):
            d = []
            lg = net.logits(m, net.run(m, mids, tap=lambda l, r: d.append((r["x"][:, at] - x[l]).square().sum(-1))))[:, at, AA] - l0[at]
        state.index_add_(1, torch.searchsorted(at, pos), torch.stack(d))
        rows.append((lg.cpu(), (mids == ids).all(1).cpu()))
    lg, own = (torch.cat(t) for t in zip(*rows))
    return l0.cpu(), lg.view(len(at), 20, len(at), 20).half(), state.sqrt().cpu(), lg[own].abs().max().item()


def recovery(c):
    """c (K, E, E), how much a replacement at one residue moves another: each row's coupling between the two
    stretches along that of the last row, as a share of the last row's."""
    h = c.shape[-1] // 2
    x = (c[:, :h, h:] + c[:, h:, :h].transpose(1, 2)).flatten(1)
    return ((x * x[-1]).sum(1) / x[-1].square().sum()).tolist()


def main():
    log = runlog.start("jump", __doc__)
    m, df = net.load(), data.sample(log.input)
    targets = json.load(open(data.path(log.input, "jump.json")))
    for name, t in list(targets.items())[log.shard[0] :: log.shard[1]]:
        ids = pick.ids_of(df, t["protein"])
        a, b = (torch.arange(c - HALF, c + HALF + 1, device=ids.device) + 1 for c in t["centres"])
        at = torch.cat([a, b])
        with torch.no_grad():
            clean, logits, state, own = zip(*[scan(m, x, at, log) for x in [masked(ids, a, b, k) for k in range(FLANKS + 1)] + [ids]])
        logits, state = torch.stack(logits), torch.stack(state)
        torch.save(dict(t, ids=ids[0].cpu(), at=at.cpu(), logits_clean=torch.stack(clean), logits=logits, state=state), f"{log.dir}/{name}.pt")
        log.note(**{name: dict(own_residue_max_abs=max(own), recovery_logits=recovery(logits.float().square().sum((2, 4)).sqrt()),
                               recovery_last_state=recovery(state[:, -1]))})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
