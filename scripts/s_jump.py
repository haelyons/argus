"""Does ESM C 6B need the residues outside two contacting stretches of a protein to couple them? For
the proteins of inputs/<input>/jump.json, as in Zhang et al. 2024 and Nainani et al. 2025: every
residue is masked but two stretches of 2 * HALF + 1 residues and the k residues outside each (before
the first, after the second), for k = 0 to FLANKS; then with as many residues unmasked at random
positions instead; then with nothing masked. Every residue of the two stretches is replaced by each
of the 20 amino acids. How do the amino-acid logits change at the residues of the two stretches, and
how far does the residual state after every block move there? What are the logits at every residue
before any replacement?
"""

import glob
import json

import torch

from argus import data, net, pick, runlog

HALF, FLANKS, BUDGET = 5, 72, 8192
AA = slice(4, 24)
KINDS = ["flank", "random", "none"]


def masked(ids, keep):
    """ids (1, T) with every residue masked but token positions `keep`."""
    m = torch.zeros_like(ids, dtype=torch.bool)
    m[:, keep] = m[:, [0, -1]] = True
    return torch.where(m, ids, data.MASK)


def inputs(ids, a, b, protein):
    """[(kind, k, ids)]: the stretches a, b and the k residues before a and after b unmasked; the stretches and
    as many of the other residues, in one random order per protein; nothing masked (k = -1)."""
    T, at = ids.shape[1], torch.cat([a, b])
    rest = torch.tensor([i for i in range(1, T - 1) if i not in at.tolist()], device=ids.device)
    rest = rest[torch.randperm(len(rest), generator=torch.Generator().manual_seed(protein)).to(ids.device)]
    out = []
    for k in range(FLANKS + 1):
        flank = torch.cat([torch.arange(max(1, int(a[0]) - k), int(a[0])), torch.arange(int(b[-1]) + 1, min(T - 1, int(b[-1]) + 1 + k))]).to(ids.device)
        out += [(0, k, masked(ids, torch.cat([at, flank]))), (1, k, masked(ids, torch.cat([at, rest[: len(flank)]])))]
    return out + [(2, -1, ids)]


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


def pattern(c):
    """c (K, E, E), how much a replacement at one residue moves another: the correlation of each row's
    coupling between the two stretches with the last row's."""
    h = c.shape[-1] // 2
    x = (c[:, :h, h:] + c[:, h:, :h].transpose(1, 2)).flatten(1)
    x = x - x.mean(1, keepdim=True)
    return ((x * x[-1]).sum(1) / (x.norm(dim=1) * x[-1].norm())).tolist()


def reduce(log, targets):
    n = FLANKS + 1
    for name, t in targets.items():
        ps = [torch.load(f) for f in sorted(glob.glob(f"{log.dir}/{name}-shard*.pt"))]
        key = torch.cat([p["kind"] * (n + 1) + p["k"] for p in ps]).argsort()
        d = {f: torch.cat([p[f] for p in ps])[key] for f in ("kind", "k", "logits_clean", "logits", "state", "own")}
        torch.save(dict(t, ids=ps[0]["ids"], at=ps[0]["at"], kinds=KINDS, **d), f"{log.dir}/{name}.pt")
        lg, st = pattern(d["logits"].float().square().sum((2, 4)).sqrt()), pattern(d["state"][:, -1])
        log.note(**{name: dict(own_residue_max_abs=d["own"].max().item(), flank_logits=lg[:n], random_logits=lg[n : 2 * n],
                               flank_last_state=st[:n], random_last_state=st[n : 2 * n])})
    log.write()
    print(open(log.path).read())


def main():
    log = runlog.start("jump", __doc__)
    targets = json.load(open(data.path(log.input, "jump.json")))
    if log.reduce:
        return reduce(log, targets)
    m, df = net.load(), data.sample(log.input)
    for name, t in targets.items():
        ids = pick.ids_of(df, t["protein"])
        a, b = (torch.arange(c - HALF, c + HALF + 1, device=ids.device) + 1 for c in t["centres"])
        at = torch.cat([a, b])
        kind, k, x = zip(*inputs(ids, a, b, t["protein"])[log.shard[0] :: log.shard[1]])
        with torch.no_grad():
            clean, logits, state, own = zip(*[scan(m, i, at, log) for i in x])
        torch.save(dict(ids=ids[0].cpu(), at=at.cpu(), kind=torch.tensor(kind), k=torch.tensor(k), logits_clean=torch.stack(clean),
                        logits=torch.stack(logits), state=torch.stack(state), own=torch.tensor(own)), f"{log.dir}/{name}-shard{log.shard[0]}.pt")
    log.write()


if __name__ == "__main__":
    main()
