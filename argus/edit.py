"""Batched unit-group edits on one protein, reusing the clean pass below the first edited block.

A row is (op, unit ids). Unit id = block * U + unit. Ops, applied at every position:
  remove  activation -> 0
  raise   activation -> 2 * activation
  copy    activation -> the contrast input's, at the position the same distance from the target;
          positions with no counterpart keep their value
"""

import torch

from argus import data, net

OPS = ["remove", "raise", "copy"]
MULT = [0.0, 2.0, 1.0]


def align(acts_off, L, shift):
    """Contrast activations laid on the target's positions: src (nu, L, U), valid (L,)."""
    pos = torch.arange(L, device=acts_off.device) + shift
    valid = (pos >= 0) & (pos < acts_off.shape[1])
    return acts_off[:, pos.clamp(0, acts_off.shape[1] - 1)], valid


def edits_for(rows, nu, U, src, valid):
    """rows: [(op index, ids)]. Returns (edits dict, first edited block)."""
    B, dev = len(rows), src.device
    mult = torch.ones(B, nu * U, device=dev)
    cm = torch.zeros(B, nu * U, dtype=torch.bool, device=dev)
    for b, (op, ids) in enumerate(rows):
        mult[b, ids] = MULT[op]
        cm[b, ids] = OPS[op] == "copy"
    mult, cm = mult.view(B, nu, 1, U), cm.view(B, nu, 1, U)
    touched = ((mult != 1) | cm).flatten(2).any(-1).any(0).nonzero()[:, 0].tolist()

    def make(l):
        def f(a, g, v):
            return torch.where(cm[:, l] & valid[None, :, None], src[l][None], a * mult[:, l])

        return f

    return {("act", l): make(l) for l in touched}, touched[0]


def run(m, ids, states, rows, nu, U, src, valid, stop, tap=None):
    """Edited states entering block `stop`, one batch row per edit row."""
    ed, start = edits_for(rows, nu, U, src, valid)
    B = len(rows)
    x = net.run(m, ids.expand(B, -1), x=states[start].expand(B, -1, -1), start=start, stop=stop, edits=ed, tap=tap)
    return x, (stop - start) * B * ids.shape[1]


def new_value(op, a, c):
    """What an op makes of clean activation a, given contrast activation c."""
    return [torch.zeros_like(a), 2 * a, c][op]


def capture(m, ids, stop=None, nu=None):
    """One protein's clean pass: states x[l] entering blocks 0..stop, and for blocks < nu (all by default)
    activations a[l] (L, U) and head outputs z[l] (L, H, dh)."""
    c = dict(a=[], z=[], x=[m.esmc.embed(ids)])

    def tap(l, r):
        c["x"].append(r["x"])
        if nu is None or l < nu:
            c["a"].append(r["a"][0]), c["z"].append(r["z"][0])

    net.run(m, ids, x=c["x"][0], stop=stop, tap=tap)
    return c


def remove_hold(removed, hold_u, hold_h, c, U, H):
    """Zero `removed`; pin units `hold_u` and heads `hold_h` to the capture c."""
    ed = {}
    for l in set((removed // U).tolist()) | set((hold_u // U).tolist()):
        rem, hold = removed[removed // U == l] % U, hold_u[hold_u // U == l] % U

        def f(a, g, v, l=l, rem=rem, hold=hold):
            a = a.clone()
            a[:, :, rem] = 0
            a[:, :, hold] = c["a"][l][:, hold]
            return a

        ed["act", l] = f
    for l in set((hold_h // H).tolist()):
        hh = hold_h[hold_h // H == l] % H

        def f(z, l=l, hh=hh):
            z = z.clone()
            z[:, :, hh] = c["z"][l][:, hh]
            return z

        ed["head", l] = f
    return ed


def moved(m, ids, at, same, edits=None):
    """ids (B, T), each row one replacement of one token, row `same` the one that leaves it as it is. The change
    of the amino-acid logits at tokens `at` (B, E, 20), the distance every block's state moves there summed over
    the rows (N, E), and row `same`'s amino-acid logits at every token (T, 20)."""
    st = []
    lg = net.logits(m, net.run(m, ids, edits=edits, tap=lambda l, r: st.append(r["x"][:, at])))[..., 4:24]
    st = torch.stack(st)
    return (lg[:, at] - lg[same, at]).half().cpu(), (st - st[:, same : same + 1]).square().sum((1, 3)).sqrt().cpu(), lg[same].half().cpu()


def at_random(ids, per, seed):
    """As many ids as `ids` in each of its blocks (id = block * per + index), at random: among the block's
    others first, and among its own when those run out."""
    g, out = torch.Generator().manual_seed(seed), []
    for l, n in zip(*(ids // per).unique(return_counts=True)):
        block = l * per + torch.randperm(per, generator=g)
        own = torch.isin(block, ids)
        out.append(torch.cat([block[~own], block[own]])[:n])
    return torch.cat(out)


def part(ids, per):
    """{block: indices within it} of ids = block * per + index."""
    return {l: ids[ids // per == l] % per for l in (ids // per).unique().tolist()}


def zero(units, heads, U, H):
    """Edits that set units' activations and heads' outputs to zero at every token."""
    idx = lambda j: (slice(None), slice(None), j)
    return {("act", l): net.put(idx(j), 0.0) for l, j in part(units, U).items()} | {("head", l): net.put(idx(j), 0.0) for l, j in part(heads, H).items()}


def _keep(own, j, cur, site):
    """Edit of per-head values (B, H, T, dh): heads j take those of the unchanged protein, own (H, T, dh), at each
    row's replaced token cur["pos"] (B,) when `site`, at every other token when not."""

    def f(v):
        v, r, p = v.clone(), torch.arange(len(v), device=v.device)[:, None], cur["pos"][:, None]
        at = own[j][:, cur["pos"]].transpose(0, 1) if site else v[r, j, p]
        if not site:
            v[:, j] = own[j]
        v[r, j, p] = at
        return v

    return f


def coupling(m, ids, sets, U, H, log, budget):
    """Every single replacement of one protein, ids (1, T), with nothing edited and then under each of `sets`,
    {name: (how, units, heads)}, ids as block * U + unit and block * H + head. how "held": the set keeps its
    values on the unchanged protein while everything else sees the replacement. how "given": the unchanged
    protein, the set alone taking its values under the replacement. how "pattern" or "value", heads only: as
    "held", but the heads keep only their attention pattern, or only the values it mixes; "value_site" and
    "value_rest" keep those values only at the replaced token, or only at the others. how "removed": the set is
    zero, with and without the replacement.
    Returns the coupling (S + 1, L, L), how much a replacement at one residue changes the amino-acid logits at
    another (norm over the 20 replacements and the 20 logits); per set, at all residues and at those not replaced
    (S, 2), the share of the unedited logit change that appears when it is given, or else goes; and the
    largest logit change under a replacement that leaves the protein as it is (S + 1)."""
    N, T, dev = len(net.blocks(m)), ids.shape[1], ids.device
    wt = capture(m, ids)
    lw = net.logits(m, wt["x"][N])[0, :, 4:24]
    need = {l for how, _, h in sets.values() if how.startswith(("pattern", "value")) for l in part(h, H)}
    own_part = {}
    if need:
        net.run(m, ids, stop=max(need) + 1, pattern=True, tap=lambda l, r: own_part.update({l: dict(pattern=r["p"][0], value=r["hv"][0])}) if l in need else None)
    plan, cur = [], {}
    for how, u, h in sets.values():
        u, h, base = u.to(dev), h.to(dev), lw
        if how == "held":
            ed = remove_hold(u[:0], u, h, wt, U, H)
        elif how == "given":
            ed = {("act", l): j for l, j in part(u, U).items()} | {("head", l): j for l, j in part(h, H).items()}
        elif how == "removed":
            ed = zero(u, h, U, H)
            first = min(l for _, l in ed)
            base = net.logits(m, net.run(m, ids, x=wt["x"][first], start=first, edits=ed))[0, :, 4:24]
        elif how in ("value_site", "value_rest"):
            ed = {("value", l): _keep(own_part[l]["value"], j, cur, how == "value_site") for l, j in part(h, H).items()}
        else:
            ed = {(how, l): net.put((slice(None), j), own_part[l][how][j]) for l, j in part(h, H).items()}
        plan.append((how, ed, min(l for _, l in ed), base))
    starts = {f for how, _, f, _ in plan if how != "given"}
    sq, own = lw.new_zeros(len(plan) + 1, T - 2, T), lw.new_zeros(len(plan) + 1)
    dot, tot = lw.new_zeros(len(plan), 2), lw.new_zeros(2)
    for pos, mids in data.mutants(ids, budget):
        if log.rec["tok_s"] is None:
            log.plain(m, mids)
        keep, got, same, cur["pos"] = {}, {}, (mids == ids).all(1), pos
        rest = torch.stack([torch.ones_like(mids, dtype=torch.bool), torch.arange(T, device=dev) != pos[:, None]])[:, :, 1:-1]

        def tap(l, r):
            if l + 1 in starts:
                keep[l + 1] = r["x"]
            for i, (how, ed, _, _) in enumerate(plan):
                for k in (("act", l), ("head", l)) if how == "given" else ():
                    if k in ed:
                        got[i, k] = r["a" if k[0] == "act" else "z"][:, :, ed[k]]

        with log.op("changed", mids.numel()):
            d0 = net.logits(m, net.run(m, mids, tap=tap))[..., 4:24] - lw
        ds = [d0]
        for i, (how, ed, f, base) in enumerate(plan):
            with log.op(how, mids.numel()):
                if how != "given":
                    x = net.run(m, mids, x=keep.get(f), start=f, edits=ed)
                else:
                    give = {k: net.put((slice(None), slice(None), j), got[i, k]) for k, j in ed.items()}
                    x = net.run(m, ids.expand(len(pos), -1), x=wt["x"][f].expand(len(pos), -1, -1), start=f, edits=give)
                ds.append(net.logits(m, x)[..., 4:24] - base)
        for i, d in enumerate(ds):
            sq[i].index_add_(0, pos - 1, d.square().sum(-1))
            own[i] = torch.maximum(own[i], (d.abs().amax((1, 2)) * same).max())
            if i:
                dot[i - 1] += ((d0 * (d if plan[i - 1][0] == "given" else d0 - d)).sum(-1)[:, 1:-1] * rest).sum((1, 2))
        tot += (d0.square().sum(-1)[:, 1:-1] * rest).sum((1, 2))
    return sq.sqrt()[:, :, 1:-1].cpu(), (dot / tot).cpu(), own.cpu()
