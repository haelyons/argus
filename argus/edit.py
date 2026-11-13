"""Batched unit-group edits on one protein, reusing the clean pass below the first edited block.

A row is (op, unit ids). Unit id = block * U + unit. Ops, applied at every position:
  remove  activation -> 0
  raise   activation -> 2 * activation
  copy    activation -> the contrast input's, at the position the same distance from the target;
          positions with no counterpart keep their value
"""

import torch

from argus import net

OPS = ["remove", "raise", "copy"]
MULT = [0.0, 2.0, 1.0]


def clean(m, ids, stop, nu):
    """States entering blocks 0..stop, and the activations of blocks < nu as (nu, L, U)."""
    states, acts = [m.esmc.embed(ids)], []

    def tap(l, r):
        states.append(r["x"])
        if l < nu:
            acts.append(r["a"][0])

    net.run(m, ids, x=states[0], stop=stop, tap=tap)
    return states, torch.stack(acts) if acts else None


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
