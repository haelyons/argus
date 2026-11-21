"""Residue pairs: a protein's contacts in its AlphaFold model, and how well a score of pairs finds them."""

import pandas as pd
import torch

from argus import data

CONTACT, SEP = 8.0, 24


def contacts(entry, n):
    """(n, n) residues within CONTACT angstrom (C-beta, C-alpha for glycine) in the AlphaFold model of a
    UniProt entry, or None when it has no model of n residues."""
    x = data.model(entry, n)
    return None if x is None else torch.cdist(x[0], x[0]) < CONTACT


def known(entry, pdb, chain, n):
    """(n, n) residues within CONTACT angstrom in one chain of an experimental structure, and the residues it
    resolves (n,)."""
    x, seen = data.structure(entry, pdb, chain, n)
    return torch.cdist(x, x) < CONTACT, seen


def precision(m, c, seen=None):
    """Share in contact of the L pairs, SEP or more apart, with the largest score: m (..., L, L) made symmetric,
    less each pair's row mean times column mean over the mean. Returns it (...) and the share of all such pairs.
    With seen (L,), only pairs of residues it marks count."""
    m = (m + m.transpose(-1, -2)) / 2
    m = m - m.mean(-2, keepdim=True) * m.mean(-1, keepdim=True) / m.mean((-2, -1), keepdim=True)
    i, j = torch.triu_indices(m.shape[-1], m.shape[-1], SEP, device=m.device)
    if seen is not None:
        i, j = i[seen[i] & seen[j]], j[seen[i] & seen[j]]
    top = m[..., i, j].topk(min(m.shape[-1], len(i))).indices
    return c[i, j][top].float().mean(-1), c[i, j].float().mean().item()


def weights(c, seen=None):
    """Two symmetric weightings (2, L, L) of the pairs SEP or more apart, each summing to one: the pairs in
    contact, equally; and the pairs not in contact, each separation weighing as much as it does among the
    contacts. Separations with no contact or with nothing but contacts weigh nothing, as do pairs with a residue
    that seen (L,) does not mark."""
    r = torch.arange(len(c), device=c.device)
    sep, both = (r[:, None] - r).abs(), torch.ones_like(c) if seen is None else seen[:, None] & seen
    n1, n0 = (torch.zeros(len(c), device=c.device).index_add_(0, sep.flatten(), (x & both).flatten().float()) for x in (c, ~c))
    ok = (sep >= SEP) & ((n1 > 0) & (n0 > 0))[sep] & both
    w1, w0 = (c & ok).float(), (~c & ok) * (n1 / n0.clamp(min=1))[sep]
    return torch.stack([w1 / w1.sum(), w0 / w0.sum()])


def gap(m, c, seen=None):
    """Mean score m (..., L, L) of the pairs in contact, and of the pairs not in contact at the same
    separations: (..., 2). Their difference is how far the score sets contacts apart."""
    return (m[..., None, :, :] * weights(c, seen)).sum((-2, -1))


def outcomes(files, df, structures=False):
    """One row per protein and set of the scans saved in `files` by an edit.coupling script: the precision and gap
    of the coupling, and the share of the logit change at all residues and at those not replaced. Contacts are
    the AlphaFold model's, or with `structures` those of the experimental chain df names in pdb and chain."""
    rows = []
    for f in files:
        d = torch.load(f)
        p, m = d["protein"], d["coupling"]
        c, seen = known(df.Entry[p], df.pdb[p], df.chain[p], m.shape[-1]) if structures else (contacts(df.Entry[p], m.shape[-1]), None)
        (hits, chance), g, share = precision(m, c, seen), gap(m, c, seen), torch.cat([torch.zeros(1, 2), d["share"]])
        for i, s in enumerate(d["sets"]):
            rows.append(dict(protein=p, group="holdout" if df.split[p] == "holdout" else "set_aside", residues=m.shape[-1], set=s, precision=hits[i].item(), chance=chance,
                             contact=g[i, 0].item(), matched=g[i, 1].item(), gap=(g[i, 0] - g[i, 1]).item(), share=share[i, 0].item(), share_not_replaced=share[i, 1].item()))
    return pd.DataFrame(rows)


def paired(t, group, a, b):
    """Per outcome of a table of outcomes: the means of sets a and b over the group's proteins, the mean of a less
    b, its standard error, and the number of proteins with a below b."""
    out = {}
    for y in ("precision", "gap"):
        v = t[t.group == group].pivot(index="protein", columns="set", values=y)
        out[y] = dict(a=v[a].mean(), b=v[b].mean(), diff=(v[a] - v[b]).mean(), se=(v[a] - v[b]).sem(), below=int((v[a] < v[b]).sum()), proteins=len(v))
    return out
