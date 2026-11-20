"""Residue pairs: a protein's contacts in its AlphaFold model, and how well a score of pairs finds them."""

import torch

from argus import data

CONTACT, SEP = 8.0, 24


def contacts(entry, n):
    """(n, n) residues within CONTACT angstrom (C-beta, C-alpha for glycine) in the AlphaFold model of a
    UniProt entry, or None when it has no model of n residues."""
    x = data.model(entry, n)
    return None if x is None else torch.cdist(x[0], x[0]) < CONTACT


def precision(m, c):
    """Share in contact of the L pairs, SEP or more apart, with the largest score: m (..., L, L) made symmetric,
    less each pair's row mean times column mean over the mean. Returns it (...) and the share of all such pairs."""
    m = (m + m.transpose(-1, -2)) / 2
    m = m - m.mean(-2, keepdim=True) * m.mean(-1, keepdim=True) / m.mean((-2, -1), keepdim=True)
    i, j = torch.triu_indices(m.shape[-1], m.shape[-1], SEP, device=m.device)
    top = m[..., i, j].topk(min(m.shape[-1], len(i))).indices
    return c[i, j][top].float().mean(-1), c[i, j].float().mean().item()


def weights(c):
    """Two symmetric weightings (2, L, L) of the pairs SEP or more apart, each summing to one: the pairs in
    contact, equally; and the pairs not in contact, each separation weighing as much as it does among the
    contacts. Separations with no contact or with nothing but contacts weigh nothing."""
    r = torch.arange(len(c), device=c.device)
    sep = (r[:, None] - r).abs()
    n1, n0 = (torch.zeros(len(c), device=c.device).index_add_(0, sep.flatten(), x.flatten().float()) for x in (c, ~c))
    ok = (sep >= SEP) & ((n1 > 0) & (n0 > 0))[sep]
    w1, w0 = (c & ok).float(), (~c & ok) * (n1 / n0.clamp(min=1))[sep]
    return torch.stack([w1 / w1.sum(), w0 / w0.sum()])


def gap(m, c):
    """Mean score m (..., L, L) of the pairs in contact, and of the pairs not in contact at the same
    separations: (..., 2). Their difference is how far the score sets contacts apart."""
    return (m[..., None, :, :] * weights(c)).sum((-2, -1))
