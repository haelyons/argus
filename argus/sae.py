"""Biohub's layer-60 SAE, opened up.

It reads state[60]. Per residue: zscore over the D dims, subtract b_dec, project on W_enc,
relu, keep the top 64. So a feature's input (pre) does not alone decide whether it fires.
"""

import torch

from esm.models.esmc import EsmcSaeModel

REPO = "biohub/ESMC-6B-sae-layer60-k64-codebook16384"
LAYER = 60


def load(device="cuda"):
    s = EsmcSaeModel.from_pretrained(REPO, device=device)
    return s.layers[str(LAYER)].requires_grad_(False)


def zscore(x):
    mu = x.mean(-1, keepdim=True)
    sd = (x - mu).std(-1, keepdim=True) + 1e-5
    return (x - mu) / sd, sd


def encode(sae, x):
    """x: (..., D) state[60]. Returns pre (..., F), top-k values and indices (..., k), sd (..., 1)."""
    z, sd = zscore(x)
    pre = (z - sae.b_dec) @ sae.W_enc
    top = torch.topk(pre.relu(), sae.params.k, -1)
    return pre, top.values, top.indices, sd


def read_weight(sae):
    """W such that pre = (x @ W) / sd + const: the encoder with the zscore centring folded in. (D, F)"""
    return sae.W_enc - sae.W_enc.mean(0, keepdim=True)
