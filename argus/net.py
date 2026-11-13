"""ESMC forward pass with every internal exposed and editable.

run() walks blocks [start, stop). Per block it hands tap() a dict:
  g, v, a   gate, value, silu(g)*v            (B, L, U)
  z         per-head context before out_proj  (B, L, H, dh)
  p         attention pattern or None         (B, H, L, L)
  attn, ffn the two writes into the state     (B, L, D)
  x         state after the block             (B, L, D)
edits maps (kind, layer) -> fn. kind 'act': fn(a, g, v); 'head': fn(z); 'state': fn(x).
States are indexed by the block they enter: state[l] enters block l, state[N] feeds the head.
Callers own the grad mode.
"""

import torch
import torch.nn.functional as F

from esm.models.esmc import EsmcForMaskedLM

from argus.data import MASK, PAD  # noqa: F401

MODEL = "biohub/ESMC-6B"


def load(device="cuda", dtype=torch.float32):
    m = EsmcForMaskedLM.from_pretrained(MODEL, device=device, dtype=dtype, attn_implementation="sdpa")
    return m.requires_grad_(False)


def blocks(m):
    return m.esmc.transformer.blocks


def out_weight(m, l):
    """Unit write vectors of block l as rows, residual scaling folded in. (U, D)"""
    b = blocks(m)[l]
    return b.ffn.fc2_weight.T / b.scaling_factor


def head_weight(m, l):
    """Head write maps of block l, residual scaling folded in. (H, dh, D)"""
    b = blocks(m)[l]
    return b.attn.out_proj.weight.T.view(b.attn.n_heads, b.attn.d_head, -1) / b.scaling_factor


def centred(w, gain):
    """Read rows with the LayerNorm gain and centring folded in (its input-dependent scale is not)."""
    w = w * gain
    return w - w.mean(-1, keepdim=True)


def put(idx, val):
    """Edit that sets t[idx] = val. idx is an index tuple over (batch, pos, unit|head)."""

    def f(t, *_):
        t = t.clone()
        t[idx] = val
        return t

    return f


def _block(b, l, x, mask, edits, tap, pattern):
    at, s = b.attn, b.scaling_factor
    B, L, _ = x.shape
    q, k, v = at.layernorm_qkv(x).chunk(3, -1)
    q, k = at._apply_rotary(at.q_ln(q), at.k_ln(k))
    q, k, v = (t.view(B, L, at.n_heads, at.d_head).transpose(1, 2) for t in (q, k, v))
    p = None
    if pattern:
        sc = (q @ k.transpose(-2, -1)) * at.d_head**-0.5
        if mask is not None:
            sc = sc.masked_fill(~mask, float("-inf"))
        p = sc.softmax(-1)
        z = p @ v
    else:
        z = F.scaled_dot_product_attention(q, k, v, mask)
    z = z.transpose(1, 2)
    if ("head", l) in edits:
        z = edits["head", l](z)
    attn = at.out_proj(z.reshape(B, L, -1)) / s
    x = x + attn

    f = b.ffn
    h = F.layer_norm(x, (f.hidden_size,), f.layer_norm_weight, f.layer_norm_bias, f.eps)
    g, u = F.linear(h, f.fc1_weight).chunk(2, -1)
    a = F.silu(g) * u
    if ("act", l) in edits:
        a = edits["act", l](a, g, u)
    ffn = F.linear(a, f.fc2_weight) / s
    x = x + ffn
    if ("state", l) in edits:
        x = edits["state", l](x)
    if tap:
        tap(l, dict(g=g, v=u, a=a, z=z, p=p, attn=attn, ffn=ffn, x=x))
    return x


def run(m, ids, x=None, start=0, stop=None, edits=None, tap=None, pattern=False):
    """Returns the state entering block `stop`; x, when given, is the state entering `start`."""
    bl = blocks(m)
    stop = len(bl) if stop is None else stop
    pad = ids != PAD
    mask = None if pad.all() else pad[:, None, None, :]
    if x is None:
        x = m.esmc.embed(ids)
    for l in range(start, stop):
        x = _block(bl[l], l, x, mask, edits or {}, tap, pattern)
    return x


def logits(m, x):
    return m.lm_head(m.esmc.transformer.norm(x))


def mlm_loss(m, x, labels):
    """Summed cross-entropy at labelled positions, per row. labels: -100 where unmasked."""
    lp = logits(m, x).float().log_softmax(-1)
    hit = labels >= 0
    nll = -lp.gather(-1, labels.clamp(min=0)[..., None])[..., 0]
    return (nll * hit).sum(-1)
