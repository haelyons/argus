"""The three exactness checks. Run them on the first batch of the next real run after any core change."""

import torch

from argus import net, sae


def checks(m, s, ids):
    """Returns the three check residuals; all should be ~0 (float noise) or exactly 0."""
    out, keep = {}, {}

    def tap(l, r):
        b = net.blocks(m)[l]
        ref = b.ffn(r["x"] - r["ffn"]) / b.scaling_factor
        rebuilt = r["a"] @ net.out_weight(m, l)
        out[l] = ((rebuilt - ref).abs().max() / ref.abs().max()).item()
        if l % 16 == 0:
            keep[l] = r["a"][:, 5:9, ::977].clone()

    x60 = net.run(m, ids, stop=sae.LAYER, tap=tap)
    x = net.run(m, ids, x=x60, start=sae.LAYER, tap=tap)
    lg = net.logits(m, x)
    res = dict(write_rebuild_max_rel=max(out.values()))

    U = net.blocks(m)[0].ffn.ffn_hidden_size
    idx = (slice(None), slice(5, 9), slice(0, U, 977))
    same = net.run(m, ids, edits={("act", l): net.put(idx, v) for l, v in keep.items()})
    res["edit_to_current_max_abs"] = (net.logits(m, same) - lg).abs().max().item()

    m.add_sae_models([s])
    ref = m(input_ids=ids)
    theirs = ref.sae_outputs[f"layer{sae.LAYER}"].to_dense()
    _, val, ind, _ = sae.encode(s, x60[ids != net.PAD])
    ours = torch.zeros_like(theirs).scatter(-1, ind, val)
    res["logits_vs_biohub_max_abs"] = (ref.logits - lg).abs().max().item()
    res["feature_vs_biohub_max_abs"] = (ours - theirs).abs().max().item()
    res["feature_support_mismatch"] = ((ours > 0) != (theirs > 0)).float().mean().item()
    res["feature_max"] = theirs.max().item()
    res["sae_idf_max_are_ones"] = bool((s.idf == 1).all() and (s.max == 1).all())
    m.esmc._sae_models.clear()
    return res
