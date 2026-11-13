"""Stage 1: the three standing checks on the first real batch, and what each access costs."""

import argparse

import torch

from argus import data, net, sae
from argus.runlog import Log


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    shard = tuple(map(int, ap.parse_args().shard.split("/")))
    log = Log("s1", shard)
    m, s = net.load(), sae.load()
    idx, ids = next(data.batches(data.sample(), budget=8192, shard=shard))
    ids = ids.cuda()
    n = ids.numel()
    with torch.no_grad():
        log.plain(m, ids)
        log.note(checks=checks(m, s, ids), batch=list(ids.shape))
        nop = lambda l, r: None
        with log.op("forward_to_60", n):
            net.run(m, ids, stop=sae.LAYER)
        with log.op("forward_tapped", n):
            net.logits(m, net.run(m, ids, tap=nop))
        with log.op("forward_with_patterns", n):
            net.logits(m, net.run(m, ids, tap=nop, pattern=True))
    one = ids[:2]
    acts = []
    with log.op("forward_backward_all_units", one.numel()):
        x = net.run(m, one, x=m.esmc.embed(one).requires_grad_(), stop=sae.LAYER, tap=lambda l, r: (r["a"].retain_grad(), acts.append(r["a"])))
        sae.encode(s, x)[0][:, 5, 0].sum().backward()
    log.note(grad_nonzero=float(sum((a.grad != 0).float().mean() for a in acts) / len(acts)))
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
