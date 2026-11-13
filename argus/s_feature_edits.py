"""Edit the feature itself and ask whether the same readers and predictions move as
when its unit group is removed.

A feature edit changes state[60] by (f' - f) * decoder direction * sd at every position where
the feature fires, which leaves the SAE's reconstruction error in place. The masked input gets
the same state change (the SAE itself is never run on masked input).
Edits compared, per example of each feature target:
  feat_remove (f' = 0), feat_raise (f' = 2f), group_remove (the unit group s_readers removed)

out/feature_edits/<target>.pt, per example:
  per edit: units     top 256 units in blocks >= 60 by absolute change * write norm: ids, signed change
            block_change (20,), heads (20, H) as in s_readers
            kl, logp (intact input, on-position true residue), mloss, mlogp, dlogits (64,) at the on-position
  clean:    logp, mloss, mlogp
  agree:    feat_remove vs group_remove, and feat_remove vs feat_raise (expect about -1): cosines of the
            full change in later units, head outputs, logits and state[60]; overlap of the top-64 readers
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from argus import data, edit, net, pick, sae
from argus.runlog import Log

OUT, K, R = "out/feature_edits", 256, 64


def run_example(m, s, ids, e, j, removed, norm, log):
    N, U, L60, pos, dev = norm.shape[0], norm.shape[1], sae.LAYER, e["on"][1], "cuda"
    H = net.blocks(m)[0].attn.n_heads
    true = ids[0, pos]
    mids, labels = data.mask(ids.cpu(), [e["on"][0]])
    labels[0, pos], mids[0, pos] = int(true), net.MASK
    mids, labels = mids.cuda(), labels.cuda()
    c, cm = edit.capture(m, ids), edit.capture(m, mids)
    x60 = c["x"][L60]
    _, val, ind, sd = sae.encode(s, x60)
    f = (val * (ind == j)).sum(-1, keepdim=True)
    step = f * s.W_dec[j] * sd
    lg0 = net.logits(m, c["x"][N]).float()[0]
    lp0 = lg0.log_softmax(-1)
    lpm = lambda x: net.logits(m, x).float().log_softmax(-1)[0, pos, true]
    none = torch.zeros(0, dtype=torch.long, device=dev)
    ca, cz = torch.stack(c["a"][L60:]), torch.stack(c["z"][L60:])

    def go(start, ed, x0, xm0):
        got = dict(a=[], z=[])

        def tap(l, r):
            if l >= L60:
                got["a"].append(r["a"][0]), got["z"].append(r["z"][0])
            if l == L60 - 1:
                got["x60"] = r["x"]

        with log.op("edited_pass", 2 * ids.numel()):
            x = net.run(m, ids, x=x0, start=start, edits=ed(c), tap=tap)
            xm = net.run(m, mids, x=xm0, start=start, edits=ed(cm))
        da, dz = torch.stack(got["a"]) - ca, torch.stack(got["z"]) - cz
        lg = net.logits(m, x).float()[0]
        lp = lg.log_softmax(-1)
        size = da.abs().sum(1) * norm[L60:]
        top = size.flatten().topk(K).indices
        rec = dict(units=(top.cpu() + L60 * U, da.sum(1).flatten()[top].cpu()), block_change=size.sum(1).cpu(),
                   heads=dz.norm(dim=-1).sum(1).cpu(), kl=(lp0.exp() * (lp0 - lp)).sum(-1).mean().item(),
                   logp=lp[pos, true].item(), mloss=net.mlm_loss(m, xm, labels).item(), mlogp=lpm(xm).item(),
                   dlogits=(lg - lg0)[pos].cpu())
        return rec, dict(units=da, heads=dz, logits=lg - lg0, x60=got.get("x60", x0) - x60, top=set(top[:R].tolist()))

    out, full = {}, {}
    for nm, k in (("feat_remove", -1.0), ("feat_raise", 1.0)):
        out[nm], full[nm] = go(L60, lambda cc: {}, x60 + k * step, cm["x"][L60] + k * step)
    start = int(removed.min()) // U
    out["group_remove"], full["group_remove"] = go(start, lambda cc: edit.remove_hold(removed, none, none, cc, U, H), c["x"][start], cm["x"][start])
    cos = lambda a, b: F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    agree = {}
    for a, b in (("feat_remove", "group_remove"), ("feat_remove", "feat_raise")):
        agree[f"{a}~{b}"] = {k: cos(full[a][k], full[b][k]) for k in ("units", "heads", "logits", "x60")} | {
            "top_overlap": len(full[a]["top"] & full[b]["top"]) / R}
    clean = dict(logp=lp0[pos, true].item(), mloss=net.mlm_loss(m, cm["x"][N], labels).item(), mlogp=lpm(cm["x"][N]).item(),
                 value=f[0, pos, 0].item(), value_sum=f.sum().item())
    return dict(e, clean=clean, agree=agree, **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    shard = tuple(map(int, ap.parse_args().shard.split("/")))
    log = Log("feature_edits", shard)
    m, s = net.load(), sae.load()
    df = data.sample()
    norm = torch.from_numpy(np.load("out/weights/norm.npy")).cuda()
    os.makedirs(OUT, exist_ok=True)
    for name, kind, t, ex in pick.targets(shard):
        if kind != "features":
            continue
        removed = torch.load(f"out/readers/{name}.pt")[0]["removed"].cuda()
        rec = []
        for split, pairs in ex.items():
            for e in pairs:
                ids = pick.ids_of(df, e["on"][0])
                if log.rec["tok_s"] is None:
                    log.plain(m, ids)
                with torch.no_grad():
                    rec.append(dict(run_example(m, s, ids, e, t, removed, norm, log), split=split))
        torch.save(rec, f"{OUT}/{name}.pt")
        log.note(done=name)
    log.write()


if __name__ == "__main__":
    main()
