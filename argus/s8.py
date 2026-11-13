"""Stage 8: what reads an edit. Apply one removal per target, record everything downstream that
moves, then repeat it with the strongest readers held at their clean values.

The removal: for a feature target, its 64 top gradient-ranked units; for a unit target, the unit.
out/s8/<target>.pt, per example:
  removed        unit ids
  units          the 1024 other units that moved most: ids, signed and absolute change summed over
                 positions (ranked by absolute change * write norm)
  block_change   (N,) absolute change * write norm, summed over each block's units
  heads          (N, H) norm of each head's output change, summed over positions
  features       the 256 features that moved most: ids, change summed over positions, change at the on-position
  held_units, held_heads   the readers held fixed: top 64 units, top 8 heads
  runs           rows = [edit, edit + units held, edit + heads held, edit + both held]
                 cols = RUN: target at the on-position where it is not itself edited, size of the
                 change over all features, mean KL of the predictions on the intact input, and on
                 the masked input the loss and the on-position log-prob
  clean          the same quantities unedited
"""

import argparse
import os

import numpy as np
import torch

from argus import data, net, pick, sae
from argus.runlog import Log

OUT, GROUP, R_UNITS, R_HEADS, K_UNITS, K_FEATS = "out/s8", 64, 64, 8, 1024, 256
RUN = ["pre", "value", "feature_change", "kl", "mloss", "mlogp"]


def capture(m, ids):
    c = dict(a=[], z=[], x=[m.esmc.embed(ids)])

    def tap(l, r):
        c["a"].append(r["a"][0]), c["z"].append(r["z"][0]), c["x"].append(r["x"])

    net.run(m, ids, x=c["x"][0], tap=tap)
    return c


def feats(s, x60):
    pre, val, ind, _ = sae.encode(s, x60[0])
    return pre, torch.zeros_like(pre).scatter(1, ind, val)


def edits(removed, hold_u, hold_h, c, U, H):
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


def run_example(m, s, ids, e, kind, t, removed, norm, log):
    N, U, H, pos, dev = norm.shape[0], norm.shape[1], net.blocks(m)[0].attn.n_heads, e["on"][1], "cuda"
    start = int(removed.min()) // U
    mids, labels = data.mask(ids.cpu(), [e["on"][0]])
    labels[0, pos], mids[0, pos] = int(ids[0, pos]), net.MASK
    mids, labels = mids.cuda(), labels.cuda()
    with log.op("capture", 2 * ids.numel()):
        c, cm = capture(m, ids), capture(m, mids)
    pre0, f0 = feats(s, c["x"][sae.LAYER])
    lp0 = net.logits(m, c["x"][N]).float().log_softmax(-1)[0]
    j = t if kind == "features" else None

    def outcome(hold_u, hold_h, tap=None):
        x60 = {}

        def tp(l, r):
            if l == sae.LAYER - 1:
                x60["x"] = r["x"]
            if tap:
                tap(l, r)

        with log.op("edited_pass", 2 * ids.numel()):
            x = net.run(m, ids, x=c["x"][start], start=start, edits=edits(removed, hold_u, hold_h, c, U, H), tap=tp)
            xm = net.run(m, mids, x=cm["x"][start], start=start, edits=edits(removed, hold_u, hold_h, cm, U, H))
        pre, f = feats(s, x60.get("x", c["x"][sae.LAYER]))
        lp = net.logits(m, x).float().log_softmax(-1)[0]
        kl = (lp0.exp() * (lp0 - lp)).sum(-1).mean()
        mlp = net.logits(m, xm).float().log_softmax(-1)[0, pos, ids[0, pos]]
        tv = [pre[pos, j], f[pos, j]] if j is not None else [torch.tensor(float("nan"))] * 2
        return torch.stack([x.cpu() for x in tv + [(f - f0).norm(), kl, net.mlm_loss(m, xm, labels)[0], mlp]]), f

    dsum, dabs, dz = (torch.zeros(N, n, device=dev) for n in (U, U, H))

    def diff(l, r):
        da = r["a"][0] - c["a"][l]
        dsum[l], dabs[l] = da.sum(0), da.abs().sum(0)
        dz[l] = (r["z"][0] - c["z"][l]).norm(dim=-1).sum(0)

    none = torch.zeros(0, dtype=torch.long, device=dev)
    r0, f = outcome(none, none, diff)
    size = dabs * norm
    size.view(-1)[removed] = 0
    top = size.flatten().topk(K_UNITS).indices
    hu, hh = top[:R_UNITS], dz.flatten().topk(R_HEADS).indices
    runs = torch.stack([r0] + [outcome(a, b)[0] for a, b in ((hu, none), (none, hh), (hu, hh))])
    df_ = (f - f0)
    ft = df_.sum(0).abs().topk(K_FEATS).indices
    clean = torch.stack([x.cpu() for x in ([pre0[pos, j], f0[pos, j]] if j is not None else [torch.tensor(float("nan"))] * 2)
                         + [torch.tensor(0.0), torch.tensor(0.0), net.mlm_loss(m, cm["x"][N], labels)[0],
                            net.logits(m, cm["x"][N]).float().log_softmax(-1)[0, pos, ids[0, pos]]]])
    return dict(e, removed=removed.cpu(), units=(top.cpu(), dsum.flatten()[top].cpu(), dabs.flatten()[top].cpu()),
                block_change=size.sum(1).cpu(), heads=dz.cpu(), features=(ft.cpu(), df_.sum(0)[ft].cpu(), df_[pos, ft].cpu()),
                held_units=hu.cpu(), held_heads=hh.cpu(), runs=runs, clean=clean, cols=RUN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    shard = tuple(map(int, ap.parse_args().shard.split("/")))
    log = Log("s8", shard)
    m, s = net.load(), sae.load()
    df = data.sample()
    norm = torch.from_numpy(np.load("out/s2/norm.npy")).cuda()
    U = norm.shape[1]
    os.makedirs(OUT, exist_ok=True)
    for name, kind, t, ex in pick.targets(shard):
        if kind == "features":
            removed = torch.load(f"out/s7/{name}.pt")["rankings"]["grad"][:GROUP].cuda()
        else:
            removed = torch.tensor([t[0] * U + t[1]], device="cuda")
        rec = []
        for split, pairs in ex.items():
            for e in pairs:
                ids = pick.ids_of(df, e["on"][0])
                if log.rec["tok_s"] is None:
                    log.plain(m, ids)
                with torch.no_grad():
                    rec.append(dict(run_example(m, s, ids, e, kind, t, removed, norm, log), split=split))
        torch.save(rec, f"{OUT}/{name}.pt")
        log.note(done=name)
    log.write()


if __name__ == "__main__":
    main()
