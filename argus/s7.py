"""Stage 7: edit groups of upstream units and measure the target and the predictions.

Rankings of upstream units, each scored from train data only (higher = supports the target):
  grad     stage 6 score          share   direct share (stage 4; for unit targets, of the gate input)
  comove   stage 3 co-movement    weight  write/read alignment (stage 2 quantity)
  random   a fixed random set
Rows per example: every ranking x sizes 1..1024 x {remove, raise, copy}, plus single removals of
each ranking's top 16. Intact input: the target at the on-position. Masked input (the seeded
mask plus the on-position): loss over masked positions, log-prob of the true residue there;
group remove/copy rows only.

out/s7/<target>.pt
  rankings {name: 1024 unit ids}; rows [(ranking, size or rank, op)]
  examples: split, on, off, coverage (share of positions the contrast covers), clean {..}, cols,
            out (rows, cols). Feature cols: pre, value, value summed over positions.
            Unit cols: activation. Then: direct (what same-position direct shares predict for the
            first col's change), mloss, mlogp (nan where not run).
Sum of single edits: add the single rows' changes for a ranking's first n units.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from argus import data, edit, net, pick, sae
from argus.runlog import Log
from argus.s2 import centred

OUT, BUDGET, SINGLES, TOPN = "out/s7", 12288, 16, 1024
SIZES = [2**i for i in range(11)]


def rows_of(rankings):
    rows = [(r, n, op) for r in rankings for n in SIZES for op in range(3)]
    return rows + [(r, k, 0) for r in rankings if r != "random" for k in range(SINGLES)]


def row_ids(rankings, row, n_group):
    r, n, op = row
    return rankings[r][:n] if n_group else rankings[r][n : n + 1]


def feature_scores(j, grad, stats):
    L = sae.LAYER
    share = np.stack([np.load(f"out/s4/train/unit_{l:02d}.npy", mmap_mode="r")[j] for l in range(L)])
    cond = np.stack([np.load(f"out/s3/train/cond_{l:02d}.npy", mmap_mode="r")[j] for l in range(L)])
    enc = np.stack([np.load(f"out/s2/enc_{l:02d}.npy", mmap_mode="r")[:, j] for l in range(L)])
    mean, std, norm = (x[:L] for x in stats)
    t = lambda x: torch.from_numpy(x.astype(np.float32)).cuda()
    return dict(grad=grad.cuda(), share=t(share), comove=(t(cond) - mean) / std, weight=t(enc) / norm)


def unit_scores(m, l, u, grad, samp, share):
    f = net.blocks(m)[l].ffn
    read = F.normalize(centred(f.fc1_weight[u], f.layer_norm_weight), dim=-1)
    weight = torch.stack([F.normalize(net.out_weight(m, k), dim=-1) @ read for k in range(l)])
    z = lambda x: (x - x.mean(0)) / (x.std(0) + 1e-6)
    tgt = z(samp[:, l, u].cuda().float())
    comove = torch.stack([(z(samp[:, k].cuda().float()) * tgt[:, None]).mean(0) for k in range(l)])
    return dict(grad=grad.cuda(), share=share, comove=comove, weight=weight)


def top(scores, seed):
    r = {k: v.flatten().argsort(descending=True)[:TOPN] for k, v in scores.items()}
    r["random"] = torch.randperm(scores["grad"].numel(), generator=torch.Generator().manual_seed(seed))[:TOPN].cuda()
    return r


class Target:
    """What differs between a feature target and a unit target."""

    def __init__(self, m, s, kind, t):
        self.m, self.s, self.kind, self.t = m, s, kind, t
        self.nu, self.stop = (sae.LAYER, sae.LAYER) if kind == "features" else (t[0], t[0] + 1)
        self.cols = ["pre", "value", "value_sum"] if kind == "features" else ["act"]

    def measure(self, x, got, pos):
        """(B, cols) from the edited run: x enters block `stop`; got holds the unit tap."""
        if self.kind == "units":
            return got["a"][:, None]
        pre, val, ind, _ = sae.encode(self.s, x)
        v = (val * (ind == self.t)).sum(-1)
        return torch.stack([pre[:, pos, self.t], v[:, pos], v.sum(1)], 1)

    def tap(self, got, pos):
        if self.kind == "features":
            return None
        l, u = self.t

        def f(k, r):
            if k == l:
                got.update(a=r["a"][:, pos, u], g=r["g"][:, pos, u], v=r["v"][:, pos, u], mid=(r["x"] - r["ffn"])[:, pos])

        return f

    def direct(self, x, got, pos):
        """Coefficients d (nu, U): d * (change in a unit's activation at pos) = predicted change in
        the feature input, or for a unit target in its (gate, value) inputs: d is then (2, nu, U)."""
        m = self.m
        if self.kind == "features":
            sd = sae.zscore(x[0, pos])[1]
            w = sae.read_weight(self.s)[:, self.t] / sd
            return torch.stack([net.out_weight(m, l) @ w for l in range(self.nu)])
        l, u = self.t
        f = net.blocks(m)[l].ffn
        U = f.ffn_hidden_size
        sig = (got["mid"][0].var(unbiased=False) + f.eps).sqrt()
        rd = centred(f.fc1_weight[[u, U + u]], f.layer_norm_weight) / sig
        return torch.stack([net.out_weight(m, k) @ rd.T for k in range(self.nu)]).permute(2, 0, 1)

    def predict(self, d, delta, got):
        """delta: (B, nu*U) activation changes at pos. Predicted change in the first col."""
        if self.kind == "features":
            return delta @ d.flatten()
        dg, dv = delta @ d[0].flatten(), delta @ d[1].flatten()
        return F.silu(got["g"][0] + dg) * (got["v"][0] + dv) - got["a"][0]


def run_example(tg, ids, e, rankings, rows, log, U):
    m, nu, pos, L = tg.m, tg.nu, e["on"][1], ids.shape[1]
    got = {}
    states, acts = edit.clean(m, ids, tg.stop, nu)
    net.run(m, ids, x=states[tg.stop - 1], start=tg.stop - 1, stop=tg.stop, tap=tg.tap(got, pos))
    base = tg.measure(states[tg.stop], got, pos)[0]
    d = tg.direct(states[tg.stop], got, pos)
    off_ids = pick.ids_of(tg.df, e["off"][0])
    src, valid = edit.align(edit.clean(m, off_ids, nu, nu)[1], L, e["off"][1] - pos)

    mids, labels = data.mask(ids.cpu(), [e["on"][0]])
    labels[0, pos], mids[0, pos] = int(ids[0, pos]), net.MASK
    mids, labels = mids.cuda(), labels.cuda()
    N = len(net.blocks(m))
    mstates, _ = edit.clean(m, mids, N, 0)
    lp = lambda x: net.logits(m, x).float().log_softmax(-1)[:, pos, ids[0, pos]]
    cl = dict(zip(tg.cols, base.tolist()), mloss=net.mlm_loss(m, mstates[N], labels).item(), mlogp=lp(mstates[N]).item(),
              masked=int((labels >= 0).sum()))

    out = torch.full((len(rows), len(tg.cols) + 3), float("nan"))
    n_group = len(rankings) * len(SIZES) * 3
    spec = [(i, r[2], row_ids(rankings, r, i < n_group)) for i, r in enumerate(rows)]
    spec.sort(key=lambda x: -int(x[2].min()))
    B = max(1, BUDGET // L)
    full = used = 0
    for c in range(0, len(spec), B):
        chunk = spec[c : c + B]
        ix = [x[0] for x in chunk]
        er = [(x[1], x[2]) for x in chunk]
        with log.op("edit_intact", len(chunk) * L):
            g2 = {}
            x, n = edit.run(m, ids, states, er, nu, U, src, valid, tg.stop, tg.tap(g2, pos))
            res = tg.measure(x, g2, pos)
        used, full = used + n, full + tg.stop * len(chunk) * L
        delta = torch.zeros(len(chunk), nu * U, device="cuda")
        for b, (op, uid) in enumerate(er):
            a = acts.view(nu, L, U)[uid // U, pos, uid % U]
            delta[b, uid] = edit.new_value(op, a, src[uid // U, pos, uid % U]) - a
        out[ix, : len(tg.cols)] = res.cpu()
        out[ix, len(tg.cols)] = tg.predict(d, delta, got).cpu()
        keep = [k for k, x in enumerate(chunk) if x[0] < n_group and x[1] != 1]
        if keep:
            with log.op("edit_masked", len(keep) * L):
                x, n = edit.run(m, mids, mstates, [er[k] for k in keep], nu, U, src, valid, N)
                out[[ix[k] for k in keep], -2] = net.mlm_loss(m, x, labels.expand(len(keep), -1)).cpu()
                out[[ix[k] for k in keep], -1] = lp(x).cpu()
            used, full = used + n, full + N * len(keep) * L
    return dict(e, coverage=valid.float().mean().item(), clean=cl, out=out, cols=tg.cols + ["direct", "mloss", "mlogp"]), used, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    shard = tuple(map(int, ap.parse_args().shard.split("/")))
    log = Log("s7", shard)
    m, s = net.load(), sae.load()
    df = data.sample()
    U = net.blocks(m)[0].ffn.ffn_hidden_size
    st = torch.load("out/s3/train/units.pt")
    mean = (st["stat"][:, 0] / st["n"]).float().cuda()
    std = ((st["stat"][:, 1] / st["n"]).float().cuda() - mean**2).clamp(min=1e-12).sqrt()
    stats = (mean, std, torch.from_numpy(np.load("out/s2/norm.npy")).cuda())
    todo = pick.targets(shard)
    samp = torch.load("out/s3/train/sample.pt")["acts"] if any(k == "units" for _, k, _, _ in todo) else None
    os.makedirs(OUT, exist_ok=True)
    used = full = 0
    for name, kind, t, ex in todo:
        tg = Target(m, s, kind, t)
        tg.df = df
        if tg.nu == 0:
            continue
        grad = torch.load(f"out/s6/{name}.pt")["rank"]
        with log.stage("rankings"), torch.no_grad():
            if kind == "features":
                scores = feature_scores(t, grad, stats)
            else:
                share = []
                for e in ex["train"]:
                    ids, got = pick.ids_of(df, e["on"][0]), {}
                    states, acts = edit.clean(m, ids, tg.stop, tg.nu)
                    net.run(m, ids, x=states[tg.stop - 1], start=tg.stop - 1, stop=tg.stop, tap=tg.tap(got, e["on"][1]))
                    share.append(acts[:, e["on"][1]] * tg.direct(None, got, e["on"][1])[0])
                scores = unit_scores(m, t[0], t[1], grad, samp, torch.stack(share).mean(0))
            rankings = top(scores, seed=sum(map(ord, name)))
        rows = rows_of(rankings)
        rec = dict(rankings={k: v.cpu() for k, v in rankings.items()}, rows=rows, examples=[])
        for split, pairs in ex.items():
            for e in pairs:
                ids = pick.ids_of(df, e["on"][0])
                if log.rec["tok_s"] is None:
                    log.plain(m, ids.expand(max(1, BUDGET // ids.shape[1]), -1))
                with torch.no_grad():
                    r, a, b = run_example(tg, ids, e, rankings, rows, log, U)
                rec["examples"].append(dict(r, split=split))
                used, full = used + a, full + b
        torch.save(rec, f"{OUT}/{name}.pt")
        log.note(done=name)
    log.cheaper("edited passes", before=f"{full} block-tokens recomputing every block",
                after=f"{used} block-tokens resuming from the clean state below the first edited block")
    log.write()


if __name__ == "__main__":
    main()
