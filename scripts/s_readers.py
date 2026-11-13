"""When the 64 units that most support an SAE feature (or one chosen unit) are removed, which later
units, attention heads and SAE features change, and how much of the change in predicted residues is
left when the most-changed units and heads are held at their unedited values?
"""

import numpy as np
import torch

from argus import data, edit, net, pick, runlog, sae

GROUP, R_UNITS, R_HEADS, K_UNITS, K_FEATS = 64, 64, 8, 1024, 256
RUN = ["pre", "value", "feature_change", "kl", "mloss", "mlogp"]


def feats(s, x60):
    pre, val, ind, _ = sae.encode(s, x60[0])
    return pre, torch.zeros_like(pre).scatter(1, ind, val)


def run_example(m, s, ids, e, kind, t, removed, norm, log):
    N, U, H, pos, dev = norm.shape[0], norm.shape[1], net.blocks(m)[0].attn.n_heads, e["on"][1], "cuda"
    start = int(removed.min()) // U
    mids, labels = data.mask(ids.cpu(), [e["on"][0]])
    labels[0, pos], mids[0, pos] = int(ids[0, pos]), net.MASK
    mids, labels = mids.cuda(), labels.cuda()
    with log.op("capture", 2 * ids.numel()):
        c, cm = edit.capture(m, ids), edit.capture(m, mids)
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
            x = net.run(m, ids, x=c["x"][start], start=start, edits=edit.remove_hold(removed, hold_u, hold_h, c, U, H), tap=tp)
            xm = net.run(m, mids, x=cm["x"][start], start=start, edits=edit.remove_hold(removed, hold_u, hold_h, cm, U, H))
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
    log = runlog.start("readers", __doc__)
    m, s = net.load(), sae.load()
    df = data.sample(log.input)
    norm = torch.from_numpy(np.load(log.read("weights", "none") + "/norm.npy")).cuda()
    U = norm.shape[1]
    for name, kind, t, ex in pick.targets(log.input, log.shard):
        if kind == "features":
            removed = torch.load(f"{log.read('group_edits')}/{name}.pt")["rankings"]["grad"][:GROUP].cuda()
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
        torch.save(rec, f"{log.dir}/{name}.pt")
        log.note(done=name)
    log.write()


if __name__ == "__main__":
    main()
