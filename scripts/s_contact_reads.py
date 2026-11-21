"""What do the heads whose attention shows contacts read to attend as they do? On the proteins of
s_contact_edits, unchanged, for the first TOP heads of s_contact_rank's attention ranking: how many of the
L residue pairs 24 or more apart that a head's attention weighs most are in contact, as s_contact_rank
scores it, (a) as it is, and on the protein with every residue masked; (b) when what the heads, or the
units, of one window of WINDOW blocks wrote is taken out of the state the head reads, for its queries
and keys, for its queries alone and for its keys alone, everything else as it was; (c) when the heads'
outputs, or the units' activations, of one window take their values on the fully masked protein and
every later block runs on?
"""

import glob

import pandas as pd
import torch

from argus import data, edit, net, pairs, pick, runlog

TOP, WINDOW = 64, 8
SIDES = ["both", "query", "key"]


def reads(m, ids, c, top, H):
    """Attention precision of the heads `top`, in their order: clean (K,), on the masked protein (K,), with a
    window's writes out of what they read (sides, kinds, windows, K), with a window at its masked values
    (kinds, windows, K). Kinds are heads, then units."""
    bl, by = net.blocks(m), edit.part(top, H)
    N, at = len(bl), lambda l: (top // H == l).nonzero()[:, 0]
    score = lambda p, l: pairs.precision(p[0, by[l], 1:-1, 1:-1], c)[0]
    st, wr = [m.esmc.embed(ids)], []
    net.run(m, ids, x=st[0], tap=lambda l, r: (st.append(r["x"]), wr.append((r["attn"], r["ffn"]))))
    off = edit.capture(m, data.masked(ids, []))
    W = N // WINDOW
    clean, masked = (torch.zeros(len(top), device=ids.device) for _ in range(2))
    direct, total = torch.zeros(len(SIDES), 2, W, len(top), device=ids.device), torch.zeros(2, W, len(top), device=ids.device)
    for l in by:
        q, k, _ = net.qkv(bl[l].attn, st[l])
        clean[at(l)] = score(net.attention(bl[l].attn, q, k), l)
        qm, km, _ = net.qkv(bl[l].attn, off["x"][l])
        masked[at(l)] = score(net.attention(bl[l].attn, qm, km), l)
        for w in range(W):
            for kind in range(2):
                gone = sum((wr[b][kind] for b in range(w * WINDOW, min((w + 1) * WINDOW, l))), torch.zeros_like(st[l]))
                q2, k2, _ = net.qkv(bl[l].attn, st[l] - gone)
                for s, (a, b) in enumerate(((q2, k2), (q2, k), (q, k2))):
                    direct[s, kind, w, at(l)] = score(net.attention(bl[l].attn, a, b), l)
    total[:] = clean
    for w in range(W):
        for kind in range(2):
            span = range(w * WINDOW, (w + 1) * WINDOW)
            ed = {("head", b): (lambda z, b=b: off["z"][b][None]) for b in span} if kind == 0 else {("act", b): (lambda a, g, v, b=b: off["a"][b][None]) for b in span}
            net.run(m, ids, x=st[span[0]], start=span[0], edits=ed, pattern=True, tap=lambda l, r: total[kind, w].index_copy_(0, at(l), score(r["p"], l)) if l in by else None)
    return clean, masked, direct, total


def main():
    log = runlog.start("contact_reads", __doc__)
    if log.reduce:
        return reduce(log)
    m, df = net.load(), data.sample(log.input)
    r = torch.load(log.read("contact_rank") + "/rank.pt")
    top, H = r["rank"]["readable"][:TOP].cuda(), net.blocks(m)[0].attn.n_heads
    order = df[(df.split == "holdout") | df.idx.isin(r["kept"])].sort_values("Length", ascending=False).idx.tolist()
    for p in order[log.shard[0] :: log.shard[1]]:
        ids = pick.ids_of(df, p)
        if log.rec["tok_s"] is None:
            log.plain(m, ids.expand(8192 // ids.shape[1], -1))
        with log.op("reads", ids.numel()), torch.no_grad():
            out = reads(m, ids, pairs.contacts(df.Entry[p], ids.shape[1] - 2).cuda(), top, H)
        torch.save(dict(protein=p, heads=top.cpu(), **dict(zip(("clean", "masked", "direct", "total"), (t.cpu() for t in out)))), f"{log.dir}/p{p}.pt")
        log.note(**{f"p{p}": dict(clean=out[0].mean().item(), masked=out[1].mean().item())})
    log.write()


def reduce(log):
    df, rows = data.sample(log.input), []
    for d in map(torch.load, sorted(glob.glob(f"{log.dir}/p*.pt"))):
        g = "holdout" if df.split[d["protein"]] == "holdout" else "set_aside"
        row = lambda what, kind, w, x: rows.append(dict(protein=d["protein"], group=g, what=what, kind=kind, window=w, precision=x.mean().item()))
        row("clean", "", -1, d["clean"]), row("masked", "", -1, d["masked"])
        for k, kind in enumerate(("heads", "units")):
            for w in range(d["total"].shape[1]):
                row("total", kind, w, d["total"][k, w])
                for s, side in enumerate(SIDES):
                    row(f"direct_{side}", kind, w, d["direct"][s, k, w])
    t = pd.DataFrame(rows)
    t.to_parquet(f"{log.dir}/reads.parquet")
    by = lambda g: t[t.group == g].groupby(["what", "kind", "window"]).precision.mean()
    log.note(**{g: {f"{what} {kind}".strip(): [round(x, 4) for x in v.tolist()] for (what, kind), v in by(g).groupby(level=[0, 1])} for g in ("holdout", "set_aside")})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
