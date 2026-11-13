"""From the weights alone, which SAE features does each unit write toward, and which later units and
attention heads could read what it writes?
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from argus import net, sae
from argus.runlog import Log

K, KH = 64, 32
OUT = "out/weights"


def top_abs(vals, ids, k, dim):
    i = vals.abs().topk(k, dim).indices
    return vals.gather(dim, i), ids.gather(dim, i)


def merge(run, vals, ids, dim, k=K):
    """Running top-k by |value| along dim; ids broadcast against vals."""
    ids = ids.expand_as(vals)
    if run is not None:
        vals, ids = torch.cat([run[0], vals], dim), torch.cat([run[1], ids], dim)
    return top_abs(vals, ids, k, dim)


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    log = Log("weights")
    m, s = net.load(), sae.load()
    bl = net.blocks(m)
    N, U, H = len(bl), bl[0].ffn.ffn_hidden_size, bl[0].attn.n_heads
    dev = "cuda"
    write = [net.out_weight(m, l) for l in range(N)]
    read = [F.normalize(net.centred(b.ffn.fc1_weight, b.ffn.layer_norm_weight), dim=-1) for b in bl]
    qkv = [net.centred(b.attn.layernorm_qkv.weight, b.attn.layernorm_qkv.layer_norm_weight).view(3, H, -1, 2560) for b in bl]
    Wr, Wd = sae.read_weight(s), F.normalize(s.W_dec, dim=-1)
    feat = torch.arange(Wd.shape[0], device=dev)

    np.save(f"{OUT}/norm.npy", torch.stack([w.norm(dim=-1) for w in write]).cpu().numpy())
    dec, readers, heads, writers = [], [], [], [None] * N
    for l in range(N):
        w = write[l]
        wn = F.normalize(w, dim=-1)
        with log.stage("features"):
            if l < sae.LAYER:
                np.save(f"{OUT}/enc_{l:02d}.npy", (w @ Wr).half().cpu().numpy())
            dec.append(tuple(t.cpu() for t in top_abs(wn @ Wd.T, feat.expand(U, -1), K, 1)))
        with log.stage("readers"):
            r, h = None, [None] * 3
            for j in range(l + 1, N):
                c = read[j] @ wn.T
                rid = j * 2 * U + torch.arange(2 * U, device=dev)[:, None]
                r = merge(r, c, rid, 0)
                wid = l * U + torch.arange(U, device=dev)[None]
                writers[j] = merge(writers[j], c, wid, 1)
                share = torch.einsum("khdD,uD->khdu", qkv[j], wn).norm(dim=2) / qkv[j].norm(dim=(2, 3))[..., None]
                hid = j * H + torch.arange(H, device=dev)[:, None]
                h = [merge(h[k], share[k], hid, 0, KH) for k in range(3)]
            if r is not None:
                readers.append((r[0].T.cpu(), r[1].T.cpu()))
                heads.append((torch.stack([x[0].T for x in h]).cpu(), torch.stack([x[1].T for x in h]).cpu()))
        print(l, flush=True)

    def pack(xs):
        return dict(val=torch.stack([x[0] for x in xs]).half(), id=torch.stack([x[1] for x in xs]).int())

    torch.save(pack(dec), f"{OUT}/dec_top.pt")
    torch.save(pack(readers), f"{OUT}/readers.pt")
    torch.save(pack(heads), f"{OUT}/heads.pt")
    torch.save(pack([(w[0].cpu(), w[1].cpu()) for w in writers[1:]]), f"{OUT}/writers.pt")
    log.note(N=N, U=U, H=H, K=K, files=sorted(os.listdir(OUT)))
    log.write()


if __name__ == "__main__":
    main()
