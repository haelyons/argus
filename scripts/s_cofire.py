"""Which unit poles are active at the same residues? Taking each unit's positive part and negative
part as two poles, how correlated over all train residues is every pair of poles, in any two blocks,
and into which 16384 groups does k-means on those correlations sort them?

Each pole's activity over residues is kept as a random projection to 4096 numbers, which keeps every
pairwise correlation to about 1/64.
"""

import glob
import os

import torch
import torch.nn.functional as F

from argus import data, net, runlog

D, K, ITERS, BUDGET, CHUNK, SEED = 4096, 16384, 50, 8192, 32768, 0


def sketch(m, df, log):
    bl = net.blocks(m)
    N, U = len(bl), bl[0].ffn.ffn_hidden_size
    sk = torch.zeros(N, 2 * U, D, device="cuda")
    s1, s2 = (torch.zeros(N, 2 * U, dtype=torch.float64, device="cuda") for _ in range(2))
    rs, n = torch.zeros(D, dtype=torch.float64, device="cuda"), 0
    for idx, ids in data.batches(df, BUDGET, log.shard):
        ids = ids.cuda()
        if log.rec["tok_s"] is None:
            log.plain(m, ids)
        keep = ids != net.PAD
        T = int(keep.sum())
        R = torch.randn(T, D, device="cuda", generator=torch.Generator("cuda").manual_seed(int(idx[0])))

        def tap(l, r):
            a = r["a"][keep]
            x = torch.cat([a.relu(), (-a).relu()], 1)
            s1[l] += x.sum(0, dtype=torch.float64)
            s2[l] += (x * x).sum(0, dtype=torch.float64)
            sk[l] += x.T @ R

        with log.op("pass_sketch", T), torch.no_grad():
            net.run(m, ids, tap=tap)
        rs += R.sum(0, dtype=torch.float64)
        n += T
        log.note(tokens=n)
    os.makedirs(f"{log.dir}/shard{log.shard[0]}", exist_ok=True)
    torch.save(dict(n=n, sk=sk.cpu(), s1=s1.cpu(), s2=s2.cpu(), rs=rs.cpu()), f"{log.dir}/shard{log.shard[0]}/sketch.pt")


def assign(v, C):
    out = [(v[i : i + CHUNK] @ C.T).max(1) for i in range(0, len(v), CHUNK)]
    return torch.cat([o.indices for o in out]), torch.cat([o.values for o in out])


@torch.no_grad()
def reduce(log):
    parts = [torch.load(f) for f in sorted(glob.glob(f"{log.dir}/shard*/sketch.pt"))]
    n = sum(p["n"] for p in parts)
    N, U2, _ = parts[0]["sk"].shape
    order = lambda x: x.view(N, 2, U2 // 2, *x.shape[2:]).transpose(0, 1).flatten(0, 2)
    mean = order(sum(p["s1"] for p in parts)) / n
    var = order(sum(p["s2"] for p in parts)) / n - mean**2
    sk = parts[0]["sk"]
    for p in parts[1:]:
        sk += p["sk"]
    v = order(sk) - mean.float()[:, None] * sum(p["rs"] for p in parts).float()
    live = var > 1e-12
    v = v[live].cuda()
    check = (v.pow(2).sum(1) / (D * n * var[live].float().cuda())).quantile(torch.tensor([0.01, 0.5, 0.99], device="cuda"))
    v = F.normalize(v, dim=1)

    g = torch.Generator("cuda").manual_seed(SEED)
    C = v[torch.randperm(len(v), device="cuda", generator=g)[:K]].clone()
    fit = []
    for _ in range(ITERS):
        with log.stage("kmeans"):
            lab, sim = assign(v, C)
            S = torch.zeros_like(C).index_add_(0, lab, v)
            C = torch.where(S.norm(dim=1, keepdim=True) > 0, F.normalize(S, dim=1), C)
            fit.append(round(sim.mean().item(), 4))
    lab, sim = assign(v, C)
    size = torch.bincount(lab, minlength=K)
    full = lambda x, fill: torch.full((len(live),), fill, dtype=x.dtype).masked_scatter_(live, x.cpu())
    cluster = full(lab.int(), -1)
    control = full(lab[torch.randperm(len(lab), device="cuda", generator=g)].int(), -1)
    torch.save(dict(cluster=cluster, control=control, sim=full(sim.half(), 0), mean=mean.float(), sd=var.clamp(min=0).sqrt().float(),
                    size=size.cpu(), order="pole (positive, negative), block, unit"), f"{log.dir}/clusters.pt")

    block = torch.arange(N).repeat_interleave(U2 // 2).repeat(2)[live].cuda()
    span = torch.zeros(K, N, device="cuda").index_put_((lab, block), torch.ones(len(lab), device="cuda"), accumulate=True)
    q = lambda x: [round(t, 4) for t in x.float().quantile(torch.tensor([0.1, 0.5, 0.9], device=x.device)).tolist()]
    log.note(tokens=n, poles=len(live), live=int(live.sum()), sketch_norm_over_true=[round(t, 4) for t in check.tolist()],
             mean_cosine_to_centre=dict(first=fit[0], last=fit[-1], final=round(sim.mean().item(), 4)), cosine=q(sim),
             size=q(size), largest=int(size.max()), empty=int((size == 0).sum()), blocks_per_group=q((span > 0).sum(1)),
             share_in_main_block=q(span.max(1).values / span.sum(1).clamp(min=1)))
    log.write()
    print(open(log.path).read())


def main():
    log = runlog.start("cofire", __doc__)
    if log.reduce:
        return reduce(log)
    df = data.sample(log.input)
    sketch(net.load(), df[df.split == "train"], log)
    log.write()


if __name__ == "__main__":
    main()
