"""Is an SAE feature a copy of one unit? For each feature, which units' activations best track whether
it fires, do they still track it on holdout proteins, and how closely does any unit's write direction
match the feature's decoder direction? For each unit, which feature does it track best?
"""

import numpy as np
import torch
import torch.nn.functional as F

from argus import net, runlog, sae

K = 8


def stats(collect, split):
    """Per unit mean and standard deviation (N, U), and per feature the share of tokens where it fires (F,)."""
    u, f = torch.load(f"{collect}/{split}/units.pt"), torch.load(f"{collect}/{split}/features.pt")
    mean = (u["stat"][:, 0] / u["n"]).float().cuda()
    sd = ((u["stat"][:, 1] / u["n"]).float().cuda() - mean**2).clamp(min=1e-12).sqrt()
    return collect, split, mean, sd, (f["stat"][0] / u["n"]).float().cuda()


def tracking(st, l):
    """r (F, U): correlation over tokens between block l's unit activations and each feature firing."""
    collect, split, mean, sd, p = st
    cond = torch.from_numpy(np.load(f"{collect}/{split}/cond_{l:02d}.npy").astype(np.float32)).cuda()
    return (cond - mean[l]) / sd[l] * ((p / (1 - p)).sqrt() * (p > 0))[:, None]


@torch.no_grad()
def main():
    log = runlog.start("copies", __doc__)
    collect = log.read("collect")
    m, s = net.load(), sae.load()
    N, U = len(net.blocks(m)), net.blocks(m)[0].ffn.ffn_hidden_size
    Wd = F.normalize(s.W_dec, dim=-1)
    train, holdout = stats(collect, "train"), stats(collect, "holdout")
    r = unit = None
    cos_max, cos_unit, unit_r, unit_feature = [], [], [], []
    for l in range(N):
        with log.stage("tracking"):
            t = tracking(train, l)
            v, i = t.abs().max(0)
            unit_r.append(t.gather(0, i[None])[0].half().cpu()), unit_feature.append(i.short().cpu())
            ids = (l * U + torch.arange(U, device="cuda")).expand_as(t)
            t, ids = (t if r is None else torch.cat([r, t], 1)), (ids if r is None else torch.cat([unit, ids], 1))
            i = t.abs().topk(K, 1).indices
            r, unit = t.gather(1, i), ids.gather(1, i)
        with log.stage("directions"):
            v, i = (F.normalize(net.out_weight(m, l), dim=-1) @ Wd.T).abs().max(0)
            cos_max.append(v), cos_unit.append(l * U + i)
    cos_max, j = torch.stack(cos_max).max(0)
    cos_unit = torch.stack(cos_unit).gather(0, j[None])[0]
    r_holdout, cos = torch.zeros_like(r), torch.zeros_like(r)
    for l in range(N):
        here = unit // U == l
        t = tracking(holdout, l).gather(1, (unit % U) * here)
        c = (Wd @ F.normalize(net.out_weight(m, l), dim=-1).T).gather(1, (unit % U) * here)
        r_holdout, cos = torch.where(here, t, r_holdout), torch.where(here, c, cos)
    fs = torch.load(f"{collect}/train/features.pt")["stat"]
    torch.save(dict(unit=unit.cpu(), r=r.cpu(), r_holdout=r_holdout.cpu(), cos=cos.cpu(), cos_max=cos_max.cpu(), cos_unit=cos_unit.cpu(),
                    unit_r=torch.stack(unit_r), unit_feature=torch.stack(unit_feature), tokens=fs[0], activity=fs[2]), f"{log.dir}/copies.pt")
    w = (fs[2] / fs[2].sum()).float().cuda()
    at = lambda x: {t: dict(features=round((x >= t).float().mean().item(), 4), activity=round(w[x >= t].sum().item(), 4)) for t in (0.3, 0.5, 0.7, 0.9)}
    log.note(r=at(r[:, 0].abs()), r_holdout=at(r_holdout[:, 0].abs()), cos_max=at(cos_max),
             units={t: round((torch.stack(unit_r).float().abs() >= t).float().mean().item(), 5) for t in (0.3, 0.5, 0.7, 0.9)})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
