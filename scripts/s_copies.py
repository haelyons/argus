"""Is an SAE feature a copy of one unit? For each feature, which units' activations best track whether
it fires, do they still track it on holdout proteins, and how closely does any unit's write direction
match the feature's decoder direction? For each unit, which feature does it track best?
"""

import torch
import torch.nn.functional as F

from argus import net, runlog, sae
from argus.collect import stats, top_units, tracking

K = 8


@torch.no_grad()
def main():
    log = runlog.start("copies", __doc__)
    collect = log.read("collect")
    m, s = net.load(), sae.load()
    N, U = len(net.blocks(m)), net.blocks(m)[0].ffn.ffn_hidden_size
    Wd = F.normalize(s.W_dec, dim=-1)
    train, holdout = stats(collect, "train"), stats(collect, "holdout")
    cos_max, cos_unit, unit_r, unit_feature = [], [], [], []

    def best_feature(l, t):
        i = t.abs().max(0).indices
        unit_r.append(t.gather(0, i[None])[0].half().cpu()), unit_feature.append(i.short().cpu())

    with log.stage("tracking"):
        unit, r = top_units(train, N, K, best_feature)
    with log.stage("directions"):
        for l in range(N):
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
