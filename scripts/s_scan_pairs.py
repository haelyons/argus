"""Are the residues that move a pole from afar in contact with its residue in the folded protein, and
does a pole move more when a residue changes its chemical class than when it keeps it? From s_scan,
with contacts read from each protein's AlphaFold model (C-beta atoms, C-alpha for glycine, within
CONTACT angstrom). Per pole: of the residues FAR or more along the chain, how many touch its residue,
and of those whose substitution moves it by HALF its clean activation, how many do? Does the one that
moves it most? How much does it move under substitutions within a class and across classes, at its
own residue and at the others? Per protein: of the L residue pairs, SEP or more apart, that the
logits couple most, and that each block's state couples most, how many are in contact?
"""

import glob
import gzip
import json
import os
import urllib.request

import pandas as pd
import torch
import torch.nn.functional as F

from argus import data, runlog

FAR, HALF, CONTACT, SEP = 12, 0.5, 8.0, 24
CLASSES = ["AVLIM", "FWY", "STNQ", "KRH", "DE", "C", "G", "P"]
AFDB, RAW = "https://alphafold.ebi.ac.uk/api/prediction/", "data/raw/afdb"


def contacts(entry, n):
    """(n, n) residue contacts of the AlphaFold model, or None when it has no model of n residues."""
    path = f"{RAW}/{entry}.pdb"
    get = lambda url: urllib.request.urlopen(url, timeout=60).read()
    if not os.path.exists(path):
        os.makedirs(RAW, exist_ok=True)
        try:
            pdb = get(json.loads(get(AFDB + entry))[0]["pdbUrl"])
        except Exception:
            return None
        open(path, "wb").write(gzip.decompress(pdb) if pdb[:2] == b"\x1f\x8b" else pdb)
    atoms = [(int(r[22:26]), r[12:16].strip(), [float(r[30:38]), float(r[38:46]), float(r[46:54])]) for r in open(path) if r.startswith("ATOM")]
    xyz = {i: x for i, a, x in atoms if a == "CA"} | {i: x for i, a, x in atoms if a == "CB"}
    if sorted(xyz) != list(range(1, n + 1)):
        return None
    x = torch.tensor([xyz[i] for i in range(1, n + 1)])
    return torch.cdist(x, x) < CONTACT


def precision(m, c):
    """Share in contact of the L pairs, SEP or more apart, with the largest score: m (L, L) made symmetric,
    less each pair's row mean times column mean over the mean."""
    m = (m + m.T) / 2
    m = m - m.mean(0, keepdim=True) * m.mean(1, keepdim=True) / m.mean()
    i, j = torch.triu_indices(len(m), len(m), SEP)
    return c[i, j][m[i, j].topk(min(len(m), len(i))).indices].float().mean().item(), c[i, j].float().mean().item()


def main():
    log = runlog.start("scan_pairs", __doc__)
    df, scan = data.sample(log.input), log.read("scan")
    cls = torch.tensor([[k for k, c in enumerate(CLASSES) if a in c][0] for a in data.VOCAB[:20]])
    pairs, maps, cont = [], [], {}
    for f in sorted(glob.glob(f"{scan}/p*.pt")):
        d = torch.load(f)
        p, t, L = d["protein"], d["pos"], d["dg"].shape[0]
        c = contacts(df.Entry[p], L)
        if c is None:
            continue
        cont[p] = c
        a0 = F.silu(d["g"]) * d["v"]
        rel = (F.silu(d["g"] + d["dg"].float()) * (d["v"] + d["dv"].float()) - a0).abs() / a0.abs()  # (L, 20, P)
        res = d["ids"][1:-1] - 4
        std = (res >= 0) & (res < 20)
        same = (cls[res.clamp(0, 19)][:, None] == cls[None]) & std[:, None]
        same[std, res[std]] = False
        other = (cls[res.clamp(0, 19)][:, None] != cls[None]) & std[:, None]
        own = (torch.arange(1, L + 1)[:, None] == t[None]).float()  # (L, P)
        mean = lambda sub, at: (rel * sub[:, :, None] * at[:, None]).sum((0, 1)) / (sub.sum(1)[:, None] * at).sum(0)
        far = (torch.arange(1, L + 1)[None] - t[:, None]).abs() >= FAR  # (P, L)
        touch = F.pad(c.float(), (0, 0, 1, 1))[t] > 0
        move = rel.amax(1).T
        hit, top = far & (move >= HALF), torch.where(far, move, -1.0).max(1)
        cols = dict(block=d["block"], pole=d["pole"], unit=d["unit"], pos=t, size=d["size"], a=a0, far=far.sum(1), far_touch=(far & touch).sum(1),
                    moved=hit.sum(1), moved_touch=(hit & touch).sum(1), mover=top.indices + 1, mover_move=top.values,
                    mover_touch=touch.gather(1, top.indices[:, None])[:, 0], own_same=mean(same, own), own_other=mean(other, own),
                    rest_same=mean(same, 1 - own), rest_other=mean(other, 1 - own))
        pairs.append(pd.DataFrame({k: v.numpy() for k, v in cols.items()}).assign(protein=p))
        score = [d["logits"].float().flatten(3).norm(dim=3).norm(dim=1)] + list(d["state"].float())
        for b, m in enumerate(score):
            hits, chance = precision(m[:, 1:-1], c)
            maps.append(dict(protein=p, residues=L, block=b - 1, precision=hits, chance=chance))
    pairs, maps = pd.concat(pairs), pd.DataFrame(maps)
    pairs.to_parquet(f"{log.dir}/pairs.parquet")
    maps.to_parquet(f"{log.dir}/maps.parquet")
    torch.save(cont, f"{log.dir}/contacts.pt")
    k = pairs[pairs.mover_move >= HALF]
    log.note(proteins=len(cont), pairs=len(pairs), far_touch=pairs.far_touch.sum() / pairs.far.sum(), moved_touch=pairs.moved_touch.sum() / pairs.moved.sum(),
             poles_moved_from_afar=len(k), mover_touch={b: float(k[k.block // 8 == b // 8].mover_touch.mean()) for b in range(0, 80, 8)}, **{c: float(pairs[c].median()) for c in ("own_same", "own_other", "rest_same", "rest_other")},
             precision_logits=maps[maps.block < 0].precision.mean(), precision_last_block=maps[maps.block == maps.block.max()].precision.mean(),
             chance=maps[maps.block < 0].chance.mean())
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
