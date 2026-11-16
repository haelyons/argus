"""Protein inputs. Each is a folder inputs/<name>/ holding sample.parquet (proteins, annotations,
split) with targets.json and examples.json beside it. Serves length-sorted batches by shard.

python -m argus.data <name>     builds inputs/<name>/sample.parquet
"""

import gzip
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import torch

PAD, MASK, CLS, EOS = 1, 32, 0, 2
RAW = "data/raw/swissprot.tsv.gz"
FEATURES = "data/features.parquet"
CHAINS = "data/raw/full_seq_dict.json"  # github.com/NainaniJatinZ/plm_circuits, data/full_seq_dict.json
PFAM = "data/raw/pfam.tsv"  # ebi.ac.uk/interpro/api/entry/pfam?extra_fields=counters: accession, proteins, ...
AFDB, MODELS = "https://alphafold.ebi.ac.uk/api/prediction/", "data/raw/afdb"
SEED, HOLDOUT = 0, 0.1
STRIDE = 2048
VOCAB = "LAGVSERTIDPKQNFYMHWCXBUZO"
TOK = np.full(128, 3, np.int64)
TOK[[ord(c) for c in VOCAB]] = np.arange(4, 4 + len(VOCAB))


def path(name, file):
    return f"inputs/{name}/{file}"


def swissprot_random(df):
    """50,000 reviewed UniProt proteins of 30 to 1022 residues, uniformly at random."""
    return df[df.Length.between(30, 1022)].sample(50_000, random_state=SEED)


def swissprot_biohub(df):
    """Every protein named in a top-20 Swiss-Prot list of Biohub's feature table, 30 to 2046 residues."""
    top = pd.read_parquet(FEATURES, columns=["top_swissprot_activations"]).top_swissprot_activations
    ids = {x["uniprot_id"] for v in top for x in v}
    return df[df.Entry.isin(ids) & df.Length.between(30, 2046)].sort_values("Entry")


def jump(df):
    """The two PDB chains of Nainani et al. 2025 (bioRxiv 10.1101/2025.08.22.671739), from the sequences of
    Zhang et al. 2024: 2B61 chain A (MetXA, P45131) and 1PVG chain A (TOP2, P06786), expression tags included."""
    seqs = json.load(open(CHAINS))
    return pd.DataFrame([dict(Entry=k, Length=len(seqs[k]), Sequence=seqs[k]) for k in ("2B61A", "1PVGA")])


def model(entry, n):
    """The AlphaFold model of a UniProt entry, fetched once: C-beta coordinates, C-alpha for glycine, (n, 3) and
    pLDDT (n,). None when it has no model of n residues."""
    file = f"{MODELS}/{entry}.pdb"

    def get(url, tries=8):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except OSError as e:
            if getattr(e, "code", None) in (400, 404):
                return None
            if tries == 0:
                raise
        time.sleep(15)
        return get(url, tries - 1)

    if not os.path.exists(file):
        os.makedirs(MODELS, exist_ok=True)
        found = get(AFDB + entry)
        if not found or not json.loads(found):
            return None
        pdb = get(json.loads(found)[0]["pdbUrl"])
        open(file, "wb").write(gzip.decompress(pdb) if pdb[:2] == b"\x1f\x8b" else pdb)
    atoms = [(int(r[22:26]), r[12:16].strip(), [float(r[30:38]), float(r[38:46]), float(r[46:54])], float(r[60:66])) for r in open(file) if r.startswith("ATOM")]
    ca = {i: (x, q) for i, a, x, q in atoms if a == "CA"}
    xyz = {i: x for i, (x, q) in ca.items()} | {i: x for i, a, x, q in atoms if a == "CB"}
    if sorted(xyz) != list(range(1, n + 1)):
        return None
    return torch.tensor([xyz[i] for i in range(1, n + 1)]), torch.tensor([ca[i][1] for i in range(1, n + 1)])


def swissprot_depth(df):
    """Reviewed proteins of 80 to 200 residues with one Pfam family, one per family, by the family's size
    (proteins in InterPro): 10 in each of four length ranges for families of at most 300, of 1,000 to 10,000
    and of at least 30,000, taken in random order. Each has an AlphaFold model with a mean pLDDT of at least
    85 in which at least 1% of the residue pairs 24 or more apart are within 8 angstrom."""
    size = pd.read_csv(PFAM, sep="\t", header=None, index_col=0)[1]
    df = df[df.Length.between(80, 200) & df.Pfam.str.count(";").eq(1) & df.Sequence.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+")]
    df = df.assign(family=df.Pfam.str.strip(";")).sample(frac=1, random_state=SEED).drop_duplicates("family")
    df = df.assign(family_size=df.family.map(size), plddt=np.nan)
    keep = []
    for lo, hi in ((0, 300), (1_000, 10_000), (30_000, np.inf)):
        for b in range(4):
            n = 0
            for i, r in df[df.family_size.between(lo, hi) & (((df.Length - 80) // 30).clip(upper=3) == b)].iterrows():
                m = model(r.Entry, r.Length)
                if m is None or m[1].mean() < 85:
                    continue
                far = torch.triu_indices(r.Length, r.Length, 24)
                if (torch.cdist(m[0], m[0])[far[0], far[1]] < 8).float().mean() < 0.01:
                    continue
                df.loc[i, "plddt"] = m[1].mean().item()
                keep.append(i)
                n += 1
                if n == 10:
                    break
    return df.loc[keep]


BUILD = {"swissprot-random": swissprot_random, "swissprot-biohub": swissprot_biohub, "jump": jump, "swissprot-depth": swissprot_depth}


def build(name):
    df = BUILD[name](pd.read_csv(RAW, sep="\t", low_memory=False)).reset_index(drop=True)
    df["split"] = np.where(np.random.default_rng(SEED).random(len(df)) < HOLDOUT, "holdout", "train")
    os.makedirs(path(name, ""), exist_ok=True)
    df.to_parquet(path(name, "sample.parquet"))
    print(name, df.split.value_counts().to_dict(), "tokens", int(df.Length.sum()) + 2 * len(df))


def sample(name, split=None):
    """Proteins of an input. idx is the protein index; a token id is idx * STRIDE + position."""
    df = pd.read_parquet(path(name, "sample.parquet"), columns=["Entry", "Length", "Sequence", "split"])
    df["idx"] = df.index
    return df if split is None else df[df.split == split]


def tokens(seq):
    return np.concatenate([[CLS], TOK[np.frombuffer(seq.encode(), np.uint8)], [EOS]])


def batches(df, budget=16384, shard=(0, 1)):
    """Yields (idx, ids): protein indices and padded token ids, at most `budget` padded tokens."""
    df = df.sort_values("Length", ascending=False)
    out, rows = [], []
    for idx, n, seq in zip(df.idx, df.Length, df.Sequence):
        if rows and (len(rows) + 1) * (rows[0][1] + 2) > budget:
            out.append(rows)
            rows = []
        rows.append((idx, n, seq))
    out.append(rows)
    for rows in out[shard[0] :: shard[1]]:
        ids = np.full((len(rows), rows[0][1] + 2), PAD, np.int64)
        for i, (_, n, seq) in enumerate(rows):
            ids[i, : n + 2] = tokens(seq)
        yield np.array([r[0] for r in rows]), torch.from_numpy(ids)


def mutants(ids, budget=8192, at=None):
    """Every single substitution of one protein, ids (1, T), or of its token positions `at`: yields (pos (B,),
    ids (B, T)) of at most `budget` tokens. Rows run over residues, then over the 20 amino acids in token
    order, the protein's own included."""
    T = ids.shape[1]
    at = torch.arange(1, T - 1, device=ids.device) if at is None else at
    pos = at.repeat_interleave(20)
    aa = torch.arange(4, 24, device=ids.device).repeat(len(at))
    for i in range(0, len(pos), budget // T):
        p = pos[i : i + budget // T]
        out = ids.repeat(len(p), 1)
        out[torch.arange(len(p)), p] = aa[i : i + budget // T]
        yield p, out


def doubles(ids, pairs, budget=8192):
    """Every replacement of two residues of one protein, ids (1, T), for each of `pairs` (P, 2) of token
    positions: yields (pair (B,), ids (B, T)) of at most `budget` tokens. Rows run over pairs, then over the 20
    amino acids at the first position, then at the second, the protein's own included."""
    T = ids.shape[1]
    k = torch.arange(len(pairs), device=ids.device).repeat_interleave(400)
    aa = torch.arange(4, 24, device=ids.device)
    a, b = aa.repeat_interleave(20).repeat(len(pairs)), aa.repeat(20 * len(pairs))
    for i in range(0, len(k), budget // T):
        s = slice(i, i + budget // T)
        out = ids.repeat(len(k[s]), 1)
        r = torch.arange(len(k[s]))
        out[r, pairs[k[s], 0]], out[r, pairs[k[s], 1]] = a[s], b[s]
        yield k[s], out


def mask(ids, idx, rate=0.15):
    """Masked copy and labels (-100 off-mask). Seeded by protein index, so it is the same in every run."""
    ids, labels = ids.clone(), torch.full_like(ids, -100)
    for i, p in enumerate(idx):
        n = int((ids[i] != PAD).sum()) - 2
        pos = 1 + np.random.default_rng(int(p)).choice(n, max(1, round(rate * n)), replace=False)
        labels[i, pos] = ids[i, pos]
        ids[i, pos] = MASK
    return ids, labels


if __name__ == "__main__":
    build(sys.argv[1])
