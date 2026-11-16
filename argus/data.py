"""Protein inputs. Each is a folder inputs/<name>/ holding sample.parquet (proteins, annotations,
split) with targets.json and examples.json beside it. Serves length-sorted batches by shard.

python -m argus.data <name>     builds inputs/<name>/sample.parquet
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

PAD, MASK, CLS, EOS = 1, 32, 0, 2
RAW = "data/raw/swissprot.tsv.gz"
FEATURES = "data/features.parquet"
CHAINS = "data/raw/full_seq_dict.json"  # github.com/NainaniJatinZ/plm_circuits, data/full_seq_dict.json
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


BUILD = {"swissprot-random": swissprot_random, "swissprot-biohub": swissprot_biohub, "jump": jump}


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
