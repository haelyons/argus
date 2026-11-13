"""The protein sample: build it once, then serve length-sorted batches by shard.

python -m argus.data            builds data/sample.parquet from data/raw/swissprot.tsv.gz
"""

import numpy as np
import pandas as pd
import torch

PAD, MASK, CLS, EOS = 1, 32, 0, 2
SAMPLE = "data/sample.parquet"
RAW = "data/raw/swissprot.tsv.gz"
N, SEED, MIN_LEN, MAX_LEN, HOLDOUT = 50_000, 0, 30, 1022, 0.1
VOCAB = "LAGVSERTIDPKQNFYMHWCXBUZO"
TOK = np.full(128, 3, np.int64)
TOK[[ord(c) for c in VOCAB]] = np.arange(4, 4 + len(VOCAB))


def build():
    df = pd.read_csv(RAW, sep="\t", low_memory=False)
    df = df[df.Length.between(MIN_LEN, MAX_LEN)]
    df = df.sample(N, random_state=SEED).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    df["split"] = np.where(rng.random(N) < HOLDOUT, "holdout", "train")
    df.to_parquet(SAMPLE)
    print(df.split.value_counts().to_dict(), "tokens", int(df.Length.sum()) + 2 * N)


def sample(split=None):
    df = pd.read_parquet(SAMPLE, columns=["Entry", "Length", "Sequence", "split"])
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
    build()
