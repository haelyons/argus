"""Is each SAE feature as strong and as frequent in our proteins as in Biohub's own statistics, and
do its strongest residues fall in the proteins Biohub lists for it?
"""

import pandas as pd
import torch

from argus import data, runlog


def main():
    log = runlog.start("compare_peaks", __doc__)
    collect = log.read("collect")
    df = data.sample(log.input)
    tr, ho = (torch.load(f"{collect}/{s}/features.pt") for s in ("train", "holdout"))
    b = pd.read_parquet(data.FEATURES).set_index("feature_id").sort_index()
    top = torch.cat([tr["hi"][1], ho["hi"][1]]) // data.STRIDE
    listed = [{x["uniprot_id"] for x in v} for v in b.top_swissprot_activations]
    entry = df.Entry.values
    out = pd.DataFrame(dict(
        ours_max=torch.maximum(tr["hi"][0][0], ho["hi"][0][0]).numpy(),
        ours_proteins=(tr["stat"][1] + ho["stat"][1]).numpy(),
        ours_residues=(tr["stat"][0] + ho["stat"][0]).numpy(),
        biohub_swissprot_max=[max(x["activation"] for x in v) for v in b.top_swissprot_activations],
        uniref90_max=b.uniref90_max_activation.values,
        uniref90_proteins=b.uniref90_frequency.values,
        threshold=b.threshold.values,
        category=b.category.values,
        strongest_proteins_listed=[len({entry[p] for p in top[:, j].tolist()} & listed[j]) for j in range(len(b))],
        strongest_proteins=[len(set(top[:, j].tolist())) for j in range(len(b))],
    ))
    out.index.name = "feature_id"
    out["max_ratio"] = out.ours_max / out.uniref90_max
    out["frequency_ratio"] = (out.ours_proteins / len(df)) / (out.uniref90_proteins / out.uniref90_proteins.max())
    out.to_parquet(f"{log.dir}/features.parquet")
    q = lambda x: x.quantile([0.01, 0.1, 0.5, 0.9, 0.99]).round(3).tolist()
    log.note(max_ratio=q(out.max_ratio), frequency_ratio=q(out.frequency_ratio), below_half=float((out.max_ratio < 0.5).mean()),
             listed_share=float(out.strongest_proteins_listed.sum() / out.strongest_proteins.sum()))
    log.write()


if __name__ == "__main__":
    main()
