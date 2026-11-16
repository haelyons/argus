"""Does ESM C 6B couple the residues in contact better in proteins of larger families? From
s_scan_pairs on an input whose proteins carry family_size (proteins in the Pfam family) and plddt
(mean of the AlphaFold model): the share in contact of the L most coupled residue pairs, for the
logits and for every block's state. What is its mean in the families of at most SMALL, of at least
LARGE and of the sizes between? What is its rank correlation with family size, alone and with length
and pLDDT taken out, and with pLDDT once length and family size are taken out?
"""

import numpy as np
import pandas as pd

from argus import data, runlog

SMALL, LARGE = 300, 30_000


def partial(t, x, y, z):
    """Rank correlation of columns x and y of t after a linear fit of each rank on the ranks of columns z."""
    r = t[[x, y, *z]].rank()
    Z = np.column_stack([np.ones(len(r))] + [r[c] for c in z])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(res(r[x].values), res(r[y].values))[0, 1])


def main():
    log = runlog.start("scan_depth", __doc__)
    df = pd.read_parquet(data.path(log.input, "sample.parquet"), columns=["Entry", "Length", "family", "family_size", "plddt"])
    maps = pd.read_parquet(f"{log.read('scan_pairs')}/maps.parquet").join(df, on="protein")
    maps["stratum"] = np.select([maps.family_size <= SMALL, maps.family_size >= LARGE], ["small", "large"], "between")
    rows = []
    for b, t in maps.groupby("block"):
        rows.append(dict(block=b, proteins=len(t), **t.groupby("stratum").precision.mean().to_dict(),
                         size=partial(t, "precision", "family_size", []), size_given=partial(t, "precision", "family_size", ["Length", "plddt"]),
                         plddt_given=partial(t, "precision", "plddt", ["Length", "family_size"]), length=partial(t, "precision", "Length", [])))
    out = pd.DataFrame(rows)
    out.to_parquet(f"{log.dir}/depth.parquet")
    maps.to_parquet(f"{log.dir}/proteins.parquet")
    log.note(strata=maps[maps.block < 0].stratum.value_counts().to_dict(), **{("logits" if b < 0 else f"block{b}"): r for b, r in out.set_index("block").drop(columns="proteins").to_dict("index").items() if b < 0 or b % 8 == 7})
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
