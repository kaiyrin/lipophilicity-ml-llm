import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from config import load_config
from data import load_raw, standardise, merge_duplicates, add_scaffold
from features import descriptors, morgan

cfg = load_config()
d, f = cfg["data"], cfg["features"]
out = cfg["paths"]["processed"]
out.mkdir(parents=True, exist_ok=True)

raw = load_raw(d)
std = standardise(raw)
mols = add_scaffold(merge_duplicates(std))

D = descriptors(mols.smiles, f["desc_abs_max"])
F = morgan(mols.smiles, f["morgan_radius"], f["morgan_bits"])
features = pd.concat([mols[["mol_id"]], D, F], axis=1)

mols.to_parquet(out / "molecules.parquet")
features.to_parquet(out / "features.parquet")
dups = mols[mols.n_dup > 1]
sc = mols.scaffold.value_counts()
print(f"Rows: raw {len(raw)} → valid {len(std)} → unique molecules {len(mols)}   invalid SMILES: {len(raw) - len(std)}")
print(f"Duplicates merged: {len(dups)} structures")
if len(dups):
    print(f"   logD spread within duplicates: median {dups.logd_spread.median():.2f}, max {dups.logd_spread.max():.2f}")
print(f"logD: min {mols.logd.min():.2f}  mean {mols.logd.mean():.2f}  SD {mols.logd.std():.2f}  max {mols.logd.max():.2f}")
print(f"Scaffolds: {len(sc)}   singletons {(sc == 1).sum()}   acyclic {sc.get('', 0)}   largest family {sc.iloc[0]}")
print(f"Features: {D.shape[1]} descriptors, {F.shape[1]} fingerprint bits")
print(f"NaN in descriptors: {int(D.isna().sum().sum())} values in {int((D.isna().sum() > 0).sum())} columns")