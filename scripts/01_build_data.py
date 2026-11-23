import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # retention_project/
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.config import load_config
from src.data import load_raw, standardise, merge_duplicates, add_scaffold
from src.features import descriptors, morgan

cfg = load_config()
d, f = cfg["data"], cfg["features"]
out = cfg["paths"]["processed"]
out.mkdir(parents=True, exist_ok=True)

raw = load_raw(d)
std = standardise(raw)
mols = add_scaffold(merge_duplicates(std)) #one row per mol + scaffold
mols = mols.reset_index(drop=True)  # mol_id is now the index, but then its needed as a column for later merge with features 


D = descriptors(mols.smiles, f["desc_abs_max"])
#Future: scale for Ridge, SVR, kNN, MLP; noscale for  rf and gb
F = morgan(mols.smiles, f["morgan_radius"], f["morgan_bits"])
D = D.loc[:, D.nunique() > 1]   # NEW: drop descriptor columns with ≤1 distinct value
D = D.drop(columns=["d_Ipc"], errors="ignore") #drop Ipc descriptor, which is NaN for 56% of molecules and has a huge range (0.1 to 1e6) that dominates scaling
F = F.loc[:, F.nunique() > 1]   # NEW: drop fingerprint bits that are all 0 (or all 1)
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
print(f"NaN in fingerprints: {int(F.isna().sum().sum())} values in {int((F.isna().sum() > 0).sum())} columns")

print("Descriptor and fingerptint features dropping summary:")
na = D.isna().mean()
print(na[na > 0.05].sort_values(ascending=False))
na = F.isna().mean()
print(na[na > 0.05].sort_values(ascending=False))
print(f"NaN in descriptors: {int(D.isna().sum().sum())} values in {int((D.isna().sum() > 0).sum())} columns")
print(f"NaN in fingerprints: {int(F.isna().sum().sum())} values in {int((F.isna().sum() > 0).sum())} columns")