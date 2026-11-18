"""Explore which molecule groups are well represented and how they behave pH 3 → pH 8.
Run after 01_build_data.py (and again after chem.py adds ion_class)."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qsrr.config import load_config

cfg = load_config()
proc, rep = cfg["paths"]["processed"], cfg["paths"]["reports"] / "tables"
long = pd.read_parquet(proc / "long.parquet")
w = pd.read_parquet(proc / "wide_ph.parquet")

# add ion_class if chem.py has run, otherwise continue without it
extra = [c for c in ["ion_class"] if c in long.columns and c not in w.columns]
if extra:
    w = w.join(long.groupby("mol_id")[extra].first())

lab = {False: "ret", True: "void"}
w["transition"] = w.cens_0434.map(lab) + "→" + w.cens_0435.map(lab)

def table(col, top=15):
    t = pd.crosstab(w[col].fillna("unclassified"), w.transition)
    t["total"] = t.sum(axis=1)
    t = t.sort_values("total", ascending=False)
    t.to_csv(rep / f"groups_by_{col.split('.')[-1]}.csv")
    print(f"\n=== {col} (top {top}) ===\n", t.head(top).to_string())

print("Overall transitions pH 3 → pH 8:\n", w.transition.value_counts().to_string())
for col in ["classyfire.superclass", "classyfire.class", "ion_class"]:
    if col in w.columns:
        table(col)
    else:
        print(f"\n(skip {col}: not available yet)")