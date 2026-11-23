"""Which features does HGB_both rely on? Permutation importance on held-out scaffold folds."""
import sys, re, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.inspection import permutation_importance
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from src.config import load_config
from src.scaffold import scaffold_folds
from src.models import build_models

from dotenv import load_dotenv
from google import genai
import json
load_dotenv()

cfg = load_config()
out = cfg["paths"]["processed"]
reports = cfg["paths"]["reports"]    
mols = pd.read_parquet(out / "molecules.parquet")
feats = pd.read_parquet(out / "features.parquet")
X, y = feats.drop(columns="mol_id"), mols.logd.to_numpy()
SEED, N_REPEATS = 0, 5                                                #SDs over seeds were tiny


def perm_importance(cols):
    """Train HGB_both on `cols` per scaffold fold, shuffle each column on TEST molecules."""
    imps = []
    for tr, te in scaffold_folds(mols.scaffold.to_numpy(), cfg["cv"]["n_splits"], SEED):
        model, _ = build_models(cfg["models"], SEED)["HGB_both"]            # fresh model per fold
        model.fit(X[cols].iloc[tr], y[tr])
        r = permutation_importance(model, X[cols].iloc[te], y[te], scoring="neg_root_mean_squared_error",
                                   n_repeats=N_REPEATS, random_state=SEED, n_jobs=-1)
        imps.append(pd.Series(r.importances_mean, index=cols))
    imp = pd.concat(imps, axis=1)                                         # columns x folds
    return pd.DataFrame({"RMSE_increase": imp.mean(1), "fold_SD": imp.std(1)}
                        ).sort_values("RMSE_increase", ascending=False)


# all features: which ones matter most to HGB_both?
_, prefixes = build_models(cfg["models"], SEED)["HGB_both"]
cols = [c for c in X.columns if c.startswith(prefixes)]
summary = perm_importance(cols)
print(summary.head(15).round(4))
summary.to_csv(reports / "permutation_importance.csv")

# correlated descriptors grouped: is a descriptor unimportant, or does HGB just use its near-copy?
D = X.filter(regex="^d_")
corr = D.corr(method="spearman").fillna(0).to_numpy()     # pandas handles NaN pairwise
link = hierarchy.ward(squareform(1 - abs(corr), checks=False))
cluster = pd.Series(hierarchy.fcluster(link, t=0.5, criterion="distance"), index=D.columns)

# representative = the most important member of each cluster (from run 1)
rep_of = cluster.groupby(cluster).transform(lambda s: summary.loc[s.index, "RMSE_increase"].idxmax())
keep = rep_of.unique().tolist()
fp = [c for c in cols if c.startswith("fp_")]

summary_grouped = perm_importance(keep + fp)
print(summary_grouped.head(15).round(4))
summary_grouped.to_csv(reports / "permutation_importance_grouped.csv")
pd.DataFrame({"cluster": cluster, "representative": rep_of}).to_csv(reports / "descriptor_clusters.csv")


# ---- hypotheses (ML only): statement, rule fixed in advance, numbers, verdict ----
imp_all, sd_all = summary.RMSE_increase, summary.fold_SD
pos = imp_all.clip(lower=0)
share = lambda k: pos.nlargest(k).sum() / pos.sum()
top2, top10 = imp_all.index[:2].tolist(), imp_all.index[:10]
n_desc_top10 = int(top10.str.startswith("d_").sum())
h4 = []
for c in top2:
    c_rep = rep_of.get(c, c)                                   # fingerprint bits have no cluster
    n_copies = int((cluster == cluster[c]).sum() - 1) if c in cluster.index else 0
    h4.append((c, n_copies, imp_all[c], summary_grouped.RMSE_increase[c_rep]))

hyp = [
    ("H1", "The model relies on very few columns", "top-2 columns hold at least 50% of the total importance",
     f"top-1 {share(1):.0%}, top-2 {share(2):.0%}, top-10 {share(10):.0%}; top column {top2[0]} "
     f"{imp_all.iloc[0]:.3f} ± {sd_all.iloc[0]:.3f} over folds", share(2) >= 0.5),
    ("H2", "Most columns are not used", "at least 50% of columns have importance <= 0",
     f"{(imp_all <= 0).mean():.0%} of {len(imp_all)} columns are <= 0; {(imp_all > 0.001).sum()} are above 0.001",
     (imp_all <= 0).mean() >= 0.5),
    ("H3", "The most important columns are mainly descriptors, not fingerprint bits",
     "at least 6 of the top-10 columns are descriptors",
     f"{n_desc_top10} of the top-10 are descriptors, {10 - n_desc_top10} are fingerprint bits", n_desc_top10 >= 6),
    ("H4", "Near-copies were hiding some importance of the top columns",
     "a top column has near-copies and its importance rises when they are collapsed",
     "; ".join(f"{c}: {n} near-copies, {a:.3f} alone -> {g:.3f} collapsed" for c, n, a, g in h4),
     any(n > 0 and g > a for _, n, a, g in h4)),
    ("H5", "The top columns stay on top after grouping", "the top-2 columns (or their representatives) are still the top-2",
     f"before {top2}, after {summary_grouped.index[:2].tolist()}",
     {rep_of.get(c, c) for c in top2} <= set(summary_grouped.index[:2])),
]
print()
for hid, text, rule, evidence, ok in hyp:
    print(f"{hid} {text}: {'SUPPORTED' if ok else 'NOT SUPPORTED'}\n   {evidence}")

