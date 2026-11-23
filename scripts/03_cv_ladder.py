"""Run the ladder on random and scaffold folds, fill the OOF table, then score it."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from src.config import load_config
from src.scaffold import scaffold_folds
from src.models import build_models, choose_alpha
from src.validate import metrics, rmse_diff_ci

cfg = load_config()
mcfg = cfg["models"]
SEEDS, N_SPLITS = cfg["cv"]["seeds"], cfg["cv"]["n_splits"]
N_BOOT, BOOT_SEED = cfg["bootstrap"]["n_boot"], cfg["bootstrap"]["seed"]
ALPHAS = np.logspace(*mcfg["ridge_alphas"])
INCLUDE_MLP = True                         # False for a quick first run
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]     # fixed in advance, NOT picked after seeing test errors
PAIRS = [("HGB_both", "HGB_desc"),         # does the fingerprint add anything?
         ("HGB_fp", "HGB_desc"),           # fingerprint vs descriptors
         ("HGB_desc", "B1_ridge_desc"),    # trees vs linear (same features)
         ("HGB_fp", "B2_knn_fp"),          # learned rules vs copy neighbours (same features)
         ("MLP_both", "HGB_both")]         # neural net vs trees (same features)

def nn_jaccard(train_fp, test_fp):
    """For each test molecule: Jaccard similarity to its most similar training molecule."""
    A, B = train_fp.astype(np.float32), test_fp.astype(np.float32)   # bool @ bool would stay bool
    inter = B @ A.T                                                  # shared bits, test x train
    union = B.sum(1)[:, None] + A.sum(1)[None, :] - inter            # bits set in either molecule
    return (inter / np.maximum(union, 1)).max(axis=1)                # max(whatever,1) avoids 0/0

def rmse(e):
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")


# Loading the processed molecules and features, and preparing the data for modeling
out = cfg["paths"]["processed"]
mols = pd.read_parquet(out / "molecules.parquet")
feats = pd.read_parquet(out / "features.parquet")
assert (mols.mol_id.values == feats.mol_id.values).all(), "molecules and features are not aligned"
X = feats.drop(columns="mol_id")                       # all features, stays a DataFrame
y = mols.logd.to_numpy()
scaffolds = mols.scaffold.to_numpy()
fp = X.filter(regex="^fp_").to_numpy(dtype=bool)       # only for the similarity flag

# -Check on KFold and scaffold folds: the first test fold should be different for each seed, otherwise the repeats are useless
first = {}
for seed in SEEDS:
    folds = list(scaffold_folds(scaffolds, N_SPLITS, seed))
    first[seed] = np.sort(folds[0][1])
    print(f"seed {seed}: test sizes {[len(te) for _, te in folds]}, first test ids {first[seed][:5]}")
if len(SEEDS) > 1 and all(np.array_equal(first[SEEDS[0]], first[s]) for s in SEEDS[1:]):
    print("WARNING: scaffold folds identical for every seed -> repeats only change the MLP")

# Split => seed => fold => model => train on tr, predict on te, record y, pred, max_sim
rows, alphas_used = [], []
for split in ("random", "scaffold"):
    for seed in SEEDS:
        folds = (scaffold_folds(scaffolds, N_SPLITS, seed) if split == "scaffold"
                 else KFold(N_SPLITS, shuffle=True, random_state=seed).split(X))
        for tr, te in folds:
            sim = nn_jaccard(fp[tr], fp[te])   # jaccard similarity to the training set, for each test molecule
            for name, (model, prefixes) in build_models(mcfg, seed).items():           # fresh models each fold
                if name == "MLP_both" and not INCLUDE_MLP:
                    continue
                Xm = X[[c for c in X.columns if c.startswith(prefixes)]]               # must stay a DataFrame
                if name == "B1_ridge_desc":                    # alpha by scaffold CV inside the training fold
                    a = choose_alpha(Xm.iloc[tr], y[tr], scaffolds[tr], ALPHAS, n_splits=N_SPLITS, seed=seed)
                    model.set_params(ridge__alpha=a)
                    alphas_used.append({"split": split, "alpha": a})
                model.fit(Xm.iloc[tr], y[tr])                  # train on this fold's training molecules only
                rows.append(pd.DataFrame({"split": split, "seed": seed, "model": name, "mol_idx": te,
                          "y": y[te], "pred": model.predict(Xm.iloc[te]), "nn_sim": sim}))   # was "max_sim"
        print(f"done: {split} seed {seed}")

oof = pd.concat(rows, ignore_index=True)
oof["err"] = oof.pred - oof.y
oof.to_parquet(out / "oof.parquet")
order = [m for m in build_models(mcfg, 0) if INCLUDE_MLP or m != "MLP_both"]

print(f"\nRidge alpha chosen per fold (grid {ALPHAS[0]:.3g} .. {ALPHAS[-1]:.3g}; always at an edge -> widen)")
print(pd.DataFrame(alphas_used).groupby("split").alpha.value_counts())

# RMSE table: mean over seeds, and the random vs scaffold gap. Scaffold split table: mean and SD over seeds (SD = fold-assignment variation, + init for MLP)
per_seed = pd.DataFrame([{"split": s, "model": m, "seed": sd, **metrics(g.y, g.pred)}
                         for (s, m, sd), g in oof.groupby(["split", "model", "seed"])])
rmse_tab = per_seed.pivot_table(index="model", columns="split", values="RMSE").loc[order]
rmse_tab["gap"] = rmse_tab["scaffold"] - rmse_tab["random"]
print("\nRMSE (mean over seeds); gap = scaffold - random")
print(rmse_tab.round(3))

sc_tab = per_seed[per_seed.split == "scaffold"].groupby("model").agg(
    RMSE=("RMSE", "mean"), RMSE_SD=("RMSE", "std"), MAE=("MAE", "mean"),
    R2=("R2", "mean"), Spearman=("Spearman", "mean")).loc[order]
print("\nScaffold split, mean and SD over seeds (SD = fold-assignment variation, + init for MLP)")
print(sc_tab.round(3))

# Paired bootstrap on scaffold OOF (resamples whole families) 
sc = oof[oof.split == "scaffold"]
err = sc.pivot(index=["seed", "mol_idx"], columns="model", values="err")
clusters = scaffolds[err.index.get_level_values("mol_idx")]
print("\nRMSE(A) - RMSE(B), 95% CI.  'A better' = A has lower error")
for a, b in PAIRS:
    if a in err.columns and b in err.columns:
        print(f"{a:10s} vs {b:14s}", rmse_diff_ci(err[a], err[b], clusters, N_BOOT, BOOT_SEED))

# Nearest-neighbour Jaccard similarity to the training set, for each test molecule. The median and 10th percentile are reported for each split type.
for s in ("random", "scaffold"):
    ns = oof[(oof.split == s) & (oof.model == "HGB_both")].nn_sim
    print(f"\nnearest-neighbour Jaccard similarity, {s} split: median {ns.median():.2f}, 10th pct {ns.quantile(.1):.2f}")

h = sc[sc.model == "HGB_both"]
e, ns = h.err.to_numpy(), h.nn_sim.to_numpy()
print("\nHGB_both, scaffold split: refuse molecules with nn_sim < t")
print(f"{'t':>4} {'coverage':>9} {'RMSE kept':>10} {'RMSE refused':>13}")
for t in THRESHOLDS:
    keep = ns >= t
    print(f"{t:4.1f} {keep.mean():9.1%} {rmse(e[keep]):10.3f} {rmse(e[~keep]):13.3f}")