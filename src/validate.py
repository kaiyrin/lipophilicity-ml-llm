"""Metrics and the paired comparison of two models (bootstrap over scaffold groups)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr


def metrics(y, p):
    """RMSE: sqrt(mean((y - p)^2)), MAE: mean(|y - p|), 
       R2: 1 - sum((y - p)^2)/sum((y - mean(y))^2), Spearman: rank correlation"""
    y = np.asarray(y)
    e = np.asarray(p) - y
    return {"RMSE": float(np.sqrt((e ** 2).mean())), "MAE": float(np.abs(e).mean()),
            "R2": float(1 - (e ** 2).sum() / ((y - np.mean(y)) ** 2).sum()),
            "Spearman": float(spearmanr(y, p).statistic)}



def rmse_diff_ci(err_a, err_b, clusters, n_boot=2000, seed=0): #err_a/err_b = pred_(a/b) - y (out-of-fold), clusters = scaffold groups
    """RMSE(a) − RMSE(b) with a 95% CI. Resamples whole clusters (scaffolds), because
    molecules in one family are not independent."""
    df = pd.DataFrame({"a": np.asarray(err_a) ** 2, "b": np.asarray(err_b) ** 2, "c": np.asarray(clusters)}) #square the errors to get squared errors for RMSE
    per = df.groupby("c").agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size")) # a, b - sum of squared errors per family, n - number of molecules per family
    # bootstrap resample the families, compute RMSE(a) - RMSE(b) for each resample, and get the 2.5% and 97.5% percentiles for the CI
    idx = np.random.default_rng(seed).integers(0, len(per), (n_boot, len(per)))
    A, B, N = per.a.to_numpy(), per.b.to_numpy(), per.n.to_numpy()
    d = np.sqrt(A[idx].sum(1) / N[idx].sum(1)) - np.sqrt(B[idx].sum(1) / N[idx].sum(1)) #for each round of bootstrap, compute RMSE(a) - RMSE(b)
    point = np.sqrt(A.sum() / N.sum()) - np.sqrt(B.sum() / N.sum()) #point estimate of RMSE(a) - RMSE(b) using all data
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5) #middle 95% of the bootstrap distribution
    verdict = "A better" if hi < 0 else "B better" if lo > 0 else "no clear difference"
    return round(float(point), 3), round(float(lo), 3), round(float(hi), 3), verdict