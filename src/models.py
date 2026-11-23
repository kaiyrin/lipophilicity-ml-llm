"""The baseline ladder. Each entry: (model, which column prefixes it uses)."""
import sys
from pathlib import Path
# ROOT = Path(__file__).resolve().parents[1]   
# sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import  make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.base import clone

from src.config import load_config
from src.scaffold import scaffold_folds

def choose_alpha(X, y, scaffolds, alphas, n_splits=5, seed=0):
    """Pick Ridge alpha by scaffold CV INSIDE the training fold (test families never seen)."""
    base = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge())
    mean_err = []
    for a in alphas:
        err = []
        for tr, va in scaffold_folds(scaffolds, n_splits, seed):        # same function, smaller data
            m = clone(base).set_params(ridge__alpha=a).fit(X.iloc[tr], y[tr])
            err.append(np.mean((m.predict(X.iloc[va]) - y[va]) ** 2))
        mean_err.append(np.mean(err))
    return alphas[int(np.argmin(mean_err))]

def to_bool(X): #for KNN with Jaccard distance, need boolean features (fingerprints)
    return np.asarray(X, dtype=bool)


def build_models(mcfg, seed):
    hgb = lambda: HistGradientBoostingRegressor(**mcfg["hgb"], random_state=seed)   # same settings for all HGB rows
    bool_transformer = FunctionTransformer(to_bool)    # jaccard needs booleans
    desc_prep = ColumnTransformer([("d", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                                    make_column_selector(pattern="^d_"))], remainder="passthrough")
    return {
        "B0_mean":       (DummyRegressor(strategy="mean"), "d_"), #predict the average logD of the training molecules
        "B1_ridge_desc": (make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge()), ("d_",)),
        "B2_knn_fp":     (make_pipeline(bool_transformer, KNeighborsRegressor(n_neighbors=mcfg["knn_k"], metric="jaccard",
                                                                     weights="distance", algorithm="brute")), ("fp_",)),
        "HGB_desc":      (hgb(), ("d_",)),
        "HGB_fp":        (hgb(), ("fp_",)),
        "HGB_both":      (hgb(), ("d_", "fp_")),
        "MLP_both":      (make_pipeline(desc_prep, MLPRegressor(**mcfg["mlp"], random_state=seed)),
                                       ("d_", "fp_")),
    }

