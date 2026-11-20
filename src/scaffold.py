"""Cross-validation fold scaffold (whole chemical families per fold)."""
import numpy as np, pandas as pd


def scaffold_folds(scaffolds, n_splits, seed): 
    """Each scaffold group goes entirely to one fold. Groups are visited in random order (seed)
    and each is put in the currently smallest fold, so folds stay balanced in size."""
    s = pd.Series(np.asarray(scaffolds))
    sizes = s.value_counts()
    order = sizes.index.to_numpy()[np.random.default_rng(seed).permutation(len(sizes))]
    fold_n, fold_of = np.zeros(n_splits, int), {}
    for g in order:
        k = int(np.argmin(fold_n))
        fold_of[g] = k
        fold_n[k] += sizes[g]
    f = s.map(fold_of).to_numpy()
    for k in range(n_splits):
        yield np.where(f != k)[0], np.where(f == k)[0]
        