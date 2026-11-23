"""Generate molecular features from SMILES strings. With 1. Descriptors (d_) and 2. Morgan fingerprints (fp_). - 1024 col"""
"""Descriptors (d_*) and Morgan fingerprint bits (fp_*) from SMILES."""
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

"""Fututre fix:
Acyclic molecules all get scaffold "", so they form one big family. 
Also, groups are visited in random order, so a big group can land 
late and unbalance the folds."""

def descriptors(smiles, abs_max=1e6): 
    """For each SMILES build RDKit molecule and compute all its descriptors. 
    Return a DataFrame with one row per SMILES and one column per descriptor. 
    Drop columns with ≤1 distinct value."""
    rows = [Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s)) for s in smiles]
    X = pd.DataFrame(rows).add_prefix("d_").astype(float)
    X = X.mask(X.abs() > abs_max)                        # ind. exploding values ((inf, huge numbers)) → NaN
    return X


def morgan(smiles, radius=2, bits=1024):
    """Calls Morgan fingerprint generator for each SMILES and returns a DataFrame with one row per SMILES and one column per bit.
    The array are stacked into matrix fp_0,fp_1...fp_n, where n = bits-1."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    A = np.stack([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles]).astype(np.uint8)
    return pd.DataFrame(A, columns=[f"fp_{i}" for i in range(bits)])
