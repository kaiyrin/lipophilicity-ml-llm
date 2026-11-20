"""Generate molecular features from SMILES strings. With 1. Descriptors (d_) and 2. Morgan fingerprints (fp_). - 1024 col"""
"""Descriptors (d_*) and Morgan fingerprint bits (fp_*) from SMILES."""
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")


def descriptors(smiles, abs_max=1e6):
    rows = [Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s)) for s in smiles]
    X = pd.DataFrame(rows).add_prefix("d_").astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.mask(X.abs() > abs_max)                        # exploding values → NaN
    return X


def morgan(smiles, radius=2, bits=1024):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    A = np.stack([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles]).astype(np.uint8)
    return pd.DataFrame(A, columns=[f"fp_{i}" for i in range(bits)])