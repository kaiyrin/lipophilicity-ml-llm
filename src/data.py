"""Load raw Lipophilicity, standardise structures, merge duplicates, add scaffolds."""
"""One table row - molecule with  mol_id | smiles | logd | scaffold | ..."""
"""Second table row features - mol_id | d_MolLogP | d_TPSA | ... | fp_0 | fp_1 | ... """


import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

_largest = rdMolStandardize.LargestFragmentChooser()

"""Parse SMILES: so RDKit reads text and build whole molecule object to later use as a row"""
def load_raw(dcfg):
    """Load the raw CSV file and rename columns to standard names."""
    df = pd.read_csv(dcfg["raw_file"])
    df = df.rename(columns={dcfg["smiles_col"]: "smiles_raw", dcfg["target_col"]: "logd"})
    return df[["smiles_raw", "logd"]]

"""Canonical SMILES: largest fragment rewriting"""
"""The flat key (no stereo) is used to merge in stereo in SMILES (@ or /), while the canonical SMILES (with stereo) is kept for later use."""
def standardise(df):
    """Parse → largest fragment → canonical SMILES (with stereo) + flat key (without stereo)."""
    mols = df.smiles_raw.map(Chem.MolFromSmiles)
    out = df[mols.notna()].copy()                                   # unparseable SMILES dropped here
    parents = mols[mols.notna()].map(lambda m: _largest.choose(rdMolStandardize.Cleanup(m)))
    out["smiles"] = parents.map(Chem.MolToSmiles)
    out["flat_key"] = parents.map(lambda m: Chem.MolToSmiles(m, isomericSmiles=False))
    return out

"""Same flat key rows become one row with avg logD, counts and spread.
   n_dup is the number of duplicates (stereo or not)."""
def merge_duplicates(df):
    """One row per structure. Mean logD; keep counts and spread so noise can be reported."""
    g = df.groupby("flat_key")
    out = g.agg(smiles=("smiles", "first"), logd=("logd", "mean"),
                n_dup=("logd", "size"), logd_spread=("logd", lambda v: v.max() - v.min()))
    out = out.reset_index().rename(columns={"flat_key": "mol_id"})
    return out


def add_scaffold(df):
    """Scafford: strip each molecule to its core ring, to identify core family. 
   It's the group label for the scaffold split, so whole families go to either train or test."""
    df = df.copy()
    df["scaffold"] = df.smiles.map(lambda s: MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))) #molecule parse with Chem.MolFromSmiles(s) then passes as mol
    return df