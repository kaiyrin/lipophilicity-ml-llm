"""Loading RepoRT files, assert method facts, clean, build long and wide tables."""
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

META = {"column.name": "column_name", "column.usp.code": "usp", "column.length": "length_mm",
        "column.id": "id_mm", "column.particle.size": "dp_um", "column.temperature": "temp_c",
        "column.flowrate": "flow", "column.t0": "t0", "eluent.A.pH": "pH"}
BENZOIC = Chem.MolFromSmarts("c1ccccc1C(=O)[OX2H1]")
CLASSYFIRE = ["classyfire.superclass", "classyfire.class", "classyfire.subclass"]
RT_KEEP = ["id", "name", "smiles", "inchikey", "rt", "comment"] + CLASSYFIRE
# ---------- keepinging on;y needed ----------
def read_rt(raw, m):
    df = pd.read_csv(raw / m / f"{m}_rtdata_canonical_success.tsv", sep="\t")
    rt_col = next(c for c in df.columns if c.lower().startswith("rt"))
    df = df.rename(columns={rt_col: "rt", "smiles.std": "smiles", "inchikey.std": "inchikey"})
    return df[[c for c in RT_KEEP if c in df.columns]]

def read_meta(raw, m):
    row = pd.read_csv(raw / m / f"{m}_metadata.tsv", sep="\t").iloc[0]
    missing = [k for k in META if k not in row.index]
    if missing:
        print(f"[{m}] metadata columns not found: {missing}  (available: {list(row.index)[:15]}...)")
    return {v: row.get(k, np.nan) for k, v in META.items()}

def read_gradient(raw, m):
    g = pd.read_csv(raw / m / f"{m}_gradient.tsv", sep="\t")
    g.columns = ["t", "flow", "A", "B"] + list(g.columns[4:])
    g["phi"] = g["B"] / 100.0
    return g[["t", "flow", "phi"]]

# ---------- fail loudly if a method is not what we think ----------
def assert_method(m, meta, grad, mcfg, exp):
    def close(a, b, tol, what):
        if pd.notna(a) and abs(float(a) - b) > tol:
            raise ValueError(f"[{m}] {what}: file says {a}, config says {b}")
    close(meta["t0"], mcfg["t0"], 1e-3, "t0")
    close(meta["flow"], exp["flow"], 1e-3, "flow")
    close(meta["temp_c"], exp["temp"], 0.5, "temperature")
    if pd.isna(meta["pH"]):
        print(f"[{m}] pH missing in metadata, using config pH {mcfg['pH']}")
    else:
        close(meta["pH"], mcfg["pH"], 0.05, "pH")
    phi_at = lambda t: np.interp(t, grad.t, grad.phi)
    for t, phi in [(0, exp["phi0"]), (exp["hold_end"], exp["phi0"]), (exp["ramp_end"], exp["phi1"])]:
        close(phi_at(t), phi, 1e-3, f"gradient phi at {t} min")
    close(grad.t.max(), exp["run_time"], 1e-3, "run time")

# ---------- cleaning ----------
def clean_rt(df, m, run_time):
    n0 = len(df)
    df = df.dropna(subset=["rt", "smiles", "inchikey"])
    df = df[np.isfinite(df.rt) & (df.rt > 0) & (df.rt <= run_time)]
    ok = df.smiles.map(lambda s: Chem.MolFromSmiles(s) is not None)
    df = df[ok]
    print(f"[{m}] rows {n0} → {len(df)} after removing missing, out-of-range RT, unparsable SMILES")
    return df

def collapse_duplicates(df):
    g = df.groupby("inchikey")
    out = g.agg(smiles=("smiles", "first"), name=("name", "first"),
                rt=("rt", "median"), n_dup=("rt", "size"),
                rt_spread=("rt", lambda x: x.max() - x.min()))
    for c in CLASSYFIRE:
        if c in df:
            out[c] = g[c].first()
    if "comment" in df:
        out["comment"] = g["comment"].agg(lambda x: ";".join(sorted(set(x.dropna().astype(str)))))
    return out.reset_index()

# ---------- tables ----------
def build_long(cfg):
    raw, exp, dcfg = cfg["paths"]["raw"], cfg["expected"], cfg["data"]
    parts = []
    for m, mcfg in cfg["methods"].items():
        meta, grad = read_meta(raw, m), read_gradient(raw, m)
        assert_method(m, meta, grad, mcfg, exp)
        df = collapse_duplicates(clean_rt(read_rt(raw, m), m, exp["run_time"]))
        df = df.assign(method=m, column=mcfg["column"], buffer=mcfg["buffer"],
                       pH=mcfg["pH"], t0=mcfg["t0"],
                       length_mm=meta["length_mm"], dp_um=meta["dp_um"])
        df["censored"] = df.rt <= dcfg["c_void"] * df.t0
        df["censored_alt"] = df.rt <= dcfg["c_void_alt"] * df.t0
        parts.append(df)
    long = pd.concat(parts, ignore_index=True)
    long["is_benzoic_acid"] = long.smiles.map(
        lambda s: Chem.MolFromSmiles(s).HasSubstructMatch(BENZOIC))
    assert not long.duplicated(["inchikey", "method"]).any(), "duplicate (molecule, method)"
    return long.rename(columns={"inchikey": "mol_id"})

def stereo_report(long):
    k14 = long.mol_id.str[:14]
    multi = long.groupby([long.method, k14]).mol_id.nunique()
    return multi[multi > 1].reset_index(name="n_stereo_variants")

def build_wide(long, big_shift):
    rt = long.pivot(index="mol_id", columns="method", values="rt").add_prefix("rt_")
    cen = long.pivot(index="mol_id", columns="method", values="censored").add_prefix("cens_")
    info_cols = ["name", "smiles", "is_benzoic_acid"] + [c for c in CLASSYFIRE if c in long]
    info = long.groupby("mol_id")[info_cols].first()
    wide = info.join(rt).join(cen)
    wide["dRT_pH"] = wide.rt_0435 - wide.rt_0434
    wide["dRT_col"] = wide.rt_0433 - wide.rt_0434
    tables = {"wide_ph": wide.dropna(subset=["rt_0434", "rt_0435"]),
              "wide_col": wide.dropna(subset=["rt_0434", "rt_0433"]),
              "wide_all3": wide.dropna(subset=["rt_0434", "rt_0435", "rt_0433"])}
    shifts = wide[(wide.dRT_pH.abs() > big_shift) | (wide.dRT_col.abs() > big_shift)]
    return wide, tables, shifts

def overlap_table(long):
    sets = {m: set(g.mol_id) for m, g in long.groupby("method")}
    sets_nc = {m: set(g.mol_id[~g.censored]) for m, g in long.groupby("method")}
    acids = set(long.mol_id[long.is_benzoic_acid])
    combos = {"0434": ["0434"], "0435": ["0435"], "0433": ["0433"],
              "pH pair": ["0434", "0435"], "column pair": ["0434", "0433"],
              "all three": ["0434", "0435", "0433"]}
    rows = []
    for label, ms in combos.items():
        s = set.intersection(*(sets[m] for m in ms))
        s_nc = set.intersection(*(sets_nc[m] for m in ms))
        rows.append({"set": label, "n": len(s), "n_non_censored": len(s_nc),
                     "n_benzoic": len(s & acids)})
    return pd.DataFrame(rows)