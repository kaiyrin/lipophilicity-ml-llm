import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qsrr.config import load_config
from qsrr.data import build_long, build_wide, overlap_table, stereo_report

cfg = load_config()
out, rep = cfg["paths"]["processed"], cfg["paths"]["reports"] / "tables"
out.mkdir(parents=True, exist_ok=True); rep.mkdir(parents=True, exist_ok=True)

long = build_long(cfg)
wide, tables, shifts = build_wide(long, cfg["data"]["big_shift_min"])
long.to_parquet(out / "long.parquet")
wide.to_parquet(out / "wide.parquet")
for name, t in tables.items():
    t.to_parquet(out / f"{name}.parquet")

ov = overlap_table(long); ov.to_csv(rep / "overlap.csv", index=False)
st = stereo_report(long); st.to_csv(rep / "stereo_variants.csv", index=False)
shifts.to_csv(rep / "big_shifts_to_inspect.csv")
cens = long.groupby("method")[["censored", "censored_alt"]].mean().round(3)

print("\nOverlap:\n", ov.to_string(index=False))
print("\nCensored fraction per method (c=1.5, c=2):\n", cens)
print(f"\nStereo-only variants: {len(st)}   Big shifts to inspect: {len(shifts)}")