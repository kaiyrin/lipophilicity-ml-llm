"""Run the whole pipeline in order. A step is skipped if its output already exists.
   python scripts/run_all.py                         # run only what's missing
   python scripts/run_all.py --force 03_cv_ladder    # rerun one step
   python scripts/run_all.py --force all             # rerun everything"""
import argparse, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config

cfg = load_config()
RAW, PROC, REP = cfg["paths"]["raw"], cfg["paths"]["processed"], cfg["paths"]["reports"]

#  comment out a line to switch a step off
STEPS = [
    ("00_download.py",         RAW / "Lipophilicity.csv"),
    ("01_build_data.py",       PROC / "features.parquet"),
    ("03_cv_ladder.py",        PROC / "oof.parquet"),                  
    ("04_analysis_interp.py",  REP / "permutation_importance.csv"),  
    ("05_llm_summary.py",       REP / "xai_facts.json")  
]

ap = argparse.ArgumentParser()
ap.add_argument("--force", nargs="*", default=[], help="steps to rerun, e.g. 03_cv_ladder, or 'all'")
force = ap.parse_args().force

for script, output in STEPS:
    name = script.removesuffix(".py")
    if output.exists() and not ("all" in force or name in force):
        print(f"[skip] {name}: {output.relative_to(ROOT)} exists")
        continue
    print(f"\n[run]  {name}")
    t = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"[fail] {name} (exit {r.returncode}): fix it, then rerun run_all")
    print(f"[done] {name} in {time.time() - t:.0f}s")