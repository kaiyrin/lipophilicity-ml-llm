"""Download the RepoRT files for the configured methods into data/raw/."""
import argparse, csv, datetime, hashlib, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qsrr.config import load_config

PER_METHOD = ["{m}_rtdata_canonical_success.tsv", "{m}_metadata.tsv", "{m}_gradient.tsv"]
SHARED = {"processed_data/studies.tsv": "studies.tsv",
          "resources/hsm_database/hsm_database.tsv": "hsm_database.tsv"}

def fetch(url, dest, force):
    if dest.exists() and not force:
        print(f"skip  {dest} (exists)"); return dest.read_bytes()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    if data.lstrip()[:1] == b"<":                       # got an HTML error page, not a TSV
        raise RuntimeError(f"not a TSV: {url}")
    dest.write_bytes(data)
    print(f"ok    {dest} ({len(data):,} bytes)")
    return data

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    base = f"https://raw.githubusercontent.com/{cfg['download']['repo']}/{cfg['download']['ref']}"
    raw = cfg["paths"]["raw"]
    jobs = [(f"{base}/processed_data/{m}/{f.format(m=m)}", raw / m / f.format(m=m))
            for m in cfg["methods"] for f in PER_METHOD]
    jobs += [(f"{base}/{src}", raw / dst) for src, dst in SHARED.items()]

    rows = []
    for url, dest in jobs:
        data = fetch(url, dest, args.force)
        rows.append({"file": str(dest.relative_to(raw)), "url": url, "bytes": len(data),
                     "sha256": hashlib.sha256(data).hexdigest(),
                     "downloaded": datetime.date.today().isoformat()})
    with open(raw / "MANIFEST.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} files, manifest at {raw / 'MANIFEST.csv'}")

if __name__ == "__main__":
    main()
    