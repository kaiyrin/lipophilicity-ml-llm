"""Download the raw Lipophilicity CSV file."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # retention_project/
sys.path.insert(0, str(ROOT))
import hashlib, urllib.request
from src.config import load_config

cfg = load_config()["data"]
dest = cfg["raw_file"]
dest.parent.mkdir(parents=True, exist_ok=True)

if dest.exists():
    print(f"Already downloaded: {dest}")
else:
    print(f"Downloading {cfg['url']}")
    urllib.request.urlretrieve(cfg["url"], dest)

md5 = hashlib.md5(dest.read_bytes()).hexdigest()
status = "OK" if md5 == cfg["md5"] else "MISMATCH (file differs from the published checksum; check before use)"
print(f"Size: {dest.stat().st_size / 1e3:.0f} kB   md5: {md5}   {status}")