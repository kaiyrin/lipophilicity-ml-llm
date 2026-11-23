import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path=ROOT / "config.yaml"):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["data"]["raw_file"] = ROOT / cfg["data"]["raw_file"]
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    for p in cfg["paths"].values():
        p.mkdir(parents=True, exist_ok=True)
    return cfg