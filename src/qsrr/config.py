from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_config(path=ROOT / "config.yaml"):
    cfg = yaml.safe_load(Path(path).read_text())
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = ROOT / v
    return cfg