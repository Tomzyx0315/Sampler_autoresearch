"""Shared configuration and small I/O helpers."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cifar10.json"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def schedule(steps, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    if steps < 2 or not 0 < sigma_min < sigma_max or rho <= 0:
        raise ValueError("Need steps>=2, 0<sigma_min<sigma_max, rho>0")
    a, b = sigma_max ** (1 / rho), sigma_min ** (1 / rho)
    return [(a + i / (steps - 1) * (b - a)) ** rho for i in range(steps)] + [0.0]


def asset_paths(directory, keys):
    specs = read_json(CONFIG_PATH)["assets"]
    paths = {key: Path(directory) / specs[key]["filename"] for key in keys}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run python -m harness.prepare")
    return paths


def git_info():
    def git(*args):
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain")}
