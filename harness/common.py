"""Shared configuration and small I/O helpers."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cifar10.json"
LOCAL_ASSETS = ROOT / "cache/assets"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def schedule(steps, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    if steps < 2 or not 0 < sigma_min < sigma_max or rho <= 0:
        raise ValueError("Need steps>=2, 0<sigma_min<sigma_max, rho>0")
    a, b = sigma_max ** (1 / rho), sigma_min ** (1 / rho)
    return [(a + i / (steps - 1) * (b - a)) ** rho for i in range(steps)] + [0.0]


def default_asset_dir():
    """Return the configured shared data root used by the research server."""
    configured = os.environ.get("SAMPLER_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    shared_root = read_json(CONFIG_PATH).get("shared_root")
    return Path(shared_root).expanduser() if shared_root else LOCAL_ASSETS


def _asset_candidates(directory, key, spec, config):
    """Yield candidates in explicit-root, configured-root, then shared order."""
    override = os.environ.get(f"SAMPLER_{key.upper()}_PATH")
    if override:
        yield Path(override).expanduser()
        return

    roots = []
    for value in (directory, os.environ.get("SAMPLER_ASSETS"), default_asset_dir()):
        if value is None:
            continue
        root = Path(value).expanduser()
        if root not in roots:
            roots.append(root)

    relative = spec.get("relative_path")
    for root in roots:
        yield root / spec["filename"]
        if relative:
            yield root / relative

    shared_root = os.environ.get("SAMPLER_DATA_ROOT", config.get("shared_root"))
    if shared_root:
        root = Path(shared_root).expanduser()
        if relative:
            yield root / relative
        yield root / spec["filename"]


def asset_paths(directory, keys):
    """Resolve assets from local overrides or the configured shared data tree.

    ``directory`` is an explicit asset-root override. The configured shared tree
    uses per-asset subdirectories from ``relative_path``.
    """
    config = read_json(CONFIG_PATH)
    specs = config["assets"]
    paths = {}
    for key in keys:
        if key not in specs:
            raise KeyError(f"Unknown asset key: {key}")
        spec = specs[key]
        candidates = list(dict.fromkeys(_asset_candidates(directory, key, spec, config)))
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            searched = ", ".join(str(candidate) for candidate in candidates)
            raise FileNotFoundError(
                f"Missing {key} ({spec['filename']}). Searched: {searched}. "
                "Run python -m harness.prepare or set SAMPLER_*_PATH."
            )
        paths[key] = path
    return paths


def git_info():
    def git(*args):
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain")}
