"""Small, dependency-free helpers for records, assets, and provenance."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from datetime import datetime, timezone
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cifar10.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialise first, so an invalid metric cannot replace the previous record.
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(data)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("Run ID must be 1–80 letters/digits/hyphens/underscores")
    return value


def schedule(steps, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    if steps < 2 or not (0 < sigma_min < sigma_max) or rho <= 0:
        raise ValueError("Need steps>=2, 0<sigma_min<sigma_max, rho>0")
    a, b = sigma_max ** (1 / rho), sigma_min ** (1 / rho)
    return [(a + i / (steps - 1) * (b - a)) ** rho for i in range(steps)] + [0.0]


def git_info():
    def git(*args):
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain")}


def source_files():
    # Deliberately exclude large artifacts, runtime records, and local environments.
    paths = [ROOT / n for n in ("sampler.py", "requirements.txt", "configs/cifar10.json")]
    for folder in ("harness", "baselines", "playground", "third_party"):
        paths.extend((ROOT / folder).rglob("*.py"))
    return sorted({p for p in paths if p.is_file()})


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for p in source_files()}


def snapshot(destination):
    files = source_files()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(ROOT)))
        for path in (ROOT / "AGENTS.md", ROOT / "programs.md", ROOT / "third_party/SOURCES.md"):
            if path.exists():
                archive.write(path, str(path.relative_to(ROOT)))
    return source_hashes()


def asset_paths(asset_dir, keys):
    asset_dir = Path(asset_dir)
    manifest_path = asset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Assets missing. Run: python -m harness.prepare")
    manifest = read_json(manifest_path)
    config = read_json(CONFIG_PATH)
    paths, records = {}, {}
    for key in keys:
        spec = config["assets"][key]
        path = asset_dir / spec["filename"]
        record = manifest.get(key, {})
        if record.get("url") != spec["url"] or not path.exists():
            raise ValueError(f"Missing/mismatched asset {key}; run harness.prepare")
        actual = sha256(path)
        if actual != record.get("sha256"):
            raise ValueError(f"Asset checksum changed: {path}")
        paths[key], records[key] = path, record
    return paths, records


def finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
