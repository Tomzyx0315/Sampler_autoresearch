"""Check the active research environment and resolve the expected assets."""
import importlib.util
import platform
import sys
from pathlib import Path

from .common import CONFIG_PATH, asset_paths, default_asset_dir, read_json


REQUIRED_MODULES = {
    "torch": "PyTorch",
    "numpy": "NumPy",
    "scipy": "SciPy (FID)",
    "PIL": "Pillow",
}


def _module_status():
    missing = []
    for module, label in REQUIRED_MODULES.items():
        if importlib.util.find_spec(module) is None:
            missing.append(f"{label} ({module})")
    return missing


def main():
    config = read_json(CONFIG_PATH)
    print(f"Python: {sys.executable}")
    print(f"Version: {platform.python_version()}")

    missing = _module_status()
    if missing:
        print("Missing dependencies: " + ", ".join(missing))
    else:
        import torch
        print(f"Torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    for key in ("network", "reference", "inception"):
        try:
            path = asset_paths(default_asset_dir(), [key])[key]
        except FileNotFoundError:
            path = None
        if path is None:
            if key == "inception":
                print(f"Asset {key}: missing (run python -m harness.prepare to download it)")
            else:
                print(f"Asset {key}: MISSING")
        else:
            print(f"Asset {key}: {path}")

    shared_root = Path(config.get("shared_root", ""))
    dataset = config.get("dataset", {}).get("relative_path")
    if dataset and shared_root:
        dataset_path = shared_root / dataset
        print(f"Raw CIFAR-10: {dataset_path} ({'present' if dataset_path.exists() else 'missing'})")

    if missing:
        return 1
    try:
        asset_paths(default_asset_dir(), ["network", "reference"])
    except FileNotFoundError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
