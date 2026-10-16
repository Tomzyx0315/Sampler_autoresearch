"""Read-only checks for a research server. Installs and downloads nothing."""
import argparse
import importlib
import sys
from .common import ROOT, asset_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", default=str(ROOT / "cache/assets"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--skip-assets", action="store_true")
    args = parser.parse_args()
    problems = []
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        problems.append("Use Python 3.10–3.12 on the server")
    for name in ("torch", "numpy", "scipy", "PIL", "requests", "click", "tqdm", "psutil"):
        try:
            module = importlib.import_module(name)
            print(f"{name}: {getattr(module, '__version__', 'available')}")
        except ImportError:
            problems.append(f"Missing dependency: {name}")
    try:
        import torch
        if args.device == "cuda":
            if not torch.cuda.is_available():
                problems.append("CUDA is not available")
            else:
                print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass
    try:
        import distutils.util  # required by the pinned NVIDIA dnnlib on Python 3.12
    except ImportError:
        problems.append("distutils unavailable; install the pinned setuptools on the server")
    if not args.skip_assets:
        try:
            paths, _ = asset_paths(args.assets, ["network", "inception", "reference"])
            print("Assets verified:", ", ".join(paths))
        except (ValueError, FileNotFoundError) as error:
            problems.append(str(error))
    for problem in problems:
        print(f"NOT READY: {problem}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
