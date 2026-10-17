"""Download official CIFAR-10 assets on the research server."""
import argparse
from pathlib import Path
import urllib.request
from .common import CONFIG_PATH, asset_paths, default_asset_dir, read_json


def download(url, destination):
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "sampler-autoresearch"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
            size = 0
            while data := response.read(1024 * 1024):
                out.write(data)
                size += len(data)
            expected = response.headers.get("Content-Length")
            if not size or (expected is not None and size != int(expected)):
                raise IOError("Incomplete download; retry prepare")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, default=default_asset_dir(),
                   help="Asset root; defaults to the configured shared data directory")
    p.add_argument("--only", choices=["network", "inception", "reference", "all"], default="all")
    args = p.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    selected = {args.only} if args.only != "all" else {"network", "inception", "reference"}
    for key, spec in read_json(CONFIG_PATH)["assets"].items():
        if key not in selected:
            continue
        destination = args.assets / spec.get("relative_path", spec["filename"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            print(f"Using {key}: {destination}")
            continue
        try:
            existing = asset_paths(args.assets, [key])[key]
        except FileNotFoundError:
            existing = None
        if existing is not None:
            print(f"Using shared {key}: {existing}")
            continue
        print(f"Downloading {key}: {spec['url']}", flush=True)
        download(spec["url"], destination)
        print(destination)


if __name__ == "__main__":
    main()
