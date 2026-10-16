"""Download the three fixed official assets on the research server."""
import argparse
from pathlib import Path
import time
import urllib.request
from .common import CONFIG_PATH, ROOT, read_json, write_json, sha256


def download(url, destination):
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "sampler-autoresearch/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
                size, last = 0, 0
                while True:
                    data = response.read(1024 * 1024)
                    if not data:
                        break
                    out.write(data)
                    size += len(data)
                    if size - last >= 32 * 1024 * 1024:
                        print(f"  {size // (1024 * 1024)} MiB", flush=True)
                        last = size
                expected = response.headers.get("Content-Length")
                if size == 0 or (expected is not None and size != int(expected)):
                    raise IOError("Incomplete download")
            partial.replace(destination)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "cache/assets")
    parser.add_argument("--only", choices=["network", "all"], default="all")
    args = parser.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    manifest_path = args.assets / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    for key, spec in read_json(CONFIG_PATH)["assets"].items():
        if args.only == "network" and key != "network":
            continue
        destination = args.assets / spec["filename"]
        if destination.exists():
            previous = manifest.get(key)
            if not previous or previous.get("url") != spec["url"] or sha256(destination) != previous.get("sha256"):
                raise ValueError(f"Untracked or changed asset: {destination}. Move it aside before retrying.")
            print(f"Verified {key}")
            continue
        print(f"Downloading {key}: {spec['url']}", flush=True)
        download(spec["url"], destination)
        manifest[key] = {**spec, "sha256": sha256(destination), "bytes": destination.stat().st_size}
        write_json(manifest_path, manifest)
    print(f"Assets ready: {manifest_path}")


if __name__ == "__main__":
    main()
