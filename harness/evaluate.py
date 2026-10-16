"""Single-GPU/CPU CIFAR-10 experiment runner with fixed evaluation semantics."""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import traceback
from .common import (ROOT, CONFIG_PATH, read_json, write_json, run_id, now,
                     asset_paths, schedule, sha256, git_info, snapshot, source_hashes)

SAMPLERS = {"dpmpp": "baselines.dpmpp:sample", "heun": "baselines.heun:sample",
            "candidate": "sampler:sample"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="Unique experiment ID; never overwritten")
    p.add_argument("--sampler", default="candidate", help="candidate, dpmpp, heun, or playground.module:sample")
    p.add_argument("--steps", type=int, default=10, help="Intervals, not NFE (Heun uses 2*steps-1)")
    p.add_argument("--preset", choices=["smoke", "screen", "full"], default="smoke")
    p.add_argument("--seed", type=int, default=0, help="First seed; samples use consecutive seeds")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--assets", type=Path, default=ROOT / "cache/assets")
    p.add_argument("--params", default="{}", help="JSON keyword arguments for the candidate sampler")
    p.add_argument("--hypothesis", required=True, help="Question or expected outcome, recorded before execution")
    p.add_argument("--parent", help="Previous experiment ID, if applicable")
    p.add_argument("--trace", type=int, default=0, help="Save first N samples at every network evaluation (smoke only)")
    p.add_argument("--max-nfe", type=int, default=128, help="Maximum forward calls per batch")
    p.add_argument("--max-seconds", type=float, default=14400, help="Budget checked between model calls and batches")
    return p


def validate(args):
    run_id(args.run)
    if args.parent:
        run_id(args.parent)
    if (args.steps < 2 or args.batch < 1 or args.max_nfe < 1 or
            not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        raise ValueError("Invalid steps, batch size, NFE limit, or time budget")
    if not 0 <= args.seed < 2**32 - 50000:
        raise ValueError("Seed must be in [0, 2**32-50000)")
    if args.trace < 0 or args.trace > 16 or (args.trace and args.preset != "smoke"):
        raise ValueError("Trace is diagnostic only: use smoke and 0<=trace<=16")
    if args.trace > args.batch:
        raise ValueError("Trace sample count must not exceed batch size")
    params = json.loads(args.params)
    if not isinstance(params, dict):
        raise ValueError("--params must be a JSON object")
    # Reject nonstandard JSON NaN/Infinity as well.
    json.dumps(params, allow_nan=False)
    entry = SAMPLERS.get(args.sampler, args.sampler)
    if entry not in SAMPLERS.values() and not entry.startswith("playground."):
        raise ValueError("Custom sampler must be a playground.module:function")
    if entry.count(":") != 1 or not all(part.isidentifier() for part in entry.replace(":", ".").split(".")):
        raise ValueError("Invalid sampler entry point")
    return entry, params


def save_grid(images, path):
    from PIL import Image
    import numpy as np
    array = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    width = min(8, len(array))
    height = (len(array) + width - 1) // width
    canvas = np.zeros((32 * height, 32 * width, 3), dtype=np.uint8)
    for i, item in enumerate(array):
        row, col = divmod(i, width)
        canvas[row * 32:(row + 1) * 32, col * 32:(col + 1) * 32] = item
    Image.fromarray(canvas).save(path)


def execute(args, entry, params, record_dir, artifact_dir, status):
    import numpy as np
    import torch
    from .model import load_network, noise_for_seeds, Denoiser, synchronize
    from .metrics import to_uint8, load_detector, FeatureStats, fid

    config = read_json(CONFIG_PATH)
    num_samples = config["presets"][args.preset]
    need_fid = args.preset != "smoke"
    keys = ["network", "inception", "reference"] if need_fid else ["network"]
    paths, assets = asset_paths(args.assets, keys)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use a GPU server or --device cpu for a small smoke run")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    net = load_network(paths["network"], device)
    detector = load_detector(paths["inception"], device) if need_fid else None
    stats = FeatureStats() if need_fid else None
    if not net.sigma_min <= config["sigma_min"] < config["sigma_max"] <= net.sigma_max:
        raise ValueError("Benchmark sigma range is not supported by checkpoint")
    sigmas = schedule(args.steps, config["sigma_min"], config["sigma_max"], config["rho"])
    rounded = net.round_sigma(torch.tensor(sigmas[:-1], dtype=torch.float64, device=device)).cpu().tolist()
    if any(a <= b for a, b in zip(rounded, rounded[1:])) or rounded[-1] <= 0:
        raise ValueError("Rounded sigmas must be strictly decreasing and positive")
    sigmas = rounded + [0.0]
    module_name, function_name = entry.split(":")
    sample = getattr(importlib.import_module(module_name), function_name)
    if not callable(sample):
        raise ValueError("Sampler entry is not callable")

    manifest = read_json(record_dir / "manifest.json")
    manifest.update({
        "assets": assets, "benchmark_sha256": sha256(CONFIG_PATH), "sigmas": sigmas,
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "platform": platform.platform(), "hostname": socket.gethostname(),
        "num_samples": num_samples, "seed_end_exclusive": args.seed + num_samples,
        "initial_noise": "per-seed CPU torch.Generator; float32 noise promoted to float64",
        "sampling_precision": "float64 state / float32 pretrained denoiser",
        "fid_protocol": "EDM uint8 quantization + NVIDIA Inception + sample covariance (N-1)",
    })
    write_json(record_dir / "manifest.json", manifest)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (record_dir / "environment.txt").write_text(freeze.stdout if freeze.returncode == 0 else freeze.stderr)

    deadline = time.monotonic() + args.max_seconds
    # One full sampling batch warm-up, excluded from throughput and included in run wall time.
    warm_n = min(args.batch, num_samples)
    print(f"Warm-up: {warm_n} images", flush=True)
    with torch.no_grad():
        initial = noise_for_seeds(range(args.seed, args.seed + warm_n), device) * sigmas[0]
        warm_model = Denoiser(net, args.max_nfe, deadline)
        warm_result = sample(warm_model, initial, list(sigmas), **params)
        check_output(warm_result, initial)
        del warm_result, initial
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        total_time, sample_evaluations, forward_calls = 0.0, 0, 0
        min_nfe, max_nfe = None, 0
        for start in range(0, num_samples, args.batch):
            if time.monotonic() > deadline:
                raise TimeoutError("Experiment time budget exhausted")
            seeds = list(range(args.seed + start, args.seed + min(start + args.batch, num_samples)))
            initial = noise_for_seeds(seeds, device) * sigmas[0]
            observed = Denoiser(net, args.max_nfe, deadline, args.trace if start == 0 else 0)
            synchronize(device)
            begin = time.perf_counter()
            samples = sample(observed, initial, list(sigmas), **params)
            synchronize(device)
            total_time += time.perf_counter() - begin
            check_output(samples, initial)
            nfe = observed.sample_evaluations / len(seeds)
            min_nfe = nfe if min_nfe is None else min(min_nfe, nfe)
            max_nfe = max(max_nfe, nfe)
            sample_evaluations += observed.sample_evaluations
            forward_calls += observed.calls
            images = to_uint8(samples)
            if start == 0:
                save_grid(images[:64], artifact_dir / "preview.png")
                if observed.trace:
                    np.savez_compressed(
                        artifact_dir / "trace.npz", seeds=np.asarray(seeds[:args.trace]),
                        sigma=np.asarray([v["sigma"] for v in observed.trace]),
                        x=np.stack([v["x"] for v in observed.trace]),
                        denoised=np.stack([v["denoised"] for v in observed.trace]),
                        final=samples[:args.trace].detach().float().cpu().numpy(),
                    )
            if need_fid:
                features = detector(images, return_features=True)
                stats.update(features.detach().double().cpu().numpy())
            status.update({"completed_samples": start + len(seeds), "updated_at": now()})
            write_json(record_dir / "status.json", status)
            print(f"{start + len(seeds)}/{num_samples}  batch NFE/sample={nfe:g}", flush=True)

    metric = None
    if need_fid:
        mean, covariance = stats.finalize()
        np.savez(artifact_dir / "inception_stats.npz", mu=mean, sigma=covariance)
        with np.load(paths["reference"], allow_pickle=False) as reference:
            metric = fid(mean, covariance, reference["mu"], reference["sigma"])
    unchanged = source_hashes() == manifest["source_hashes"]
    if not unchanged:
        raise RuntimeError("Source changed during experiment; results are not accepted. Rerun unchanged code.")
    result = {
        "run": args.run, "preset": args.preset, "sampler": entry, "steps": args.steps,
        "num_samples": num_samples, "fid": metric,
        "fid_label": "FID-50000" if args.preset == "full" else ("screen-FID-2048" if need_fid else None),
        "nfe_per_sample": sample_evaluations / num_samples,
        "min_batch_nfe_per_sample": min_nfe, "max_batch_nfe_per_sample": max_nfe,
        "forward_calls": forward_calls, "warmup_forward_calls": warm_model.calls,
        "sampling_seconds": total_time, "sampling_ms_per_image": total_time * 1000 / num_samples,
        "timing_valid_for_comparison": args.trace == 0,
        "timing_scope": "synchronized sampler only, excludes warmup/loading/noise/FID/PNG; includes trace if enabled",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "source_unchanged": unchanged,
    }
    write_json(record_dir / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def check_output(samples, initial):
    import torch
    if not isinstance(samples, torch.Tensor) or samples.shape != initial.shape:
        raise ValueError("Sampler must return a tensor with the input shape")
    if samples.device != initial.device or samples.dtype != initial.dtype:
        raise ValueError("Sampler must preserve device and float64 state dtype")
    if not bool(torch.isfinite(samples).all()):
        raise ValueError("Sampler returned NaN/Inf")


def main():
    args = parser().parse_args()
    entry, params = validate(args)
    record_dir, artifact_dir = ROOT / "experiments" / args.run, ROOT / "artifacts" / args.run
    # Exclusive creation prevents accidental reruns from replacing evidence.
    record_dir.mkdir(parents=True, exist_ok=False)
    status = {"state": "running", "pid": os.getpid(), "hostname": socket.gethostname(),
              "started_at": now(), "completed_samples": 0}
    write_json(record_dir / "status.json", status)
    started = time.monotonic()
    try:
        artifact_dir.mkdir(parents=True, exist_ok=False)
        hashes = snapshot(artifact_dir / "source.zip")
        manifest = {"schema_version": 1, "run": args.run, "hypothesis": args.hypothesis,
                    "parent": args.parent, "created_at": now(), "sampler_entry": entry,
                    "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "params": params, "git": git_info(), "source_hashes": hashes,
                    "source_archive_sha256": sha256(artifact_dir / "source.zip"),
                    "command": [sys.executable, "-m", "harness.evaluate", *sys.argv[1:]]}
        write_json(record_dir / "manifest.json", manifest)
        execute(args, entry, params, record_dir, artifact_dir, status)
        status["state"] = "completed"
    except BaseException as error:
        status.update({"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       "error": f"{type(error).__name__}: {error}"})
        (record_dir / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        status.update({"ended_at": now(), "wall_seconds": time.monotonic() - started})
        write_json(record_dir / "status.json", status)


if __name__ == "__main__":
    main()
