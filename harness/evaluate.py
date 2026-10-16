"""Generate CIFAR-10 samples; optionally compute FID and save a trace."""
import argparse
from datetime import datetime
import importlib
import json
from pathlib import Path
import time
from .common import ROOT, CONFIG_PATH, read_json, write_json, asset_paths, schedule, git_info

SAMPLERS = {"dpmpp": "baselines.dpmpp:sample", "heun": "baselines.heun:sample",
            "candidate": "sampler:sample"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    p.add_argument("--sampler", default="candidate", help="candidate, dpmpp, heun, or module:function")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--num", type=int, default=16)
    p.add_argument("--fid", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--assets", type=Path, default=ROOT / "cache/assets")
    p.add_argument("--params", type=json.loads, default={})
    p.add_argument("--trace", type=int, default=0, help="Trace first N samples in the first batch")
    args = p.parse_args(argv)
    if args.steps < 2 or args.num < 1 or args.batch < 1 or (args.fid and args.num < 2):
        p.error("Need steps>=2, num>=1 (>=2 for FID), batch>=1")
    if not 0 <= args.seed < 2**63 - args.num or not 0 <= args.trace <= min(args.batch, args.num):
        p.error("Invalid seed or trace count")
    if Path(args.run).name != args.run or args.run in ("", ".", ".."):
        p.error("--run must be a directory name")
    if not isinstance(args.params, dict):
        p.error("--params must be a JSON object")
    return args


def save_grid(images, path):
    from PIL import Image
    import numpy as np
    array = images.cpu().numpy().transpose(0, 2, 3, 1)
    width = min(8, len(array))
    canvas = np.zeros((32 * ((len(array) + width - 1) // width), 32 * width, 3), dtype=np.uint8)
    for i, item in enumerate(array):
        row, col = divmod(i, width)
        canvas[row * 32:(row + 1) * 32, col * 32:(col + 1) * 32] = item
    Image.fromarray(canvas).save(path)


def evaluate(args, out):
    import numpy as np
    import torch
    from .model import load_network, noise_for_seeds, Denoiser, synchronize
    from .metrics import to_uint8, load_detector, FeatureStats, fid
    config = read_json(CONFIG_PATH)
    paths = asset_paths(args.assets, ["network", "inception", "reference"] if args.fid else ["network"])
    net = load_network(paths["network"], args.device)
    detector = load_detector(paths["inception"], args.device) if args.fid else None
    stats = FeatureStats() if args.fid else None
    module, function = SAMPLERS.get(args.sampler, args.sampler).split(":")
    sample = getattr(importlib.import_module(module), function)
    levels = schedule(args.steps, config["sigma_min"], config["sigma_max"], config["rho"])
    sigmas = net.round_sigma(torch.tensor(levels[:-1], dtype=torch.float64, device=args.device)).cpu().tolist() + [0.0]
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sampling_seconds, evaluations = 0.0, 0
    with torch.no_grad():
        # Warm up one batch before measuring sampler time.
        warm = noise_for_seeds(range(args.seed, args.seed + min(args.batch, args.num)), args.device)
        sample(Denoiser(net), warm * sigmas[0], list(sigmas), **args.params)
        synchronize(args.device)
        del warm
        for start in range(0, args.num, args.batch):
            seeds = range(args.seed + start, args.seed + min(start + args.batch, args.num))
            x = noise_for_seeds(seeds, args.device) * sigmas[0]
            denoise = Denoiser(net, trace_samples=args.trace if start == 0 else 0)
            synchronize(args.device)
            begin = time.perf_counter()
            samples = sample(denoise, x, list(sigmas), **args.params)
            synchronize(args.device)
            sampling_seconds += time.perf_counter() - begin
            if (samples.shape != x.shape or samples.device != x.device or samples.dtype != x.dtype
                    or not bool(torch.isfinite(samples).all())):
                raise ValueError("Sampler must return finite samples with the input shape/device/dtype")
            evaluations += denoise.sample_evaluations
            images = to_uint8(samples)
            if start == 0:
                save_grid(images[:64], out / "preview.png")
                if denoise.trace:
                    np.savez_compressed(out / "trace.npz", seeds=list(seeds)[:args.trace],
                        sigma=[v["sigma"] for v in denoise.trace],
                        x=np.stack([v["x"] for v in denoise.trace]),
                        denoised=np.stack([v["denoised"] for v in denoise.trace]),
                        final=samples[:args.trace].float().cpu().numpy())
            if args.fid:
                stats.update(detector(images, return_features=True).double().cpu().numpy())
            print(f"{start + len(seeds)}/{args.num}", flush=True)
    value = None
    if args.fid:
        mean, covariance = stats.finalize()
        np.savez(out / "inception_stats.npz", mu=mean, sigma=covariance)
        with np.load(paths["reference"], allow_pickle=False) as ref:
            value = fid(mean, covariance, ref["mu"], ref["sigma"])
    return {"fid": value, "nfe_per_sample": evaluations / args.num,
            "sampling_ms_per_image": sampling_seconds * 1000 / args.num,
            "timing_includes_trace": bool(args.trace), "sigmas": sigmas,
            "torch": torch.__version__}


def main():
    args = parse_args()
    out = ROOT / "experiments" / args.run
    out.mkdir(parents=True, exist_ok=False)
    result = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              **git_info(), "status": "running"}
    write_json(out / "result.json", result)
    print(f"Output: {out}", flush=True)
    started = time.monotonic()
    try:
        result.update(evaluate(args, out), status="completed")
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_json(out / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
