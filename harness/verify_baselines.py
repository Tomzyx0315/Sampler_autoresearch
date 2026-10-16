"""Compare both baselines against pinned upstream code on the real checkpoint.

Run once on the server before research. Writes evidence; never invents a FID target.
"""
import argparse
import importlib.util
from pathlib import Path
from .common import ROOT, CONFIG_PATH, asset_paths, read_json, write_json, now, git_info


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--assets", type=Path, default=ROOT / "cache/assets")
    p.add_argument("--output", type=Path, default=ROOT / "experiments/baseline-verification.json")
    args = p.parse_args()
    import torch
    from .model import load_network, noise_for_seeds, Denoiser
    from baselines.heun import sample as heun
    from baselines.dpmpp import sample as dpmpp
    if args.output.exists():
        raise FileExistsError(f"Use a new --output to preserve previous evidence: {args.output}")
    paths, records = asset_paths(args.assets, ["network"])
    net = load_network(paths["network"], args.device)
    upstream_edm = load_file("reference_edm_generate", ROOT / "third_party/edm/generate.py")
    upstream_k = load_file("reference_k_sampling", ROOT / "third_party/k_diffusion_reference/sampling.py")
    config = read_json(CONFIG_PATH)
    steps, rho = 6, config["rho"]
    indices = torch.arange(steps, dtype=torch.float64, device=args.device)
    levels = (config["sigma_max"] ** (1 / rho) + indices / (steps - 1) *
              (config["sigma_min"] ** (1 / rho) - config["sigma_max"] ** (1 / rho))) ** rho
    levels = torch.cat([net.round_sigma(levels), torch.zeros_like(levels[:1])])
    latents = noise_for_seeds([1000000, 1000001], args.device)
    result = {"created_at": now(), "git": git_info(), "assets": records, "checks": {}}
    with torch.no_grad():
        for name, ours, expected_nfe in (("heun", heun, 2 * steps - 1), ("dpmpp", dpmpp, steps)):
            counter = Denoiser(net)
            actual = ours(counter, latents * levels[0], levels.cpu().tolist())
            if name == "heun":
                reference = upstream_edm.edm_sampler(net, latents, num_steps=steps,
                    sigma_min=config["sigma_min"], sigma_max=config["sigma_max"], rho=rho,
                    randn_like=torch.zeros_like)
            else:
                reference = upstream_k.sample_dpmpp_2m(net, latents * levels[0], levels, disable=True)
            passed = bool(torch.allclose(actual, reference, atol=1e-4, rtol=1e-4)) and counter.calls == expected_nfe
            result["checks"][name] = {"passed": passed, "nfe": counter.calls,
                "expected_nfe": expected_nfe, "max_abs_error": float((actual - reference).abs().max()),
                "atol": 1e-4, "rtol": 1e-4}
    result["passed"] = all(item["passed"] for item in result["checks"].values())
    write_json(args.output, result)
    print(result)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
