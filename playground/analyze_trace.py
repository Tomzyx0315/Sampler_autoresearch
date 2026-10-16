"""Example diagnostic: norm, direction change, and residual rank in a saved trace.

Run: python -m playground.analyze_trace artifacts/RUN/trace.npz
Heun traces include predictor/corrector evaluation states, not only accepted states.
This is an editable example; it does not prescribe what the agent must investigate.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(path):
    with np.load(path, allow_pickle=False) as data:
        sigma, x, denoised = data["sigma"], data["x"], data["denoised"]
    v = ((x - denoised) / sigma[:, None, None, None, None]).reshape(len(sigma), len(x[0]), -1)
    norms = np.linalg.norm(v, axis=-1)
    cosine = (v[1:] * v[:-1]).sum(-1) / np.maximum(norms[1:] * norms[:-1], 1e-30)
    ranks = []
    # Per-trajectory rank after removing its temporal mean; avoid conflating
    # different seeds' distinct subspaces, or a large shared velocity component.
    for j in range(v.shape[1]):
        centered = v[:, j] - v[:, j].mean(axis=0)
        energy = np.linalg.svd(centered, compute_uv=False) ** 2
        ranks.append(float(energy[:2].sum() / energy.sum()) if energy.sum() > 1e-25 else None)
    return {"network_evaluations": len(sigma), "traced_samples": v.shape[1],
            "sigma": sigma.tolist(), "mean_velocity_norm": norms.mean(axis=1).tolist(),
            "mean_adjacent_cosine": cosine.mean(axis=1).tolist(),
            "centered_top2_energy_fraction_per_trajectory": ranks,
            "interpretation": "Descriptive only. Full-trajectory PCA uses future information; it is not an online sampler."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    text = json.dumps(analyze(args.trace), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
