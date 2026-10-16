"""Deterministic EDM Heun, Algorithm 2 with S_churn=0.

See third_party/SOURCES.md and third_party/edm/LICENSE.txt.
The endpoint at sigma=0 uses Euler (equivalently, the last denoiser).
"""


def sample(denoise, x, sigmas, **kwargs):
    if kwargs:
        raise ValueError(f"Unknown Heun parameters: {sorted(kwargs)}")
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        d = (x - denoise(x, sigma)) / sigma
        proposed = x + (next_sigma - sigma) * d
        if next_sigma > 0:
            d_next = (proposed - denoise(proposed, next_sigma)) / next_sigma
            x = x + (next_sigma - sigma) * (d + d_next) / 2
        else:
            x = proposed
    return x
