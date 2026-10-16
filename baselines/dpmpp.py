"""DPM-Solver++(2M), midpoint variant, in EDM's x=x0+sigma*noise coordinates.

Equivalent to the pinned k-diffusion implementation (see third_party/SOURCES.md).
One network evaluation per interval, including the final denoising interval.
"""
import math


def sample(denoise, x, sigmas, **kwargs):
    if kwargs:
        raise ValueError(f"Unknown DPM++ parameters: {sorted(kwargs)}")
    old_denoised = None
    old_h = None
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        denoised = denoise(x, sigma)
        if next_sigma == 0:
            x = denoised
        else:
            h = math.log(sigma / next_sigma)
            estimate = denoised
            if old_denoised is not None:
                ratio = old_h / h
                estimate = (1 + 1 / (2 * ratio)) * denoised - old_denoised / (2 * ratio)
            x = (next_sigma / sigma) * x - math.expm1(-h) * estimate
            old_h = h
        old_denoised = denoised
    return x
