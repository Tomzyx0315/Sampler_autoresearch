"""Editable sampler: x is already noise*sigma_max; denoise returns D(x,sigma).
ODE: dx/dsigma=(x-D)/sigma. Never call denoise at sigma=0.
"""
from baselines.dpmpp import sample as baseline


def sample(denoise, x, sigmas, **params):
    return baseline(denoise, x, sigmas, **params)
