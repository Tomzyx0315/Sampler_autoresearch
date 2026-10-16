"""Agent-editable sampler. Initially identical to the DPM++(2M) baseline.

Contract:
  denoise(x, sigma) -> D(x, sigma), counted by the evaluator.
  x: [B,3,32,32], float64, already noise * sigmas[0].
  sigmas: descending positive Python floats, followed by exactly one zero.
  Return a finite tensor with the same shape/device/dtype; do not clamp it.

The physical ODE is dx/dsigma=(x-D(x,sigma))/sigma; never evaluate at zero.
Additional algorithm modules may live in playground/. Network weights stay fixed.
"""
from baselines.dpmpp import sample as baseline


def sample(denoise, x, sigmas, **params):
    return baseline(denoise, x, sigmas, **params)
