# Sources

- **NVlabs/edm**, commit `008a4e5316c8e3bfe61a62f874bddba254295afb`:
  https://github.com/NVlabs/edm/tree/008a4e5316c8e3bfe61a62f874bddba254295afb
  Model-loading support files are unmodified; `sampler_reference.py` extracts the
  unchanged `edm_sampler` function with minimal imports. `baselines/heun.py` adapts
  its deterministic branch; image quantization and FID follow upstream generate.py/fid.py.
  License: CC BY-NC-SA 4.0, see `edm/LICENSE.txt`.
- **crowsonkb/k-diffusion**, commit `4601bf085320592473f681a62808ed873d17fad5`:
  https://github.com/crowsonkb/k-diffusion/tree/4601bf085320592473f681a62808ed873d17fad5
  `k_diffusion_reference/sampling.py` extracts the unchanged `sample_dpmpp_2m`
  function body (no-grad supplied by callers). `baselines/dpmpp.py` is equivalent
  with Python scalar time coefficients and an explicit zero endpoint.
  License: MIT, see `k_diffusion_reference/LICENSE`.

Upstream notices and licenses remain applicable. Checkpoints retain their source terms.
