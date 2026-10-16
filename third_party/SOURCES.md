# Pinned upstream sources and licenses

## NVlabs/edm

- Repository: https://github.com/NVlabs/edm
- Commit: `008a4e5316c8e3bfe61a62f874bddba254295afb`
- Vendored: Python files in dnnlib/, torch_utils/, training/, plus generate.py,
  fid.py and LICENSE.txt. Source files are unmodified.
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0;
  see `edm/LICENSE.txt`. This includes noncommercial and share-alike conditions.
- `baselines/heun.py` adapts the deterministic portion of Algorithm 2 in
  generate.py. FID quantization/statistics follow generate.py and fid.py.
- Paper: https://arxiv.org/abs/2206.00364

## crowsonkb/k-diffusion

- Repository: https://github.com/crowsonkb/k-diffusion
- Commit: `4601bf085320592473f681a62808ed873d17fad5`
- `k_diffusion_reference/sampling.py` contains the unmodified function body of
  `sample_dpmpp_2m`, with minimal imports and a provenance docstring added.
  The upstream no-grad decorator is omitted; callers use `torch.no_grad()`.
- License: MIT, see `k_diffusion_reference/LICENSE`.
- `baselines/dpmpp.py` is an equivalent implementation using Python scalar time
  coefficients, with an explicit sigma=0 endpoint to avoid log(0).
- DPM-Solver++ paper: https://arxiv.org/abs/2211.01095
- Authors' solver repository: https://github.com/LuChengTHU/dpm-solver

`manifest.json` records the vendored file hashes. Preserve upstream notices.
This repository combines code under different licenses and does not relicense the
upstream files. Model weights and detector downloads retain their source terms.
