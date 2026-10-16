---
name: cifar10-eval
description: Run and interpret this repository's fixed CIFAR-10 sampler evaluation, baseline verification, and trace collection on the research server.
---

# CIFAR-10 evaluation

Use the repository root as the working directory. Read `documents/state.md` to
check whether server setup and baseline verification have actually passed.
Do not install dependencies or download models on the laptop.

## First server run

Follow `README.md` for environment setup, then run:

```bash
python -m harness.doctor
python -m unittest discover -s tests -v
python -m harness.verify_baselines
```

The real-model parity report checks both Heun and DPM++ against pinned upstream
implementations. A skipped runtime test is not a passed runtime check.

## Evaluate

```bash
python -m harness.evaluate --run UNIQUE_ID --sampler candidate --steps 10 \
  --preset smoke --hypothesis "The change is finite and respects the interface"
```

Replace `candidate` with `dpmpp` or `heun` for a baseline, or with
`playground.module:sample` for another candidate. `--params` accepts a JSON object.
The presets are fixed: smoke=16 (no FID), screen=2048, full=50000.
Use new run IDs; records are never overwritten. Check actual NFE in `metrics.json`.

For trajectory diagnosis, add `--trace 4` to smoke. Analyze the resulting
`artifacts/UNIQUE_ID/trace.npz` with `playground.analyze_trace` or a new probe.
These are network evaluation states; Heun includes correction evaluations.
Trace timings are diagnostic, not benchmark latency.

For promising candidates, run full with `--seed 100000` or the next unused
confirmation range, together with the matching baseline. A small screen improvement
does not justify a final claim. Preserve all repeats, not just the best FID.

## Read and record

- `experiments/ID/manifest.json`: configuration, code snapshot hashes, asset hashes.
- `status.json`: completion or failure; `metrics.json` exists only after success.
- `artifacts/ID/`: source snapshot, preview, optional trace and Inception statistics.
- Report actual NFE, sample count, seed range, timing scope, and comparator.
- Write the scientific interpretation under `documents/experiments/ID.md` using
  the small template there. Update `documents/state.md` and `findings.md` as useful.

On failure, read `error.txt`. Fix setup errors separately from sampler failures.
Before rerunning a supposedly active task, inspect its recorded host and process.
Do not edit sources while an evaluation is active; a source change invalidates it.
