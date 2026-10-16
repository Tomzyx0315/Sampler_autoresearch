# Working in this repository

Read `programs.md`, `documents/state.md`, and recent entries in
`documents/findings.md` before starting or resuming research. Before evaluation,
read `skills/cifar10-eval/SKILL.md` explicitly (stored as a repo workflow, without
requiring a global skill installation). Commands run from the repository root.

## Scope

- This is a CIFAR-10 sampler research project with a frozen EDM checkpoint.
- Edit `sampler.py` freely. Use `playground/` for analysis scripts, new algorithm
  modules, hypothesis tests, and temporary experiments. Add useful tests.
- Keep `harness/`, `baselines/`, `configs/cifar10.json`, the dependencies, and
  `third_party/` stable during research. Infrastructure setup and repairs are a
  separate task: explain the bug, retain old evidence, and rerun affected baselines.
- Do not change pretrained weights, FID preprocessing, reference statistics,
  sample selection, or official evaluation seeds to improve a score.
- Call the supplied `denoise` callback for every model evaluation. Do not access
  its underlying network to bypass counting. Include online probes, correctors,
  learned auxiliary modules, and their overhead when reporting computational cost.
- The benchmark fixes initial noise and the base sigma grid. A candidate may
  choose its own internal evaluation times or parameterization within the model's
  supported positive sigma range, subject to the same cost accounting.
- Extra training/calibration is allowed for lightweight sampler components in
  playground, but record its data, cost, artifacts and scope. Keep calibration
  seeds separate from final evaluation. Never silently switch the backbone.

## Experiment discipline

- Before changing a candidate, inspect existing results. A diagnostic experiment
  is a valid research step even if it produces no new sampler.
- Use a unique run ID and write the hypothesis before running. Standard eval:
  `python -m harness.evaluate --help`.
- Commit code before a substantial run when Git is available. Do not edit source
  while that run is active. The runner snapshots source and rejects changed code.
- Preserve unsuccessful experiments, raw metrics, errors and source snapshots.
  Summarize each meaningful experiment under `documents/experiments/<run>.md`.
- Report sample count, actual NFE, timing scope, seeds, and the baseline with each
  quality claim. Screen-FID-2048 is a screening statistic, not FID-50000.
- Use smoke with `--trace` for diagnosis. Trace-enabled timings are not valid
  speed comparisons. Future trajectory states are not available to an online sampler.
- Run one GPU experiment at a time unless the user allocates independent GPUs.
  Inspect the process on the recorded host before rerunning a stale `running` job.
  Failed/interrupted runs receive a new ID when rerun; this MVP has no batch resume.
- Finish each work session by updating `documents/state.md` and commit the useful
  code and small records. Do not push to a remote unless requested.

## Environment

The local laptop environment must not be modified. Install dependencies, download
model assets, and run expensive model experiments only on the designated research
server after the user starts that work there. Do not install packages just to make
local tests pass. Local standard-library checks are available:
`python3 -B -m unittest tests.test_core -v`.

On the server, run `python -m harness.doctor`, the complete unit suite, and
`python -m harness.verify_baselines` before autonomous algorithm research.
Missing assets/runtime or an unverified baseline is a setup issue, not evidence
for or against a sampler. Do not fabricate GPU validation or paper-level FID.
