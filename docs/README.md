# `docs/`

Design/task planning documents — the "why" behind a version jump or pipeline change,
kept alongside the code instead of only in commit messages.

| File | Contents |
|---|---|
| `wavepinn_v12_dt_fix_and_deployment_plan.md` | Design doc for the v10b → v12 work: the per-sample `dt` batch-averaging bug fix, GT-free deployment velocity-scale estimation, the v12 retrain/ablation plan, and the train/val leakage fix (grouped-by-acquisition splitting). Explains *why* several non-obvious decisions in `src/architectures/wavepinn_v6.py`, `src/utils/deployment_scale.py`, and `scripts/regroup_splits.py` were made. |

Add new planning docs here as `<topic>_plan.md` rather than as loose files in the repo
root.
