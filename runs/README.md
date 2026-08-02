# Conductor retrain run reports

This directory contains human-readable run reports and manifest copies for
`mallochio/mantis` Conductor retraining jobs. The actual checkpoints and
metrics live in `outputs/conductor_retrain/<timestamp>/`, which is `gitignore`d.

## Manifest schema

Every Conductor retrain output now writes `manifest.json` with these fields:

| Field | Meaning |
|---|---|
| `run_id` | Unique run identifier |
| `timestamp` | ISO-8601 start/end timestamp |
| `git_commit` | `git rev-parse HEAD` of `mantis` at run time |
| `base_model` | HuggingFace id or local path used as the GRPO base |
| `real_checkpoint_loaded` | `true` only if `base_model` is `di-zhang-fdu/openfugu-conductor-3b` |
| `device` | `cuda` or `cpu` |
| `gpu_type` | GPU name (or `none`) |
| `gpu_count` | Number of GPUs used |
| `dtype` | Torch dtype used for training |
| `dataset` | Dataset source |
| `task_limit` | Number of tasks used |
| `steps` | GRPO optimizer steps |
| `generations` | `num_generations` per prompt |
| `per_device_batch` | `per_device_train_batch_size` |
| `pool` | 7-slot worker pool spec |
| `output_checkpoint_path` | Path to the saved checkpoint/adapter |
| `acceptance_passed` | `true` if the reloaded checkpoint produced a parseable DAG |
| `valid_for_runtime` | `true` only for the real 3B checkpoint on GPU with passing acceptance |

## Important rule

Only outputs with `real_checkpoint_loaded=true`, `device=cuda`, and
`acceptance_passed=true` may be marked `valid_for_runtime=true`. CPU runs or
runs with a substituted smaller base model are always `valid_for_runtime=false`.

## Run reports

- [`20260801-135M-smoke.md`](reports/20260801-135M-smoke.md) — pipeline-only CPU smoke with a 135M proxy model. Not valid for runtime.
- [`20260801-real-3b-gcp.md`](reports/20260801-real-3b-gcp.md) — real-3B acceptance smoke on GCP L4 spot. `valid_for_runtime=true`.
