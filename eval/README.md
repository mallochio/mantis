# eval/

Evaluation harness and results for Mantis.

## Files

- `run_eval.py` — runs a config against fixtures and writes raw results.
- `score.py` — scores raw results and writes a scored projection plus `report.md`.
- `ab_compare.py` — compares two result sets for an A/B report.
- `report_luna.py` — builds the luna-conductor report against the native-v2 baseline.
- `fixtures.jsonl` — the fixed set of eval prompts/cases used by all configs.

## Regeneration commands

```
# run a config, raw results land in eval/results.jsonl (or --output)
python3 eval/run_eval.py --config direct --fixtures eval/fixtures.jsonl --output eval/results.jsonl
# score raw results; writes <stem>-scored.jsonl and eval/report.md
python3 eval/score.py --results eval/results.jsonl
# luna report; reads results-native-v2-scored.jsonl as baseline
python3 eval/report_luna.py
```

## Boundary rule

Commit fixtures, code, and canonical reports. Do not commit raw run output or
derived `-scored` projections; raw runs belong in `runs/` (gitignored) and
derived projections are regenerated on demand.

`eval/results-native-v2-scored.jsonl` is a documented exception: `report_luna.py`
reads it as its direct/trinity baseline, so it stays tracked until a future
change adds a `--baseline` option that points at raw results instead.
