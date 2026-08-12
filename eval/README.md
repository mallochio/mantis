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
# run a config through the local Bifrost or Mantis endpoint
uv run python eval/run_eval.py --config direct --fixtures eval/fixtures.jsonl --output eval/results.jsonl
# score raw results; writes <stem>-scored.jsonl and eval/report.md
python3 eval/score.py --results eval/results.jsonl
# luna report; reads tracked native-v2 raw results as baseline
python3 eval/report_luna.py
```

## Boundary rule

Commit fixtures, code, and canonical reports. Do not commit raw run output or
derived `-scored` projections; raw runs belong in `runs/` (gitignored) and
derived projections are regenerated on demand. Reports must be reproducible
from tracked fixtures and raw results.

Reports display a quality or latency mean only when an arm has at least four
successful rows. Comparisons additionally require matching successful counts;
otherwise the report uses `n/a` and omits derived deltas and percentages.
