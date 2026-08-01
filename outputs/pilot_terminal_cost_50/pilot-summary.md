# Pilot retrain summary (TerminalBench 2.1, limit 50, cost mode)

- Date: 2026-08-01
- Pool: ['anthropic/claude-sonnet-5|medium', 'anthropic/claude-opus-5|medium', 'openai/gpt-5.6-sol|medium', 'openai/gpt-5.6-luna|max', 'openai/gpt-5.6-terra|xhigh', 'deepseek/deepseek-v4-flash|none', 'z-ai/glm-5.2|none']
- Dataset: s3://external-datasets-archive/terminal-bench-2.1/
- Label mode: cost
- Train tasks: 45
- Validation tasks: 5
- Best validation worker accuracy: 0.8000
- Avg validation reward: 0.0260

## Label distribution (gold worker per training task)

- slot 0 `anthropic/claude-sonnet-5|medium`: 0
- slot 1 `anthropic/claude-opus-5|medium`: 0
- slot 2 `openai/gpt-5.6-sol|medium`: 0
- slot 3 `openai/gpt-5.6-luna|max`: 5
- slot 4 `openai/gpt-5.6-terra|xhigh`: 0
- slot 5 `deepseek/deepseek-v4-flash|none`: 40
- slot 6 `z-ai/glm-5.2|none`: 0

## Worker API calls

- Total worker calls: 315
- Failed calls: 56
- Estimated worker API spend (from configs/worker-costs.json, 2K-in/1K-out assumption): **$4.6021 USD**

### Failures per model

- `openai/gpt-5.6-sol|medium`: 17 failures
- `openai/gpt-5.6-terra|xhigh`: 15 failures
- `anthropic/claude-opus-5|medium`: 11 failures
- `anthropic/claude-sonnet-5|medium`: 7 failures
- `openai/gpt-5.6-luna|max`: 6 failures

## Notes

The cost-aware gold label heavily favors the cheapest workers because the
TerminalBench reward is a difflib similarity (0..1). With a 70x cost spread
between deepseek-v4-flash ($0.00056/call) and gpt-5.6-sol ($0.04/call), a
small absolute reward difference is not enough to overcome the price ratio.
For a production router, either use a stronger code-execution reward (e.g.
SWE-bench / LiveCodeBench pass rate) or rescale costs so quality can win on
harder tasks.

All reasoning models (Claude/GPT-5.6) recorded at least some API failures
during this run; the non-reasoning deepseek-v4-flash and glm-5.2 calls
succeeded for every task.
