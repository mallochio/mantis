# Conductor 500 Diagnosis

Date: 2026-08-02
Environment: mantis Docker stack on Linux x86_64, no GPU
Conductor device: `cpu`, dtype: `float32` (from `.env`)

## Method

1. Started the full Docker stack: `docker compose up -d`
2. POSTed to `localhost:8088/v1/chat/completions` with `model: conductor` and a short prompt:
   ```json
   {"model":"conductor","messages":[{"role":"user","content":"what does git rerere do?"}],"max_tokens":256}
   ```
3. Tested three conductor configurations by varying `MANTIS_LOCAL_CONDUCTOR`.

## Results

### A. No local checkpoint (LiteLLM planner only)

`MANTIS_LOCAL_CONDUCTOR` unset.

- HTTP status: **200**
- `fugu_trace`: `steps:3:conductor`
- Response: full, correct answer about `git rerere`
- Cost: several worker calls, estimated ~$0.04

This shows the orchestration code and worker calls work fine when the planner is a hosted model through LiteLLM.

### B. Base checkpoint `di-zhang-fdu/openfugu-conductor-3b`

`MANTIS_LOCAL_CONDUCTOR=di-zhang-fdu/openfugu-conductor-3b`

- HTTP status: **500**
- Response body:
  ```json
  {"error":"step 1 references future/own step 1 (not a DAG order)"}
  ```
- Docker logs: checkpoint loads successfully, no OOM, model weight loading completes in ~1s, then the above error is returned.

**Root cause for this checkpoint:** the local 3B checkpoint emits a workflow whose `access_list` contains a self-reference or forward-reference. `ConductorExecutor` validates the DAG and rejects it.

### C. Retrained checkpoint `outputs/conductor_retrain/retrain-conductor-20260802_003213/checkpoint`

`MANTIS_LOCAL_CONDUCTOR=/app/checkpoint` (mounted via `eval/docker-compose.conductor.yml`)

- HTTP status: **500**
- Response body (truncated):
  ```json
  {"error":"Conductor did not emit a parseable workflow. Raw: `git rerere` is a Git feature that allows you to squash and replay commits..."}
  ```
- Docker logs: checkpoint loads successfully, no OOM.

**Root cause for this checkpoint:** the retrained checkpoint emits a plain-text answer instead of the required three Python lists (`model_id`, `subtasks`, `access_list`). `openfugu.ultra.parse_workflow` cannot parse it.

## Memory check

The Llama-3.2-3B checkpoint loads in `float32` on CPU and completes without an OOM error. The container has access to the host's full RAM in this Linux environment, so the 500s are not caused by weight loading memory exhaustion. On Docker Desktop for Mac with a 7.7 GB VM limit, loading `float32` could OOM, but that was not observed here.

## Conclusion

The 500s are **model-output format errors**, not memory errors:

1. Base checkpoint produces a parseable but invalid DAG (self/forward reference).
2. Retrained checkpoint produces an unparseable direct answer.

Both checkpoints need the conductor to generate a valid, acyclic DAG of three equal-length lists. Native GPU/MPS serving with `bfloat16` and sampling may change generation behavior enough to fix this, but the core issue is the checkpoint output format, not the container runtime.
