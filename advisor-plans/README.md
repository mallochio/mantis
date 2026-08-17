# Advisor plans

Advisory design and direction documents. Repository experiment plans live in
`plans/`.

## Archived history

| Plan | Title | Status |
|------|-------|--------|
| 001 | Fusion prototype | Rejected; superseded by 002 |
| 002 | Fusion spike | Completed; superseded by the current implementation |
| 003 | Fusion critique | Superseded by 005; resolved findings were removed |

## Active plans

| Plan | Status | Implement when |
|------|--------|----------------|
| [005 — Fusion hardening](005-fusion-hardening.md) | Active, prioritized P0–P2; deferred items explicit | P0/P1/P2 work is justified by the confirmed risks and focused tests; deferred work only under the conditions in the plan |
| [004 — Mantis routing cost control](004-mantis-routing-cost.md) | Partial; Phase 1 implemented, Phases 2/3 deferred | Phase 2 after logs show cheap-tier deterministic failures materially affecting coding-agent completion; Phase 3 when a real per-session/window cost cap is required |
