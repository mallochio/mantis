# 004 — Mantis routing cost control

**Status:** Superseded — Base now uses NVIDIA NeMo Switchyard stage routing
**Priority:** P2

Phase 1 (idle-time downgrade and periodic rescoring) shipped in the retired
Supra gateway. Phases 2 (failure-triggered cascade) and 3 (budget pacing) were
never implemented there.

`mantis/base` now uses Switchyard's stage router: efficient-first with
signal-driven escalation on tool errors and stalled coding turns. That covers
the cascade intent without a custom Python policy. Budget pacing remains out of
scope until a real per-session spend cap is required.
