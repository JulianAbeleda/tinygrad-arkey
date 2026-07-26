# Retirement verdict: `shared_attention_evidence_bundle.v1` gate

Retired 2026-07-26 by organization-audit action A3. Recovered with
`git log --diff-filter=A -- extra/qk/shared_attention_evidence_gate.py`.

## What it was

`extra/qk/shared_attention_evidence_gate.py` was a CPU-only, fail-closed classifier for a selected fused
prefill-attention artifact. It defined schema `tinygrad.shared_attention_evidence_bundle.v1`, required
`MIN_TIMING_SAMPLES = 200`, admitted only the profiles `qwen3_8b_q4k_m_gfx1100` and `qwen3_14b_q4k_m_gfx1100`, and
exposed `classify_shared_attention_evidence` / `summarize_checkpoint` returning a `GateResult(status, reasons)`.

## Why it was retired

**Nothing ever produced the bundle it validated.** The schema string appeared exactly once in the repository — its own
definition at line 12. Its only importer was its own unit test. It was a validator with no producer, so it never gated
anything: a passing test proved the classifier self-consistent, not that any artifact had been admitted.

## What actually gates shared-attention evidence

`extra/qk/shared_attention_promotion.py`, which gates `COMPOSITE_ADMISSION_SCHEMA` — a schema with a real producer
(`extra/qk/shared_attention_evidence.py`) and a real consumer (`extra/qk/benchmark_shared_attention.py`). That path is
unaffected by this retirement.

## The durable rule, if it is ever needed again

A fused prefill-attention artifact should not be admitted on fewer than 200 timing samples, and admission should be
restricted to profiles with captured compiler, allocator, and benchmark evidence. Re-express that rule on a schema
something writes.
