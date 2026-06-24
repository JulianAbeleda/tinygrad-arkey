# Attention Tail After B5 — Scope (2026-06-22)

**Phase 3 of 8B decode-gap exhaustion.** Audit only; no kernel/default change.

## Question
The diff's **attention qk/softmax/pv** bucket is the **ctx-growing** gap (+1.5 ms @ctx512 → +2.6 ms @ctx4096,
ratio 3–5×). Reopening attention is gated by the hard rule: **"Do not reopen attention without explaining
B5 saturation."** Determine whether the remaining attention gap is **critical-path** or **overlapped /
mapping-artifact**, and whether a **bounded** reopen is justified after the B4/B5 arc.

## Required: explain B5 saturation first
B4 = the owned AMDGCN flash-decode tile entering the TinyJit decode graph as `Ops.PROGRAM` nodes
(`B4_WD_FAIL_INTEGRATION`). B5 = the cheaper split-KV combine (`B5_COMBINE_LOCAL_PASS_WD_FAIL`). The
saturation must be characterized from `docs/b4-cheaper-combine-result-20260622.md`,
`docs/decode-time-tax-audit-result-20260622.md`, `docs/split-kv-economics-audit-result-20260621.md`,
and the codegen closeouts (`fused-flash-concrete-gate`, `matmul-pv-diagnostic`,
`north-star-decode-attention-redesign-audit`).

## Method
- Confirm the attention bucket is **correctly mapped** from the rendered kernels (the diff already maps
  `flash_*` correctly; verify and add the ctx-growing `start_pos` QK/PV reduce).
- Measure attention's **ctx-scaling** from `bench/qk-decode-kernel-probe/latest.json`.
- Record the **B5 transfer ground-truth** (in-model W==D %) that bounds any reopen.
- Decide: critical-path vs overlapped, and reopen-justified vs lever-exhausted.

## Deliverables
`extra/qk_attention_tail_after_b5_audit.py`, `bench/qk-attention-tail-after-b5-audit/latest.json`, this scope + result.

## Stop condition
If the bounded attention lever is exhausted (owned tile < +7% gate, deeper fused-flash codegen-blocked),
record `NO_REOPEN` and rank attention below any lane with a bounded transferable lever.
