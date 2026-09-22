# NV decode final accounting audit

Date: 2026-08-05. Scope: adversarial review of the d512 Qwen3-8B-Q4_K_M
RTX 5090 / 595.84 native-NV versus llama token ledger. This review ran no GPU
work and changed no production code.

## Verdict

**PASS for physical location of the 1.646170 ms/token gap; FAIL for a fully
causal, independently recoverable explanation.** There is no unbooked
*outer-boundary* term above 50 us. The remaining failure is a causality and
booking issue, not missing elapsed time.

The exact compatible-clock reconciliation is:

```text
authority wall gap
= (profiled support delta + profiled quant-core delta - llama internal gaps
   - profile-to-unprofiled device reconciliation)
  + outside-window delta + outer reconciliation
= (1108.082 + 302.788 - 8.111 - 1.143)
  + 239.804933 + 4.749067
= 1646.170000 us/token
```

The `-1.143 us` term is required: the semantic partition is a scaled
per-node-profile composition, whereas 1401.616 us is the independent
marker-light device-window authority. Omitting it double-counts 1.143 us.
The outer reconciliation is likewise a residual between independently
medianed native and llama runs, not an optimization target.

## What is established

- **Device location:** 1401.616 us (85.14%) is inside the native device
  window. Its top-level comparison is 1108.082 us support-work critical-path
  exposure, 302.788 us quantized-core difference, and -8.111 us llama graph
  gaps, adjusted by -1.143 us profiling reconciliation.
- **Outside location:** 239.805 us (14.57%) is native-versus-llama time
  outside the device window. Native's own 321.784 us outside boundary has a
  bounded, disjoint operational decomposition; its largest known paths are
  predispatch and scalar copyout/item.
- **No material launch-gap term:** native's measured five-group device window
  is continuous at the accounting level; CUDA's six inter-group gaps are only
  2.4 us. The broad difference is work exposed on a serialized native graph
  versus llama work overlapped with MMQ, not graph-launch bubbles.
- **Quant direction:** the Q6 attention and Q4 FFN-down CUDA substitutions
  show real substrate opportunity, while native-Q6 ordinary-UOp and included
  Q8+DP4A gates currently fail. Those are implementation directions, not
  native-NV recovery credits.

## Findings that prevent a stronger verdict

### 1. A 69.166-us causal native A/B is absent from the master booking

`nv-decode-native-d512-predispatch-ab-record-20260805.md` reports an
adjacent-control, token-stream-identical native-NV combined A+B result of
`-69.1655 us/token`. The P6 ledger still sets
`accepted_native_recovery_us = 0` and calls the entire 1646.170 us
“unexplained.” That is internally inconsistent with P6's stated admission
rule unless full-logit equality is an explicit required gate. The A/B record
only has the sampled-token contract, so either:

1. add an explicit **provisional native recovery** of 69.166 us with its
   weaker correctness label, yielding an engineering remainder of
   **1577.004 us**, or
2. retain zero booking, but change the P6 rule to require a full-logit oracle
   and label the 69.166 us `CAUSAL_NATIVE_WALL_SIGNAL_UNBOOKED_FOR_LOGITS`.

Do not add its individual 65.536-us and 28.372-us rows: their combined arm is
only 69.166 us and they overlap. This is a material accounting defect
(69.166 us > 50 us), though it does not invalidate the wall-location equation.

### 2. The Shapley rows are ownership, not independent causes

The 574.654-us norms, 247.989-us flash, and 240.319-us
residual/cast/contiguous rows are valid *additive critical-path ownership*
under the stated symmetric interval convention. They are not independently
recoverable savings. In particular, llama's Q8 quantization, casts and
residuals cross fusion boundaries with the MMQ calls, while tinygrad materializes
some of them as separate nodes. Calling the 662.128-us “raw implementation /
topology cost” too strongly implies a matched like-for-like kernel comparison
that has not been made.

Required wording: call this term **fusion/dataflow and exposed-work
attribution**, not raw implementation cost. A native family A/B, or an exact
dataflow-preserving lowerer that changes one disjoint population, is needed
before booking any such row. This is the largest unresolved causal ambiguity,
but it is not a missing time bucket.

### 3. Flash’s conclusion must be narrowed

The flash record correctly shows that a llama raw-flash advantage is *not
required* to explain the +247.989-us ownership allocation: its class interval
union is 363.716 us versus native's calibrated 305.581 us. However, those are
different measurement mechanisms and fusion contexts. They do not falsify a
per-kernel or data-layout advantage. The supported statement is:

> Present evidence makes overlap/exposure sufficient to explain the localized
> flash ownership gap; it does not establish flash-body parity.

This distinction matters because a flash rewrite can still be useful, but its
wall credit remains zero absent a real-token native A/B.

### 4. The master ledger is stale after subsequent experiments

Its P6-A row says the packed-int8 provider is pending, but the provider now
passes and the included-cost Q6 gate fails by +1.172 to +1.352 us. Its status
also says finer native causes are open despite the new role partition. Update
the master record before treating it as a self-contained final brief.

## Bottom line

The evidence now fully accounts for **where** llama wins at d512 and identifies
the dominant mechanism: llama overlaps a large amount of support/fusion work
behind MMQ, while native NV exposes it serially; the remaining device portion
is concentrated in a few quantized populations, and the host portion is
specific replay/copy machinery. It does **not** yet justify a claim that the
role-attribution numbers are a complete sum of independently actionable kernel
savings. The decisive missing experiment is not another broad trace: it is an
exact-output native A/B for each fusion/dataflow population, plus an
implementation route for native multi-queue overlap.

Machine-readable audit:
`docs/task_workflow/output/nv-decode-final-accounting-audit-20260805.json`.

## Post-audit disposition

The master P6 ledger was amended after this review. It now labels the 69.166-us
predispatch result as a causal native wall signal unbooked for the missing
full-logit oracle, reports the 1577.005-us provisional engineering remainder,
uses fusion/dataflow/body *attribution* rather than raw-cost wording, narrows
the flash statement to overlap sufficiency, and records the completed DP4A
Gate-1 failure. The audit's physical-location PASS and independent-recovery
FAIL remain unchanged.
