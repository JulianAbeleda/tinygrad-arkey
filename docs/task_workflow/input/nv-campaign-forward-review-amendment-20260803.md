# NV campaign forward-path review - amendment

Date: 2026-08-03

Status: review amendment. This document records the review verdict on
`nv-campaign-forward-review-20260803.md`. It supersedes that document's sections 2-6 where
the lifecycle state, L1/Path-1 decision, endpoint expectation, or forward ordering differs.
It does not authorize implementation, route promotion, or promotion to `dev`/`exp`/`master`.
Branch boundary: tinygrad `nvidia-bringup-20260731` at review commit `1d668e3bb`.

The original review scope remains unchanged as the request-of-record. This amendment is the
response-of-record.

---

## 1. Verdict

The proposed forward sequence is **not approved as written**. It rewinds work that is already
implemented and measured, treats a measured boundary failure as an open design question, and
carries an endpoint estimate that a later campaign amendment explicitly withdrew.

The campaign is not waiting to implement L1. Its current state is:

- M1 admission exists.
- M2 Q6K down-coop in-kernel merge is landed and promoted on NV.
- M3 fused decode RMSNorm is landed closed-default and measured non-landing.
- M4 Q4K epilogue variants are landed closed-default and measured non-landing.
- M5 flash-combine fp16 output is landed closed-default and measured non-landing.
- Path 3 semantic RMSNorm is landed closed-default and measured non-landing at
  d512/d2048/d4096 with byte-identical tokens.

Therefore the next decision is **which existing closed variant, if any, earns a narrowly
scoped prerequisite or redesign that can reopen it**. It is not whether to implement the
original L1 design.

---

## 2. Corrections to the review scope

### 2.1 Lifecycle correction - the nine L1 questions are no longer the blockers

`l1-decode-plumbing-fusion-design-20260802.md` is the historical design that preceded M1-M5.
Its HARD STOP and nine questions were consumed by the subsequent implementation, review, and
measurement sequence. The authoritative later records are:

- `decode-norm-fusion-paths-forward-20260802.md` sections 9-10;
- `m4-q4k-epilogue-measurement-record-20260802.md`;
- `m5-flash-combine-normalization-measurement-record-20260802.md`;
- `path3-semantic-rmsnorm-measurement-record-20260802.md`; and
- `nv-decode-parity-final-20260802.md`.

The forward review's instructions to resolve the nine questions and then "Implement L1" would
duplicate already-landed additive variants. Any future implementation scope must start from
the current closed records and name a reopen condition, not restart the original design.

### 2.2 L1 versus Path 1 - selection admission does not solve transport

The review scope asks whether per-emitter opt-in is enough to let L1 land before Path 1. The
forced-open measurements answer this: **no**.

| variant | measured boundary result | wall result |
| --- | --- | --- |
| M3 fused RMSNorm | 144 input copies + 72 output materializations | -3% at d512 |
| M4 Q4K epilogues | 126 opaque-boundary copies | -18.8% at d512 |
| M5 fp16 combine | 36 absorbed casts replaced 1:1 by 36 fp16 copies | noise / no win |
| Path 3 semantic RMSNorm | +110 kernels/token, including per-call boundary classes | -0.9% to -1.3% at all depths |

Per-emitter opt-in controls **which emitter is selected**. It does not change
`UOp.custom_kernel` / `execute_promoted_program` materialization semantics. For the variants
above, a transport or producer/consumer contract change is a prerequisite to reopening.

This does **not** reverse the standalone Path-1 NO-GO in
`non-norm-copy-inventory-census-20260802.md`. That census correctly found zero category-A tax
in the default M2-on graph. The current justification is narrower: copies introduced by a
specific forced-open variant. To avoid conflating the two claims, a future scope should call
this a **variant-reopen boundary P0**, not a general Path-1 transport campaign.

The P0 must name, for every proposed consumer:

1. producer, consumer, logical shape, dtype, and exact UOp chain;
2. copy class, count, median time, and whether it exists only with the variant open;
3. the typed opt-in contract that replaces materialization;
4. the legacy route/hash that must remain byte-identical; and
5. the fixed-depth wall and sha gate required before the route record may change.

### 2.3 M4 has a second blocker independent of the boundary copies

The M4 record attributes the non-landing to the 126 boundary copies, but its own totals show
that this is incomplete:

- total kernel-time regression: **+1264 us/token**;
- boundary-copy cost: approximately `126 * 1.5 us = 189 us/token`;
- unexplained residual after deleting every copy arithmetically: approximately
  **+1075 us/token**, before crediting the elementwise work M4 intended to remove.

The implementation explains the missing mass. In
`decode_kernels.py:q4k_g3_lanemap_gemv_kernel`, the `ffn_down_fused` variant loads
`gate_out[idx]` and `up_out[idx]` and evaluates `_silu_uop(g) * u` inside the reduction for
each output row. The legacy path computes the 12288-element activation once, then every down
row reads it. The fused prelude recomputes the nonlinear activation across the 4096 output
rows. Removing boundary copies cannot make that shape economical.

M4 must be decomposed before any reopen claim:

- **o-proj residual epilogue**: eligible for a narrow boundary P0 and isolated measurement;
- **k/v fp16 output**: overlaps M5's producer/output-layout problem and needs an isolated
  output-contract measurement;
- **ffn-down SiLU/multiply prelude**: reject the current shape for performance. Redesign it
  so the activation is computed once (for example producer-side gate/up fusion or a separately
  materialized activation), then measure that design independently;
- **ffn-down residual epilogue**: may be separable from the rejected prelude and should not be
  forced to share its verdict.

The current combined M4 record must stay closed even if a boundary P0 succeeds.

### 2.4 Endpoint correction - withdraw the old L1 arithmetic

The `0.9-1.0ms` L1 claim and `1.07-1.21x` decode endpoint in the forward review are not a
current forecast. `decode-norm-fusion-paths-forward-20260802.md` section 9 reconciled the norm
hypothesis to approximately **-144 launches / -0.16ms node-sum**, and section 9.6 explicitly
requires forecasts to be recomputed from measured M4/M5 wall results rather than carrying the
old L1 arithmetic forward.

The subsequent measurements did not supply the missing wall evidence:

- M3: non-landing;
- M4: large regression and a second mechanism defect;
- M5: net zero;
- Path 3: non-landing at all depths.

Accordingly:

- `0.9-1.0ms` remains historical node-sum opportunity attribution, not an available lever;
- `1.07-1.21x` is withdrawn as a forward expectation;
- no composed L1 endpoint is stated until each redesigned/reopened component has an isolated
  wall measurement; and
- `nv-decode-parity-final-20260802.md` remains the current wall authority: 1.44x/1.46x/1.52x
  behind llama, with the next evidenced wall lever in decode GEMV efficiency.

### 2.5 Parity-record reproducibility correction

`nv-decode-parity-final-20260802.md` says the default runtime has "all promotion records
closed", but the same campaign intentionally keeps M2's
`decode_epilogue_fusion` record promoted for `NV:sm_120`, and the record later describes the
baseline as M2-on.

The reproducible state is:

- M2 Q6K down-coop merge: **open/promoted for NV sm_120**;
- M3 norm, M4 Q4K epilogue, M5 combine, and Path 3 semantic RMSNorm: **closed**.

The parity document should be corrected before a later measurement cites it as a fully closed
baseline.

---

## 3. Answers to the five review questions

### Q1 - Is the nine-question L1 set complete?

No. It is no longer the applicable gate. Later implementations and measurements answered or
superseded the questions. The live blockers are variant-specific boundary contracts, the M4
FFN-down recomputation defect, and missing isolated wall evidence for any reopen candidate.

### Q2 - Does "no new kernel" forbid a fused RMSNorm emitter?

This is resolved by history, not still open. M3 and Path 3 already added fused/native RMSNorm
kernel families behind closed records and proved correctness. Their failure is economic, not a
campaign-policy violation. Do not reopen the old policy question as an implementation blocker.

### Q3 - Does the boundary gate epilogue absorption, or is opt-in sufficient?

The boundary gates the existing M3/M4/M5/Path-3 variants. Per-emitter opt-in is necessary for
safe migration but insufficient for copy-free execution. The next transport work, if pursued,
must be a typed, variant-specific opt-in with the default flat-buffer contract unchanged.

### Q4 - Should B3 wait for decode L1?

No. B3 is an independent prefill runtime lever with its own AMD control requirement. Nothing in
the decode boundary decision licenses serializing it behind decode. It may proceed under its own
scope while decode work follows the corrected sequence below.

### Q5 - Are the endpoint expectations evidence-disciplined?

Prefill parity statements are measured and may stand with their exact per-depth ratios. The
decode `1.07-1.21x` statement is not licensed: it promotes superseded node-sum opportunity into
an end-state expectation after all relevant variants measured non-landing. Withdraw it until
new same-session wall measurements exist.

---

## 4. Corrected forward sequence

The decode and prefill paths are independent after the documentation corrections.

### 4.1 Decode

1. **Correct the state-of-record.** Amend the parity protocol to name M2 open and
   M3/M4/M5/Path 3 closed. Replace "implement L1" language with "redesign/reopen existing
   variants".
2. **Decompose M4 without changing default behavior.** Produce isolated census/wall rows for
   residual-add, k/v fp16-output, and FFN-down variants. Reject the current SiLU/multiply
   prelude unless a new design proves the activation is computed once rather than once per
   output row.
3. **Choose one minimal boundary P0.** M5 is the cleanest first probe because its failure is a
   measured 1:1 cast-to-copy substitution with zero confounding wall gain. Scope one typed
   output-layout/view-preservation mechanism, closed-default and consumer-specific.
4. **Re-measure before composing.** A successful M5 boundary P0 does not automatically reopen
   M3, M4, or Path 3. Each route needs its own d512/d2048/d4096 wall and sha record. M4's
   combined record cannot reopen until its FFN-down defect is redesigned.
5. **Prioritize the current wall authority.** Continue the decode GEMV-efficiency work named by
   `nv-decode-parity-final-20260802.md`; do not park it behind the old L1 forecast. L2
   single-pass partial, L4 vocab substrate, and flash substrate remain separate scopes ranked
   by newly measured wall opportunity, not the superseded gross budget.
6. **Recompute an endpoint only from landed pieces.** No node-sum stack or wall forecast is
   published until the components have isolated same-session measurements.

### 4.2 Prefill

1. Keep the measured pp512+ parity record as the baseline.
2. Scope B3 independently with the required AMD control.
3. Measure the polling change against both NV and AMD before landing; do not use decode progress
   as its gate.

---

## 5. Review boundary

This amendment approves only the corrected interpretation and ordering above. It does not
authorize:

- a general change to `custom_kernel`'s default flat-buffer ABI;
- reopening any closed promotion record;
- reusing the current M4 FFN-down prelude after a boundary-only fix;
- carrying the old `0.9-1.0ms` or `1.07-1.21x` figures into a new forecast; or
- promotion to `dev`/`exp`/`master`.

HARD STOP. The next implementation requires a separate, variant-specific scope with its
settling command, legacy hash controls, correctness pins, and fixed-depth wall gate.
