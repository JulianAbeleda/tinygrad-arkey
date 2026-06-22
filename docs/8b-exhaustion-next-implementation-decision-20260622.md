# 8B Decode Exhaustion — Next-Implementation Decision (2026-06-22)

## Verdict: **NEXT_IMPL_NORM_ROPE_KV**

Specifically: **eliminate the full-`max_context` KV-cache rematerialization copy** (`tinygrad/llm/model.py:952`).
This is the largest **bounded, transferable** remaining 8B decode opportunity (~1.2–1.4 ms/token, measured
transfer +1.5 ms / +8 tok/s), and it was **completely hidden** by a bucket mislabel in the time-tax diff.

Audit only — this doc names the failing layer and scopes the work; it does **not** implement it.

## How the audit changed the picture
The time-tax diff ranked the remaining gap as: (1) attention, (2) norm/rope/small-ops, (3) FFN activation.
Rendering the actual kernels (`bench/qk-decode-kernel-probe/latest.json`: AST fingerprints + source flags)
showed **two of those three were mislabels**:

| diff bucket | diff gap_ms @1024 | what it actually is | audit verdict |
|---|---|---|---|
| FFN activation (10–20×) | +1.21 | **KV-cache copy** (`E_49152`/`E_1536` are pure buffer moves; silu is fused into the gate/up GEMV) | `FFN_ACTIVATION_GAP_IS_MAPPING_ARTIFACT` |
| norm/rope/small ops (2.1×) | +1.79 | **~55% mislabeled** KV-proj + q8-quant; genuine norm/rope is −0.21 ms (parity) | `SMALL_OPS_BUCKET_MOSTLY_MISLABELED_GENUINE_NORM_NEAR_PARITY` |
| attention qk/softmax/pv (3–5×) | +1.64 → +2.64 | correctly mapped flash-decode; ctx-growing | `ATTENTION_BOUNDED_LEVER_EXHAUSTED_NO_REOPEN` |

The real silu activation is **fused** (`q4k_gemv_partial_12288_4096` has the exp); genuine RMSNorm is at/below
llama parity. So the "activation" and "norm/rope" lanes have **no bounded primitive** — and a previously
invisible **KV-cache copy** is the actual tax.

## Ranked candidates — by gap_ms, boundedness, transfer

| candidate | bounded gap_ms @1024 | transfer | bounded? | failing layer named | rank |
|---|---|---|---|---|---|
| **KV-cache rematerialization copy** | **~1.2–1.4 ms** (flat; grows with `max_context`) | **HIGH** (measured +1.5 ms/+8 tok/s via MAXC-shrink) | **MEDIUM** — named idiom fix, an in-place `.assign()` already exists (commented) | `model.py:952` `cache_kv.uop.after(slice.store(...))` | **1** |
| attention flash-decode | +1.6 → +2.6 ms (ctx-growing) | LOW (bounded lever exhausted) | **NO** (owned tile +5.7%@4096 < +7% gate; fused-flash codegen-blocked) | — | 2 (un-actionable) |
| FFN activation | ~0 (artifact; silu fused) | n/a | n/a | — | — |
| norm/rope (genuine) | −0.21 ms (parity) | n/a | n/a | — | — |

**Ranking by gap_ms alone** would pick attention (largest raw gap_ms), but attention's bounded lever is
exhausted (B5: overlap/off-critical-path, codegen-blocked deeper) — so its actionable gap_ms is ~0. The
**only** lane with a large gap_ms **and** a bounded transferable lever is the **KV-cache copy**.

## The chosen scope: KV-cache copy elimination

### Failing layer (named)
`tinygrad/llm/model.py:952` (upstream idiom, refactor #15780):
```python
assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).uop)))
k = assigned_kv[0, :, :, 0:start_pos+T, :]
v = assigned_kv[1, :, :, 0:start_pos+T, :]
#self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(...)   # <- in-place form, disabled (lines 956-958)
```
`.after()` is taken on the **full** `cache_kv` buffer, so the JIT materializes the entire MAXC×kv_dim
buffer (`E_49152`, ~1.4 ms/token @ MAXC 4608) every decode step — O(MAXC) where an in-place append into
the `[start_pos:start_pos+T]` slice is O(1). The in-place `.assign()` is commented out as a
`@function(precompile=True)` purity workaround.

### Why bounded (MEDIUM confidence)
- The store already targets only the new slice; only the `.after()`/read-back forces the full-buffer copy.
- An in-place form already exists (the disabled `.assign()`), i.e. a known intended fix — not an open-ended primitive.
- Lossless (greedy byte-identical expected — it's a data-movement reorganization, not a math change).
- Transfer is measured, not projected.

### Why not higher confidence
- The `.after()`-on-full-buffer is the upstream JIT idiom; making the write in-place under
  `@function(precompile=True)` purity is the reason the `.assign()` was disabled. The fix may be a local
  restructure (take `.after()` on the `[0:start_pos+T]` slice, not the full buffer) **or** may require a
  JIT/scheduler interaction — unknown until tried.

### Required first step of the implementation (audit-first discipline)
Before any default change, a **bounded tractability probe**: re-enable / restructure the in-place KV write
(slice-scoped `.after()` or `.assign()`), verify (a) greedy byte-identical decode, (b) `E_49152` shrinks to
O(1), (c) W==D wall transfer ≥ +5%@ctx1024 with no regression, (d) JIT capture is stable. If the probe fails
(purity/JIT blocker), the verdict downgrades to a deferred capability item — but the gap_ms (~1.4 ms,
the single largest bounded item) justifies the probe.

## Boundaries respected
Audited before naming the layer; ranked by gap_ms (not tinygrad share); did not reopen attention (B5
saturation explained: overlap + bounded-lever-exhausted); no defaults changed; stayed on 8B. The FFN-activation
and norm/rope lanes are **closed** (mapping artifact / parity).

## Artifacts
- `extra/qk_decode_audit_common.py` → `bench/qk-decode-kernel-probe/latest.json` (sources + AST fingerprints + timeline).
- Phase tools + bench: `qk_ffn_activation_gap_audit`, `qk_small_ops_time_tax_audit`, `qk_attention_tail_after_b5_audit`.
- Phase docs: `ffn-activation-gap-audit-{scope,result}`, `small-ops-time-tax-sub-audit-{scope,result}`,
  `attention-tail-after-b5-audit-{scope,result}` (all `-20260622`).
