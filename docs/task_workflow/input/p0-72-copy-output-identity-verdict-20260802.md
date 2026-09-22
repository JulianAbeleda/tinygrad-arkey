# P0 verdict - 72-copy output identity investigation

Date: 2026-08-02

Status: verdict issued. Analysis performed by an investigation agent and independently
re-verified against the code before this document was written; the missing-deliverable
lesson is recorded in section 5. Branch boundary: tinygrad `nvidia-bringup-20260731`.
No code changes were made by this scope.

## 1. The question

The M3 fused-decode-RMSNorm run adds 72 output materialization kernels per token to the
flash decode graph (2 x 36: the attn and ffn layer norms), with the trace-order signature
`copy -> decode_rmsnorm -> copy`. This P0 asks: where do the 72 come from, who owns the
copy, can they be removed without moving the token sha256, and is removal an independent,
well-scoped piece of work?

## 2. Verified mechanism facts (all re-checked in code)

1. `UOp.has_buffer_identity` (tinygrad/uop/ops.py:995-999) walks `RESHAPE`/`MULTI` and
   `GETTUPLE(TUPLE)` to a base of `BUFFER`/`SLICE`/`PARAM`. It does NOT follow `AFTER`.
   An opaque-kernel output that is later reshaped therefore does NOT have buffer identity
   through the reshape.
2. `UOp.custom_kernel` (tinygrad/uop/ops.py:1260-1271) preserves an argument only when it
   is exactly `Ops.AFTER` or a `MEMORY_SEMANTIC` whose base has buffer identity; every
   other argument (including `RESHAPE(AFTER)`) is passed through `.contiguous()`, which is
   a materialization site.
3. Every Q4_K decode GEMV route rebuilds its activation input as
   `x[:, 0, :].reshape(K).cast(dtypes.float16).contiguous()` (tinygrad/llm/decode_routes.py:78),
   so the consumer side owns a per-route contiguous request on the norm output it receives.
4. The 72 kernels in the fused trace are exactly the attn+ffn norm outputs (2 x 36); the
   q/k norm outputs (36 + 36) reshape to `(1,32,1,128)`/`(1,8,1,128)` and show no companion
   materialization (m3 census: `E_32_32_4_3b0fcfbc...` x72 at 1.54us median =
   ~110.9us node-sum; `decode_rmsnorm_1_4096` x72 at 4.96us).

## 3. Verdict

GO - the 72-copy removal is a scoped, independent P0, with two conditions:

1. The copies are consumer-owned today (`decode_routes.py:78` contiguous-on-reshape plus
   `custom_kernel`'s preserve-or-materialize rule), so the fix belongs to the transport/ABI
   layer, not to the norm emitter. Candidate fix shapes to evaluate in the P0 itself:
   (a) emit the fused norm output already laid out and dtyped for the GEMV input contract
   (data-driven output-spec change; zero consumer changes), or (b) an opt-in identity
   preservation path in `custom_kernel` for single-consumer reshape-of-AFTER arguments.
   Shape (a) is preferred because it does not touch the default flat-buffer contract.
2. The P0 must be gated and measured exactly like M3: new variant name/hash, legacy paths
   byte-identical, fixed-depth sha256 discipline, and a fused-trace census before/after.
   If the removal changes the token sha256, STOP and report the delta; the fused path's
   first-token digits must match the baseline `151936`.

If the P0 removes the 72 (~110.9us node-sum), the M3 planning basis improves from
`-144 launches / ~-0.16ms node-sum` to `-216 launches / ~-281us node-sum` (72 fewer
kernels at ~1.54us each). That materially changes whether the M3 story can beat wall time;
the P0 verdict lands BEFORE Path 3 sequencing is finalized, per the review amendment
(decode-norm-fusion-paths-forward-20260802.md section 10.3).

## 4. What this does NOT authorize

- No transport/ABI implementation yet - this P0 only names the candidate shapes and the
  gate. Implementation is a separate scope.
- No M3 reopen. The 72-copy fix alone must not flip M3 on without the full fixed-depth
  protocol beating M2 in wall time.

## 5. Process note

The investigation agent confirmed the provenance (consumer `.contiguous()` at
decode_routes.py:78; `has_buffer_identity` not following `AFTER`) but its verdict document
was never written (agent reported completion without the file landing). Every claim above
was therefore re-verified directly before this document was written. For the remaining
agents in this campaign: deliverable-first, write the doc incrementally.

