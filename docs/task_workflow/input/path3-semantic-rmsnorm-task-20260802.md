# Path 3 - semantic RMSNorm lowering (task handoff)

Status: scoped, awaiting M5 review completion. Implements the reviewer-approved semantic
RMSNorm contract (`decode-norm-fusion-paths-forward-20260802.md` section 9.3, accepted in
section 10.2). M3's opaque fused-norm emitter stays closed and untouched; this is a different
mechanism with its own closed record.

## 0. HARD BANS

- No decode- or model-name-specialized rule in global `jit_lower`/rangeify. Admission is keyed
  to target/shape/dtype/capability records only.
- Do not touch M2's `decode-epilogue-fusion-route-policy.json` (NV sm_120) or M4's
  `decode-q4k-epilogue-fusion-route-policy.json` (closed). Do not touch
  `tinygrad/llm/flash_decode_attention.py` or M5's record if it has landed.
- Do not reopen M3's `decode_norm_fusion` record, and do not delete the M3 opaque emitter.
- No prefill promotion from decode evidence. Decode and prefill are separate admission classes.
- No dtype/precision cleanup outside the semantic's own contract.
- Do not reverse-match generic ADD reductions (rangeify's attention-lowering rule).

## 1. Established state (verified facts)

- `nn.RMSNorm` is the ordinary `square -> mean -> rsqrt -> multiply` expression with no
  semantic today. Verified: no RMSNorm semantic op exists.
- The repo precedent: `Ops.ATTENTION` semantic with a fail-closed lowering
  (`tensor.py _semantic_attention`, `rangeify.py lower_attention_semantic` + PatternMatcher).
  The marker's first source is the ordinary graph; an unadmitted lowering returns the source
  unchanged. Path 3 mirrors this exactly.
- M3's opaque fused norm measured non-landing (whole-decode `1021 -> 1093` kernels, +142us):
  the opaque boundary materialized 144 input copies + 72 output materializations. The 72-copy
  P0 verdict (`p0-72-copy-output-identity-verdict-20260802.md`) is consumer-owned: the
  boundary's flat output + `RESHAPE(AFTER)` identity gap plus downstream `.contiguous()`.
- llama's graph has one generic kernel per norm (`rms_norm_f32`, 145 nodes, 1.3-3.4us) - the
  shape Path 3 produces: ONE scheduler kernel, no boundary, no copies.
- In-kernel reduction machinery exists: M2's q6k coop in-kernel merge uses the same
  staged-shfl + smem-barrier building blocks Path 3 needs.

## 2. The work

1. Add an RMSNorm semantic op carrying reduction axis, epsilon, input/output dtype, and
   optional affine weight. Create it from `nn.RMSNorm.__call__` (or a `_semantic_rmsnorm`
   helper mirroring `_semantic_attention`), with the ordinary expression as the marker's
   first source (the universal fallback).
2. Add `lower_rmsnorm_semantic` + a PatternMatcher in `rangeify.py` mirroring the attention
   precedent: for admitted shapes/targets it lowers to ONE scheduler-owned kernel whose
   reduction result feeds its epilogue in-kernel; for everything else it returns the source
   unchanged (fail-closed).
3. Admission: a new closed `boltbeam.route_policy.v1` record (e.g.
   `decode-rmsnorm-native-lowering-route-policy.json`, `promoted_targets: []`), resolved from
   device facts like the M3/M4 gates. Separate decode and prefill records; decode evidence
   never authorizes prefill.
4. Shared semantic tests: the fallback and native lowering must satisfy the same reference
   values (isolation parity), so they cannot silently become two definitions of RMSNorm.
5. Reduction-order parity gate: fixed-depth token sha (pin `9d6b3787...`, first token
   `151936` on Qwen3-8B Q4_K_M) must hold with the native lowering forced open.

## 3. Measurement protocol (before any promotion decision)

Fixed-depth protocol at d512/d2048/d4096: census (norm family must become ~145 kernels with
NO copy/materialization kernels around the norms; the P0's `E_32_32_4_3b0f` count must be 0 on
the native path), kernel-us node-sum, wall tok/s vs the M2-on baseline (d512: 1021 kernels,
~6179us, ~173 tok/s), token sha pins. Compare against llama's recorded d512/d2048/d4096
numbers (`nv-performance-campaign-scope-20260801.md`). The record reopens only when the full
fixed-depth protocol beats M2; a norm-family-only win is not a promotion.

## 4. Deliverable

- `[nn]` semantic op + fail-closed lowering + closed record + model wiring, default runtime
  byte-identical (prove with the fixed-depth sha).
- `[test]` shared semantic parity tests + gate tests (closed default, loader, explicit-target
  naming).
- `[docs]` measurement record with the census and wall tables and the verdict
  (`path3-semantic-rmsnorm-measurement-record-20260802.md`).
- Force-open measurement probe (monkeypatched record, pattern of `/tmp/m3_census_probe.py`).
- Report: commits, pytest, census default vs forced-open, wall tok/s, sha, verdict,
  deviations, blocked on.

## 5. One-line job

Give RMSNorm the attention-semantic treatment: one semantic op, fail-closed scheduler-native
lowering, closed per-target record, shared parity tests, measured at d512/d2048/d4096 before
any promotion.
