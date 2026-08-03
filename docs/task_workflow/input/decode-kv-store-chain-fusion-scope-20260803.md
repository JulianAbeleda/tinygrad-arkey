# Decode kv-store chain fusion scope

Date: 2026-08-03
Status: implementation scope, authorized by `nv-decode-gap-decomposition-record-20260803.md`
(section 5, lever 1). HARD STOP after implementation + same-session measurement; no
promotion to `exp`/`dev`/`master` and no composed performance endpoint.
Branch: tinygrad `nvidia-bringup-20260731`, HEAD `894c08c48`.

## 1. Evidence (OBSERVED, same session as this scope)

The decode kv-store chain (`model.py:640-700`) materializes 270 kernels / 568 us per
token (7.5 per layer, 36 layers): q-rope, k-rope, k/v casts, layout reshapes,
`Tensor.stack(k, v)`, and the cache `uop.store` are each separate kernels. The k/v
store side (k-rope, k-cast, v-cast, stack, store) is ~5 kernels / ~380 us per token;
q-rope + q-cast is the remaining ~2.5 kernels/layer and is OUT OF SCOPE here.

Same-session d512 baseline (as-committed w1w3 scalar): 177.2 tok/s ours vs 251.8
llama; 948 kernels / 6021 us; GEMV-class 3.775 ms; pins token sha
`9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9`, first token
151936. See the decomposition record section 2/4 for the full table.

## 2. Design: Option A, one fused kv-store kernel

New kernel `decode_kv_rope_store_kernel` (emitted by `decode_kernels.py`, route
selection in `decode_routes.py` / `model.py` kv path): inputs are the fp32 k and v
GEMV outputs (1024,) plus the freqs table and start_pos; the kernel applies k-rope
in-kernel (fp32, same `apply_rope` arithmetic), casts k and v to fp16, and writes
both directly into `cache_kv` at slot `start_pos`. It replaces the k-rope + k-cast +
v-cast + `Tensor.stack(k, v)` + store chain per layer.

Why this shape:

- It is downstream of BOTH the q4k and q6k k/v GEMV families (54x q4k + 18x q6k
  k/v GEMVs per token, mixed quant), so ONE kernel covers both families.
- The alternative (Option B, store epilogue fused into the GEMV emitters) needs
  edits in two emitters and repeats M4's boundary-copy risk; rejected as primary.
- The kernel is a pure elementwise rope+cast+store over 2048 fp16 outputs; no
  reduction, no shared memory, no cross-lane communication.

Expected outcome (INFERRED from the OBSERVED chain): ~180 kernels removed (~380 us
of the 568 us chain), kernels/token 948 -> ~770, +5-7% wall at d512. The conversion
is expected to be near 1:1 because the chain is sequential and not overlapped, but
that is an expectation to be MEASURED, not claimed.

### 2.1 Kernel contract

- Inputs: `k` fp32 (B*Hkv*Hd = 1024,), `v` fp32 (1024,), `freqs` fp32
  (rope_dim, MAXC) precomputed table, `start_pos` (runtime scalar via the decode
  graph's `start_pos` variable).
- Output: writes `cache_kv[0, :, :, start_pos, :]` (roped k, fp16) and
  `cache_kv[1, :, :, start_pos, :]` (v, fp16); returns the cache tensor AFTER the
  store so the flash route reads the written bytes.
- The store is issued through the existing custom-kernel boundary
  (`Tensor.uop_program` / `UOp.custom_kernel`) exactly like the w1w3 fused GEMV;
  the fused program's output is the cache slice (slot 0), inputs are k, v, freqs.

### 2.2 Admission (closed default)

- New route record `generated/decode-kv-store-fusion-route-policy.json`
  (boltbeam.route_policy.v1), CLOSED default: no `promoted_targets` key or empty
  list promotes nothing. The checked-in record promotes NOTHING until the
  same-session measurement lands; a follow-up commit opens it for the measured
  target only.
- Loader `load_decode_kv_store_fusion_promotion(path)` +
  `decode_kv_store_fusion_promoted(target)` in `model_route_plan.py`, same schema
  family as the w1w3 loader; absence of the key or an empty list returns
  `frozenset()`.
- The model resolves the flag ONCE at load (same pattern as
  `_decode_q4k_w1w3_fusion_promoted`, `model.py:1412`); blocks read their own copy.
- The fused store fires only when ALL hold: decode (T==1), B==1, fp16 cache dtype,
  not kv_quant, not rope-at-read (ring), and the route record promotes the target.
  Any miss falls back to the legacy chain byte-for-byte.

## 3. Files touched

- `tinygrad/llm/decode_kernels.py`: new `decode_kv_rope_store_kernel` builder
  (+~60 lines, additive only, legacy emitters untouched).
- `tinygrad/llm/decode_routes.py`: new `decode_kv_rope_store_route(...)` selecting
  the program when admitted; fallback = legacy graph unchanged.
- `tinygrad/llm/model.py`: fused branch in the non-quant store site
  (`model.py:685` region) behind the resolve-once block flag.
- `tinygrad/llm/model_route_plan.py`: loader + promotion predicate.
- `tinygrad/llm/generated/decode-kv-store-fusion-route-policy.json`: CLOSED record.
- `test/unit/test_decode_kv_store_fusion_gate.py`: gate tests (new, ~8 tests).

## 4. Gates (landing requires all)

1. `pytest test/unit/test_decode_kv_store_fusion_gate.py` green (CPU-only).
2. pg3 decode render-equality (`scratchpad/pg3_decode_rendered_source_equality.py`)
   byte-identical to the pinned `/tmp/pg3.log` table; the new kernel is additive
   (new row, legacy hashes untouched).
3. Pins 3/3 in BOTH arms of the same-session A/B: token sha
   `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9`, first token
   151936, census row present.
4. Same-session interleaved wall A/B at d512 (arms: legacy closed vs fused active,
   interleaved, census harness); fused must not regress and should land inside the
   expected +5-7% band.
5. `git diff --check` clean; `sz.py` budget respected (40143/100000).

## 5. HARD STOP conditions

- Any pin moves (token sha or first token differ) in the fused arm: STOP, do not
  promote, report the delta and the exact diff.
- The fused arm regresses wall vs legacy closed: STOP, do not promote.
- The fused kernel's rendered source differs on AMD/Metal (render-only control):
  STOP; the kernel must be target-agnostic by construction (elementwise, no
  target-specific intrinsic).

## 6. Out of scope (do NOT do)

- No q-rope/q-cast fusion (follow-up).
- No flash score/combine changes, no rmsnorm changes.
- No dtype or precision changes anywhere; no `dtypes.*` literal edits.
- No prefill changes of any kind (separate scope exists).
- No AMD/Metal promotion: they need their own measured record plus pg3 parity.
- No BOUNDED_PACKED_TILES, no new subsystem, no prefill_routes.py edits.
- No `[refactor]` commits; prefixes are `[nn]` (kernel/route/model), `[test]`
  (tests), `[docs]` (this scope + measurement record). One owning prefix per commit.

## 7. Deliverable

Implementation + measurement record with the same-session A/B table (both arms),
kernels/token delta, and pin rows, following the house report format (commits,
pytest, e2e, deviations, blocked on).
