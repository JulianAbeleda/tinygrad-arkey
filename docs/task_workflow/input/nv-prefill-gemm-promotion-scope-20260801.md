# NV prefill GEMM promotion - scope

Date: 2026-08-01

Status: scoped, not implemented. Branch boundary: tinygrad `nvidia-bringup-20260731`. Does not
authorize promotion to `dev`/`master`.

Motivation and measurements: `docs/nv-prefill-decode-diagnosis-20260801.md`. Short version: NV
prefill runs at ~1% of its compute roofline because the dense Q4_K GEMMs fall through to
scalarized CUDA-core kernels (no `mma.sync`/`dp4a`/`half2`), while AMD runs the same shapes on
promoted BoltBeam WMMA full-kernel candidates. The sm120 candidate set exists and now compiles
on the 5090 (C5, `948b26318`). This scope promotes it into the production graph-GEMM path.

## 1. Established state

- The production promoted artifact is
  `tinygrad/llm/generated/prefill_wmma_lds_dbuf_candidate_set.json`, and
  `promoted_candidate_set()` (`tinygrad/llm/prefill_candidate_runtime.py:157`) is pinned to
  `AMD gfx1100 wave32` via `_PINNED_COMPACT_ARTIFACT_TARGET`; any other target raises "compact
  target is unsupported". `automatic_promoted_prefill_graph_policy` therefore returns None on NV
  and the admission census records `prefill_overlay_promotion: "no-promoted-candidate"`.
- The NV candidate set exists at
  `bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-sm120-v1/candidate-set.json`
  (schema `boltbeam.full_kernel_candidate_set.v1`): 4 entries covering `attn_kv`, `attn_qo`,
  `ffn_down`, `ffn_gate_up`, all m=512 fp16 (`cuda_mma_f32_8x16x16_f16_lds2_static`,
  `wmma_f32_8x16x16_f16`, tile 128x128x32, threads 256, buffer2). It is not the promoted
  artifact.
- C5 (`948b26318`, `[codegen]`) fixed the two lowering gates: the accumulator carrier
  `vec(8)` vs `vec(4)` mismatch (`kernel_pipeline.py:181`) and the CUDA operand lane-layout
  derivation (`kernel_lds.py:171`). Both NV buffer shapes admit, compile, dispatch, and measure
  `max_abs_error 0.0` on the 5090 through `scratchpad/c5_nv_canonical_lane_probe.py`.
- The production consumption chain is already generic: `_graph_gemm_registry` /
  `_graph_gemm_binding` (`tinygrad/llm/model.py:275,282`) decode a policy's
  `graph_gemm.candidate_set` via `decode_prefill_graph_candidate_set` and bind exact rows to
  linears; `route_pf16_graph_gemm` (`tinygrad/llm/prefill_graph_gemm.py:140`) runs the attached
  candidate through `_install_candidate_matmul` (fp16 overlay, TC warmstart). The bench row
  (`extra/llm/bench/model_e2e_bench.py:170`) already surfaces
  `prefill_overlay_promotion` from `model.config.admit`.

## 2. The work - four pieces, each ending at a HARD STOP

### P0 - Mint the NV compact promoted artifact

Produce a checked-in NV compact artifact (same schema and layout as the AMD one: `schema`,
`route_id`, `candidate_set_identity`, `profile`, `target`, `template`, 4 `entries` with
`canonical_identity`/`legacy_identity`) whose `target` is `CUDA sm_120 wave32` and whose
template is the sm120 typed schedule. The template derivation exists:
`extra/llm_research/mint_typed_candidate_template.py` (T6, `derive_target_schedule` with
`NV_SM120_TWO_BUFFER_STAGE1_CAPABILITY`). Use the mint, do not hand-edit a JSON.

Deliverable: artifact file in `tinygrad/llm/generated/` with an exact `candidate_set_identity`
computed by `canonical_candidate_set_identity`, and a unit test that loads it through
`decode_prefill_graph_candidate_set` and admits all 4 roles with exact-shape match for
`(512, 1024, 4096)`, `(512, 4096, 4096)`, `(512, 4096, 12288)`, `(512, 12288, 4096)`.

### P1 - Make promotion selection target-parametric

`promoted_candidate_set()` is `@cache`d on zero args and pins one target. Promotion must select
by live scanned facts (`backend`/`architecture`/`wave_size`), so NV can resolve the NV artifact
while AMD still resolves the AMD artifact byte-identically. Keep the fail-loud guard: an
artifact whose `target` disagrees with its schedule must raise, never silently admit. Do not add
an `if backend == "NV"` branch; make it a data lookup over declared per-artifact targets
(the TG8 shape: both operands are data).

Deliverable: `automatic_promoted_prefill_graph_policy` returns a `FULL_RESIDENT_OVERLAY` policy
on NV for Qwen3-8B with the 4 roles bound to the NV canonical identities, and still returns the
AMD policy on AMD (six-route rendered-source hashes unmoved, pg2 control re-run).

### P2 - Census and bench row

On NV the admission census must stop saying `no-promoted-candidate` and instead report the
promoted candidate/route id, in BOTH `model.config.admit` and the e2e bench row. Keep the S4
labeled fallback for a registry miss (no promoted candidate on an expressible target), but the
success path must be labeled too: e.g. `prefill_overlay_promotion: <candidate_set_identity>`.
Update/extend the admission consistency unit test to cover the NV success row.

### P3 - Correctness and measurement gate

1. NV 8B e2e (`extra/llm/bench/model_e2e_bench.py`, same command as `/tmp/qwen3-8b-p5-final2.json`
   provenance): first-token digits must be unchanged (`50994, 82, 31109, ...`), decode token
   sha256 unchanged (`0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`).
2. AMD control re-run on the same commit: prefill strategy/census and first tokens unchanged.
3. Measure prefill pp512 head-to-head (same session, warm steady-state, PROFILE=0). Report
   tok/s, ttft, census, and the per-kernel mix (fused attention vs GEMM routes) before and
   after.

If digits move, STOP and report the delta and the exact diff, per the standing rule.

## 3. Guardrails

- No commits to `master`/`dev`/`exp`; all work on `nvidia-bringup-20260731`. Prefixes
  `[nn]`/`[test]`/`[docs]` only, one owning prefix per commit, never mix NFC with functional.
- No `if backend == "NV"` branches in lowering; data lookups only (declared per-target facts).
- No touching `tinygrad/llm/prefill_routes.py` or the per-call dispatch path (parked scope).
- No dtype/precision cleanup of any kind (separate parked scope:
  `dtype-authority-decomposition-scope-20260731.md`, explicitly not scheduled).
- Do not revert `948b26318` or the C5 resolution; it is the load-bearing lowering fix.
- 5090 is shared: serialize GPU work, keep runs bounded, no background processes left behind.
- AMD control re-run after every commit that touches promotion/admission.

## 4. Deliverable + HARD STOP

NV 8B prefill e2e with the promoted sm120 candidates active: census shows the promoted
candidate in the admission report and bench row, first tokens unchanged, AMD control unchanged,
and a measured head-to-head prefill number reported with the kernel mix. Hard stop after P3's
report for review; nothing beyond.

## 5. One-line job

Promote the already-compiling sm120 candidate set into the production graph-GEMM path via a
target-parametric artifact selection, then prove NV prefill correctness and measure it.
