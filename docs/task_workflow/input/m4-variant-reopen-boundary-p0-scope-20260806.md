# M4 variant-reopen boundary P0 scope - o-proj residual_add via typed input ABI

Date: 2026-08-06
Branch boundary: tinygrad `nvidia-bringup-20260731`
Status: **scope document only. Authorizes two test-first probes; no implementation,
no route-record change, no promotion.**

## 1. Why this P0 exists

The fusion A/B campaign (2026-08-06) proved both ordinary-UOp epilogue constructions
(norms, residual/cast/contiguous) are `CONSTRUCTION_GAP`: the consuming q4k GEMV is an
opaque custom program, so no ordinary in-core epilogue can fuse into it without a CUSTOM
boundary. The custom-boundary route already exists (`Q4KGEMVEpilogue`, M4 record) and was
closed because of two measured defects:

1. opaque-boundary copies: the boundary materializes each extra input as a separate buffer
   (`E_32_32_4_86a2`, 36x/token for the residual x input alone); and
2. ffn_down SiLU recompute: the fused prelude recomputes the activation per output row
   (3.74x per kernel, +1295 us/token mass).

This P0 reopens ONLY the o-proj residual_add variant (`kind="residual_add"`), the one the
amendment (`nv-campaign-forward-review-amendment-20260803.md` section 2.3) and the M4
decomposition record (`m4-decomposition-measurement-record-20260803.md` section 6) both
marked "eligible for a narrow boundary P0 and isolated measurement". It does NOT reopen the
ffn_down prelude (rejected until activation-once redesign) or the combined M4 record.

## 2. Boundary of record (amendment 2.2 item 1)

- Producer: the block input `x`, fp32 `(1,1,N)` = `(1,1,4096)`, the ordinary residual
  stream (embedding output or previous block output, `model.py` `FFNBlock._run`).
- Consumer: `q4k_g3_lanemap_gemv_epi_resadd_4096_4096` (route_role `attn_qo`, o-proj GEMV,
  `binding.N = binding.K = 4096`), epilogue `total + residual[row]`.
- Consumer chain today (`decode_routes.py:97-98`):
  `epi_inputs["residual"][:, 0, :].reshape(binding.N).cast(fp32)` passed through
  `execute_promoted_program`; `UOp.custom_kernel` `.contiguous()`s every non-AFTER arg
  (`tinygrad/uop/ops.py`), which materializes the boundary copy.
- Logical shape/dtype: producer writes fp32 `(4096,)` (flat, row-major, contiguous, offset 0);
  consumer reads flat fp32 `(N,)`. No layout change is asked for; only the materialization
  between them.
- Exact UOp chain with the variant open: `x (fp32 (1,1,4096))` -> `[:, 0, :]` (slice) ->
  `reshape(4096)` -> `cast(fp32)` (identity) -> boundary `.contiguous()` -> PARAM of the
  epi GEMV. The movement legs are pure; the copy is produced by the opaque boundary's
  preserve-or-materialize rule, exactly the Path-1 copy class.
- Unlike the M5 combine case, the producer is an ORDINARY tensor (not an opaque AFTER with a
  declared typed output). The M5 validator (`kernel_program.py:_validated_typed_view`)
  hard-requires `chain.base.op is Ops.AFTER`, `requires_combine_fusion`, and
  `route_role == "attn_qo"` on the O-PROJ ACTIVATION slot. **The current validator cannot
  fold the residual slot as written; extending it for an ordinary-producer fp32 residual
  slot is the object of probe 1.**

## 3. Copy class of record (amendment 2.2 item 2)

- Class: `E_32_32_4_86a2`, fp32->fp32 elementwise boundary copy (M4 record).
- Count: 36x/token (one per layer for the residual x input), d512 census.
- Median time: ~1.47 us/kernel (M4 census).
- Exists only with the variant open: baseline (gate closed) counts 0 of this class for the
  residual x input; variant open counts 36. Removing the copies is worth
  `36 x 1.47 = ~53 us/token` before any fused-kernel economics are considered.

## 4. Test-first probes (before any implementation)

The two load-bearing unknowns are tested BEFORE any production code changes:

### Probe 1 - ordinary-producer typed-view fold (CPU-only, hermetic)

Question: can the fp32 residual slot fold to a zero-copy view of the ordinary producer under
an extended typed-view contract, without touching the M5 combine contract?

Deliverables (owned by the probe, hermetic, no GPU):

- `extra/llm_research/decode/m4_residual_boundary_fold_probe.py`: builds the exact
  producer->consumer UOp chain from the real model (block input `x` -> slice -> reshape ->
  cast -> boundary), and evaluates an EXTENDED validator contract: pure movement chain,
  offset-0 row-major view, dtype/numel preserved, base is an ordinary producer with buffer
  identity (or a declared typed output), residual-slot opt-in gated to `attn_qo`
  `_epi_resadd`. Fail-closed: any mismatch rejects to the generic flat-buffer ABI.
- `test/unit/test_m4_residual_boundary_fold_probe.py`: hermetic assertions - fold fires for
  the real residual chain with zero materialization; fail-closed contrast without the ABI
  still materializes the copy; the M5 combine contract (route_role/slot/dtype/after-base)
  is NOT weakened (M5 tests must still pass).
- `docs/task_workflow/input/m4-residual-boundary-fold-probe-record-20260806.md`: record.

Gate: PASS requires the fold removes the copy class in the hermetic graph probe with zero
materialization, plus all existing M5 typed-boundary tests still passing. NO-GO is recorded
with evidence if the residual chain cannot be proven pure.

### Probe 2 - fused epi_resadd per-kernel economics (GPU, serialized)

Question: is `q4k_g3_lanemap_gemv_epi_resadd_4096_4096` economical per kernel vs the legacy
`q4k_g3_lanemap_gemv_4096_4096`, in isolation (one variant open)?

Deliverables (owned by the probe):

- `extra/llm_research/decode/m4_resadd_kernel_microgate.py`: same-session isolated
  one-variant-open census + per-kernel medians, M4 decomposition protocol
  (`/tmp/m4_decomp_probe.py` pattern: Qwen3-8B-Q4_K_M, d512, nmeas 20, reps 3, tokens pinned).
- `test/unit/test_m4_resadd_kernel_microgate.py`: hermetic assertions on the harness pieces.
- `docs/task_workflow/input/m4-resadd-kernel-microgate-record-20260806.md`: record.

Gate: fused kernel median must not regress the legacy per-kernel time materially
(the M4 isolated row measured +69 us/token total INCLUDING 36 copies; the copy-free ceiling
must be shown positive or at parity, not negative). All timing under
`flock -w 600 /tmp/gpu-bench.lock`; no promotion record touched.

## 5. Legacy route/hash byte-identity (amendment 2.2 item 4)

- Legacy `q4k_g3_lanemap_gemv_4096_4096`: rendered HIP sha `27857cb8ca03`, src_len unchanged;
  must remain byte-identical. The epi variant is a NEW kernel name, so no legacy hash moves.
- pg3 10-row baseline immutable; only additive rows may be introduced (the M4 epi kernel
  row already exists in the pg3 set if re-derived).
- Token sha `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9`, first token
  `151936`, decode sha `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`:
  all must hold at every depth and mode.

## 6. Fixed-depth wall and sha gate before any record change (amendment 2.2 item 5)

The `decode_q4k_epilogue_fusion` record may change (promoted targets gain NV sm_120 for the
residual_add variant ONLY) after ALL of the following on a same-session run:

- d512/d2048/d4096 wall, nmeas=20, reps=3, median tok/s, must not regress the M2-on baseline
  (172.80 / 161.50 / 149.00 tok/s) and must show a positive delta attributable to the
  copy-free residual_add fusion.
- M4-specific census assertion with the P0 active: `E_32_32_4_86a2` count 0 for the residual
  x slot, `E_32_32_4_02a` (residual add) count 0 for attn_qo, epi_resadd count 36,
  legacy attn_qo gemv count 0.
- All pins in section 5 hold 3/3 at every depth, both modes.

## 7. HARD STOP

This scope authorizes ONLY the two probes above. It authorizes no implementation, no
production code change to `kernel_program.py` / `decode_routes.py` / `decode_kernels.py`,
no route-record change, and no promotion. A successful probe pair does NOT automatically
open the record; the section 6 gate is a separate, subsequent decision. The ffn_down prelude
stays rejected (activation-once redesign required); the combined M4 record stays closed;
M3/M5/Path 3 stay closed. No GPU probe runs outside `flock -w 600 /tmp/gpu-bench.lock`.

## 8. References

- `m5-variant-reopen-boundary-p0-scope-20260803.md` (template: typed ABI contract)
- `m5-typed-boundary-p0-implementation-record-20260803.md` (landed substrate)
- `nv-campaign-forward-review-amendment-20260803.md` sections 2.2, 2.3, 4.1, 5
- `m4-decomposition-measurement-record-20260803.md` section 6 (per-piece verdicts)
- `m4-q4k-epilogue-measurement-record-20260802.md` (combined record, stays closed)
- `nv-fusion-exhaustive-scope-20260805.md` (fusion ledger authority)
- `nv-fusion-norms-ab-record-20260806.md` / `nv-fusion-residual-ab-record-20260806.md`
- `tinygrad/llm/kernel_program.py` (`_validated_typed_view`, `_fold_typed_input_views`)
- `tinygrad/llm/decode_routes.py` (`_Q4KDecodeCandidate.execute`, typed_input_views)
- `tinygrad/llm/decode_kernels.py` (`Q4KGEMVEpilogue`, residual_add body)
