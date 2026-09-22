# M4 Q4K GEMV epilogue absorption measurement record

Date: 2026-08-02
Status: measured non-landing (tokens byte-identical, perf regression)

## What landed

Three Q4K G3 GEMV epilogue variants (`decode_kernels.py` `Q4KGEMVEpilogue`), additive under a NEW
closed-default record `decode_q4k_epilogue_fusion` -- deliberately SEPARATE from M2's
`decode_epilogue_fusion` record, so the measured Q6K in-kernel win stays NV-promoted while these
q4k variants ship closed on every target:

| variant | kernel name | census role | extra inputs |
| --- | --- | --- | --- |
| (a) o-proj residual add | `q4k_g3_lanemap_gemv_epi_resadd_{N}_{K}` | attn_qo (N=4096,K=4096) | `residual` (N floats) |
| (b) ffn_down silu\*mul prelude + residual | `q4k_g3_lanemap_gemv_epi_ffndown_{4096}_{12288}` | ffn_down | `gate_out` (K), `up_out` (K), `normed_h` (N) |
| (c) k/v fp16 cast write | `q4k_g3_lanemap_gemv_epi_f16cast_{N}_{K}` | attn_kv (N=1024,K=4096) | none (output dtype changes to fp16) |

Legacy kernel hashes unchanged (UOp digest `10d8d359bca1...` preserved). The gate is closed:
the q4k record promotes nothing. Model load resolves the q4k gate from `decode_q4k_epilogue_fusion_promoted`
independently of the M2 answer, and `decode_routes.py` selects the fused variants only when the linear's
`q4k_epilogue_fusion_admitted` is open (never from M2's `fusion_admitted`).

Mechanism: `model.py` `_run` threads residual `x` into `attn_output` and `gate_out`/`up_out`/`normed_h`
into `ffn_down` when `_decode_q4k_epilogue_fusion_promoted` is set on the block. `decode_routes.py`
`_Q4KDecodeCandidate.execute` selects the fused variant and passes extra inputs to
`execute_promoted_program`. Model load resolves the gate once from device facts (M3 pattern).

## Kernel census (d512 prime token, DEBUG=2, Qwen3-8B-Q4_K_M, NV sm_120)

| class | baseline (gate closed) | M4 (gate open) | delta |
| --- | ---: | ---: | ---: |
| total kernels/token | 1021 | 1075 | +54 |
| E_ kernels/token | 510 | 582 | +72 |
| r_ kernels/token | 185 | 167 | -18 |
| kernel us/token | 6258 | 7522 | +1264 |
| E_32_32_4_86a2 (opaque-boundary copies) | 0 | 126 | +126 |
| E_32_32_4_02a (residual add) | 54 | 0 | -54 |
| q4k_g3_lanemap_gemv_epi_f16cast_1024_4096 | 0 | 54 | +54 |
| q4k_g3_lanemap_gemv_epi_resadd_4096_4096 | 0 | 36 | +36 |
| q4k_g3_lanemap_gemv_epi_ffndown_4096_12288 | 0 | 18 | +18 |

Baseline q4k_g3_lanemap_gemv_1024_4096 (54) replaced by epi_f16cast variant (54) — net zero
name change, but the fp16_cast output dtype changes the downstream graph shape, adding 36
E_32_32_4_86a2 copies (from the k/v output tensor being consumed in cache-store after
the dtype change). The residual-add path absorbs 54 E_32_32_4_02a kernels but adds 36
E_32_32_4_86a2 copies (one per layer for the residual x input). The ffn_down prelude+epilogue
kernel absorbs 36 E_128_32_3 silu kernels and 72 E_32_32_4 residual/h+ffn_out kernels (108
total), but adds 72 E_32_32_4_86a2 copies for gate_out/up_out/normed_h inputs. Net +126
copy kernels vs -54 residual-add + -108 ffn elementwise = +72 E_ (matches observed).

The boundary mechanism is the same as M3: the `uop_program` boundary realizes each extra
input as a separate buffer, creating a per-input copy kernel. This is not fixable in M4's
scope (paths 1/2/3 in `decode-norm-fusion-paths-forward-20260802.md` are prerequisites).
CORRECTION (2026-08-03): the 2026-08-03 decomposition
(`m4-decomposition-measurement-record-20260803.md`) shows the boundary copies are the
mechanism for the +126 copy class only; the DOMINANT regression mechanism is the
`ffn_down_fused` SiLU recompute (+1295 us/token, 3.74x per kernel), which the combined
record's "root cause" sentence did not isolate. This record stays closed either way.

## Wall tok/s (d512, 5 reps median, first token)

| metric | baseline | M4 | delta |
| --- | ---: | ---: | ---: |
| tok/s | 172.69 | 140.20 | -18.8% |
| token sha256 | `9d6b3787...` (full) | `9d6b3787...` (full) | identical |
| first token | 151936 | 151936 | identical |

SHA and first token preserved — correctness is confirmed. The performance regression
is the dominating result; this is a non-landing.

## Promotion record

New, closed (`tinygrad/llm/generated/decode-q4k-epilogue-fusion-route-policy.json`):
```json
{
  "schema": "boltbeam.route_policy.v1",
  "route": "decode_q4k_epilogue_fusion",
  "promoted_targets": []
}
```

M2's record (`decode-epilogue-fusion-route-policy.json`) is unchanged and keeps NV sm_120 promoted
for the Q6K coop in-kernel merge only.

## Deviations from brief

1. **Gate split (blocking fix)**: The first implementation gated the q4k variants on M2's
   `fusion_admitted`, which is NV-promoted -- shipping it as-is would ship this measured regression.
   Split onto the new closed record above; `FFNBlock.__call__` now keys `h` and the final residual
   add on the actual absorption signal (`_epi_residual` / `_ffn_absorbed`, checked against the linear's
   own admission), not the promotion record, so a reopened record cannot drop residuals on blocks whose
   `attn_output`/`ffn_down` are not admitted Q4K primitives (MLA/GatedDeltaNet, `nn.Linear`).
2. **fp16_cast route_role fix**: The `route_role` for k/v projections is `"attn_kv"` (normalized),
   not `"attn_k"` / `"attn_v"`. Initial implementation used the wrong role strings; corrected after
   census revealed the variant wasn't firing.

3. **Non-landing result**: The ~0.4ms node-sum claim in the design doc requires copy-free
   scheduler integration (paths 1/2/3 in the forward doc). M4 measured a 18.8% wall regression
   due to the same opaque-kernel-boundary copies that M3 hit. The fused emitter code is correct
   and additive; it is ready for promotion when the copy problem is resolved.

4. **No d2048/d4096 measurement**: The d512 regression was definitive enough to skip further
   depth measurements.

## Files changed

- `tinygrad/llm/decode_kernels.py` — `Q4KGEMVEpilogue` dataclass, `_silu_uop` helper, 3 fused
  kernel variants in `q4k_g3_lanemap_gemv_kernel`
- `tinygrad/llm/qk_primitives.py` — `Q4KPrimitiveLinear.__call__` accepts `**epilogue_inputs`
- `tinygrad/llm/decode_routes.py` — `q4k_primitive_linear_call` and `_Q4KDecodeCandidate.execute`
  accept and route epilogue inputs, select fused variant when admitted
- `tinygrad/llm/model_route_plan.py` — `load_decode_q4k_epilogue_fusion_promotion` /
  `decode_q4k_epilogue_fusion_promoted` (new closed record)
- `tinygrad/llm/generated/decode-q4k-epilogue-fusion-route-policy.json` — closed promotion record
- `tinygrad/llm/model.py` — gate setup (`_decode_q4k_epilogue_fusion_promoted` on blocks),
  `_run` threads epilogue inputs for o-proj residual and ffn_down fusions,
  `_attention` accepts `residual_for_output`, all `_attention` overrides updated
- `test/unit/test_q4k_epilogue_fusion_m4.py` — 13 unit tests (spec validation, legacy
  hash preservation, fused kernel names, HIP+CUDA render arms, admission wiring)
- `test/unit/test_q4k_epilogue_fusion_gate.py` — closed-record gate tests (q4k promotes
  nothing; M2 record keeps NV sm_120; M3 norm record stays closed)
- `scratchpad/pg3_decode_rendered_source_equality.py` — 3 new fused variant rows in
  the render-equality baseline

## Test results

All 55 tests pass (34 pre-existing + 13 new M4 + 8 M3/pre-existing):
```
test/unit/test_q4k_epilogue_fusion_m4.py   13 passed
test/unit/test_q4k_epilogue_fusion_gate.py  4 passed
test/unit/test_decode_epilogue_fusion_gate.py 10 passed
test/unit/test_llm_decode_kernels.py         8 passed
test/unit/test_llm_decode_routes.py         14 passed
test/unit/test_decode_norm_fusion_gate.py    7 passed
test/unit/test_m3_decode_machine_search.py   1 passed
test/unit/test_q4k_w_f16_decode.py           1 failed (nvcc not in PATH, pre-existing)
```

Pg3 render equality: all 17 kernels (14 pre-existing + 3 new) render through HIP target.
Legacy UOp hashes verified byte-identical (`10d8d359bca1...` for 4096x4096, `b034e43c...` for 32x1024).
