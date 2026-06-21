# Decode Fusion Build — Final Report

Date: 2026-06-20

Executor: Claude

Scope: `docs/decode-fusion-build-scope-20260620.md`

Outcome: **one custom fusion candidate built and gated (Phase B1), with a decisive negative result that reframes
both phases.** Phase B1 fuses the FFN `silu(gate)*up` into the up GEMV — **byte-exact** but **0% faster**, because
decode fusion **conserves work** (the activation was real work, not recoverable launch overhead). That mechanism,
plus a kernel-shape reframe of the attention split and the documented flash linearizer wall, makes Phase A a
**no-go**. **No decode default changed.** This meets the scope's minimum success ("one custom fusion candidate
built and locally gated, even if it fails, with clear evidence").

## 1. Attention fusion candidate results (Phase A)

**Not built — evidence-based no-go** (`docs/decode-attention-fusion-analysis-result-20260620.md`). The cost-split
data reframes the target:

- `reduce_fixup` (the dominant bucket) is the **Q@Kᵀ score reduction** (`r_*`), and its ctx-slope is the O(KV)
  symbolic-length score reduce (`r_2_…start_pos…`: 0.32 → 1.10 ms @512→4096). Intrinsic compute, not fusible fixup.
- `softmax_stats`' slope is `flash_prob` (O(KV) exp/key: 0.19 → 0.71 ms). Intrinsic.
- Only ~0.5 ms of *fixed* helper kernels (`flash_max/den/gmax`) are fusible — and Phase B proved fusing fixed
  overhead is **work-conserved** (no win).
- The real lever (fully-fused Q@Kᵀ+softmax+P@V) is **linearizer-walled** by design (`qk_flash_decode.py:73-80`).

## 2. FFN activation fusion candidate results (Phase B1) — BUILT

`docs/decode-ffn-activation-producer-fusion-result-20260620.md`. Built a fused up-GEMV that writes
`silu(gate[row]) * (Σ w·x)` directly (two kernel variants: REG/flattened and buffer-accumulator/scratch).

| variant | opts | correctness | baseline µs | fused µs | delta |
|---|---|---:|---:|---:|---:|
| v1 (REG, flattened) | LOCAL:0:64 | byte-exact (rel 0.0) | 172 | 171 | +0.3% |
| v2 (buffer-accum, scratch) | LOCAL:0:64 | byte-exact (rel 0.0) | 168 | 167 | +0.3% |
| v2 | LOCAL:0:128 | byte-exact | 168 | 168 | −0.1% |

`E_49152` is eliminated; net speedup **~0%**. The activation work moves into the GEMV and runs serially for the
same ~33 µs.

## 3. W==D full-route timing for passing candidates

None. No candidate cleared its **local** gate (Phase B1 local ~0%; Phase A not built), so no candidate was
promoted to full W==D. The local A/Bs used clock-pinned same-process interleaved timing (diagnostic, not a product
benchmark, per scope policy 4).

## 4. Correctness / quality status

Phase B1 is **byte-identical** to the baseline `silu(gate)*up` (max|abs diff = 0.0; the UOp
`silu = g/(1+exp(-g))` matches `Tensor.silu()` bit-for-bit on the tested weights). No quality regression — but
also no speedup. No default changed, so production decode output is unaffected.

## 5. Stacked route timing or reason not stacked

**Not stacked.** Stacking requires ≥1 attention and ≥1 FFN candidate to pass local gates (scope Phase C). Neither
passed: Phase A was not built (no-go), Phase B1 was ~0%. There is nothing to stack.

## 6. Lifecycle-search encoding status (Phase D) — DONE

Encoded into `bench/qk-lifecycle-search/`:
- Candidates (`generated_candidates.json`): `decode_ffn_activation_producer_fusion` (state `refuted`),
  `decode_attention_reduce_stat_fusion` (state `not_built_no_go`).
- Refutations (`refutations.json`): `ffn_activation_fusion_work_conserved`,
  `attention_dominant_cost_is_intrinsic_okv`, `flash_fused_multireduce_linearizer_wall`, `FLASH_L_256_512`,
  `FLASH_DECODE_0_SDPA`, `remove_contiguous_no_target_movement`.

## 7. Exact commands

```bash
# Phase B1 FFN activation fusion A/B (built, byte-exact, ~0% — work conserved)
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_ffn_activation_producer_fusion_ab.py
# Cost splits + cheap-candidate refutations (prior scope, reused as evidence)
PYTHONPATH=. python3 extra/qk_decode_attention_cost_split.py --modes baseline,q8 --ckpts 512 1024 2048 4096 ...
```

## 8. Artifact paths

- Kernels: `extra/q4_k_gemv_primitive.py` (`q4k_gemv_silu_gate_kernel`, `q4k_gemv_silu_gate_v2_kernel` — research,
  not wired into the model).
- Harness: `extra/qk_decode_ffn_activation_producer_fusion_ab.py`,
  `bench/qk-decode-fusion-build/ffn_activation_producer_fusion_ab.json`.
- Docs: `docs/decode-ffn-activation-producer-fusion-result-20260620.md`,
  `docs/decode-attention-fusion-analysis-result-20260620.md`, this report.
- Lifecycle search: `bench/qk-lifecycle-search/{generated_candidates,refutations}.json`.

## 9. Default behavior changed: **NO**

`tinygrad/llm/model.py` is unmodified (`git diff` empty). The new fused kernels live in `extra/` and are never
routed in-model. GPU perf-state pinned only for measurement windows; `auto` restored after (verified).

## 10. Recommendation: **STOP the bounded fusion builds; the recoverable gap requires deeper codegen**

The decisive, reproduced Phase B1 result reframes the whole decode-fusion direction:

- **Decode is GPU-execution-WORK-bound** (D≈W, host-sync 0%). Removing a kernel launch saves ~nothing; only doing
  the work *faster* or *latency-hidden* helps. Naive fusion conserves work.
- The Deliverable-0 gap families (attention +2.7 ms, elementwise +1.8 ms) are **real**, but tinygrad spends that
  time because it runs the activation/softmax/score work **serially**, whereas llama hides it under HBM-bound
  mmvq/flash latency. Closing the gap is a **latency-hiding / fully-fused-tiled-kernel** problem (interleave the
  activation and softmax ALU into the memory-load loops), which is the **linearizer / software-pipelined-K-loop
  wall** already documented across the prefill arc.
- Concretely: **keep current decode defaults** (baseline 60.8–68.0 tok/s, q8 opt-in 64.5–72.8). The bounded
  fusion levers are exhausted (cheap env levers refuted in the prior scope; FFN fusion built+refuted here;
  attention fusion no-go). The next real decode lever is **either** the fully-fused flash kernel **or** an
  activation-in-GEMV latency-hiding rewrite — both deep codegen projects gated by the linearizer, not bounded
  custom-kernel patches. Recommend taking that on only as an explicit multi-week codegen effort, or resting decode
  at the current route.

## Success-criteria assessment

- **Minimum success: ACHIEVED** — one custom fusion candidate (Phase B1) built, byte-exact, locally gated
  (FAILED, ~0%), with a clear, reproduced mechanism (work conserved).
- **Strong success: NOT achieved** — no candidate recovered ≥0.5 ms / ≥3% (the recoverable gap is latency-hiding,
  not fusion).
- **Best success: NOT achieved** — no stacked ≥80 tok/s route (correctly not pursued on refuted candidates).
