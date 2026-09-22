# NV decode first-principles re-audit (2026-08-19)

Date: 2026-08-19
Branch: `nvidia-bringup-20260731`, HEAD `d14e6964e`
Status: **corrected loss trace.** Re-derived the loss from the same session,
same model, same GPU, and found the earlier flash-shape search measured the
wrong backend and the wrong lever.

## 1. What changed this session

The flash-shape P3/P4 work was measured on `DEV=CUDA`, not the production
backend `DEV=NV`. Two clean-session measurements expose the difference:

| route | tok/s | wall us/token | note |
| --- | ---: | ---: | --- |
| tinygrad `DEV=NV` (production) | 205.0 | 4877 | census + A/B harness agree |
| tinygrad `DEV=CUDA` | ~177 | ~5660 | the backend the P3/P4 search used |
| llama.cpp `build-cuda` llama-bench | 254.4 | 3931 | decode tg128, `-ngl 99` |

The `DEV=CUDA` backend is ~28 tok/s slower than `DEV=NV`. The P3 isolated-body
search (`nv_flash_geometry_search.py`) hard-codes `Device.DEFAULT == "CUDA"` and
`Device["CUDA"]`, so its "4.19 us control / 3.968 us best" numbers and the P4
wall bracket verdict were all measured on the non-production backend. On
`DEV=NV` the P4 control/candidate re-bracket for the bitwise-identical
`stage_width=2` candidate is neutral (candidate ~203.9 vs control ~206.5 tok/s),
so the flash-shape lever is still not promotable.

The P4 harness also sets
`_CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()`, which disables the
production fused prefill attention and changes the token baseline
(`e3f81cdb...` vs production census `227ad3ce...`). That does not change the
decode-kernel wall delta, but it means "token identity" was checked against a
non-production prefill path.

## 2. The corrected loss decomposition

Same-measurement CUPTI ledgers (`nv-tinygrad-node-ledger-gap-record-20260813.md`,
`nv-internal-gap-first-principles-20260818.md`) plus the fresh anchors:

| term | llama | tinygrad | gap |
| --- | ---: | ---: | ---: |
| GPU kernel work (node-sum) | ~4774 us | ~4519 us | tinygrad does LESS work |
| overlap (work hidden behind GEMV chain) | ~946 us | 0 us | the whole inversion |
| GPU busy (span/union) | ~3835 us | ~4519 us | +684 us |
| host gap | ~212 us | ~269 us | +57 us |
| wall | ~3931 us (254 tok/s) | ~4877 us (205 tok/s) | +946 us |

tinygrad is already doing less GPU work than llama. It is slower only because
it runs that work fully serialized while llama hides ~946 us of support kernels
(norms, quantize, rope, residual, vocab aux) behind its long GEMV chain.

Class-level correction: the flash score BODY is at parity.
`nv-llama-fattn-matched-isolated-record-20260816.md` pins tinygrad S=48 at
4.160-4.192 us vs llama grid.y=2 at 4.096 us (~0.09 us/launch, ~3 us/token).
The installed in-graph delta (tinygrad ~6.5 us/launch vs llama ~3.16 us in-situ)
is launch/L2-cold/overlap cost, not score math. `nv-flash-score-codegen-diff-20260819.md`
details the structural mapping: both kernels are 128 threads / 4 warps; the
difference is warp ownership, not the dot body.

## 3. Verdict

The loss is **serialization (HIDE), not kernel shape (ELIMINATE) or vocab**.

- Flash score: body at parity; the open ~39 us is launch/overlap, not the
  searched geometry.
- Vocab argmax: hidden mass; honest recoverable wall ~5.6 us/token. The packed
  u64 route is the wrong codegen answer; a native `(fp32 max, int32 index)`
  composite reduce is the correct fix (`nv-vocab-top1-codegen-diff-20260819.md`).
- The dominant lever is llama's ~946 us of overlap. Recovering it requires
  exposing the decode graph's intra-layer parallelism and executing it with the
  PDL/multi-stream substrate, not retuning the flash tile.

## 4. Promotion path

1. **Overlap substrate (primary).** Restructure the decode graph so independent
   branches (Q/K/V projections, FFN gate/up, per-head work, and the
   norms/quantize/rope support kernels) can run concurrently like llama's graph,
   then execute with the already-proven PDL launch substrate. This targets the
   ~946 us overlap gap, by far the largest lever.
2. **Vocab codegen (small, clean).** Replace the packed-u64 argmax with a native
   fp32+int32 composite reduce (`tinygrad/mixin/__init__.py:argmax`,
   `tinygrad/uop/ops.py`, `tinygrad/codegen/late/composite_combines.py`); ~5.6 us.
3. **Search generalization.** Remove the `DEV=CUDA` hard-coding from
   `nv_flash_geometry_search.py`, add an NV timing path, and only then re-open
   `score_group_width`/`warps`/`split_count` if the overlap substrate is not
   enough (`nv-bubblebeam-search-gap-20260819.md`).

## 5. Companion analyses (same date)

- `nv-flash-score-codegen-diff-20260819.md`
- `nv-vocab-top1-codegen-diff-20260819.md`
- `nv-bubblebeam-search-gap-20260819.md`

## 6. Overlap-substrate proof update (same date)

The HIDE lever was tested end to end on the production route. The planner was
reintroducing a false WAR/WAW edge when `reduce_output_rmsnorm_8_128` reused
the arena slot of a sibling norm, which kept it serialized on the primary
queue. Landing reuse-lane arena coloring in `tinygrad/schedule/memory.py`
removes that false edge and reproduces the `NO_MEMORY_PLANNER` ceiling with
the planner still enabled.

| arm | tok/s | decode census |
| --- | ---: | ---: |
| serial control | 205.99 | 32/0 |
| planner-off ceiling | 211.50 | 21/11 |
| landed reuse lanes | 212.12 | 21/11 |

Token sha identical across arms (`1d299b89...`). Evidence:
`docs/task_workflow/evidence/nv-overlap-substrate-reuse-lanes-20260819.json`.

This recovers the small slice the currently qualified two-GPFIFO substrate can
expose (~6 tok/s, ~140 us this session), not llama's full ~946 us shadow. The
promotion path is unchanged for the remaining gap: deeper graph concurrency
beyond two GPFIFOs plus the vocab codegen fix, then the generalized search.
