# Sibling-output fusion and row-local lowering: status at pause (2026-09-26)

## Landed on exp (all flag-gated, default off)
| commit | what |
|---|---|
| e72668106 | `[codegen]` a reduce of a widened product forms the products in the accumulation dtype (bf16 dot with dtype=float matches fp64 to about 1e-7). **On by default.** |
| 3747f3588 | `[schedule]` SIBLING_FUSE=1: independent stores over the same output ranges share one kernel (invariants I1-I5, tests in `test/unit/test_sibling_fusion.py`) |
| d2111ba7a | `[schedule][codegen]` SIBLING_FUSE=2: output ranges shared in rangeify, plus the row-local lowering (`codegen/opt/row_local.py`) |
| b792e79c1 | `[schedule]` SIBLING_FUSE=2: a partially realized value goes to a global buffer, never a shared-memory stage |
| 327bc54db | `[codegen]` row-local never uses a lane count equal to a whole extent (an L*L shared stage otherwise); `compile_cached` never stores an empty binary |

## Gate results
SIBLING_FUSE=1 (v1): **all gates clean.**
- Full test/unit: failure set identical to flag-off (196).
- Real 4B NV: prefix, prefill and seeded generate are bit-identical.
- capture_check: 3016/3016. window_check (k=41/36/30/12): 3264/3264 each.
- Warmstart route hits: unchanged, 7/7 applied.
- Kernels per generate: 10692 -> 10184.
- Rollout step time: -0.3 to -1.1%.

SIBLING_FUSE=2 (v2), GPU gate run on b792e79c1. This run predates 327bc54db and was not repeated:
- Real 4B prefix: failed to compile. The prefill softmax kernel `r_8_5_72_200` used 160 KB of shared memory (L=200 = the whole key extent, giving an L*L stage). The cause was reproduced on CPU by rendering all tiny-model prefix kernels for sm_120, and fixed in 327bc54db (largest shared buffer now 400 B). **Not re-verified on GPU.**
- window_check k=41/36/30/12: 3264/3264 bit-equal each. These paths did not hit the bad kernel.
- Full test/unit: 196 failed, the same count as flag-off, but 12 are new and 2 are fixed. The new ones are:
  - status-quo pins that now see the fusion happen: `test_nv_norms_fusion_ab` x2 and `test_nv_boundary_free_ordinary_uop_gate` x4;
  - bitwise-vs-ordinary RMSNorm comparisons, which the tree reduction order changes: `test_reduce_output_rmsnorm` x3;
  - `test_generic_reduce_output::test_per_site_admission_spelling_lowers_to_one_call_body_free`;
  - `test_decode_graph_position_invariance::test_eager_full_logits_realize_honors_bound_position`, **needs a look**;
  - `test_sibling_fusion::test_v2_no_workgroup_spanning_shared_stage` (under a global SIBLING_FUSE=2 env).
- Timing, flag off, `rstep.py B 4096 256`: 11.13 / 16.98 / 28.79 ms at B=32/64/128.
- Timing, flag 2: failed at every B. B=32/64 died on the same compile failure; B=128 ran out of device memory (NV_ERR_NO_MEMORY). The pattern bench failed with "blob is not an ELF" because unit tests had written empty stub binaries into the NV compile cache. Those 8 entries are purged, and 327bc54db prevents a repeat.
- On CPU, v2 fuses the residual+RMSNorm+hi/lo pattern into 1 kernel (was 5). The NV render has vLLM's fused_add_rms_norm structure: one block per row, 448 lanes, one barrier. Against flag-off: x and hi are bit-identical, lo differs in about 0.6% of elements (reduction order), and the error against fp64 is unchanged.

## Next step on resume
1. Re-run `/home/ubuntu/scratchpad/corefix/v2check.sh` then `v2time.sh` (both under gpu-run) on exp HEAD, to check whether 327bc54db clears the 4B compile failure.
2. Look at `test_decode_graph_position_invariance` under SIBLING_FUSE=2. Update the fusion-pin tests only if the flag is ever turned on by default.
3. If the pattern timing shows the 448-read shared-memory broadcast matters, replace it with a tree or warp-shuffle broadcast.
4. Root-cause the degenerate group-reduce stage (a size-1 split indexes the stage by the lane twice) in `fix_group_for_reduce`, rather than only avoiding it in lane_count.
