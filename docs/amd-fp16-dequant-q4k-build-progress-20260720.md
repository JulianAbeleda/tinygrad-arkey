# BUILD PROGRESS / CONTINUATION: generated fp16-dequant Q4_K primitive (AMD)

## PRIMITIVE-SUBSTRATE PIVOT (2026-07-20) — read this first
The bespoke hand-authored `mmq_llama_*` UOp stack (below) is stuck on a multi-wave writeback codegen bug + spill wall. A cleaner path was proven: the **plain Tensor primitive** `x @ q4_k_reference(bytes).reshape(N,K).cast(half).transpose()` (USE_TC=1, no `.contiguous()`) is the *substrate* — the tinygrad scheduler produces it CORRECT (rel_rmse ~4.1e-4 at ALL real 14B shapes) and FUSED/non-materializing (single kernel, no dense fp16 weight ever allocated, even at the 178MB ffn shapes). Multi-wave correctness + non-materialization — the entire reason the bespoke stack existed — are **free** here.

**But throughput is the whole story now — TWO independent gaps (measured on ffn_gate_up 512×17408×5120, gfx1100):**
- **Gap 1 — dequant feed: ~4.1 → ~14.5 TFLOP/s.** Default fused primitive emits NO WMMA (Q4_K block scale/min → multiple reduce axes → TC heuristic `heuristic.py:67` skips it → scalar accum). `TC_OPT=2` forces WMMA but is WORSE (2.2 TFLOP/s): dequant recomputed per-K, per-lane along M (reuse factor 2 not 512), no LDS staging → ~80 dequant VALU ops / 2 WMMA calls, tensor cores starved.
- **Gap 2 — WMMA GEMM itself: ~14.5 → llama's 61 TFLOP/s.** Even DENSE fp16 WMMA (no dequant) tops out ~14.5 (~12% of RDNA3 fp16 peak). Scheduler's generic TC lowering is weak on these shapes; closing this needs int8 (~2×) + better tiling (~2×).

**The lever for Gap 1 already exists in-tree:** `build_precontract_lds_stage` (`tinygrad/codegen/opt/kernel_lds.py:394-470`), staged dequant panel in LDS + M-reuse via WMMA, activated by `candidate_context`/geometry at `postrange.py:372-408`. The plain Tensor expr never attaches it → falls into the naive branch. **Plan: keep the primitive as substrate, wire it through the existing LDS-staging machinery** (no hand-authored ISA; reuses the good scheduler part, drops the buggy hand-written writeback). Repro scripts: `scratchpad/qk_wmma_prefill_bench.py`, `scratchpad/exp1_dense_fp16.py`, `scratchpad/exp2_min.py`.

### HEAD-TO-HEAD RESULT (2026-07-20) — packed-WMMA (fork B) IS THE ROUTE; matches/beats llama per-kernel
The `candidate_context`/`PackedWeightTransform`/`build_precontract_lds_stage` machinery is ALREADY wired via `extra/qk/prefill/current_prefill_execution_adapter.py` (scheduler-native "fork B" = fp16-dequant-in-register + LDS-staged WMMA), exercised by `extra/qk/prefill/packed_wmma_correctness_canary.py`. Correct (canary max_abs_error 0.0), fused (50MB packed B, never 178MB dense). **Steady-state (min-of-30 after 200-dispatch warmup — the earlier "~33 TFLOP/s wash" was a COLD-CLOCK artifact: packed-WMMA's first ~7 dispatches sit at 2.76ms then drop to 1.33ms):**

| role | direct-packed | dense-fp16 | **packed-WMMA** | llama isolated |
|---|---:|---:|---:|---:|
| ffn_gate_up 512×17408×5120 | 14.6 | 20.0 | **68.8** | ~56–65 |
| ffn_down 512×5120×17408 | 14.1 | 26.9 | **62.2** | ~56–65 |
| attn_qo 512×5120×5120 | 14.0 | 27.8 | **39.6** | — |

(TFLOP/s.) Packed-WMMA beats direct-packed **4.2–4.6× every role**, matches/beats llama on the dominant ffn roles WITH fp16 (no int8). llama rocprof: `/home/ubuntu/BoltBeam/outputs/llama-prefill-qwen14b-pp512-20260714-fresh/rocprof/qwen14b_pp512_kernel_trace.csv`, ffn_gate_up-class = Grid 4352×32, 1.41–1.62ms → 56–65 TFLOP/s.

**PIVOT: fork-B packed-WMMA REPLACES the C5-faulted int8 `mmq_llama_five_buffer_*` stack** (separate impl — hand-built UOp graph + native PM4, zero cross-imports; confirmed via grep). The int8 grind (C5 fault, spill wall, multi-wave writeback bug) is MOOT — this scheduler-native fp16 path already works and is faster. Bench harness: `scratchpad/bench_variant.py`; ISA diag: `scratchpad/diagnose_wmma.py`.

**Open (do not over-call yet):** (1) direct_packed measured 6.27ms vs stale doc §13.2 2.762ms (~2.3× disc, flag not reconcile; relative win is same-harness solid).

### WHOLE-MODEL ESTIMATE (2026-07-20) — ~1460–1650 tok/s; gap to llama = attention GEMM only
Bottom-up steady-state (200 warmup, min-of-30, host-dispatch removed), all linear roles at real 14B dims × 40 layers + measured non-GEMM overhead. Per-layer GEMM 6.412ms → ×40 = 256.5ms; non-GEMM (norms/rope/qk-norm/attn-scores/PV/residual/swiglu) 1.363ms/layer → 54.5ms. Body 311ms → **1646 tok/s** (last-token lm_head); 1463 w/ full dense-fp16 lm_head. vs llama **1837**, default **366** → **~4–4.5× default, ~80–90% llama.** GEMM 82.5% / overhead 17.5%.

**The gap is entirely the small-N attention GEMMs (packed-WMMA):**
- **attn_kv 512×1024×5120 = 12.0 TFLOP/s** — N=1024 → ~32 workgroups, cannot fill 96 CUs. GRID-limited.
- **attn_qo 512×5120×5120 = 38.1 TFLOP/s** — occupancy-limited (40KB LDS → 1 wg/CU).
- FFN roles fine: ffn_gate/up 69.2, ffn_down 62.0 (at/above llama). Attention-linear = 36% of GEMM time despite trivial FLOPs.

**Lever:** if attn roles hit FFN-like efficiency, attn GEMM ~92ms→~31ms → body ~250ms → **~2050 tok/s (past llama 1837 AND project ≥2000 target).** Fix = make small-N attention GEMMs fill the machine: smaller tiles→more workgroups (grid-limited attn_kv), shrink LDS→2 wg/CU (occupancy attn_qo), possibly split-K.

**Tier-2 (real end-to-end) BLOCKED — glue needed (packed-WMMA isolated under `extra/qk/prefill/`, zero refs in `tinygrad/llm/*`):** (1) `Q4KPackedWMMAPrefillCandidate` matching `PrefillLinearRouteSpec` in `tinygrad/llm/prefill_routes.py:151`; (2) real GGUF Q4_K→`PackedWeightTransform` materializer (today only fed `Tensor.empty` synthetic); (3) route registration in `model_route_plan.py`/`prefill_policy.py`; (4) lm_head role (no `model_profiles.py` entry). NEXT: close attn_qo/attn_kv occupancy+grid (the crux to beat llama), then wire Tier-2. Bench: `scratchpad/bench_variant2.py`, `bench_overhead.py`, `bench_lmhead.py`.

---

### REAL END-TO-END + PRIMITIVE/GRAPH VERIFICATION (2026-07-20)
**Real pp512 measured: 799 tok/s** (median, real Qwen3-14B Q4_K_M GGUF `/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf`, full forward under TinyJit, all 4 Q4_K geometries correctness-gated before dispatch, finite output). = **2.26× the default (354, same harness)** but **43% of llama 1837** — the 1911 estimate was ~2.4× optimistic. Bench: `scratchpad/e2e_packed_wmma_bench.py`.
- **Why estimate was optimistic:** (1) MIXED-QUANT — real Q4_K_M keeps attn_v Q6_K in ½ layers, ffn_down Q6_K in ½ layers → those fall through to slow direct-packed (census: ffn_down 50% routed, attn_kv 75%); (2) estimate removed host-dispatch overhead + assumed steady-state clocks + uniform Q4_K.
- **GRAPH CAPTURE INTACT (dispatch theory REFUTED):** our route IS captured — all ~1130 kernels/pass replay inside 6 HCQGraph batches, ZERO eager dispatch. Route uses pure lazy Tensor chain + `warmstart_candidate_state` (`postrange.py:535`); `compile_current_prefill_program`/`ExecutableHandle` only in the OFFLINE correctness gate, never in-forward. So 641ms/pass is REAL GPU COMPUTE across 1130 kernels, not launch latency. TinyJit at `model.py:781`; graph-GEMM binding route (`route_pf16_graph_gemm`) is a different (fp16-overlay) scheme, not needed.
- **ALL 8 PRIMITIVES VERIFIED PASS:** (1) scheduler WMMA `v_wmma_f32_16x16x16_f16` (+iu8 desc `tc.py:145`); (2) PackedWeightTransform Q4_K AND Q6_K canary max_abs 0.0; (3) LDS staging lds_bytes=40960, 56 ds_load/store_b128; (4) prefill_packed_weight zero-copy *in sidecar/shared mode only* (else clones); (5) non-8 sm*sn geometry compiles; (6) compile ABI globals(0,1,2)/outs(0)/ins(1,2); (7) route override fires first; (8) canary fails-closed on corruption.
- **NEXT: profile the 641ms/1130-kernel breakdown by bucket** (Q4 WMMA ~220ms vs Q6 fallback vs E_/r_ helpers vs attention vs norms) → biggest lever. Levers in-frame: Q6_K coverage (extend primitive 2 into route), kernel fusion (fold movement-chain helpers), FFN tiling. Missing primitive: none new needed (graph exists).

### PRIMITIVE-ROUTE VIABILITY SWEEP (2026-07-20) — only #2 viable; int8 moot; overhead second-order
Tested all primitive-aligned routes to beyond-parity (scheduler-native, no bespoke stack):
- **#2 codegen fix — VIABLE, LANDED (commit `da28e5efd`).** 17-line diff, 2 files: `postrange.py:449` validate accumulator ownership vs `subtiles_m*subtiles_n*8` not literal 64 (was silently forcing sm*sn==8); `kernel_lds.py:340` `_wave_ok` accepts CONST-0 for size-1 wave axes (was requiring live RANGE → wm=1 crash). Unlocks attn_kv machine-filling geometry (32×64/waves(1,2)=256 wg; 64×64/waves(2,2)=128 wg), correct on real GPU (max_abs 0.0), FFN path green, unit suite 76 pass/1 fail (pre-existing, confirmed by revert). Separate degenerate sm=1/sn=1 `shift_to` wall scoped out (not needed for ≥96-wg). **BENCH RESULT: whole-model est 1646 → 1911 tok/s — BEATS llama 1837 (+4%), 96% of 2000 target, +16% baseline.** Production geometries (all correctness-gated): attn_qo 128×32/waves(4,1)/bc=1 = 42.4 TFLOP/s (+11%); attn_kv 64×32/waves(2,1)/bc=1 = 20.2 (+68%); BONUS bc=1 lifts FFN too: ffn_gate_up 69.2→79.6 (+15%), ffn_down 62→73.6 (+19%). **CORRECTNESS CAVEAT: some compiled geometries are SILENT-WRONG (max_abs 52.5, e.g. attn_qo 128×64/waves(4,2)/bc=1 and 128×128/waves(4,4)/bc=1) — not crashes. Baseline geoms unaffected (no regression); but every geometry candidate MUST be correctness-gated before use.** Remaining >2000 lever = FFN tiling (66% of GEMM, ~66% of peak) + 54.5ms overhead, NOT attention (no longer bottleneck).
- **#3 int8-through-substrate — VIABLE but MOOT on gfx1100.** The iu8 WMMA descriptor/renderer/LDS-staging already exist and are dtype-generic in CORE tinygrad (`tc.py:140-147` `(char,int)` entry, `cstyle.py:445` `wmma_i32_16x16x16_iu8`, `kernel_lds.py:24-31` dtype-agnostic) — NO bespoke stack forced. BUT RDNA3 iu8 uses the SAME `elements_per_thread=(16,16,8)` as fp16 → int8 issues at the SAME rate → no 2× edge (`docs/prefill-lessons-ledger.md:90-95`; 2× is RDNA4/CDNA only). Even fully built, int8 would NOT exceed fork B's ~69 TFLOP/s fp16 ceiling. Gaps if ever wanted (RDNA4): int8 output in `packed_weight.py` dequant_tile + block_q8_1 activation + DS4 as scheduler ops + a register-pressure spill at `test_stage1_int8_candidate_compiles_end_to_end`. **DECISION: skip — no throughput gain here.**
- **#4 non-GEMM overhead — VIABLE but SECOND-ORDER.** SwiGLU already 1 fused kernel; residuals structurally pinned by multi-consumer CSE (`.contiguous()` are legit realize boundaries, dropping = recompute across 40 layers); codebase already hand-tuned for warmstart TC-kernel isolation (`model.py:403`). Recoverable ~1-4ms of 311ms body → single-digit tok/s. **DECISION: skip.**

**Roadmap collapsed: #2 (attention occupancy, landed) is the sole primary lever → ~1900 est (beats llama). The >2000 push is NOT int8 — it's FFN-tiling headroom (fp16 WMMA 69 TFLOP/s ≈ 57% of ~120 peak).**

---

Living doc — resume the implementation from here if context is lost. Plan: [`amd-fp16-dequant-q4k-primitive-implementation-plan-20260720.md`](amd-fp16-dequant-q4k-primitive-implementation-plan-20260720.md). Decision/context: handoff §1.16 (AMD primitive = fp16-dequant-in-register), §1.15 (occupancy routing). Last updated 2026-07-20.

## CORRECTNESS FAIL-FAST RESULT (2026-07-20) — decode CORRECT, full-kernel GPU output WRONG
Verified on a minimal real config (BN=16 single subtile, real decode, K=256; **compiles spill-free**: emitted=True, VGPR=224, 0 spills, `v_wmma_f32_16x16x16_f16`, LDS 16384).
- **Decode math: PASS, bit-exact.** `q4_k_fp16_decode_group_callback` vs `gguf.ggml_data_to_tensor` (GGML_Q4_K), CPU: **max abs diff 0.0**. The novel dequant (`d·sc·code−dmin·mn`, f32 intermediate, single f16 cast) + the `.bitcast(half)` are CORRECT. The highest-risk part works.
- **GPU dispatch: FAIL.** 2048/2048 mismatch, 368 NaN, max_abs_err ~1.29e3. Deterministic pattern: weight-rows 0–22 (waves 0–1) = NaN; rows 23–127 (waves ≥2) = **exactly 0.0, never written**. Same with a single unchained K32 group → NOT a chaining/lifetime bug. Fault is in the WMMA-fragment/cooperative-LDS-staging/multi-wave-writeback wiring — a path never before run on real GPU.
- **Next discriminator:** a genuinely single-wave (32-thread) config to separate "generator wiring logic bug" from "multi-wave LDS race" (needs parameterizing the hardcoded 128-row/8-wave cooperative schedules in `mmq_llama_record_producers.py`).
- **Repro:** `scratchpad/min_fp16_q4k_kernel.py` (build/compile), `scratchpad/gpu_correctness_check.py` (dispatch+reference+compare).
- **So: two open problems on the FULL kernel** — (1) full-geometry spill-free (deferred, register-window structural), (2) the GPU correctness wiring bug above. The decode/approach is validated; the generated kernel's integration is not.

## One-line state
Converting the MMQ generator int8→fp16-dequant IN PLACE (per §1.16; int8 source preserved in git history for later NVIDIA recovery = task #15). Kernel now RENDERS correct-structure fp16 (16 KB LDS, f16 WMMA, 3-buffer ABI); the register-pressure spill is SOLVED; currently closing the last compile blockers. **Numerical correctness NOT yet measured** (that's phase 3/4).

## Commits on master (in order)
- `4eef43945` [wip] convert generator int8→fp16-dequant (renders, not spill-free) — phases 1-2.
- `f6c5c4f6a` [wip] chain accumulator across K32 groups (fixed 512→64 VGPR accumulator blowup; 64 chain heads → 8, matching hand kernel).
- `27e1f0f25` [amd] generalize `_frag_b128_loads` lane stride by dtype itemsize (keeper; superseded by the ratio fix in 65c16271b).
- `65c16271b` [wip] **fix fp16 fragment DS_LOAD vectorization — REGISTER WALL DOWN.** `_fragment_at` load raw bytes (`uint8.vec(esz)`) then `.bitcast(half)` the VALUE (was `UOp.index(dtype=half)` mislabeling a uint8 ptr → legalizer shattered into 16 uchar scalar loads); `_wmma_half_addr` unwraps the BITCAST; `_frag_b128_loads` stride = loaded/pointer itemsize RATIO. Result: scalar DS_LOAD 1950→0 (28 vectorized DS_LOAD_B128), peak vregs 491→63, compile 6.5min→75s. `test_amd_isa_wmma.py` 4 failures are PRE-EXISTING (verified on baseline 51cce914c), not regressions.

## CURRENT BLOCKER (register pressure SOLVED, peak 69) — core regalloc fix approved, in progress
HEAD `849bd9e2c` (group-0-only cross-element guard; fast ~75s compile; peak_virtual 69; still `emitted=False`). Register PRESSURE is solved — remaining is a **fixed-register LIFETIME** conflict: the reused WMMA A/B fragment window (`v201-215`, isel `ab_key="wmma_ab"`) is pinned live the WHOLE kernel `[579,22100]` instead of released per group → spill at uop 855. Real `Ops.BARRIER` between groups already exists (15, hand-kernel cadence — NO correctness race). Neither graph-side variant closes it (group-0-only=fast-but-blocked; every-group=breaks the rewrite engine with a superlinear/infinite-loop blowup — see reverted `ae02946e3`). **Confirmed: closing it needs a CORE change** (not scopeable to `amd.py`; an isel-side `ctx`-dep attempt crashes `unified_rewrite` with `KeyError: replace[SINK]` on cross-sibling refs). **User approved option (a): narrow `tinygrad/codegen/late/regalloc.py` `_pressure_schedule_block` change** (split blocks at `.after()`-chained fixed-lease boundaries / deeper lookahead for lease_width==1), TEST-GATED: baseline → change → zero new failures vs baseline → int8-WMMA spot-check (shares wmma_ab) → f16 emits spill-free. Fallback (b) = fix `unified_rewrite` cross-chain limit in `tinygrad/uop/ops.py` (wider blast radius).

### (superseded) earlier note — real barriers
Full kernel still `emitted=False`. Real target (the "SGPR/PARAM(0)" label was cosmetic — `Register.index` is always 0): an **intra-chain `DS_LOAD_B128` fixed-lease (`v200`) conflict** — 7 reloads within one subtile element's 7 non-head groups are hoisted together by the greedy `pressure_schedule`/`_pressure_schedule_block` (`tinygrad/codegen/late/regalloc.py`) because there is **no real `Ops.BARRIER` between K32 groups** to split the block (only `.after()` pseudo-ops).
- **FIX IN PROGRESS (graph-side, preferred over core-scheduler change): emit real `Ops.BARRIER` between the 8 K32 groups** (hand-kernel cadence, wmma.py:600-631). This splits the block for the scheduler AND — **likely also a real correctness bug** — provides the cross-wave workgroup barrier the single-buffered LDS (DBUF=0, 8 waves sharing LDS) requires; `.after()` only orders within one wave, so without it waves race on shared LDS → wrong runtime results. Agent verifying whether any real barrier exists today.
- Fallback if barriers insufficient: minimal `_pressure_schedule_block` change (split at `.after()`-chained fixed-lease boundaries) — core-scheduler, blast radius, needs review.

## Verification commands
- Full compile (~75s now): `build_llama_five_buffer_full_kernel(128,128,256)` then `compile_llama_five_buffer_full_kernel(k)`; check `.emitted` (True=spill-free), `.blocker`, `.program.arg` (VGPR/LDS). Confirm ISA has `v_wmma_f32_16x16x16_f16`, LDS group_segment 16384.
- Cheap ~65s probe: a 2-K32-group synthetic mirroring `_full_grid_sink` (isel/regalloc only, no GPU) — reproduces the same pressure/blocker at 1/8 cost.
- Pressure introspection: `REGALLOC_DEBUG` env prints peak live vregs + PEAK_CONTRIBUTORS + spill point.

## What's left (ordered)
1. **Finish phase 2b (spill-free):** fix the 8 `test_amd_isa_wmma.py` regressions (fragment-load fix must cover int8 AND fp16; the BITCAST unwrap must be a strict no-op when no bitcast); fix the SGPR PARAM(0) blocker → `emitted=True`, 0 spills. Then commit.
2. **CORRECTNESS FAIL-FAST (pulled forward):** the moment it emits, before building the family, do a small-shape numeric parity of the dequant vs the authority. The `.bitcast(half)` reinterprets raw bytes as half — if byte order/layout is off it compiles but outputs garbage. MUST verify before trusting.
3. **Phase 3 — correctness authority + CPU parity:** author `ffn_gate_up_fp16_dequant_reference` on the GGML `d*sc*code−dmin*mn` math (`tinygrad/llm/gguf.py:76-84` / `extra/qk/layout.py:157` `q4_k_reference`; existing analogue `mmq_ffn_gate_up_guarded_correctness.py:357-375` `ffn_gate_up_direct_dense_reference`). Feed into the same `_validate_numeric_comparison`/`_validate_full_comparison` (`:223-299`). NOT the int8 authority (`mmq_q4k_q8_reference.py`) — different rounding path (§2.5, needs new authority + C0A sign-off). Accept: `rtol=atol=3e-3`, zero mismatch, finite.
4. **Phase 4 — new frozen family + GPU:** new 2-3-buffer ABI family (checklist = plan PART III; canonical ABI constants `extra/qk/prefill/frozen_exact_role_runtime.py:37-39`); C4 no-target canary; then guarded reduced-grid ladder `(1,1,1)…(8,4,1)` zero-mismatch, then the FULL 544-wg dispatch that MUST now pass (16 KB LDS) where int8 wedged at 64.
5. **C6-C8** (full correctness / memory / timing → CERTIFIED_WIN or FALLBACK).

## Follow-ons (separate tasks)
- #10: occupancy-based route admission axis (§1.15) — routes int8→NVIDIA, lean→AMD from facts.
- #15: recover int8 generator as renamed NVIDIA-only, NOT-selectable modules (source is at pre-conversion git history / was HEAD `51cce914c`).

## Key files
- `extra/qk/mmq_llama_candidate_plan.py` (`_geometry` two fp16 16KB regions, `_rdna3_f16_tc`).
- `extra/qk/mmq_llama_oracle_recurrence.py` (`_fragment_at` bytes+bitcast; fp32-accumulate recurrence).
- `extra/qk/mmq_llama_group_chain.py` (chained accumulator across 8 K32 groups; seed src[2] via O(1) DAG re-point).
- `extra/qk/mmq_llama_record_producers.py` (`q4_k_fp16_decode_group_callback` — the decode).
- `extra/qk/mmq_llama_five_buffer_graph.py` / `_full_kernel.py` (3-buffer ABI, per-K32 epoch loop 8× per K256).
- `tinygrad/renderer/isa/amd.py` (`_frag_b128_loads` stride, `_wmma_half_addr` bitcast unwrap).

## Non-negotiables (don't regress)
No dense fp16 weight materialization (§2.4 — decode stays per-tile in-register). Preserve llama Q4_K rounding: f32 intermediate, SINGLE final f16 cast (`d*sc*code−dmin*mn`). New correctness authority signed off before trusting numbers (§2.5). Route stays research/not-promoted, strict fallback to direct-packed, until C8.
