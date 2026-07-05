# HANDOFF — 14B Qwen3 Q4_K prefill speedup (int8 MMQ track)

Date: 2026-07-05. Branch: `int8-wmma-vocab`. For: codex picking up mid-track.
Hard rule from user: **NO hand-written GPU kernels** — every speedup must come from tinygrad
codegen / scheduler / vocab. GPU-wedge hazard: **NEVER `timeout`/`pkill` a live `DEV=AMD` run**
(jams the MES ring → needs GPU reset/reboot). Interpreter is `.venv/bin/python` (NOT `python`).
Standing rule: on any failure/blocker, spawn a Fable 5 agent to review (high-level then drill-down)
rather than solo trial-and-error.

## Mission & the number to beat

14B Qwen3 Q4_K **prefill (pp512)** is the target. Authority: `extra/qk/prefill_whole_synced.py`.
- Current tinygrad ceiling: **~365 tok/s**, and it is **VALU-bound** — the VALU work IS the per-element
  fp16 dequant of Q4_K weights (fp16 GEMM after dequant). WMMA rate was never the limiter.
- llama.cpp reference: **~1849 tok/s** (its MMQ keeps weights int8, does int8 dot + per-group scale
  correction — no fp16 dequant, half the weight bytes).

## What is SOLVED this session (with evidence)

1. **iu8 int8 WMMA works, bit-exact, via codegen (no hand kernel).** A plain
   `a.matmul(b.transpose(), dtype=dtypes.int)` tensorizes to `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`
   on the DEFAULT path (no TC forcing). max_abs_diff=0 on real gfx1100 at 16³/64³/512³/512×4096×4096.
   Three bugs fixed (all in the tree, see Files): `str(tc)` C-identifier space; rendered vector-type
   names + `int4` typedef; iu8 sign flags are `true`=signed (comment was inverted).

2. **DECISIVE: iu8 WMMA has NO raw-GEMM throughput win on RDNA3.** The iu8 descriptor reuses fp16's exact
   dims=(16,16,16)/epc=(16,16,8)/swizzle → identical instruction count & rate. Measured 512×4096×4096
   kernel-only: fp16=33.8 TFLOP/s vs int8=26.3 TOP/s (int8 marginally SLOWER). **The "~2× from int8 WMMA"
   thesis is REFUTED** (2× is RDNA4/CDNA, not RDNA3). Consequence: the Q4_K prefill win must come from
   **killing the fp16-dequant VALU + halving weight bandwidth in a fused MMQ**, NOT a faster tensor core.
   int8 WMMA's only lever is that it runs on separate silicon that can OVERLAP the scale-correction VALU.

3. **The scalar-`_sdot4` (v_dot4_i32_iu8) MMQ path already exists AND is already wired into prefill AND is
   numerically bit-accurate.** Kernels: `q4k_q8_1_sdot4_gemm_kernel`, `q4k_q8_1_sdot4_coop_gemm_kernel`
   (extra/qk/quant/q4_k_gemv_primitive.py). Wired in `tinygrad/llm/prefill_routes.py:route_direct_packed_prefill`
   under `PREFILL_Q4K_Q8=sdot4` (and `=mmq` for the coop variant). Weights stay int8, activations q8_1, dot via
   `_sdot4`, per-group d/dmin scale correction. Verified this session: mmq-vs-sdot4 rel=0.0, both match the
   q8-dequant-activation reference to **1.3e-7**, and match full-precision to ~4.6e-3 (= the q8_1 activation
   quant noise floor — expected, format property not a kernel bug).

4. **These MMQ kernels are now validatable GPU-free.** `_sdot4` is a HIP-only `CUSTOMI` (native
   `__builtin_amdgcn_sudot4`); the PYTHON backend can't render it. A non-HIP fallback `pm_expand_sdot4`
   (tinygrad/codegen/__init__.py) expands it to equivalent int UOps, gated on `ren.target.device != "AMD"`
   so the HIP intrinsic is untouched. TRAP fixed: the naive `((a>>8i)&0xff).cast(int8).cast(int32)`
   sign-extend gives GARBAGE (~6.3 rel_rmse) inside the reduce on PYTHON — must use pure-int32 arithmetic
   sign-extend `ub - ((ub & 0x80) << 1)`. (This was the mysterious 6.304 seen mid-session.)

## The ONE decisive OPEN experiment (Route A) — needs a GPU run

Does keeping weights int8 + scalar v_dot4 (which frees the fp16-dequant VALU) beat 365 tok/s at pp512?
This answers the whole strategy with ZERO new kernels — the route is already wired. Run on the 14B:

    PREFILL_Q4K_Q8=sdot4 DEVICE_IN_FUNCTION_BUG=1 ALLOW_DEVICE_USAGE=1 .venv/bin/python extra/qk/prefill_whole_synced.py
    # also try PREFILL_Q4K_Q8=mmq (coop variant)

Compare vs the 365 fp16-dequant baseline (same harness, PREFILL_Q4K_Q8 unset).
- If sdot4/mmq **beats 365** → direction proven; Route B (WMMA) is optional polish.
- If it **doesn't** → scalar v_dot4 is itself VALU-bound → justifies Route B (move the dot to the WMMA
  unit to overlap the correction VALU).
NOTE: `_sdot4` had a prior "kernel 1.77x over fp but e2e tok/s UNCHANGED" null result in DECODE
(commit cc9cacdf2) — prefill (compute-bound, big M) is a different regime; measure, don't assume.

## Route B (iu8-WMMA fused MMQ) — SPEC'd, contingent on Route A

Full design + algebra: `docs/route-b-iu8-wmma-mmq-design-20260705.md`. Summary: per-32-elem-group int8
dot `RAW = xq@q4.T` (dtype=int, tensorizes to iu8 WMMA; 2 WMMAs per Q4_K group, int32-accumulate) then fp
scale-fold `d8·(D·SC·RAW − DMIN·MN·QSUM)`. Codegen-only (matmul + TC opt, no hand kernel). New
`PREFILL_Q4K_Q8=wmma` route branch; do NOT change the default until pp512 beats scalar sdot4.
Build ON the same q4_k_gemv_primitive.py + the parity gate below. Task #7.

## Files changed this session (all uncommitted → being committed on this branch)

- `tinygrad/codegen/__init__.py` — `pm_expand_sdot4` non-HIP fallback (Fable agent) + wired in full_rewrite_to_sink.
- `tinygrad/codegen/opt/tc.py` — `TensorCore.__str__` space-sanitize for C identifier; iu8 in amd_rdna3 TC list.
- `tinygrad/renderer/cstyle.py` — iu8 WMMA wrapper (rendered vec-type names, int4 typedef, `true` sign flags).
- `extra/qk/prefill_mmq_parity_gate.py` (NEW) — GPU-free + gguf-free numeric gate for the mmq/sdot4 kernels.
- `docs/route-b-iu8-wmma-mmq-design-20260705.md` (NEW), `docs/codegen-wmma-lds-staging-design-20260705.md` (mod).
- `scratchpad/iu8_probe.py`, `iu8_dbg16.py`, `iu8_vs_fp16_perf.py` — iu8 correctness/layout/perf probes.

## Canonical-harness debt (user flagged this — address it)

The canonical numeric validators are `q4k_q8_1_gemv_primitive.py __main__` and `q4_k_bench.correctness_gate`
— **both need a real gguf file**. `prefill_mmq_parity_gate.py` is the GPU-free/gguf-free complement
(synthesizes Q4_K bytes). It should NOT stand alone long-term: fold it into
`test/external/test_prefill_codegen.py` (the deferred consolidation target) and/or unify with correctness_gate.
Deferred consolidation also includes: unify the 3 schedule sources; route the (dead) `_stage` LDS work through
`codegen/experimental.py`.

## Dead ends (don't re-explore)

- **Single-buffer LDS input-tile staging (Track 2A): RULED OUT.** ~2× SLOWER at streamed shapes
  (512×17408×5120: 1837µs baseline vs 4127µs staged). Barrier+LDS-roundtrip per K-tile serializes
  load/compute. A win needs double-buffering (Track 2B, deprioritized: fp16/8B-only long-pole).
- **iu8 WMMA for raw GEMM speedup** — refuted (see Solved #2).

## Task state
#1–#5 done. #6 (in_progress) = Route A pp512 measurement (needs GPU). #7 (pending, blocked by #6) = Route B build.
Memory: see `int8-wmma-no-rdna3-throughput-win`, `14b-prefill-valu-ceiling-wmma-solve` in the auto-memory index.
