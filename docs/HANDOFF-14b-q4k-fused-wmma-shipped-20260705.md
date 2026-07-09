# HANDOFF - 14B Q4_K prefill: fused dequant->WMMA SHIPPED

Date: 2026-07-05. Branch: `master`. Commits:

```text
c5531fad5 [prefill] add fused Q4_K decode wmma builder
c6caf391c [prefill] route fused Q4_K wmma for 14B prefill
```

## Result

14B Qwen3-Q4_K_M prefill, pp512, authority harness (`extra/qk/prefill_whole_synced.py`):

```text
packed VALU baseline (before):    359 tok/s   (the VALU ceiling; re-measured same-session)
fused Q4_K WMMA (this work):      808 tok/s    (WHOLE-PREFILL@512, authority)
llama.cpp reference:             ~1849 tok/s
```

Same-session head-to-head: 808 / 359 = **2.25x**.

- **2.2x over the 365 baseline**, ~44% of llama. Tapers 808@512 -> 800@1024 -> 770@2048 -> 731@4096.
- **Bit-exact**: rel RMSE ~3e-4 vs `q4_k_reference` on all four role shapes (attn_kv, attn_qo, ffn_down, ffn_gate_up).
- **Fits 14B in memory**: weights stay PACKED 4-bit resident (~9GB), no fp16 materialization (the ~31GB fp16 copy that OOMs and forced the VALU path).
- Per-kernel ~66 TFLOPS (DEBUG=2 smoke).

## What was built

This is the quantized analog of the proven 8B resident-fp16 graph-GEMM win (`build_gemm_lds2`/`build_gemm_pipe`),
i.e. a `gen_sched`-style hand-asm substrate builder — NOT a tinygrad-codegen kernel.

- `extra/qk/prefill/wmma.py::build_gemm_lds2_q4k` — forks `build_gemm_lds2` (fp16 GEMM). A path, LDS layout,
  `compute0`, epilogue are the proven fp16 ones. Only the B side is new: B is packed Q4_K bytes `[N, (K//256)*144]`;
  the K-loop runs over 256-elem SUPER-BLOCKS with the 8 sub-groups Python-unrolled (so group index g -> static
  byte/nibble layout + `get_scale_min_k4`); each group is decoded to fp16 and stored into the SAME fp16 LDS B-tile,
  so WMMA/epilogue are unchanged. BK fixed 32 = one Q4_K sub-group. Requires BN==THREADS (holds for W2x2 T4x4).
- `extra/qk/prefill_graph_gemm_route.py::route_q4k_graph_gemm` — packed-resident route; `tinygrad/llm/route_ops.py`
  exposes it; `tinygrad/llm/prefill_routes.py::route_prefill_linear` dispatches it when `PREFILL_Q4K_WMMA_FUSED=1`
  and the linear is Q4_K (falls through to the packed VALU route if a shape can't bind).
- Test/microbench: `test_lds_gemm2_q4k` in wmma.py (`LDSGEMM2Q4K=1`), random packed bytes + `q4_k_reference`.

## How to run

Microbench (correctness + per-kernel TFLOPS), any role shape:

```bash
DEV=AMD LDSGEMM2Q4K=1 M=512 N=5120 K=5120 WAVES_M=2 WAVES_N=2 WM=4 WN=4 \
  PYTHONPATH=. .venv/bin/python extra/qk/prefill/wmma.py
```

Full 14B authority (the 808 number):

```bash
DEV=AMD PREFILL_Q4K_WMMA_FUSED=1 ALLOW_DEVICE_USAGE=1 DEVICE_IN_FUNCTION_BUG=1 \
  PYTHONPATH=. .venv/bin/python extra/qk/bench.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf --prefill
```

(`ALLOW_DEVICE_USAGE=1 DEVICE_IN_FUNCTION_BUG=1` are required or the packed-weight realize trips the
JIT-capture device guard. Never kill a live DEV=AMD run — wedges the MES ring.)

## The four gfx1100 raw-INS traps that this cost (~30 debug iterations)

A raw `Ops.INS` stream via `custom_kernel` bypasses tinygrad's scheduler, so you own all hazard/alloc rules:

1. VGPRs >= ~238 read back GARBAGE (ELF descriptor doesn't size to the highest reg). Keep temps low (FA/FB region).
2. FP/cvt result feeding a dependent VALU op is NOT hw-interlocked -> declare RAW with `s_delay_alu(simm16=1)`
   (`s_nop` does NOT satisfy the scoreboard). cvt->store is safe; cvt->VALU is not.
3. fp16 scalar-arith ops (`v_mul_f16`/`v_fma_f16`/`v_cvt_f16_u16`/`v_pack_b32_f16`) proved unreliable here; decode
   in f32 (manual fp16->f32 bit-expand for d/dmin, `v_cvt_f32_i32` for ints, `v_mul_f32`/`v_sub_f32`,
   `v_cvt_f16_f32` for the store, integer pack). `v_cvt_f16_f32` is the one cvt confirmed reliable.
4. A per-super-block value in a reg that `compute0` reuses gets clobbered mid-loop; recompute d/dmin per group
   from HDR (persists across the inner loop).

## Next levers (to close the gap to llama ~1849)

- Schedule tuning: current is a FIXED, un-searched schedule (W2x2 T4x4 BK32, DBUF=0, no PLR) with `s_delay_alu`
  fences on every FP op. BoltBeam schedule search over {BK, tile, WAVES, DBUF/PLR} + minimizing fences is the
  obvious next win (the 8B gen_sched route gets DBUF/PLR/reloc tuning it doesn't have here).
- Reduce per-element fence/decode overhead (batch decode, packed cvt) once ops are trusted.
- int8/MMQ is NOT expected to help (iu8 WMMA is throughput-neutral vs fp16 on gfx1100).

## Tension to be aware of

An older mandate (see memory `14b-prefill-valu-ceiling-wmma-solve`) was "match-then-DELETE `build_gemm_lds2` via
tinygrad codegen (no hand kernels)". This work EXTENDS the hand kernel. The 2026-07-05 session pivoted to the
`gen_sched` substrate-builder model after establishing that the 8B fast path is ALSO hand-asm `build_gemm_lds2`/
`pipe` (so a hand-asm quantized analog is consistent with how 8B actually ships). If the codegen-LDS-staging track
is revived, this route becomes the reference to match/retire.

## Deletion track (2026-07-06): int8 DEAD, Route B is the path (deep codegen, not yet built)

Goal was to replace this hand kernel with tinygrad codegen (no-hand-kernel mandate). Findings:

- **int8/Q8_1 WMMA MMQ route: DEAD.** iu8 WMMA is throughput-NEUTRAL vs fp16 on RDNA3 (no tensor-core win),
  so int8-MMQ = compute-neutral core + MMQ overhead (q8 quant, qsum, per-group affine) that the fp16-fused
  kernel avoids (decode already amortized once-per-tile into LDS). Best case <= 808. Evidence: scalar sdot4
  MMQ = 237 tok/s authority; naive full-role iu8-WMMA microbench (scratchpad/mmq_fullrole_probe.py) bit-exact
  but 0.75 TFLOPS. The MMQ atom (int8 shaped_wmma K-accumulate + Q4_K scale, rel RMSE 0.000000) is kept as a
  correctness asset only. Substrate fix landed en route: renderer WMMA-helper dedup (commit 84efd5172),
  unblocks any multi-WMMA codegen kernel.
- **Route B (fp16-fused-decode-LDS codegen): the only credible deletion route.** Replicate THIS kernel's
  algorithm (Q4_K decode -> fp16 -> LDS-stage the decoded tile -> WMMA) in codegen via
  `bufferize(LOCAL, removable=False)`. Payoff is clear (this hand kernel proves 66 TFLOPS is achievable with
  LDS-staging); the risk is the build. Track-1 codegen (schedule search, shipped) alone is ~40-48 TFLOPS
  global-direct; the 40->66 gap IS the LDS input-staging this kernel does by hand.
- **GO/NO-GO decider:** resolved as NO-GO for the register-resident/targeted-wait shortcut. On 2026-07-08,
  `DEV=AMD:ISA AMD_ISA_REG_ACCUM=1 AMD_ISA_WAITCNT_TARGETED=1 AMD_ISA_WMMA_B128_FRAG=1 REGALLOC_ADDR_REMAT=1`
  on attn_qo `(512x5120x5120)` measured `u0=2,u1=2,loc=0,unr=8 -> status=ok, 16.79 TFLOPS`; full-drain control
  was `16.35 TFLOPS`. The intended table shape `u0=4,u1=4,loc=4,unr=8` is not a measurable GO candidate in the
  current native-ISA path: with `REGALLOC_END_NO_SOURCE_LIVE=1` it returns `WRONG rr=nan`; without that flag it
  faults on GPU. The `u0=4,u1=4,loc=0,unr=8` register-resident shape also faults. Targeted/deferred `vmcnt` alone
  therefore does not lift plain fp16 codegen toward ~58 TFLOPS; keep this hand kernel unless a deeper staging/lifetime
  primitive changes the substrate.
- **Why not built yet:** the LDS-staging application code (WARP address-key + explicit CONTRACT fold +
  removable=False; see docs/codegen-wmma-lds-staging-design-20260705.md:77-99) was cleared from scratchpad and
  must be reconstructed; the cooperative-partition perf step (store_keys != read_keys) is unfinished with
  "uncertain payoff" per that doc. This is dedicated multi-session tinygrad-codegen work
  (`bufferize_to_store` at rangeify.py:397 is the machinery hook). Until then, this hand kernel (808 tok/s) is
  the honest shipped state.
