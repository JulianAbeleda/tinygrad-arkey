# Exhaustive structural-delta scope — generated decode tile vs owned hand-asm (2026-06-27)

Method: 8 parallel read-only delta deep-dives + a completeness critic over the measured transfer + ISA disasm
(`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128` + full stack vs `owned_flash_tile_gqa_whole`).
Lineage: `docs/decode-cooperative-stage-lanemap-result-20260627.md` (the 2.35× isolated / 1.75× in-model stack).

## THE REFRAME (headline — this changes the closure strategy)

The 56× isolated / 3–15× in-model gap is **latency / ILP-bound, NOT throughput-bound**. The codegen scope's
"scalar loads / global_load_d16=0 / more cross_lane" framing is **superseded** — three of its premises are now
*false* against the real disasm:

- **The generated tile emits FEWER instructions** than owned (disasm 414 vs 561 lines; global_load 10 vs 23,
  v_add_co 22 vs 80) — yet is ~56× slower. So instruction *count*/throughput is not the gap.
- **Its loads are WIDER, not narrower**: generated `global_load_b64`×10 (half4 staging via the M2 LaneMap) +
  `global_store_b128`; owned is `global_load_d16`×22 (scalar fp16). The "load-width delta" was a premise error.
- **Occupancy/VGPR/LDS are MATCHED**: generated vgpr56 ≤ owned vgpr64, LDS 8192B = 8192B, sgpr8 = sgpr8,
  4.0 wg/CU, **zero scratch** (no spills). Occupancy is not the gap.

The gap is the **serial online-softmax recurrence critical path** (acc/den/mx carried serially across the token
axis) **executed without software pipelining**. Owned (hipcc -O3 .co) and generated (tinygrad→HIPRenderer→comgr)
see the *same* per-token-reduce topology, so it is **not** a comgr-vs-hipcc quality gap on identical structure —
it is the *emitted structure*: staged-`ds_bpermute` REG round-trips, `STACK`-of-`CAST` fdot2 packing, exp
range-reduction and per-token predication all sitting *on the serial carry chain*, none of which a downstream
reorder can hide. This is why the gap **grows with ctx** (the carry critical-path length = per-workgroup token
count = Tc/S) and why the attention tile — the *only* ctx-scaling term in the decode step (the 30+ layer GEMVs
are ctx-independent) — swamps the step at ctx4096.

## Delta ledger (prioritized)

### Tier 1 — the real levers (latency / critical-path; all Track-A codegen, default-off)

| # | Delta | Lev | Evidence | Closure (one line) |
|---|---|---|---|---|
| 1 | **Serial online-softmax carry under unroll** (no 2-level/split accumulator) | **med** | carry crit-path = Tc/S → drives the gap-grows-with-ctx; isolated ctx-slope 7.2× vs owned 3.9× | `SCHED_UNROLL_SPLIT=<U>`: duplicate the carry `DEFINE_REG` **private per copy** → independent partials + online-softmax combine epilogue |
| 2 | **Recurrence not software-pipelined** | **med** | `s_waitcnt 19`; SCHED_LIST-alone refuted (7023→7075µs); SCHED_UNROLL stack = 2.35× so far | **measure first** (hotloop-diff), then mature SCHED_UNROLL pipeline depth; decide scheduling-vs-structural |
| 3 | **Q not register-hoisted + per-use f32→f16 convert + per-block reload** | med | owned hoists Q once as `half2` (hip:237-238), 0×`v_cvt_f16_f32`; generated reloads + converts per block | hoist Q to REG in the prologue (kernel-authoring), keep `half2`, drop the per-use cast |
| 4 | **exp2 range-reduction on the carry chain** | low-med | tinygrad `_fexp=(x·LOG2E).exp2()` → `v_cmp 0xc2fc0000 + 2×v_cndmask + v_exp + v_ldexp` (gen v_ldexp×4, owned×0), on the serial path | a cheaper/clamped exp2 lowering (no range-reduction when score range is bounded), or move corr-exp off the carry |
| 5 | **Masked predication (`in_r.where`) on the carry chain** | low | per-token `where(t<Tc, …)` → `v_cndmask` on sc/corr/p, on the critical path | with the runtime-bound scan (Tier 2 #6) the tail mask vanishes for full blocks → fewer cndmask |

### Tier 2 — real but small (cleanups; structural)

| # | Delta | Lev | Closure |
|---|---|---|---|
| 6 | **Split policy**: generated pins L=86 (compile-time 96-pos masked scan) vs owned S=48 + runtime `ceildiv(Tc,48)` scan | low | ctx-bucketed S=48 pinning (`DECODE_ATTN_BLOCK_TILE_FIXED_S` already exists; derive `l_route=ceildiv(Tc_bucket,48)`) → recovers low-ctx occupancy (ctx512: 48→384 wg) + removes ≤12% over-scan. Deeper: a symbolic/data-dependent REDUCE trip-count codegen primitive. **Does NOT bend the ctx-slope** (grids already match at ctx4096). |
| 7 | **Combine**: 3 post-tile programs (tile→gmax→combine) vs owned 2 (tile→combine) | low | fuse the per-head global-max into the combine kernel (a generic vertical-fusion of two custom kernels reducing the same S axis). Worth <1% at the current operating point; do it as a generic primitive, not for speed. |
| 8 | **LDS lifecycle**: 1 barrier (missing WAR barrier; correctness) | low | a generic `LDS_WAR_BARRIER` pass (correctness); separately an opt-in `LDS_DOUBLE_BUFFER` authoring transform (the only perf-positive item here) to overlap block b+1 staging with block b compute |

### Tier 3 — NON-deltas / corrections (do NOT chase — the critic refuted these)

| Item | Why it's not a delta |
|---|---|
| **cross_lane 10 vs 5** | a **2-token-unroll artifact**: two independent 5-step `ds_bpermute` ladders (one per unrolled token). **Per-token it is 5 = matched.** The 10 is a *side effect* of the ILP-positive unroll. |
| **Load width / cache dtype** | generated is **ahead** (b64 half4 + b128 store) vs owned d16-scalar; cache is already fp16 on the measured path. Pushing to b128 (W=8/`ALLOW_HALF8`) is a tiny optional refinement, not a gap. |
| **Occupancy / VGPR** | matched (vgpr56≤64, LDS=8192, no spills). Convert to a **default-off occupancy guardrail gate** that fails if a Tier-1 change (more unroll → more VGPR) drops below 4 wg/CU or induces spills. |
| **GQA K/V broadcast** | already handled — 4 warps share one kv-head, K/V staged to LDS once per block. |

## Structural-floor risk (the thing that could cap Track A)

The **staged `ds_bpermute` cross-lane reduce** (`qk_warp_reduce_lowering.py:33-37`) forces every shuffle through
a REG slot → `ds_read`+`ds_write`+`lgkmcnt` drains that **no UOp reorder removes**. If the hotloop-diff (#2)
shows the residual is this REG round-trip rather than schedulability, the next primitive is a **register-resident
cross-lane reduce** (not more scheduling). Note: **Track B (`qk_asm_scheduler.py`) is NOT wirable to decode** —
it consumes `list[Inst]` before `Ops.INS`, which only exists for the prefill hand-asm GEMM; decode emits C++ via
HIPRenderer with no ISA path (`ops_amd.py:1026`, parked `P0_3_SCHEDULE_ASM_PARK`). A decode `ISARenderer`/`Ops.INS`
backend is a separate large project, funded only if Track A ceilings.

## Closure sequence (measure-first, leverage-ordered)

1. **MEASURE (gate everything on this):** build `extra/qk_decode_hotloop_schedule_diff.py` (named in
   `docs/decode-codegen-scheduler-capability-scope.md:31-43`, absent) — disasm both tiles' backward-branch loop
   body, report in-loop `s_waitcnt lgkmcnt(0)`/`vmcnt(0)` counts, `ds_bpermute→first-consumer` issue-distance,
   and whether next-iteration independent work is interleaved into the current reduce-wait. **Converts the wall
   from argument to fact** and decides scheduling-vs-structural-floor.
2. **Tier-1 #1 (highest leverage, the ctx-slope lever):** `SCHED_UNROLL_SPLIT` 2-level accumulation. Target:
   isolated ctx-slope (ms@4096/ms@512) drops from 7.2× toward owned 3.9×, and the in-model gap stops growing
   with ctx. Watch VGPR (U private states cost U·(R+2) regs; gate with the occupancy guardrail).
3. **Tier-1 #3/#4/#5 (shorten the per-token critical path):** Q register-hoist, cheaper exp2, drop tail
   predication via the runtime-bound scan. Each is a small, independent, microgate-gated authoring/codegen change.
4. **Tier-2 cleanups** (#6 split policy → recovers low-ctx occupancy; #7 combine fusion; #8 double-buffer) as
   generic default-off primitives.
5. **If Track A ceilings** (hotloop-diff says structural): register-resident cross-lane reduce primitive; only
   then consider the decode-ISARenderer/Track-B backend.

## Discipline (every step)

- **Authority = W==D + ctx-slope, not isolated.** Isolated tile timing is not decode authority (measurement
  confounds). The gate is: (a) `BLOCK_TILE_MICROGATE_PASS` max_abs unchanged 1.526e-05; (b) W==D
  (`qk_decode_runtime_overhead.py`) moves toward owned **and the gap stops growing with ctx**; (c) default-off
  byte-identical when flags unset (cache key already carries them).
- **This is a search-CAPABILITY goal, not a shipped-throughput win.** Owned already ships at parity and is
  HBM-bound; the win is "the *machine* generates a competitive tile," measured as the ctx-slope flattening and
  the generated-route W==D closing on owned — not a new default. Do not re-derive the already-refuted cheap
  levers (SCHED_LIST-alone, occupancy-forcing, inline-reduce).
