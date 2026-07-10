# Prefill / Machine-Search Lessons Ledger

Durable learnings distilled from ~63 prefill/machine-search docs that were **decoupled and removed** on 2026-07-10
(we keep the learnings, not the paperwork). Organized by theme, deduped across docs. Where a `route_manifest.py`
authority/provenance comment or a lowering-registry `reuse_files` entry pointed at a removed doc, it now points here.

Live/current docs were NOT folded in here (see `prefill-current-state.md`, the S10/S10.5 scopes, and the
principles files, which remain as files). This ledger is the graveyard of *closed* scopes, refutations, and shipped
results.

---

## gfx1100 4x4 WMMA — parked, and why

- **Parked as of 2026-07-08.** Active generated path is 2x2 / 4x2 / 2x4 only. Root cause: 128 VGPRs are exhausted by
  the C accumulators before A/B fragments, DBUF state, addresses, and the epilogue can fit; 4x4 needs
  `PREFILL_ALLOW_PARKED_4X4=1` to even attempt.
- **The 4x4 generated fault is value-neutral and emergent — do not re-litigate single causes.** It is remu
  bit-exact, survives `AMD_ISA_SCHED=0` and `AMD_ISA_WAITCNT_CONSERVATIVE=1`, and every fixed-register hand replica
  (A0–A5) of each static feature/layout/VGPR-range PASSES. So the scheduler, waitcnt, and every single static feature
  are **exonerated**; the trigger is an emergent property of the generator's *dynamic* register-assignment / linear
  order that no hand kernel reproduces. The fault is hardware-timing/datapath — invisible to a functional emulator.
- **Exoneration ledger (all remu-pass, GPU-fail-unchanged → none is the primary trigger):** direct load addr==dest
  reuse, pack→WMMA spacing, cross-backedge load-scratch reuse, stale packed-fragment age, pack-source provenance.
- **The original 4x4 NaN (earlier, separately fixed):** post-loop store epilogue reused high WMMA-loop scratch VGPRs
  v201/v202 — NOT "disasm cvt", "s_delay_alu timing", or "high-VGPR VALU source" (all disproven).
- **Structural delta that defines the gap:** generated = 1163 instr / 16 WMMA / 64 pack / 128 scalar
  `global_load_u16` with 80 direct load addr/dest overlaps, vs hand b128 control = 808 instr / 64 WMMA / 0 pack / 64
  `global_load_b128` / 0 overlaps. B fragments are strided under `B[K,N]`, so A-only b128 lowering is **not** a
  complete equivalent to the hand control.
- **Terminal-isolation method (if ever reopened):** march from the closest PASSING fixed-register hand analog toward
  the exact generated stream one difference at a time; the first PASS→FAIL perturbation is the trigger. If a
  byte-faithful replica still passes, the trigger is purely the dynamic allocator/scheduler output.

## DBUF / LDS operand staging

- **DBUF is not a numeric/correctness blocker** in the central route-bound harness (rel_rmse ~2.08e-4). The real
  blocker is structural: postrange builds all-stores → barrier → current-LDS-load → WMMA with **no future-stage group
  in the body**, so there is nothing for the scheduler to overlap.
- **Measured DBUF cost:** DBUF ≈7.7 TFLOPS vs non-DBUF ≈11.5 (65536 group-segment bytes); on a bounded worker
  (loc=2,unr=2) generated baseline DBUF ≈7.6–7.9 TFLOPS **beats** D3A ≈7.0.
- **Safe DS-offset folding is a loss.** `PREFILL_DBUF_LDS_CONST_IMM=1` produces more VALU, more/larger DS immediate
  offsets, and LOWER TFLOPS than keeping LDS offsets as materialized VALU arithmetic. Keep the materialized route.
- **Both-operand LDS staging fix:** staging both WMMA operands regressed to spills because
  `PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1` disabled LOCAL pointer grouping too broadly; the fix narrows the
  `devectorizer.py` guard to disable grouping only for the B tile-key local placeholder (slot `993`), after which
  both-side native staging compiles no-spill.
- **Cross-operand pressure is a lifetime/order problem** (address temps, global-load temps, pack carriers,
  LDS-store operands, LDS-load/WMMA operands all live at once). Fix is a verifier-clean order primitive that drains
  one stage (stage A → store A to LDS → kill A temps → stage B …) before starting the next — not "clone two levels
  down" (a probe of that hit rel_rmse 1.22 = wrong).

## K-major stage ownership (the current primitive direction)

- **Waitcnt tuning cannot create missing overlap** — it comes AFTER stage ownership. Targeted wait variants moved
  static instruction counts but not TFLOPS.
- **Destructive suppression is invalid at renderer-matcher scope.** Rewriting matched prologue LDS stores to NOOP by
  slot/addr/epoch/owner key all produce WRONG output (`rr=nan` / `rr=1.4`) because stage identity is gone by late
  lowering and slots are reused across K-phases. K-major stage *ownership* (constructed upstream in postrange) is the
  primitive; additive D3A is refuted; schedule-table and DS-offset tweaks are not it.
- **Rotated-DBUF is the right long-term primitive but is the hard part** — it needs a behavior-changing
  prologue/body/tail rewrite. Every after-the-fact B tile-key suppression variant (drop-global, generic-layout,
  small-N) was REFUTED (WRONG rr≈1.2, density worsened): the B layout/consumer mapping is wrong, not just LDS size.

## Fragment reuse / proof keys

- **Reuse keys must be fail-closed on role/slot/phase/epoch**, else fall back to per-WMMA reloads. AMD isel currently
  receives only the arithmetic address expression, losing the proof tags — so address-only reuse groups exist but no
  proof metadata reaches operand lowering. Closing this is what unlocks hand-LDS2 fragment-reuse density.

## Why hand beats generated (the density thesis)

- **The hand kernel wins purely on WMMA issue density** — it amortizes staging, reduces waits, and avoids per-WMMA
  reloads. That density gap IS the TFLOPS gap. This is the north-star target for the pure-generated transport.
- **Baselines:** harness-fixed generated 8B pp512 = 2549 tok/s vs hand 4413. The default `DEV=AMD` prefill GEMM is
  emitted by HIPRenderer with register-resident operands — tinygrad has NO waitcnt/pipelining authority on that path.
  attn_output out-proj runs at ~16 TFLOPS (~19%) due to its strided A-operand.

## Quant / int8 (14B Q4_K)

- **"int8 gives 2× over fp16" is REFUTED on RDNA3.** iu8 WMMA runs at the *same* rate as fp16 (identical descriptor
  dims (16,16,16), epc (16,16,8)); the 2× is RDNA4/CDNA only. iu8's only lever over scalar `_sdot4` is running on
  separate silicon from VALU (overlap with per-group scale correction), not faster tensor throughput. The Q4_K
  prefill win must come from staying-quantized / bandwidth, not int8 compute.
- **Fused Q4_K dequant→WMMA SHIPPED:** 14B pp512 359→808 tok/s (2.25×, same-session A/B; llama.cpp ~1849), bit-exact
  (rel RMSE ~3e-4 on all four role shapes), ~66 TFLOPS/kernel. The old 359 ceiling was VALU-bound on a separate
  dequant pass; fusing per-element fp16 dequant into the WMMA feed removed it.
- **llama MMQ math (validated, rel_rmse 4.8e-3 vs fp32):** `W = d*sc*q4 − dmin*mn`; quantize activations per-32-group
  to int8 (xq,xsc,xsum); `DOT = Σ q4*xq` (int32, range-safe); `out = Σ_g(d*sc*xsc*DOT − dmin*mn*xsc*xsum)`. Hazard:
  int8 sdot4 sign-extend inside the reduce gives garbage (~6.3 rel_rmse) — use pure-int32 arithmetic.
- **Full-role lowering taxonomy (keep separate):** (1) Tensor oracle = correct algebra but graph-explodes at 14B;
  (2) bounded small lifecycle = proves tiled algebra, still Tensor-graph composed; (3) full-role scheduler lowering =
  the production path (runtime loops own tile_m/tile_n/group, RAW is tile-local, WMMA via tinygrad's SHAPED_WMMA
  substrate). Build level 3, don't stretch level 2. Constraint: no route-local HIP/asm/`__builtin_amdgcn_wmma`/direct
  `Ops.WMMA`.

## Decode

- **Generated decode-attention tile is correct but ~99× slower** than the owned hand kernel; the gap is codegen
  strategy (owned block-tiles: TK=16, 8KB LDS, 128 threads). The W==D variant was REFUTED (6.5/3.6/0.9 tok/s vs
  baseline 103.5/101.8/94.6 @ ctx 512/1024/4096).
- **Split-KV combine is a separate lifecycle/economics tax, distinct from the tile.** A tile can pass every
  kernel-quality and graph-integration layer and still miss W==D because: (a) low combine bytes do NOT imply a cheap
  combine (it can be latency/occupancy-bound), (b) the combine can be the binding W==D lever even when the tile is a
  large local win, (c) Amdahl (attention ~17% of decode) can make a real local win non-promotable. B4 measured
  +5.6–5.85%@ctx4096 (below the +7% bar), verdict `COMBINE_TAX_DOMINATES`. Now enforced permanently via
  `split_kv_economics_audit_v1` + the principles file.
- **Owned-tile buffer-identity KV read (shipped, DEFAULT-ON 2026-06-23):** removing the full-MAXC slice
  materialization gained +18.7/17.4/16.3/13.3%@ctx512/1024/2048/4096 (largest gain at smallest context), byte-identical.
- **Batch-1 decode megakernel** (fuse all layers / one token into one persistent generated kernel) is the
  decode-latency frontier — an L3 descriptor-owned / L5 backend-lowered artifact, never an L0/L1 hand template.

## Machine-search philosophy (embodied in S10.5)

- Target is neither "no ASM" nor "hand kernels everywhere" but **machine search over reusable compiler primitives**;
  DBUF stays fail-closed to single-buffer unless slot/phase/epoch proof exists.
- **S9 exhaustive search** (the first non-byte-identical phase, after the LDS2 oracle was decomposed into
  layout→wait→cadence→lifecycle→emitter→lowerer) is complete only when the whole axis table resolves, not after one
  wait-policy search. The conservative first lifecycle set ran with no new win; LDS memory layout intentionally
  deferred.
- **Pure-substrate north-star:** recover hand fp16 graph-GEMM perf with a pure tinygrad-codegen substrate (no
  Ops.INS, no wmma.py), then delete the hand kernel, keeping `PURE_MACHINE_SEARCH_ONLY=1` green. PLRA/PLRAB prefetch
  is 2nd-order / deferred.
- **Harness discipline:** phase-0 and route-bound measurement must reuse the existing whole-prefill authority harness
  (`extra/qk/prefill_whole_synced.py`), not a new benchmark path. The route-bound correctness gate is NOT native-ISA
  evidence (its runner forces `DEV=AMD` → HIP/C renderer); a compile-only fail-closed LDS-allocation estimator
  (`--resource-search`) is what drives schedule choice before launch.

## Operational hazards & ownership

- **NEVER `timeout`/`pkill` a live `DEV=AMD` run.** Hard-killing mid-kernel jams the RDNA3 MES ring (`dmesg`: "MES
  ring buffer is full"; ~41 kworker/u65 threads in D-state; every `DEV=AMD` realize hangs while `DEV=PYTHON` still
  works). Recover by reboot (cleanest) or `echo 1 > /sys/kernel/debug/dri/*/amdgpu_gpu_recover`. Instead, bound the
  work (smaller model/context) or `run_in_background` and wait.
- **Ownership boundary:** tinygrad owns execution, kernels, compiler/backend lowering, and hardware evidence;
  BoltBeam owns model facts, candidate/search schema, evaluation policy, ledgers, roofline attribution, and reports;
  BubbleBeam/FutureSight is the only current route path (old Beam/FutureSign wording is historical/compat-only).
  FutureSight applies its COALESCE via `opts_to_apply` → `apply_opt` in `postrange.py`, never through the timing beam
  (which was removed).
