# Track B → 100%: generated prefill WMMA GEMM on the AMD native-ISA path

The buildable plan to take the incomplete AMD layers (registry L3/L4/L5/L6/L7) to 100% for **8B** (pure fp16 WMMA
GEMM) and **14B** (fused Q4_K dequant), recovering the hand kernel's perf with NO handwritten kernel — then delete
`extra/qk/prefill/wmma.py`. Completion criteria + coverage: `docs/substrate-layer-completion-registry.md`.

## Guiding principle: REUSE / CENTRALIZE / MODULARIZE — no duplication
This is a **completion of the existing renderer/codegen**, not a parallel stack. Every task below builds on existing
code (tagged "reuse:"). `extra/qk/prefill/wmma.py` is a **spec reference only** — the kernel we delete; mirror its
instruction structure, never import/call it. New shared logic (fragment-VGPR ranges, waitcnt packing, Q4_K decode)
each lives in exactly ONE helper consumed by all callers.

## Dependency chain (forced order)
```
B0 (WMMA GEMM on ISA)  →  B1 (pipelining+waitcnt)  →  8B DONE (~58 TFLOPS, ~4413 tok/s)
        └───────────────────────────────────────────→  14B delta (Q4_K B-source swap) → 14B DONE (~23 TFLOPS, ~808)
```
**B0 is the entire weight of Track B.** B1 is thin and mostly in-place; 14B is a small reuse-heavy delta. 8B must
land first — 14B has nothing to swap its B-source into until the generated 8B GEMM exists.

---

# B0 — WMMA GEMM on `tinygrad/renderer/isa/amd.py` (the long pole)

Closes **L5 (40→100), L7 (5→100), L3 (70→100)**. Build order **L5 → L7 → L3**: fragment ranges are the enabling
primitive; L7 emits WMMA using a `V_PACK` bridge for first bit-exact correctness *without* b128; L3 then replaces the
bridge with `2× ds_load_b128` for bandwidth. Cross-cutting: keep full-drain `s_waitcnt(0)` (targeted is B1); add a
`v_wmma` latency (~16) to `_sched_lat` (`amd.py:669-674`).

**How `Ops.WMMA` arrives (reuse — do not re-derive):** `_apply_tc_opt` (`postrange.py:308-320`) builds it;
`no_vectorized_wmma` (`devectorizer.py:235-244`) splits to per-group WMMAs `dtype=float.vec(8)`, srcs vec(16)/vec(16)/vec(8).
The `amd_rdna3` `TensorCore` descriptor + LaneMap/swizzle (`tc.py:140-147`) already define the fragment layout —
**the ISA renderer consumes the same tc descriptor the HIP/LLVM renderers do.** The encoder is ready
(`rdna3/ins.py:1764`); `op_regs` auto-sizes each WMMA operand to 8 VGPRs from the base (`dsl.py:379-381`).

## L5 — fragment register allocation
Reuse: the pinned-accumulator (`_accum_pin` `amd.py:158-167`) + LDS-bump (`_lds_byte_offset` `:142-150`) pattern —
fixed deterministic assignment, never a linear-scan virtual. Do NOT extend `LinearScanRegallocContext`.
| # | Task | file:line | reuse |
|---|---|---|---|
| L5.1 | `FRAG_BASE=200`/`FRAG_TOP` constants (cap base+7 ≤ 237, the VGPR≥238 trap) | `amd.py:38` | mirrors `ACCUM_PIN_BASE/TOP` |
| L5.2 | `_frag_base(ctx,key,n,align)` bump allocator → aligned base or `None` | `amd.py:152-167` | mirrors `_accum_pin` |
| L5.3 | extend `_vpool` to exclude the fragment region when WMMA present | `amd.py:52-55` | same as ACCUM does |
| L5.4 | pressure guard: `None` → `NotImplementedError` (no-spill contract) | — | matches `amd.py:625,628` |
Gate: one `Ops.WMMA` → 3 non-overlapping 8-VGPR ranges, all base+7 ≤ 237.
**Top risk (invariant):** scheduler/waitcnt key hazards on a SINGLE `r.offset` (`amd.py:693,779`), not the 8-reg span.
Fragment regs MUST live only in the reserved region so no ordinary virtual targets `base+k`; keep full-drain waitcnt.

## L7 — tensor-core emit
| # | Task | file:line | reuse |
|---|---|---|---|
| L7.1 | `AMDOps.V_WMMA=44` + WMMA/bf16/iu8 encoder imports | `amd.py:18-27,83` | autogen `rdna3/ins.py:1764` |
| L7.2 | `tensor_cores = amd_rdna3` on the class; confirm `half` not rejected | `amd.py:614-622` | `tc.py:140` descriptor |
| L7.3 | `isel_wmma` rule (before the catch-all INS): pack vec(16) A/B → 8 pinned VGPRs via `V_PACK`; C init via `V_CONST`; emit `V_WMMA` | `amd.py:399` | existing `V_PACK` `:545`, `V_CONST` `:512`, constrained-vreg path `:424` |
| L7.4 | `lower_inst` `V_WMMA` → `v_wmma_f32_16x16x16_f16(vdst/src0/src1/src2=_V[b:b+8])` | `amd.py:~486` | slice→`Reg` `dsl.py:23` |
Gate: 16×16×16 half GEMM bit-exact on `DEV=AMD:ISA` (rmse ≤ 3e-4); then 32×32×32 (4 accumulators) for pressure.
Risk: vec(16)→8-VGPR **element order** — pack pairs `(2i,2i+1)`→reg `i`; RDNA3 A=`A[l][0:16]`, B transposed
(`wmma.py:3-8` spec); `.h/.l` half-select (`dsl.py:41`) is the fix if halves swap.

## L3 — wide memory b128 (perf replacement for the V_PACK bridge)
| # | Task | file:line |
|---|---|---|
| L3.1 | `DS_LOAD_B128/DS_STORE_B128/GLOBAL_LOAD_B128` enum + imports | `amd.py:18-27` |
| L3.2-3 | b128 branch in `isel_load`/`isel_store` targeting 4-aligned fragment sub-ranges | `amd.py:264-268,281-285` |
| L3.4 | `lower_inst` b128 → `_V[base:base+4]` | `amd.py:502-509,517-520` |
| L3.5 | switch `isel_wmma` fill from V_PACK → `2× *_b128`; `align=4` for b128 dests | — |
Gate: same GEMM, b128-sourced, byte-identical to V_PACK path, V_PACK count → 0. Risk: 16-byte alignment (depends on L5).

---

# B1 — pipelining + targeted waitcnt (thin, mostly in-place)

Closes **L6 (40→100), L4 (25→100)**. Depends on B0 exposing WMMA fragments as `Reg` **ranges** (Risk R1 below).

## L6 — targeted `vmcnt(n)`/`lgkmcnt(n)`
Reuse: **extend `_insert_waitcnt` in place** (`amd.py:747-792`) — it already tracks pending loads + hazards; add
targeting as a mode (default full-drain unchanged). Centralize simm16 packing in one helper (spec: `wmma.py:19-28`).
| # | Task | file:line |
|---|---|---|
| L6.1 | `pend_vm`/`pend_lgkm`: sets → issue-ordered lists with monotonic `seq`; record FULL def ranges (not base) | `amd.py:764,787-789` |
| L6.2 | `AMD_ISA_WAITCNT_TARGETED` (default 0); `vmcnt(n)`/`lgkmcnt(n)`/packed-both emitters | `amd.py:755` |
| L6.3 | per-class `n` = count of loads issued after the newest dependency; prune landed; packed dual-field wait | replace `amd.py:784` |
| L6.4 | keep full-drain at `s_barrier`/`s_endpgm`/`ds_load`-RMW; no targeted store waits | `amd.py:781-783` |
| L6.5 | carry pending loads across the loop backedge (the overlap-enabler); full-drain only if a store is outstanding | `amd.py:775` |
Gate: bit-exact vs full-drain on `DEV=PYTHON`, then TFLOPS lift on AMD; disassembly shows `vmcnt(LPB)`-style waits.

## L4 — software-pipelined double-buffer (NO new modulo pass — recommended)
Reuse: the existing `_schedule` list scheduler (`amd.py:676`) already front-loads height-200 loads; the existing
DBUF unroll-by-2 peel (`postrange.py:521-539`) already emits two K-copies in one block. Overlap = (front-loaded loads)
× (targeted waitcnt leaves next-iter loads in flight) × (DBUF shape). Two fixes only:
| # | Task | file:line |
|---|---|---|
| L4.1 | add `v_wmma` latency case (~16) BEFORE the generic `v_` in `_sched_lat` | `amd.py:669` |
| L4.2 | use register-buffered fragments (build_gemm_pipe style) so the conservative mem-chain (`:708`) doesn't over-serialize LDS double-buffer | codegen/B0 decision |
| L4.3 | assert the unrolled K-body is a single basic block (no label/branch between copies) | verify `amd.py:679` |
Escalation ONLY if it plateaus short of ~58: explicit modulo pass behind `AMD_ISA_SWPIPE=1` (spawn a Fable review
first, per standing rule). Risk R8: back-to-back dependent `v_wmma` may need `s_delay_alu` on gfx1100 — confirm.

---

# 14B delta — fused Q4_K→fp16 dequant (small, reuse-heavy)

Closes the **14B-specific L1-2 addition**; rides B0+B1 unchanged (reduce/WMMA/epilogue byte-identical to 8B).
Reuse (almost everything already exists):
- **Q4_K decode UOp graph** — `extra/qk/quant/q4_k_gemv_primitive.py` (`_q4k_weight`/`_q4k_group_params`/`_q4k_quant`),
  reference-pinned bit-exact (`layout.py:161 q4_k_reference`), already used by the VALU path. **Reuse verbatim.**
- **Residency** — `prefill_packed_weight()` (`qk_primitives.py:65`) keeps ~9GB packed resident, no OOM. Route wired
  (`prefill_routes.py:274`, `prefill_graph_gemm_route.py:151`).
- **LDS fusion** — `_tc_local_stage` bufferize (`postrange.py:397,429`) stages *whatever the B-source computes*, so
  the decode flows into the B-tile automatically.
| # | Task | file |
|---|---|---|
| T1 | `w_f16()` = `_q4k_weight(...).cast(float16)` (only new line) | `q4_k_gemv_primitive.py` |
| T2 | UOp WMMA GEMM kernel: A=fp16 x, B=`w_f16` over (blk,grp,pos), `Ops.REDUCE(ADD)` for the TC matcher; `grp` UPCAST → static | new fn, mirror `q4k_gemm_packed_load_reduce_out_kernel:153` |
| T3 | fp16 TC opts → `v_wmma`; `PREFILL_TC_LOCAL_STAGE=b`/`both` stages B to LDS | route env |
| T4 | generated route (no `Ops.INS`) replacing `build_gemm_lds2_q4k` call | `prefill_graph_gemm_route.py:164` |
| T5 | wire behind `PREFILL_Q4K_WMMA_FUSED` + a "generated" sub-flag | `prefill_routes.py:274` |
| T6-7 | verify B1 double-buffer composes; bit-exact vs `q4_k_reference` | gate |
What 14B needs that 8B does NOT: packed uint32 B PARAM; the decode subtree; Q4_K-structured reduce axes (blk/grp/pos).
**Sidesteps all 4 hand-asm gfx1100 gotchas** (VGPR≥238, s_delay_alu, fp16-arith, per-group reg clobber) — renderer
concerns, not the author's. Ceiling ~23 TFLOPS (decode-VALU-bound) → win is escaping the 365 tok/s fallback.

---

# Master ordered task list
1. **B0.L5** (fragment regalloc: FRAG pool + `_frag_base` + `_vpool` + pressure guard)
2. **B0.L7** (V_WMMA op + tensor_cores + isel_wmma via V_PACK + lower_inst) → **GATE: 16³ then 32³ bit-exact**
3. **B0.L3** (b128 loads; swap V_PACK→2× b128) → **GATE: byte-identical, 0 V_PACK**
   → *8B generated WMMA GEMM exists at ~40 TFLOPS (single-buffer, full-drain)*
4. **B1.L6** (targeted waitcnt in `_insert_waitcnt`) → **GATE: bit-exact + TFLOPS lift**
5. **B1.L4** (`v_wmma` latency + register-buffered frags + DBUF peel) → **GATE: TFLOPS 40→~58, bit-exact**
   → *8B DONE: pp512 → ~4413 tok/s target*
6. **14B** (T1-T7: reuse decode + source-swap) → **GATE: bit-exact vs q4_k_reference; ~808 tok/s**
7. **Delete** `extra/qk/prefill/wmma.py` + the `PREFILL_GRAPH_GEMM` raw-INS route; confirm `PURE_MACHINE_SEARCH_ONLY=1`.

# Cross-cutting invariants & top risks
- **R1 (B0↔B1):** targeted waitcnt is correct ONLY if pending-load tracking keys on the FULL fragment VGPR range —
  hard dependency on L5's range representation. Silent under-wait = wrong answer, not a crash. Keep full-drain default.
- Fragment regs live only in the reserved region (single-offset hazard tracking stays sound).
- No-spill: fail loudly on VGPR pressure (B0 scope); real spill is out of scope.
- Prove correctness on `DEV=PYTHON` before every `DEV=AMD` run (never force-kill a live AMD run — MES-ring wedge).
- Each gate is bit-exact FIRST, TFLOPS second.

# Net new surface (the no-duplication payoff)
Everything else reuses/extends existing code. Total NEW surface across all of Track B:
- **B0:** ONE module-local helper `_frag_base` (centralized contiguous-VGPR allocator, mirrors `_accum_pin`/`_lds_byte_offset`); FOUR `AMDOps` (`V_WMMA`, `GLOBAL_LOAD_B128`, `DS_LOAD_B128`, `DS_STORE_B128` — each non-redundant, routed through the existing isel/lower dispatch); ONE isel rule (`isel_wmma`); ONE attribute (`tensor_cores = amd_rdna3`, pure reuse of the shared tc descriptor). ZERO new encoders, ZERO changes to `regalloc.py`/`elf.py`/`ins.py`, ZERO imports from `wmma.py`.
- **B1:** ONE staticmethod `_waitcnt_simm16` (centralizes the packing; all emit sites route through it); `_insert_waitcnt` extended in place (set→ordered-list); ONE `_sched_lat` line (`v_wmma`→16). NO new waitcnt pass, NO new scheduler.
- **14B:** ONE thin index adapter calling the EXISTING `_q4k_weight` decode verbatim (+`.cast(f16)`); ONE generated route replacing the `Ops.INS` call. ZERO new dequant math, ZERO new LDS-staging/residency code.
Centralized single-source-of-truth helpers: `_frag_base` (fragment ranges), `_waitcnt_simm16` (wait packing), `_q4k_weight` (Q4_K decode — shared by VALU fallback AND fused-WMMA). Anti-duplication guard: the Tensor-family Q4_K decode in `prefill_int8_wmma_spec.py` is for the REFUTED int8 path — do NOT use it; reuse the UOp-family `_q4k_weight` only.

Source scopes: agent B0/B1/14B outputs (2026-07-06); census `docs/prefill-substrate-layer-census-20260706.md`.
</content>
