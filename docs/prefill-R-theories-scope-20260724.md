# 8B prefill attention: R theories (scope, 2026-07-24)

Successor to `docs/prefill-needle-theories-20260724.md`, whose three theories are all now tested: T1 dead,
T2 padding-exhausted, T3 shipped at +1.7%. That work established **where the time is not**: loads are only
15.1% of the loop body and 23.6% of SQ busy cycles, and every load-side lever measured small or negative
(V-vectorization −3.4%, G2 LDS staging −1.65%, work-per-wave VGPR-infeasible).

**R — everything that is not a load and not a WMMA — is 792 of 952 instructions (83%) and 76.4% of SQ busy
cycles.** It has never been attacked. This scope decomposes it and names three theories.

## The shipped loop body, exactly

`amd_gfx1100_q16_grid_hd128_loop_attention`, 8B/gfx1100, `PREFILL_V_TRANSPOSED=0` (the SHIPPED arm),
per KV tile:

| group | instrs | share | |
|---|---:|---:|---|
| **sync/sched** (`s_waitcnt`, `s_delay_alu`, `s_clause`) | **213** | **22.4%** | T4 **DEAD** (97 are T6's LDS waits) |
| other VALU math (softmax arithmetic, rescale, alpha) | 175 | 18.4% | — |
| **max/min** (`v_max_f32` ×135) | **135** | **14.2%** | T6 |
| global loads (16 `b128` K + 128 `d16` V gather) | 144 | 15.1% | *closed* |
| **mask** (`v_cndmask` ×56, `v_cmp` ×47) | **103** | **10.8%** | T5 |
| **cross-lane reduce** (`ds_bpermute_b32`) | **96** | **10.1%** | T6 (+ its 97 waits, see T4 verdict) |
| transcendental (`v_exp_f32` ×16, `v_ldexp` ×16) | 32 | 3.4% | — |
| other SALU/branch | 28 | 2.9% | — |
| LDS P repack (`ds_store_b16` ×8, `ds_load` ×2) | 10 | 1.1% | *closed, innocent* |
| **WMMA (the only useful work)** | **16** | **1.7%** | — |
| total | 952 | | |

Reproduce with `scratchpad/kv_tile_amortization_probe.py` (compile-only, no GPU).

---

## THEORY 4 — over-conservative waitcnt insertion (largest single group, pure codegen)

**Claim:** 213 instructions (22.4%) are dependency-tracking scaffolding, and the count is **invariant to
the number of loads it is tracking**.

**The evidence that makes this more than a guess:** the V-vectorization arm cut global loads from **144 to
32 (−78%)** and `s_waitcnt` went **112 → 113**. Unchanged. 112 waitcnts for 32 loads is >3 per load. If
waitcnts tracked load completion they would have fallen with the loads; they did not, which points at
per-use rather than per-dependency-group emission.

**Where:** the waitcnt insertion pass in the AMD ISA backend, post-regalloc section of
`tinygrad/renderer/isa/amd.py` (banner at "post-regalloc: build real rdna3 Insts + waitcnts";
`lower_inst`). Also relevant: `test/unit/test_amd_isa_frag_waitcnt.py`, `test_amdllvm_waitcnt.py`.

**Lever:** batch a load clause and emit one `s_waitcnt vmcnt(N)` per dependency group instead of per use;
drop `s_delay_alu` hints that guard already-satisfied dependencies.

**Risk: this is the one theory here that can produce silent wrong answers.** A missing waitcnt is a race,
not a crash — it may pass once and fail under different timing. Numerics passing is necessary but NOT
sufficient. Require: repeated numerics runs, and treat any nondeterminism across runs as a hard fail.

**Prize:** if half of the 213 are removable and cycles track instructions in this group, ~10% of the loop
body. Note the standing lesson: cycles/instruction differs by class (`d16` 3.02 vs `b128` 6.61), so an
instruction-count delta is NOT a cycle delta — measure, do not extrapolate.

### VERDICT 2026-07-24: **DEAD — irreducible, and the theory above is wrong in three places.** No code changed.

**(a) Wrong file. `tinygrad/renderer/isa/amd.py` is not in this kernel's compile path at all.**
`amd_gfx1100_q16_grid_hd128_loop_attention` is emitted as **HIP C source by `HIPRenderer` and compiled to
gfx1100 by `HIPCompiler` (LLVM)**. `AMDISARenderer` — and therefore `_insert_waitcnt`, `lower_inst`, and
the whole "post-regalloc: build real rdna3 Insts + waitcnts" section — is registered *only* under
`DEV=AMD:ISA` (`tinygrad/runtime/ops_amd.py:994`, comment: "Native ISA is research tooling, not part of
ordinary AMD execution"). `DEV=AMD` gives `renderer=HIPRenderer compiler=HIPCompiler`; none of
`prefill_whole_synced.py` / `prefill_hd_sweep_numerics.py` / `prefill_flash_e2e_parity.py` mentions
`AMD:ISA`. The tell is in the disassembly itself: `v_dual_mov_b32` (VOPD packing), `s_delay_alu`, and
`s_clause` are LLVM `SIInsertWaitcnts` / `SIInsertHardClauses` / `GCNHazardRecognizer` output — `amd.py`
emits no such instruction anywhere. **No edit to `amd.py` can change one of the 213.**

**(b) Wrong attribution — 97 of the 112 `s_waitcnt` do not track loads.** Exact taxonomy of the 213
(shipped arm, 952-instr body):

| | count | what it guards |
|---|---:|---|
| `s_waitcnt lgkmcnt(0)` | **97** | **LDS**: `ds_bpermute_b32` returns (Theory 6's reductions) |
| `s_waitcnt vmcnt(*)` | 15 | the 144 global loads = **0.10 waits/load** |
| `s_delay_alu` | 78 | VALU dep *hints* — not waits |
| `s_clause` | 12 | 8 K b128 pairs + 4 V bursts of 32 |
| `s_waitcnt_depctr 0xfff` | 11 | VALU→SGPR→VALU hazard (no HW interlock on gfx11) |

90 of the 97 `lgkmcnt(0)` guard exactly **one** `ds_bpermute`, 7 guard two — 96 bpermutes, 97 waits. gfx11
`lgkmcnt` has no per-register scoreboard and DS returns may reorder, so a wait-to-zero before each
consumer is mandatory. **The "112 waitcnts for 32 loads is >3 per load" figure compares LDS-return waits
against a global-load count.** Decomposing the V-vectorization A/B that the theory rests on:

| arm | loads | `ds_bpermute` | lgkm-domain waits | vm-domain waits |
|---|---:|---:|---:|---:|
| shipped `vT=0` | 144 | 96 | 98 | 15 |
| `vT=1` (V-vec) | **32** | 96 | **98** | **16** |

The invariant 98 tracks the invariant LDS traffic; the load-tracking 15 was already near-optimal and stayed
there. Nothing is unexplained, and nothing points at per-use emission.

**(c) No redundancy exists.** All 17 candidate back-to-back waits are legitimate. Six are the PV
software pipeline `vmcnt(7)→(6)→(5)→(4)→(3)→(2)→(0)`, one WMMA per wait, each counter strictly tighter than
the last — collapsing them into one `vmcnt(0)` is a **pessimization**. The rest are cross-domain (lgkm then
depctr) or guard distinct `v_cmp_gt_f32_e64 sN` → `v_cndmask_b32_e64 …, sN` pairs, where the depctr is
architecturally required. A 32-load `s_clause 0x1f` burst already gets **one** wait, not 32 — the lever's
premise ("one wait per clause group instead of per use") is already what LLVM does. `s_delay_alu` is a hint
that *suppresses* hardware stall detection, and 25 of the 78 carry `instskip(SKIP_3)` covering four
instructions each; deleting them adds bubbles. `s_clause` and `s_delay_alu` are throughput optimizations
and arguably should never have been in an "overhead" bucket.

**(d) Consequence for Theory 6's prize — T4 and T6 are the same instructions.** 97 of T4's 213 are the
waits on T6's 96 bpermutes. The two theories are not independent and their shares must not be added. Killing
a bpermute kills its wait, so **T6's real group is 96 bpermute + 97 lgkm wait + 135 `v_max_f32` = 328
instrs = 34.5% of the body**, not the 24.3% listed below — the largest addressable group in R, and the only
route to the 213.

Probe: `scratchpad/t4_dump.py` (compile-only). Working tree unchanged; no numerics/throughput run was
warranted because no code was modified.

---

## THEORY 5 — mask specialization: only the diagonal tile needs masking

**Claim:** 103 instructions (10.8%) apply the causal/validity predicate to **every element of every KV
tile**, but for a given wave a tile is one of three kinds:
- **fully masked** (entirely past this wave's last query row) — already eliminated by T3's dynamic bound
  (`PREFILL_CAUSAL_TILE_SKIP`, `c44905a18`),
- **fully valid** (entirely at or before this wave's first query row) — needs NO masking at all,
- **the diagonal tile** — the only one that needs per-element masking.

With T3 shipped, every tile a wave still visits is fully valid except the last one. So the mask is being
computed for `N-1` tiles that cannot be masked.

**Lever:** peel the boundary tile. Loop tiles `0 .. tiles_needed-2` with `validity_mode="all_v1"` (no
predicate), then emit one peeled final tile with `validity_mode="causal_v1"`. Both the loop and the peel
are constructed in `tinygrad/schedule/wmma/kernels.py` — `AMDRowSoftmaxRepackSpec(validity_mode=...)` is
built there, so this needs no change to `amd_attention_abi.py` or `uop/ops.py`.

**Composes with T3** (T3 removed tiles after the diagonal; T5 removes masking from tiles before it), and
is exact for the same reason: a fully-valid tile's predicate is a tautology.

**Prize:** ~10% of the loop body on `(N-1)/N` tiles. Same order as T3's +1.7%. Do not project higher —
T3's own estimate (+2.8%) overshot its measurement (+1.66%) by ignoring the cost of the machinery.

**Watch for:** the peel duplicates the tile body, so instruction *count* per kernel rises even as work per
tile falls. Judge on throughput, not on the probe's total.

---

## THEORY 6 — online-softmax reduction structure (231 instrs = 24.3%)

**Claim:** `96 ds_bpermute + 135 v_max_f32` per tile is more cross-lane reduction than the algorithm needs.

**Arithmetic:** `AMDRowSoftmaxRepackSpec.xor_masks = (1,2,4,8)` = 4 butterfly steps over a 16-wide row,
and the C-fragment gives each lane 8 elements (`qk_c_lanes=8`). One full reduction is therefore
`4 × 8 = 32` bpermute. Measured **96 = 3 × 32**, i.e. three full cross-lane passes per tile. Online softmax
needs two (row max, row sum). Likewise 135 `v_max_f32` against the ~40 a single max-reduction plus an
old-`m` compare would need.

**Questions to answer before changing anything:**
1. What are the three passes? (`Ops.MAX` is lowered to bpermute+max pairs by `_hip_native_bpermute_max` —
   confirm whether the row-sum also lowers through MAX, and whether `dynamic_kv_v1` adds a third.)
2. Can max and sum share one butterfly traversal instead of two?
3. Is any pass redundant given the fused `alpha` rescale?

**Where:** `expand_native_row_softmax_repack` and `_hip_native_bpermute_max` in
`tinygrad/renderer/isa/amd_attention_abi.py`; the spec in `tinygrad/uop/ops.py`
(`AMDRowSoftmaxRepackSpec`, ~line 1613).

**Risk:** online softmax is numerically delicate — the max-subtract is what keeps `exp` in range. A
"redundant" pass may be load-bearing for numerical stability at long context. `max_abs_err=6.104e-05` at
Hd=64 AND Hd=128 is the floor; any drift is a rejection, not a tradeoff.

---

## Gates (all theories)
1. Compile-only probe first: `scratchpad/kv_tile_amortization_probe.py` — confirms it builds and shows the
   static effect. Baseline **952 instrs / 16 wmma / 144 loads**.
2. Numerics: `PYTHONPATH=. python3 extra/qk/prefill_hd_sweep_numerics.py` → `max_abs_err=6.104e-05 PASS`
   at Hd=64 **and** Hd=128.
3. Real-model 8B parity: `extra/qk/prefill_flash_e2e_parity.py` → `8B: SDPA=198 FUSED=198 MATCH PASS`.
   (The 14B arm fails on in-process VRAM and `AUTHORITY_GATE: FAIL` — pre-existing, control-verified.)
4. Throughput: `extra/qk/prefill_whole_synced.py --mode authority --whole-lengths 512,1024,2048,4096`.
   **Paired same-session A/B, repeated at least twice** — this box drifts ~5% in absolute throughput across
   a session while back-to-back noise is 0.59%, so a recorded baseline is NOT a valid comparator.
5. Default-OFF env flag, per `fd654024e` / `c44905a18` house style.
6. GPU is a single resource: wrap every GPU run in `flock /tmp/gpu-bench.lock -c '…'`. Never run 14B.
   Always `TINYGRAD_PREFILL_PACKED_WMMA=0`.

## Standing lessons that apply to all three
- Price the machinery, not just the work removed. Every projection so far overshot because the delivery
  cost was omitted.
- An instruction-count delta is not a cycle delta.
- A clean negative result, committed with its numbers, is a success.
