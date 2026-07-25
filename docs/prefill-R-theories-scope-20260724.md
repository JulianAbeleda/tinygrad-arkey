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

## THEORY 6 — online-softmax reduction structure (231 instrs = 24.3%) — **RESOLVED & PROMOTED, see below**

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

### THEORY 6 — RESOLVED, and `PREFILL_SOFTMAX_REDUCE_FUSE` is now **DEFAULT ON**

**The count was right; the diagnosis was wrong in an important way.** `4 × 8 = 32` bpermute per full
reduction is correct, and there really were three traversals. But the algorithm emits **exactly two**
(`expand_native_row_softmax_repack` contains one `Ops.MAX` butterfly and one `Ops.ADD` butterfly, and
`dynamic_kv_v1`/`causal_v1` add none). All the excess was **rendering**, not algorithm:

1. **Pass 1 — row max.** 32 `ds_bpermute` + 32 `v_max_f32`. Necessary.
2. **Pass 2 — row sum of the weights.** 32 `ds_bpermute` + `v_add_f32`. Necessary. It does *not* go
   through `Ops.MAX`; in the disassembly it is the fully-interleaved 32-bpermute block that LLVM groups by
   xor mask, which is why a per-row read of the listing misses it.
3. **Pass 3 — a rematerialized copy of pass 1.** `new_m = max(old_m, row_max)` is not a native HIP op, so
   `decompositions.py` rewrites it to `(a<b).where(b,a)`. The C renderer inlines `Ops.CUSTOMI`
   **unconditionally, ignoring `child_count`**, and every rung of the ladder has two consumers (the next
   `fmaxf` and the next `bpermute`) — so the emitted source grows as 2ⁿ: **272 textual `ds_bpermute` for
   64 distinct ones**. LLVM CSEs the pure part but the `where` lowers to a `v_cmpx_lt_f32` /
   `s_cbranch_execz` exec-masked region that CSE will not cross, so the whole 4-step ladder is
   recomputed inside the guard. 8 rows × 4 = the third 32.

**And the 135 `v_max_f32` are not doing double duty as masking.** Masking is `v_cndmask_b32` (53 in the
body, unchanged by the fix). The 135 decompose as 32 real ladder maxes + ~31 from the remat ladder +
**72 degenerate `v_max_f32 vX, vX, vX` canonicalizations** — IEEE `fmaxf` on gfx11 requires canonical
operands, and the duplicated-expression source denied LLVM the chance to prove an operand was already the
result of a max. Restoring SSA form collapses all 72.

**Can max and sum share one traversal?** No, and it is not needed. They are strictly ordered — the sum is
over `exp2((value − new_m)·log2e)`, which cannot be formed until the max reduction has finished — so a
shared butterfly would have to carry two independent accumulators through the same 4 steps and would still
issue 2 `ds_bpermute` per step. The bpermute count is unchanged; only the (already-optimal) address
arithmetic would be shared. Both remaining passes are necessary and irreducible.

**The change** (`PREFILL_SOFTMAX_REDUCE_FUSE`, default OFF, two hunks in
`tinygrad/renderer/cstyle.py`): give a multi-use float `Ops.CUSTOMI` an SSA name instead of inlining it,
and accept an already-rendered `fmaxf` as the `_hip_native_bpermute_max` peer so the `new_m` carry renders
as `__builtin_fmaxf` rather than a select. **Nothing in the emitted algorithm changes** — no reordering,
no reassociation, no dropped reduction.

**Measured, per KV tile (compile-only probe, 8B, `PREFILL_V_TRANSPOSED=0`):**

| | OFF | ON | Δ |
|---|---:|---:|---:|
| loop-body instructions | 952 | **660** | **−30.7%** |
| `ds_bpermute_b32` | 96 | 64 | −32 |
| `s_waitcnt lgkmcnt(0)` | 98 | 51 | −47 |
| `v_max_f32` | 135 | **33** | −102 |
| `s_delay_alu` | 78 | 33 | −45 |
| `v_cmpx_lt_f32` / `s_cbranch_execz` | 8 / 8 | 0 / 0 | branch-free |
| textual `ds_bpermute` in HIP source | 272 | 64 | −76% |
| global loads / WMMA / `v_exp_f32` / `v_cndmask` | 144 / 16 / 16 / 53 | 144 / 16 / 16 / 54 | unchanged |
| VGPR spills | 0 | 0 | — |

The addressable group T4 handed over (96 bpermute + 98 lgkm waits + 135 `v_max_f32` = 329 instrs, 34.5%
of the body) went to 148. Loads, WMMA, transcendentals and masking are untouched.

**Throughput, 8B `--mode authority`, paired same-session interleaved A/B, two reps:**

| whole-prefill | OFF | ON | Δ |
|---|---:|---:|---:|
| 512 | 3611 / 3575 | 3650 / 3637 | **+1.4%** |
| 1024 | 3516 / 3481 | 3581 / 3566 | **+2.1%** |
| 2048 | 3329 / 3299 | 3449 / 3432 | **+3.8%** |
| 4096 | 3010 / 2981 | 3201 / 3192 | **+6.7%** |

Deepest chunk (`start_pos=3584`) 198.0/199.9 ms → 180.4/180.0 ms, −9.5%. The gain rises with context
because the KV loop is a larger share of the chunk there — the expected shape for a loop-body win, and
the reason this is worth more than T3's +1.7%.

**Gates:** `prefill_hd_sweep_numerics.py` `max_abs_err=6.104e-05 PASS` at Hd=64 **and** Hd=128, exactly
the floor. Long-context numerics (added: `q=512` chunk against a per-head fp32 numpy reference) at
kv=512/1024/2048/4096 are **bit-identical between ON and OFF** — `6.558e-05 / 2.655e-06 / 1.865e-06 /
1.053e-06`, all finite. `prefill_flash_e2e_parity.py` `8B: SDPA=198 FUSED=198 MATCH PASS` in both arms
(`AUTHORITY_GATE: FAIL` is the pre-existing 14B VRAM arm). `test/unit/test_online_softmax_tile.py`:
6 failed / 81 passed with the flag ON, OFF, and at HEAD — identical set, all `DEV=AMD:ISA` final-ISA
tests. Flag OFF reproduces the 952-instruction body and the byte-identical disassembly.

**Two standing lessons this corrects:**
- *`amd.py` is not in the shipped path, but `amd_attention_abi.py` is.* `DEV=AMD` → `HIPRenderer`, which
  lazily imports `expand_native_row_softmax_repack` / `expand_loop_fragment` / `native_repack_matcher` /
  `native_state_lane_matcher` from the isa package. You are steering LLVM by choosing what HIP source to
  emit, and **how the C renderer names intermediates is a first-class performance decision**, not
  cosmetics. Everything T4 attributed to LLVM's `SIInsertWaitcnts` moved when the source shape moved.
- *Count textual occurrences in the generated source before blaming the algorithm.* The one cheap check
  that would have found this on day one is `grep -c` on the emitted `.cpp`: 272 vs 64.

**Rejected on measurement** (kept here so nobody re-tries them): rendering `new_m` as `fmaxf` *without*
the SSA-naming hunk makes it **worse** — 952 → 1558 instrs, 96 → 192 bpermute, 26 VGPR spills, because
removing the exec-masked region removes the only thing that was forcing the ladder into a register.
Algebraically folding `old_m` into the butterfly seed (exact, since row state is replicated across all 16
columns, so `max(old_m, rowmax(v)) == rowmax(max(old_m, v))`) is a wash once the SSA hunk is in: 691 vs
660. The renderer was the whole problem.

---

## Gates (all theories)
1. Compile-only probe first: `scratchpad/kv_tile_amortization_probe.py` — confirms it builds and shows the
   static effect. Baseline **952 instrs / 16 wmma / 144 loads**.
2. Numerics: `PYTHONPATH=. python3 extra/qk/prefill/prefill_hd_sweep_numerics.py` → `max_abs_err=6.104e-05 PASS`
   at Hd=64 **and** Hd=128.
3. Real-model 8B parity: `extra/qk/prefill/prefill_flash_e2e_parity.py` → `8B: SDPA=198 FUSED=198 MATCH PASS`.
   (The 14B arm fails on in-process VRAM and `AUTHORITY_GATE: FAIL` — pre-existing, control-verified.)
4. Throughput: `extra/qk/prefill/prefill_whole_synced.py --mode authority --whole-lengths 512,1024,2048,4096`.
   **Paired same-session A/B, repeated at least twice** — this box drifts ~5% in absolute throughput across
   a session while back-to-back noise is 0.59%, so a recorded baseline is NOT a valid comparator.
5. Default-OFF env flag on landing, per `fd654024e` / `c44905a18` house style; a default flip is a separate,
   separately-gated step (see the THEORY 6 promotion section below).
6. GPU is a single resource: wrap every GPU run in `flock /tmp/gpu-bench.lock -c '…'`. Always
   `TINYGRAD_PREFILL_PACKED_WMMA=0`.
   **Correction (2026-07-24): "Never run 14B" is too strong and cost this effort real evidence.** 14B runs
   fine *with* `TINYGRAD_PREFILL_PACKED_WMMA=0` — that flag disables the packed-WMMA path that faults, so
   14B falls back to graph-GEMM and completes (`docs/BOLTBEAM_GPU_HANG_DIAGNOSIS_HANDOFF_20260724.md`). The
   thing to never do is run 14B *without* it. The other 14B blocker, `fp16 KV admits 0 ... free 5.2GB`, was
   just `prefill_flash_e2e_parity.py` holding both models in one process; use `--only 14B`. Both 14B legs of
   THEORY 6's promotion were collected this way.
   Note the instrument limit that follows: with packed-WMMA off, 14B prefill is ~94% GEMM-bound (~1420 ms
   per chunk), so 14B *whole-model* throughput cannot resolve an attention-local change. Use
   `extra/qk/prefill/prefill_flash_perf.py` for the 14B grid and treat the whole-model number as corroboration.

## THEORY 6 promotion (2026-07-24)

`PREFILL_SOFTMAX_REDUCE_FUSE` is **default ON**. Rollback `PREFILL_SOFTMAX_REDUCE_FUSE=0`, which reproduces
the old 952-instruction / 272-bpermute body byte-identically.

Beyond the 8B numbers in the THEORY 6 section above, promotion required two things that section did not
cover, both because the change is in the **shared** HIP renderer rather than the attention emitter:

- **Decode non-regression.** `extra/qk/decode/decode_codegen_identity_check.py` compiles the real decode graph
  both ways for both decode-admitted geometries and compares code-object sha256: **byte-identical**, 8
  kernels per arm, all executed. Decode's cross-lane reduce is a linear ladder, so the `child_count > 1`
  predicate never fires there.
- **14B.** Output-sha bit-identical numerics on `Hq=40`, real-model token parity `90310 == 90310`,
  attention-local A/B −25% to −31%, and a whole-model measurement whose deepest chunk (−1.45%) matches the
  −1.50% predicted from the attention-local win.

8B whole-model, re-measured as three same-session interleaved pairs: pp512/1024/2048/4096
**+1.37% / +2.17% / +3.71% / +6.72%** (2.3×–11.4× the 0.59% noise floor), deepest chunk **−9.9%**.
Whole `test/unit/` failure set equal off / on / at the new default (51 failed, 1274 passed each).
Gate: `extra/qk/prefill/prefill_softmax_reduce_fuse_promotion_gate.py`. Full write-up:
`docs/prefill-softmax-reduce-fuse-promotion-readiness-20260724.md`.

One correction to the THEORY 6 section above: its claim that ON and OFF are "bit-identical" was inferred
from matching `max_abs_err` scalars, which cannot establish bit-identity. It is now *actually* established,
by output sha256 on both grids — `prefill_long_context_numerics.py` prints `out_sha`.

## Standing lessons that apply to all three
- Price the machinery, not just the work removed. Every projection so far overshot because the delivery
  cost was omitted.
- An instruction-count delta is not a cycle delta.
- A clean negative result, committed with its numbers, is a success.
