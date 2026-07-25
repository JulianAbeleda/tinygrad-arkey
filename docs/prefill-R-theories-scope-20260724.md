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
| **sync/sched** (`s_waitcnt`, `s_delay_alu`, `s_clause`) | **213** | **22.4%** | T4 |
| other VALU math (softmax arithmetic, rescale, alpha) | 175 | 18.4% | — |
| **max/min** (`v_max_f32` ×135) | **135** | **14.2%** | T6 |
| global loads (16 `b128` K + 128 `d16` V gather) | 144 | 15.1% | *closed* |
| **mask** (`v_cndmask` ×56, `v_cmp` ×47) | **103** | **10.8%** | T5 |
| **cross-lane reduce** (`ds_bpermute_b32`) | **96** | **10.1%** | T6 |
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
