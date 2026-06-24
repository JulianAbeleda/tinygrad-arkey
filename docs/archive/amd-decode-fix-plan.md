# Phase F — close the standalone→e2e gap (full scope, both hypotheses (a) and (b))

Date: 2026-06-15. State after narrowing (NARROW_RESULT): the decode GPU is 100% busy (no gaps); the e2e
wall is the in-graph Q4_K GEMV running at **~105 GB/s = 12% peak** (267 us for the 12288×4096 FFN matmul),
while a competent standalone kernel does **482 (fp) / 686 (v_dot4) GB/s** for the identical op. Two readings
remain and this phase chases BOTH to ground:
- **(a) kernel-source**: the bad codegen is the wall; a competent kernel placed in-graph runs at its
  standalone rate and e2e jumps toward llama.cpp (57%). D1/E0 looked neutral only because quant overhead
  canceled the win.
- **(b) in-graph execution**: even our good kernel collapses to ~12% single-shot in-graph (occupancy / quant
  coupling); standalone bandwidth never transfers and the lever is occupancy/structure (llama.cpp's mmvq).

The whole point of F2 is to MEASURE the in-graph GB/s of the good kernel directly (D1/E0 only reported e2e
tok/s, which can't tell (a) from (b)). Everything else branches off that one number.

## Pre-flight invariants (every step)
- Flags default-off; baseline = current decode (~21–30 tok/s, 12%); bar = llama.cpp 105.66 tok/s (57%).
- Always `rocm-smi --setperflevel high` + warmup before timing (the clock-ramp confound, see
  [[amd-decode-measurement-confounds]]). Reset `auto` after.
- Report the IN-GRAPH kernel GB/s (DEBUG=2, the `r_toks_*`/custom GEMV line) AND e2e tok/s AND an accuracy
  check (int8 activation quant is lossy — X0-style per-layer error + output coherence on a fixed prompt).
- Pre-registered gates below; honest reporting whichever way each falls. No retuning after seeing a null.

## F0 — ground truth: WHICH kernel does the real JITted decode use, and why is it 12%? (diagnosis, cheapest)
The narrowing profiled a non-JIT first forward (fallback `x.linear` → generated `r_toks_64_16_4` reduce).
The actual JITted decode (decode_enabled=True) may instead run `q4k_gemv_partial_kernel` (custom). Resolve:
1. Under the REAL decode path (JIT-warmed, `Q4K_PRIMITIVE=1`, DEBUG=2), identify the Q4_K GEMV kernel name
   and its in-graph GB/s. Confirm it is the 12% kernel (or find the true one).
2. Dump that kernel's source + applied OptOps (`to_program`, the disassembly). Diagnose the deficiency vs our
   fp standalone (56%): is it (i) serial fp-add dequant chain, (ii) no vectorized/wide loads, (iii) bad
   reduce split (LOCAL/UPCAST/UNROLL), (iv) low occupancy (VGPR pressure), (v) scales recompute?
3. **Output**: the named in-graph GEMV, its GB/s, and the 2–3 concrete codegen deficiencies. This tells us
   whether F1 (opts-only) is even plausible and gives F2 a like-for-like comparison point.

## F1 — (a1) codegen-only fix: better OptOps on the SAME kernel, no custom kernel
If F0 shows the deficiency is schedule (reduce split / no upcast), try to lift it WITHOUT a hand kernel —
the cleanest, most general win (it would help every Q4_K model, not just this path).
- Mechanism: the warmstart hook in `postrange.py` (`_WARMSTART_OPTS`, `_warmstart_match`) forces OptOps
  (UPCAST/UNROLL/LOCAL/PADTO) on the matched Q4_K reduce — a targeted stand-in for BEAM (which hangs gfx1100).
- Sweep a small opt set informed by F0 (e.g. UPCAST the K-reduce, LOCAL across rows, UNROLL the inner block).
  Measure in-graph GB/s per config.
- **Gate**: kernel ≥ ~40% (toward fp's 56%) → codegen-tuning is a real lever; measure e2e and proceed.
  kernel stays ~12% → the generated dequant structure itself is the wall (not schedule) → custom kernel (F2)
  is required; record that opts-only can't fix it.

## F2 — THE FORK RESOLVER: wire the competent kernel + amortized quant, measure in-graph GB/s
Wire the v_dot4 path (exists: `Q4K_VDOT` + `Q4K_VDOT_AMORT`, `q4k_q8_1_vdot_builtin_partial_kernel`,
`q8_1_quantize`, `_VDOT_QUANT_CACHE`) and — the new step D1/E0 skipped — read the GEMV's **in-graph GB/s**
under DEBUG=2, not just e2e tok/s.
- Verify the quant cache HIT-rate first (q/k/v share one quant, gate/up share one → 4 quants/layer, not 7).
  `_VDOT_QUANT_CACHE["h"]/["m"]`. If the cache doesn't hit, it degrades to D1 and the comparison is invalid.
- Measure: (i) the vdot GEMV in-graph GB/s, (ii) the q8-quant + bias-pack kernels' time/token, (iii) e2e
  tok/s, (iv) accuracy.
- **Fork gate (the decisive read):**
  - in-graph GEMV ≈ 480–680 GB/s → **(a) confirmed: codegen was the wall.** The good kernel transfers.
    - if e2e also jumps (toward 40–57%) → SHIP IT (first real e2e decode win). Done; write the win.
    - if e2e does NOT jump despite the fast GEMV → the quant/bias-pack overhead eats it → **F3**.
  - in-graph GEMV ≈ 105 GB/s (collapses to default) → **(b) confirmed: in-graph single-shot execution caps
    it.** Standalone bandwidth (200-rep-amortized, warm) does not transfer to a single in-graph launch → **F4**.

## F3 — (a-branch) quant amortization accounting + reduction
Reached if the GEMV is fast in-graph but e2e doesn't move → the per-token activation quant is the new cost.
- Per-token breakdown (DEBUG=2 / profile): GEMV us vs q8-quant us vs bias-pack us vs everything else.
- Levers: (i) raise cache hit-rate to the theoretical 4 quants/layer; (ii) fuse quant INTO the GEMV kernel
  (llama.cpp mmvq quantizes activations inside the matmul — one kernel, no separate quant pass); (iii) skip
  bias-pack by folding it into the quant. Re-measure e2e.
- **Gate**: e2e clears baseline by ≥20% → quant fusion is the lever; iterate toward llama.cpp. No movement →
  the int8 path's quant cost is structural at batch-1 → honest stop; the fp competent kernel (F1/F2-fp) may
  be the better e2e target (no quant needed).

## F4 — (b-branch) occupancy / structure fix
Reached if the good kernel collapses to ~12% single-shot in-graph → the difference from standalone is
launch occupancy, not source. Diagnose then fix:
- Diagnose: standalone uses LOCAL=64, one wave/CU sustained over 200 reps; in-graph it's one launch. Check
  wavefronts-in-flight / VGPR occupancy of the single in-graph launch (DEBUG=2 reg usage; the kernel's
  global/local size). Likely too few workgroups or too-low occupancy for a single cold pass.
- Levers, cheapest first: (i) increase LOCAL / rows-per-workgroup so one launch has enough waves to hide
  latency; (ii) horizontal fusion (B1 `Q4K_FUSE`: q/k/v→one GEMV, gate/up→one) — fewer, fatter launches,
  more rows each → higher occupancy; (iii) persistent/multi-row kernel (the mmvq structure: one workgroup
  walks many output rows, keeping memory in flight). Measure in-graph GB/s after each.
- **Gate**: in-graph GEMV → 300+ GB/s and e2e clears baseline → occupancy was the wall; the mmvq-style
  structure is the fix. Still ~105 → escalate to a faithful mmvq port (the known-good reference structure).

## Decision tree (one line)
F0 (what/why) → F1 (opts-only?) → **F2 (in-graph GB/s of good kernel = the fork)** → fast: F3 (quant) →
ship · collapsed: F4 (occupancy) → ship. Every leaf ends in a measured in-graph GB/s and an honest gate.

## Touch points (all exist, gated)
- `tinygrad/llm/model.py`: `Q4KPrimitiveLinear.__call__` (vdot path L85-102), `_VDOT_QUANT_CACHE`, flags
  `Q4K_VDOT`/`Q4K_VDOT_AMORT`/`Q4K_FUSE`.
- `extra/q4_k_gemv_primitive.py`: `q4k_q8_1_vdot_builtin_partial_kernel`, `q4k_gemv_partial_kernel`,
  `q8_1_bias_pack_u32_kernel`.
- `extra/qk_layout.py`: `q8_1_quantize`. `tinygrad/codegen/opt/postrange.py`: warmstart opts (F1).
- `extra/qk_cold_perlayer.py` / `extra/qk_decode_profile.py`: the in-graph + standalone measurement harnesses.

## Honest framing / stop rules
- The WIN (kernel beats llama.cpp standalone, 76% vs 57%) is banked and independent of this phase's outcome.
- This phase decides whether that standalone win becomes an e2e win. (a) → yes, ship. (b) → the gap is
  occupancy/structure (mmvq), a concrete next target, not a fundamental wall (llama.cpp's 57% proves it).
- Pre-registered: if F2's good kernel is fast in-graph AND quant-fused AND e2e still doesn't clear baseline,
  that is the definitive "standalone bandwidth doesn't transfer at batch-1 in tinygrad's structure" result —
  report it; do not keep retuning. Each gate is measured once, reported as it falls.
