# B5-lite v2: Cheaper Split-KV Combine (tiered targets) — Scope

Date: 2026-06-22

Re-issue of `docs/b4-cheaper-combine-scope-20260621.md` with the **tiered combine targets** now in the principles
(§ "Split-KV Reduction Economics Are Part Of The Decode Primitive"):
- combine **≤ 8µs** = diagnostic/borderline,
- combine **≤ 6–7µs** = preferred for a confident W==D,
- combine **≈ 5µs** = stretch with real margin.

The first B5-lite pass (`hd64`, ~1.7× combine compute) reached the diagnostic tier but **W==D moved only +0.3%@4096**
(`docs/b4-cheaper-combine-result-20260621.md`, `B5_COMBINE_LOCAL_PASS_WD_FAIL`). The Amdahl projection
(half-combine → ~+7.0%) and the measurement (+5.71%) **disagree** → the central open question is whether the combine is
actually on the decode critical path or **overlaps** in the JIT graph. This pass pushes the combine to the preferred/
stretch tier and **re-measures W==D** to resolve that with a second data point.

## Current combine (target) — see the 20260621 scope for full geometry
`owned_flash_combine`: grid Hq=32 wg, block 32 (one warp/head), each lane 4 dims, serial S-loop, **redundant**:
(a) all 32 lanes reload `meta` for the `gm` max; (b) each lane recomputes `exp(m_s−gm)` for its dims. ~64 GB/s,
~6.7% HBM peak, 32×32 threads under-occupy 96 CUs. `part`=[Hq,S,Hd] fp32, `meta`=[Hq,S,2] fp32, `out`=[Hq,Hd] fp32.

## Variants (combine-only; same math, contract, graph-node injection; `DECODE_ATTN_AMDGCN_COMBINE` selects, default `base`)
- `hd<CWD>` (`owned_flash_combine_hd`): thread-per-output-dim + meta staged in LDS once + 2D grid (Hq, Hd/CWD). ~1.7×.
- **`hw<CWD>` (`owned_flash_combine_hw`): `hd` PLUS the per-split weights `exp(m_s−gm)` precomputed ONCE into LDS**
  (cooperatively, S exps/head instead of CWD·S) → main loop is pure FMA over `part`. **The new lever** (the redundant
  per-dim `exp` was a real cost). Local: combine compute ≈ 1.4–1.9µs (raw ~5µs) — stretch tier.
- `sr<CWD>x<CSR>` (`owned_flash_combine_sr`): split-reduction across CSR threads — refuted (sync/LDS overhead > gain).

## Gates
- **Local (`extra/qk_b4_combine_ab.py`, launch-corrected compute):** combine ≤ 7µs preferred (≤5µs stretch) at the
  W==D-relevant split (S48 ≤ctx2048 / S64 @ctx4096), `rel_rmse ≤ 1e-3`, no tile regression. Emits combine us/compute,
  tile us, total, combine fraction, effective bandwidth, workgroup count, correctness, audit class.
- **W==D (`extra/qk_b4_decode_eval.py`, the truth):** ≥+7%@ctx4096 OR ≥+5%@ctx1024, no ctx512 regression, tokens
  match/dNLL ≤ 0.01, route-firing includes the new combine node, default-off.

## Stop conditions / verdicts
Baseline not reproduced → `B5_COMBINE_BLOCKED_MEASUREMENT`. No variant reaches preferred → `B5_COMBINE_FAIL_LOCAL_AB`.
Local preferred met but W==D misses → `B5_COMBINE_LOCAL_PASS_WD_FAIL` (and classify the limiter: combine overlap /
Amdahl). W==D clears with margin → `B5_COMBINE_WD_PASS`.

## Boundaries
Only the combine. No new tile, no Route-A codegen, no KV repack/transpose, no default change, no closed-lane reopen.
`gqa_coop_vec` comparator SSOT. No headline from local GPU-busy alone (clock-state-sensitive — lean on ratios + W==D).
Preserve the B4 graph-node route + fallback. Do not revert unrelated dirty work.
