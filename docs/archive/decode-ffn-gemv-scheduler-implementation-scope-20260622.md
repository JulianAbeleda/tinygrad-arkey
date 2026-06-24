# Decode FFN-GEMV Scheduler — Implementation Scope (bounded, W==D-gated)

Date: 2026-06-22

Follow-on to `docs/decode-ffn-gemv-scheduler-diagnostic-result-20260622.md`
(`FFN_GEMV_DIAGNOSTIC_BOUNDED_SCHEDULE_SCOPE_READY`, `GEMV_SCHEDULE_BOUND`). Scope only — no kernel built here.

## Objective
Build a **lossless** FP Q4_K decode GEMV with llama's work decomposition — **128 threads/row + K-block-parallel +
in-kernel warp-shuffle reduce + one output write** — for FFN **gate/up** (then **down**), env-gated default-off, and
prove it via **W==D** (not % peak). Target: gate/up 51% → ~65–70% peak → projected **+6.5% / ~+9–11%** whole-decode.

## Why this is the lever (from the diagnostic)
- Gap is **work decomposition**, not math: `dot4`/packed-extract already matched; the untried structure is 128-thread/
  row + K-parallel + warp-reduce (llama 70% vs tinygrad ~50%).
- **Lossless** (FP Q4_K dequant; exact-vs-default up to fp reassoc) → avoids the q8-activation lifecycle that made the
  int-dot path null in-model (+1.25%).
- **Transfers**: q8 (+6%) proves gate/up is on the critical path; B5 showed attention does not — so FFN, unlike
  attention, is the right bucket.
- **Bounded**: `extra/amd_warp_reduce.warp_reduce_sum` (`ds_bpermute`) + the existing `_q4k_block_dot*` extract exist;
  no renderer/backend change.

## Phases / gates
- **F1 — kernel:** `q4k_gemv_warp_partial_kernel` in `extra/q4_k_gemv_primitive.py`: row → workgroup; 128 threads
  cooperatively walk the row's K-blocks (K-block-parallel, no per-thread serial blk loop); each thread accumulates its
  block subset in a register; **in-kernel warp-shuffle reduce** to one output. Correctness `rel<=1e-2` vs the reference.
- **F2 — local A/B:** standalone GB/s + % peak at the gate/up shape (12288×4096) vs default `q4k_gemv_partial`. Gate:
  ≥ ~60% peak (≥1.2×), correct. **% peak is diagnostic only — not a ship signal.**
- **F3 — env-gated route + W==D (the gate):** `Q4K_FFN_WARP=1` routes gate/up (then down) in `model.py`; default `base`.
  **Gate: W==D ≥ +5%@ctx1024 OR ≥ +7%@ctx4096, no ctx512 regression, greedy byte-identical.** If local wins but W==D
  saturates (the B5 failure mode) → bank `FFN_GEMV_WARP_LOCAL_PASS_WD_FAIL`, do not ship.
- **F4 — down + lifecycle:** extend to FFN down (Q4_K + Q6_K blend); register decode_eval candidate, default-off.

## Stop conditions
- Local < ~60% peak or incorrect → `FFN_GEMV_WARP_FAIL_LOCAL`.
- W==D saturates < +5% despite local win → `FFN_GEMV_WARP_LOCAL_PASS_WD_FAIL` (rest; q8 opt-in is the practical cap).
- Needs a renderer/backend change to express the warp-reduce → stop, reclassify `GEMV_BACKEND_PROJECT_LEVEL`.

## Boundaries
Lossless FP only (no q8 weights, no int-dot lifecycle — that lane stays closed). Default-off, shape-guarded, fallback
to the current `q4k_gemv_partial`. W==D-gated (never ship on % peak). `gqa_coop_vec`/llama comparators; q8 stays opt-in.
