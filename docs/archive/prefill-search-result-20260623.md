# Prefill Search — Execution Result (2026-06-23)

## Verdict: Phase A `PREFILL_SEARCH_READY_ROLE_SPECIFIC` → Phase B `PREFILL_SEARCH_EXECUTED_ORACLE_REMAINS_BEST`
The attribution gate **correctly unlocked** (a stable, real ~4–5% whole-prefill gap to vendored Tensile exists), and
the bounded per-shape GEMM **tile-config** search then found **no recovering config** — because that gap lives in
**K-loop scheduling**, not in the searchable tile-config space. The default graph-GEMM route remains best within the
searchable space. **No default flip; synced whole-prefill the only authority.**

## Phase A — attribution gate: `PREFILL_SEARCH_READY_ROLE_SPECIFIC` (with a measurement correction)
Fresh synced whole-prefill + per-role attribution, then a **clock-pinned, 3-round interleaved** graph-GEMM-vs-Tensile
gap measurement (`rocm-smi --setperflevel high`):

| ctx | gap to Tensile (3 rounds) |
|----:|---|
| 512  | +5.9%, +4.6%, +4.5% |
| 1024 | +5.2%, +4.1%, +4.1% |

**Correction:** the earlier "graph-GEMM ~99.5% of Tensile / at-rest" was a **noise-inflated single un-pinned
measurement** (an under-measured Tensile run). Clock-pinned repeats show a **stable ~4–5% gap** — so prefill *is*
searchable, attributable to the below-parity roles (ffn_down 89% of parity, qo_proj 87%; ffn_gate_up at parity,
kv_proj fixed). The gate did its job: **repeats + clock-pin surfaced a real gap a single read had hidden.** This is
the recurring project lesson applied to the gate itself.

## Phase B — per-shape tile-config search: no recovering config
Parametrized the graph-GEMM `_kernel` for additive per-shape config overrides (`PREFILL_GEMM_CFG_{out_f}_{in_f}`),
searched the two below-parity roles, ranked by **synced whole-prefill** (clock-pinned):

| role | config | whole-prefill @512/1024 | vs default |
|---|---|---|---|
| ffn_down (4096×12288) | default | 3522 / 3448 | — |
| | bk16_dbuf | 3498 / 3422 | −0.7% |
| | bk32_dbuf | 3428 / 3353 | −2.7% |
| | bk16+plra | (invalid: PLRA needs KT==2) | crash→rejected |
| qo_proj (4096×4096) | default | 3551 / 3465 | — |
| | bk16 | 3524 / 3446 | −0.5% |

**Every config is ≤ default.** The roles are well-occupied (down/qo make ~128 workgroups — *not* WG-starved like
kv_proj was), so the occupancy/tile lever that fixed kv_proj does nothing here. Their gap to Tensile is **K-loop
scheduling** — Tensile's DepthU=16 + SIA1 instruction scheduling + PGR1/PLR1 prefetch + the +23% VALU
address-arithmetic leanness (prior PMC hard-audit) — which is **hand-asm-level, not a tile-config (BK/PAD/DBUF/waves)
knob**. The bounded search space cannot reach it.

## What this means
- Prefill **is** searchable (the gate was right to unlock), but the **searchable tile-config lane does not close the
  gap** for the deep-K/proj roles. The kv_proj win was an occupancy fix (WG-starvation); down/qo need scheduling work.
- The ~4–5% to Tensile is recoverable only by (a) **hand-asm K-loop scheduling/VALU-leanness** (deterministic, not a
  search — the named "Lever A" from the PMC audit), or (b) the **vendored Tensile dependency** (declined). Neither is
  a tile-config search.
- `learned_rule`: *prefill GEMM gap to Tensile is K-loop SCHEDULING, not tile-config — occupancy/tile search only
  helps WG-starved shapes (kv_proj); well-occupied deep-K/proj roles need hand-asm scheduling.*

## Harness compliance
Synced whole-prefill the only authority (clock-pinned, interleaved/repeated for the gap); isolated GEMM TFLOPS
excluded (host-bound); nosync `qk_prefill_v2_measure` not used. Artifacts stamped via `qk_harness_contract`; ledger
updated; the override is additive (default unset = shipped route, byte-identical greedy).

## Files changed
Modified (additive): `extra/qk_prefill_graph_gemm_route.py` (`PREFILL_GEMM_CFG_*` per-shape override, default
unchanged). New: `extra/qk_prefill_search_execute.py`, this doc, artifacts under `bench/qk-prefill-search/`, ledger
entries. **No default flip; no decode change.**
