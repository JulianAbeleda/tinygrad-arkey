# Prefill structural-emit stress study — RESULT (2026-06-23)

## Verdict: `STRUCTURAL_EMIT_TRANSFERS` — cross-iteration double-buffering (DBUF) WINS +2.84% whole-prefill (byte-identical)
A statistically-defensible stress study (full 8B, clock-pinned, 6 repeats, 5 contexts incl @8192) tested whether
**structural emit-level** changes (not scheduling) can move whole-prefill. Answer: **YES.** Swapping the route's
substep-prefetch (`PLRA`) for **cross-iteration double-buffering (`DBUF`)** gives a significant **+2.84% ± 0.11%**
whole-prefill@4096 with **byte-identical output** (logit `max_abs_diff = 0`). Adding occupancy-gated relocation on top
(now in-regime because DBUF lowers occupancy) reaches **+3.87% ± 0.25%**. Pure scheduling (relocation alone) stays at
noise (+0.1%). This **rejects** "the residual is not recoverable by scheduling transforms alone" — in the *structural*
direction it is, while *scheduling-only* is confirmed non-transferable.

## Method
- `DEV=AMD JIT=1 PREFILL_V2=1`, `PYTHONPATH=.`, clock-pinned (`rocm-smi --setperflevel high`). Model Qwen3-8B-Q4_K_M
  (no OOM; full 8B), max_context 8704 for @8192. Harness: synced multi-chunk whole-prefill; a FRESH `TinyJit` is
  captured per start_pos (attention key-extent is baked at capture).
- Each candidate = an isolated subprocess (one model load), global emit-config knobs added to
  `extra/qk_prefill_graph_gemm_route.py` (`PREFILL_GEMM_{DBUF,BK,PLRA,PLRAB,LEANADDR}`, additive, default-preserving).
- 6 repeats (discard 1st), whole-prefill@{512,1024,2048,4096,8192}; significance = |Δ|>1% AND |Δ|>2×CI AND p<0.05.
- The headline win was re-confirmed with an **interleaved paired A/B** (baseline/C1/C1+reloc measured back-to-back each
  round) to cancel clock drift, plus a byte-identical correctness check.

## Whole-prefill median tok/s, Δ% vs baseline (`*` = significant)
| candidate | @512 | @1024 | @2048 | @4096 | @8192 |
|---|---|---|---|---|---|
| baseline (`dbuf0 plra1`) | 3700 | 3615 | 3382 | 2983 | 2407 |
| **C1 DBUF (`dbuf1 plra0`)** | **+2.7\*** | **+2.6\*** | **+2.6\*** | **+2.9\*** | **+2.6\*** |
| C2 DepthU `bk16` (shallower) | −18.6\* | −18.1\* | −17.2\* | −15.8\* | −13.2\* |
| C3 `dbuf1 bk16` | −0.3 | −0.3 | −0.2 | −0.2 | −0.2 |
| C4 8-wave PLRAB | +3.6\* | −14.2 | −5.6 | −0.3 | +1.8\* (UNSTABLE) |
| C8 LEANADDR | FAILED — VGPR overflow 263 | | | | |
| S relocation (scheduling) | +0.3 | +0.3 | +0.2 | +0.1 | +0.1 |
| **C1 DBUF + relocation** | +3.6 | +3.6 | +3.6 | **+4.0\*** | **+3.1\*** |

## Interleaved paired confirmation (@4096, drift-controlled, byte-identical)
- **Correctness:** C1_dbuf logits vs baseline `max_abs_diff = 0` → identical computation.
- **C1 DBUF:** paired Δ = **+2.84% ± 0.11%** (per-round [2.9, 2.7, 2.8]) — SIGNIFICANT.
- **C1 DBUF + reloc:** paired Δ = **+3.87% ± 0.25%** (per-round [4.0, 3.6, 3.9]) — SIGNIFICANT.

## Evidence verdict matrix
| candidate | class | effect (best) | significance | verdict |
|---|---|---|---|---|
| C1 DBUF | cross-iteration pipeline | +2.84% all ctx | p<0.05, ±0.11% CI, byte-identical | **WON** |
| C1 DBUF + reloc | structural + scheduling | +3.87% | p<0.05, ±0.25% CI | **WON** |
| S relocation | scheduling only | +0.1–0.3% | n.s. (within noise) | INCONCLUSIVE (non-transfer) |
| C3 dbuf+bk16 | DepthU↓ + pipeline | −0.2% | n.s. | INCONCLUSIVE |
| C4 8-wave PLRAB | tile-geometry retime | unstable (−14%…+3.6%) | erratic variance | INCONCLUSIVE (unstable) |
| C2 DepthU bk16 | shallower DepthU | −15…−18% | p<0.05 | REGRESSED |
| C8 LEANADDR | load/address emit | — | — | BLOCKED (VGPR overflow 263) |
| C2b DepthU bk64 | deeper DepthU | — | — | BLOCKED (VGPR overflow 268) |
| C3b PLRAB 4×4 | full A+B prefetch | — | — | BLOCKED (VGPR overflow 300; only fits 8-wave=C4) |
| C9 acc-partition | dependency break | — | — | BLOCKED (doubling 128-reg acc → >256 VGPR) |
| C6 full reg-pool | live-range compaction | — | — | BLOCKED (HW register-limited; register-lifetime arc) |

## Pareto ranking (speedup@4096, stability, risk)
1. **C1 DBUF + reloc** — +3.87%, tight, 2 knobs (best speedup; relocation adds risk/complexity).
2. **C1 DBUF** — +2.84%, tightest, 1 structural knob, byte-identical → **recommended default** (best risk-adjusted).
3. S relocation — ~0 (non-transfer). C3/C4 — ~0/unstable. C2 — regress.

## Root cause
`DBUF` (double-buffer: prefetch the next K-block's global loads into the other LDS buffer while computing the current
block, removing the inner barrier = full block-level software pipelining) exposes more independent work across LDS
wait-points than `PLRA` (substep-level A-prefetch only, single-buffer). The route default was `plra1` — chosen on
*isolated* kernel benchmarks where PLRA's lighter LDS footprint looked better. In-model, DBUF's fuller pipelining wins
(+2.84%), the isolated→integrated reversal. DBUF doubles LDS (→ lower occupancy), which is also why relocation now
transfers on top (+1% more): DBUF moves the roles into the exposed-LDS-latency regime where relocation helps.

## Decision
**Promote DBUF as the route default** (`dbuf=1, plra=0`): byte-identical output (dNLL=0), significant +2.84%
whole-prefill on the authority benchmark, decode untouched, fully reversible via `PREFILL_GEMM_DBUF=0
PREFILL_GEMM_PLRA=1`. Relocation stays opt-in (`PREFILL_GEMM_RELOC=1`) for the extra ~+1% (now in-regime under DBUF).

## Caveats
Single GPU/model (gfx1100, 8B/Q4_K_M); chunked-512 whole-prefill methodology; @8192 via interpolated start_pos points.
C4 (8-wave) showed erratic variance (auto-boost/clock-lottery artifact) — not pursued. Raw timing logs under
`/tmp/prefill-emits/` (non-deterministic, not committed).
