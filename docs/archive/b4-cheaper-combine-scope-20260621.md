# B5-lite: Cheaper Split-KV Combine for the B4 Graph-Node Route — Scope

Date: 2026-06-21

Follow-on to `docs/b4-split-kv-combine-tax-result-20260621.md` (`COMBINE_TAX_DOMINATES + NO_POLICY_CLEARS_GATE`:
the combine is a fixable latency-bound floor; projected halving → ~+7.4%@ctx4096). **Optimize only the combine
(`owned_flash_combine`)** so the existing owned-AMDGCN graph-node route can clear W==D. **No new attention tile, no
Route-A codegen, no KV repack, no default change.**

## Current combine (the target)
`owned_flash_combine(part, meta, out, S)`: **grid = Hq = 32 workgroups, block = 32 threads (one warp/head)**. Each of
the 32 lanes owns 4 output dims (`lane*4..+3`) and serially walks all `S` splits: a max-reduction over `meta` (all 32
lanes redundantly reload every `m_s` from HBM), then a weighted-sum over `part`. Memory layout: `part` =
`[Hq, S, Hd]` fp32 (the tile's un-normalized per-split PV), `meta` = `[Hq, S, 2]` fp32 (`m`, `l`); `out` = `[Hq, Hd]`.
Per token it reads ~`Hq·S·(Hd+2)·4` bytes (~0.8 MB @S48).

### Measured limits (from `bench/qk-decode-attention-route-b-b4-combine-tax/latest.json` + this task)
- combine ~12.6µs @S48 / ~16.2µs @S64 standalone (`wait=True`), **flat in ctx** (merges S partials/head).
- ~64 GB/s = **6.7% of HBM peak**; 32 wg × 32 threads **under-occupies** gfx1100.
- **A measured 6.46µs launch/sync floor** is in every standalone `wait=True` number (a trivial write-zero kernel reads
  ~6.46µs) → the in-graph-relevant cost is **combine compute = standalone − floor** (base ≈ 9.8µs @S64).

## Candidate variants
| variant | geometry | idea |
|---|---|---|
| `base` | grid (Hq,1,1), block 32 | current — warp/head, redundant meta, serial S |
| **`hd<CWD>`** | grid (Hq, Hd/CWD), block CWD | **thread-per-output-dim + meta staged in LDS once** + more workgroups (≥64) to hide the partial-read latency |
| `sr<CWD>x<CSR>` | grid (Hq, Hd/CWD), block (CWD,CSR) | additionally parallelize the S-reduction across CSR threads (LDS tree-reduce) |

Same log-sum-exp math, same inputs/outputs (`part`, `meta`, `out`), same graph-node injection (only the combine
kernel symbol + launch geometry change; `DECODE_ATTN_AMDGCN_COMBINE` selects it, default `base`).

## Gates
- **Local (combine A/B, `extra/qk_b4_combine_ab.py`):** launch-corrected combine **compute ≤ 8µs** at the W==D-relevant
  split (S48 ≤ctx2048 / S64 @ctx4096), correctness `rel_rmse ≤ 1e-3`, no tile regression, total attention improves.
- **W==D (`extra/qk_b4_decode_eval.py`, the truth):** `≥+7%@ctx4096 OR ≥+5%@ctx1024`, no ctx512 regression, tokens
  match / dNLL ≤ 0.01, route-firing proof includes the new combine node.

## Phases / stops
C0 scope (this) · C1 reproduce baseline (stop if not reproduced → `B5_COMBINE_BLOCKED_MEASUREMENT`) · C2 implement
variant(s) · C3 local A/B (stop if no variant improves compute meaningfully → `B5_COMBINE_FAIL_LOCAL_AB`) · C4 env-gated
integration (default `base`) · C5 W==D (only after C3 passes) · C6 register/document.

## Verdict (one of)
`B5_COMBINE_WD_PASS` · `B5_COMBINE_LOCAL_PASS_WD_FAIL` · `B5_COMBINE_FAIL_LOCAL_AB` · `B5_COMBINE_BLOCKED_MEASUREMENT`
· `B5_COMBINE_SCOPE_ONLY`.

## Boundaries
Only the combine. No new tile, no Route-A codegen, no KV repack/transpose, no default change. `gqa_coop_vec` comparator
SSOT. No broad split/policy sweep unless a variant first improves local combine. No headline from local GPU-busy alone.
Do not revert unrelated dirty work.
