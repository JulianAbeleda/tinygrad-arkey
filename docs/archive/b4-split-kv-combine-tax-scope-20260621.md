# B4 Split-KV Combine-Tax Attribution + Policy — Scope

Date: 2026-06-21

Follow-on to `docs/decode-attention-route-b-b4-external-graph-node-result-20260621.md`
(`B4_WD_FAIL_INTEGRATION`: the owned AMDGCN tile+combine replay inside tinygrad's JIT graph and are byte-identical, but
whole-decode W==D is **+5.6–5.85%@ctx4096 / ~0%@ctx1024**, below the +7%@4096 / +5%@1024 bar — Amdahl-suspected).

## Objective
Determine **why** B4 misses, by attributing the split-KV cost, and decide whether a **ctx-aware split policy** or a
**cheaper combine** can make it promotable. **No new tile kernel, no Route-A codegen, no default change.**

## Primary question
Is the B4 ceiling set by **(a) the split-KV combine tax**, **(b) over-splitting at short/mid ctx**, or **(c) the
Amdahl share of attention (~17% of decode)** — i.e. is it a kernel-economics lever or a fundamental ceiling?

## Background (why a tax exists)
Flash-Decoding splits the KV cache into `S` chunks → `Hkv·S` workgroups (fills the GPU at T=1), each writing a partial
`(m, l, PV[D])`. A **combine** kernel then does the log-sum-exp merge: `m=max_s m_s; l=Σ exp(m_s−m)·l_s;
out=Σ exp(m_s−m)·PV_s / l`. That merge is extra HBM traffic + a second kernel. More splits ⇒ more occupancy in the tile
but more **combine debt**: `Hq·S·(Hd+2)·4` bytes read + `Hq·Hd·4` written. The lever is the `S` where occupancy benefit
> combine debt (and that crossover is ctx-dependent). Literature: Flash-Decoding (PyTorch), FlashDecoding++
(partial-softmax sync ≈20% of attn overhead → unified/async max), FlashInfer (decode templates), FA4 (split-KV + a
separate combine kernel).

## Phase 1 — Attribution (cheap, standalone, no model)
Per-kernel **GPU-busy** time (signal timestamps, `wait=True`) for the B4 single-kernel ELFs, by ctx × S:
- `tile_us` (`owned_flash_tile_gqa`), `combine_us` (`owned_flash_combine`), `total_us = tile+combine`,
- `combine_bytes` (read+write estimate), `tile_workgroups = Hkv·S`, correctness vs numpy.
- ctx ∈ {512, 1024, 2048, 4096}; S ∈ {8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96}.
- Output the **per-ctx optimal S** (min `total_us`) and the combine fraction `combine_us/total_us`.

## Phase 2 — W==D policy (reuse existing artifacts + bounded targeted runs)
The prior B4 sweep (`bench/qk-decode-attention-route-b-b4/*.json`) already has whole-decode W==D for S∈{24…128} under
`ctx4096_only`/`ctx2048_only`/`adaptive`. **Reuse it** for the high-S region; add only the **untested small-S** points
(S∈{8,12,16}) that Phase 1 flags as combine-cheaper, at ctx1024/4096, via `extra/qk_b4_decode_eval.py`. Gate:
`≥+5%@ctx1024 OR ≥+7%@ctx4096`, no ctx512 regression, tokens match / dNLL ≤ 0.01.

## Phase 3 — Decision (classify)
| finding | verdict |
|---|---|
| combine small, tile≈total, gap tracks the ~17% attn share | `COMBINE_NOT_MAIN_LIMIT_AMDAHL` |
| combine is a large/growing fraction of total | `COMBINE_TAX_DOMINATES` (→ scope a cheaper combine) |
| a ctx-gated policy clears +7%@4096 with no regression | `POLICY_PASS_OPT_IN` (owner opt-in, default-off) |
| best is a real but sub-bar long-ctx gain | `POLICY_ONLY_OWNER_KNOB` |
| nothing clears the gate, no cheap lever | `B4_SPLIT_KV_TAX_REST` |

## Deliverables
`extra/qk_b4_combine_tax.py` · `bench/qk-decode-attention-route-b-b4-combine-tax/latest.json` ·
`docs/b4-split-kv-combine-tax-result-20260621.md` (+ recommendation).

## Boundaries
No new tile, no Route-A codegen, no KV repack, no default change. `gqa_coop_vec` comparator SSOT. Attribution is
GPU-busy diagnostic (not a headline). Bounded: Phase 1 is launch-only (fast); Phase 2 reuses prior W==D + a small
targeted add — no open-ended full ctx×S W==D sweep.
