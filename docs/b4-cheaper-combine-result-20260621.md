# B5-lite: Cheaper Split-KV Combine for the B4 Graph-Node Route — Result

Date: 2026-06-21

Executes `docs/b4-cheaper-combine-scope-20260621.md`: optimize **only** the split-KV combine (`owned_flash_combine`) so
the existing owned-AMDGCN graph-node route can clear W==D. No new tile, no Route-A codegen, no KV repack, no default
change.

## Decision: **`B5_COMBINE_LOCAL_PASS_WD_FAIL`** — a cheaper combine (`hd64`, 1.7× compute) PASSES the local gate but moves whole-decode W==D only **~+0.3%@ctx4096** (+5.41% → +5.71%, still < +7%). The combine-tax Amdahl projection was over-optimistic: in the JIT graph the combine **overlaps**, so it is far less on the critical path than its standalone GPU time implied. Rest Route B attention.

## C1 — Baseline reproduced
`extra/qk_b4_combine_tax.py`: base combine **12.6µs @S48 / 16.2µs @S64** standalone, correctness clean. ✅

## C2 — Variants (combine only; same math, inputs/outputs, graph-node injection)
Added to `extra/qk_owned_flash_decode.hip`, selected via `DECODE_ATTN_AMDGCN_COMBINE` (default `base`):
- **`hd<CWD>`** (`owned_flash_combine_hd`): **thread-per-output-dim + meta staged in LDS once** (no redundant global
  meta reloads) + 2D grid `(Hq, Hd/CWD)` for more workgroups. Block `CWD`.
- `sr<CWD>x<CSR>` (`owned_flash_combine_sr`): additionally parallelize the S-reduction across `CSR` threads (LDS
  tree-reduce). **Refuted** — the extra `__syncthreads` + LDS reduction cost more than the saved chain latency.

## C3 — Local combine A/B (`extra/qk_b4_combine_ab.py`) — **PASS**
**Critical methodology fix:** the standalone `wait=True` number carries a **measured 6.44µs launch/sync floor** (a
trivial write-zero kernel reads ~6.44µs). The in-graph-relevant cost is **combine compute = standalone − floor**.
Launch-corrected (median-of-40, no model):

| ctx | S | base combine µs (compute) | best `hd64` µs (compute) | compute speedup |
|---|---|---|---|---|
| 1024 | 48 | 12.7 (6.2) | 11.0 (4.6) | 1.36× |
| 1024 | 64 | 16.3 (9.9) | 12.3 (5.8) | **1.69×** |
| 2048 | 64 | 16.2 (9.8) | 12.2 (5.8) | **1.68×** |
| 4096 | 64 | 15.8 (9.3) | 11.7 (5.2) | **1.78×** |

`hd64` cuts combine **compute 1.7×** (9.9→5.8µs @S64), all ≤ 8µs, correct (`rel_rmse ≤ 1e-3`), no tile regression →
**local gate PASS**. (`hd32/hd64/hd128` are within noise — *not* workgroup-count-bound; the win is LDS-meta +
thread-per-dim. `sr` variants are slower.)

## C4 — Integration
`DECODE_ATTN_AMDGCN_COMBINE=hd64` (default `base`) threads through `amdgcn_flash_decode(..., combine=...)` →
`_combine_spec`/`_specialize_combine` select the kernel symbol + launch geometry. Default route unchanged; greedy
byte-identical (tokens match at every ctx).

## C5 — W==D (the truth, `extra/qk_b4_decode_eval.py --policy adaptive --splits 48 64 --ckpts 512 1024 2048 4096`)
Routed best-S whole-decode delta vs `gqa_coop_vec`, base combine vs `hd64`
(`bench/qk-decode-attention-route-b-b5-combine/policy_or_wd.json`):

| ctx | base combine Δ | `hd64` combine Δ | gain |
|---|---|---|---|
| 512 (off) | −0.14% | −0.15% | — |
| 1024 | +0.20% | +0.25% | +0.05% |
| 2048 | +1.84% | +2.07% | +0.23% |
| 4096 | +5.41% | **+5.71%** | **+0.30%** |

Tokens match throughout. **W==D gate MISS:** +5.71%@4096 < +7%, +0.25%@1024 < +5%.

## Why the cheaper combine barely moved W==D (overturns the projection)
The combine-tax doc projected **+7.4%@4096 if the combine were halved**, from an Amdahl model that treated the combine's
**standalone GPU time** (16µs) as fully on the serial critical path. The measurement refutes that: a **1.7× combine
(−4µs/layer compute)** moved whole-decode only **+0.3%@4096**. In the JIT graph the combine **overlaps** with other
work, so its marginal contribution to the token wall is ~6× smaller than its isolated GPU time. **At the measured
transfer rate, even a FREE combine projects ~+6%@4096 — still below +7%.** This is the recurring "isolated kernel wins
don't transfer to in-model integration" finding: the standalone-GPU-time Amdahl projection over-estimated the lever.

## Verdict & recommendation: `B5_COMBINE_LOCAL_PASS_WD_FAIL`
The combine *is* cheaper (1.7×, banked as the `hd64` owner-knob variant, default-off — it's a strict local improvement
and slightly better W==D), but **no combine-only optimization makes B4 promotable**: the B4 attention route is
**Amdahl-capped below the bar** (even a free combine ≈ +6%@4096). **Rest Route B attention.** The remaining
whole-decode W==D lever is the **non-attention FFN/GEMV share** of the decode step, not the attention primitive.

## Lifecycle
`hd64` registered as the cheaper-combine variant on the existing default-off candidate
`decode_attention_llama_flash_tile_owned_amdgcn_b4` (combine knob `DECODE_ATTN_AMDGCN_COMBINE=hd64`); still
`default_eligible=false`. No default change.

## Deliverables
`docs/b4-cheaper-combine-scope-20260621.md` · `extra/qk_b4_combine_ab.py` ·
`bench/qk-decode-attention-route-b-b5-combine/{latest,policy_or_wd}.json` · the `_hd`/`_sr` combine kernels in
`extra/qk_owned_flash_decode.hip` + the variant registry in `extra/qk_owned_flash_decode_graph_node.py` · this doc.

## Boundaries honored
Only the combine changed. No new tile, no Route-A codegen, no KV repack/transpose, no default change, no closed-lane
reopen. `gqa_coop_vec` comparator SSOT. W==D is the gate (local GPU-busy is launch-corrected diagnostic, not a
headline). Unrelated dirty work untouched.
