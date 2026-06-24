# North-Star Flash-Attn-Tile Candidate — Execution Result

> ⚠ **CORRECTION (see `docs/north-star-decode-attention-redesign-audit-20260621.md`).** The `FAIL_LOCAL_AB` verdict
> stands (confirmed by a throughput probe), but this doc's **"the combine is HBM-bandwidth-bound"** attribution
> (below) is **WRONG**. Traffic accounting shows pout is ~1 MB (~1 µs at HBM peak) — negligible; the latency-measured
> "combine cost" was 2nd-raw-dispatch overhead (the candidate runs 2 un-batched raw dispatches vs coop's batched JIT
> graph). The real ceiling is the **cooperative-dot q·k partial** (flat ~163 µs throughput vs coop's scaling
> 75–144 µs); coop's matmul q·k is near-optimal for tinygrad primitives. Read the redesign audit for the corrected
> diagnosis.

Date: 2026-06-21

Scope: implement the missing pieces (kernel, local-A/B runner, decode_eval binding) so
`gen_north_star_flash_attn_tile` moves from `PRUNE_NEEDS_TEMPLATE` to **EXECUTE** through
lifecycle-search → decode_eval → artifact → verdict. First real north-star performance attempt.

## Final decision: **`NORTH_STAR_FAIL_LOCAL_AB`**

The candidate is now **real and executable** (no longer deferred), was measured against the current winner
`gqa_coop_vec`, and **missed the local A/B gate: 0.58× @ctx1024, 0.89× @ctx4096** (byte-exact, err 0.0,
clock-pinned). Per the discipline, the local gate failed → **stopped before any W==D route**, classified, and banked
a refutation. This is a clean, measured negative — and it **corrects the project's prior hypothesis**.

## Phase 0 — design note (what must not be repeated)

Binding `north_star_flash_attn_tile_v0`; comparator `gqa_coop_vec` (never a weak baseline). Prior failures:
- **raw fused flash tile** (0.21×) — per-thread q·k redundancy (128 d-threads each recompute the full dot).
- **scalar LDS+GQA tile** — occupancy collapse / same q·k redundancy.
- **FLASH_L=64** — local pass (~1.08×) but W==D fail (combine ceiling, attention only ~23% of decode).
- **WMMA decode** — refuted (llama decode is non-WMMA vector).

The candidate **differs**: it reuses the **warp-cooperative q·k** (ds_bpermute butterfly dot — NO 128× redundancy)
+ LDS K/V staging + GQA query-head packing (4 warps) + register online softmax + many KV-splits (128–1024
workgroups, GROWS with ctx, ≥ comparator), and adds the one unexploited lever the binding names: a
**many-workgroup combine** (`streamk_combine`, Hq·(Hd/DT)=128 wg) replacing the serial `flash_reduce` (Hq=32 wg).
First concrete gate: ≥1.05× vs gqa_coop_vec @ctx1024; stop + classify on miss.

## Phase 1/2 — structural + local A/B (`extra/qk_north_star_flash_attn_tile_ab.py`)

Design preserves T=1 parallelism (partial workgroups GROW with ctx and far exceed gqa_coop_vec); GQA heads are
warp-parallel (not serialized); no WMMA. The differentiation gate (G1/G2) passed → implemented + measured.

**Measured (clock-pinned, byte-exact err 0.0; `bench/qk-north-star-flash-attn-tile/latest.json`):**

| ctx | comparator `gqa_coop_vec` µs | **partial alone** µs | serial combine total | streamk combine total | best speedup |
|---:|---:|---:|---:|---:|---:|
| 512 | 99.4 | ~108 | 190 | 190 | **0.52×** |
| 1024 | 109.4 | ~111 | 188 | 187 | **0.58×** |
| 4096 | 169.7 | **~152** | 191 | 190 | **0.89×** |

(best split S∈{16,32,64,96}; partial workgroups = Hkv·S = 128–768.)

## Classification — the prior hypothesis is REFUTED

The binding/audit predicted the **serial combine** was the ceiling and a stream-k/many-workgroup combine would lift
it. Direct attribution refutes that and locates the real ceiling:

1. **The combine is HBM-bandwidth-bound, not launch/occupancy-bound.** The `streamk_combine` (128 workgroups) is
   **byte-for-byte as slow as** the serial `flash_reduce` (32 workgroups) at every ctx (187 vs 188µs @1024). More
   workgroups do **not** help — the cost is **reading the partials** `pout` (Hq·S·Hd floats, which GROW with S), a
   bandwidth cost no amount of parallelism removes. This is the true "split-combine ceiling": bandwidth on the
   materialized partials, not the serial launch.
2. **At ctx1024 the partial itself isn't faster than coop.** The warp partial alone (~111µs) ≈ coop's *whole*
   attention (~109µs) — so even a free combine could not clear 1.05×. coop's **separated** q·k (an optimized
   tinygrad GEMM into a scores buffer) + weighted-sum partial + optimized multi-kernel combine beats the **fused**
   cooperative-dot partial at small ctx.
3. **At ctx4096 the partial IS faster (152 vs 170µs ≈ 1.12×), but the bandwidth-bound combine (+38µs) drags the
   total to 0.89×.** So the combine genuinely caps the long-ctx win — but it is bandwidth-bound, so the named
   "stream-k" lever cannot fix it.

**Stop-condition hit:** binding stop-condition #4 ("local A/B ≤ comparator"). Reason class: *combine overhead
(HBM-bandwidth-bound on materialized partials) + q·k partial mapping (fused cooperative dot ≤ separated matmul at
small ctx)*. A future candidate must produce a **faster q·k partial** and/or **avoid materializing large split
partials** (fuse combine into the partial without collapsing workgroups) — not merely a more-parallel combine.

## Phase 3 — decode_eval binding (candidate is now executable)

- `bench/qk-decode-eval/candidates.json`: added `north_star_flash_attn_tile` (local_ab via `ab_script` →
  `extra/qk_north_star_flash_attn_tile_ab.py`, byte-exact gate). Verdict **`FAIL_LOCAL_AB`**.
- `binding_templates.json`: `north_star_flash_attn_tile_v0` → `concrete_runner_status: implemented`,
  `decode_eval_binding: north_star_flash_attn_tile`, `local_ab_result` recorded; `missing_for_executable` reduced
  to the W==D route (NOT pursued — local gate failed first).
- Generator: `gen_north_star_flash_attn_tile` now carries `decode_eval_candidate_id` (executable),
  `expected_verdict: FAIL_LOCAL_AB`, `executable_status: local_ab_implemented_failed_gate`.

## Phase 4 — W==D route: **NOT added** (local gate failed; discipline = stop before W==D).

## Phase 5 — lifecycle-search integration

`gen_north_star_flash_attn_tile` now **EXECUTEs** (was `PRUNE_NEEDS_TEMPLATE`) → decode_eval → **`FAIL_LOCAL_AB`** →
`refute_candidate`. `gen_north_star_binding_selftest` still `SELFTEST_PASS`; `gen_north_star_missing_binding` still
`PRUNE_MISSING_EVALUATOR_BINDING`; closed lanes still pruned; policy guard PASS. (Run artifact:
`bench/qk-lifecycle-search/runs/decode_v0-*.json`.)

## Acceptance gates

| gate | result |
|---|---|
| G1 design differs from prior failed tiles | PASS (warp-coop q·k + many-wg combine) |
| G2 candidate preserves/enlarges T=1 parallelism vs winner | PASS (partial wg 128–768 ≫ coop) |
| G3 local A/B runner emits schema'd artifact (9 binding fields) | PASS |
| G4 local correctness passes or classified | PASS (byte-exact err 0.0) |
| G5 local perf fails → committed as refutation, no W==D route | PASS |
| G6 (W==D only if local passes) | n/a (local failed) |
| G7 gen_north_star no longer vague — executes or precise blocked | PASS (EXECUTE → FAIL_LOCAL_AB) |
| G8 no defaults changed | PASS |
| G9 no closed lanes reopened | PASS |
| G10 policy guard passes | PASS |
| G11 tree clean after commit | PASS (commit below) |

## Refutation banked

`bench/qk-lifecycle-search/refutations.json` → `north_star_warp_tile_partial_exceeds_coop_combine_bandwidth_bound`
(applies_to decode_vector_flash_tile_high_kvsplit + north_star_flash_attn_tile).

## Next unlocked project

A north-star candidate that targets the **measured** ceiling: a **faster q·k partial** (e.g. keep coop's separated
optimized matmul for scores, then a better tiled weighted-sum) and/or a **combine that does not re-read large
materialized split-partials** (fuse the LSE reduction into the partial while keeping workgroups high). The combine
is HBM-bandwidth-bound, so the lever is *less partial traffic*, not more combine parallelism. This would be a new
binding-conformant candidate, evaluated the same way.

## Changed files

`extra/qk_north_star_flash_attn_tile_ab.py` (new), `bench/qk-decode-eval/{candidates.json, binding_templates.json}`,
`extra/qk_candidate_template_gen.py`, `bench/qk-lifecycle-search/refutations.json`, this doc + handoff/READMEs.

## Boundary

No `tinygrad/` change, no model route/default, no kernel shipped, no WMMA/MMVQ, no W==D route (local failed), no
tuning sweep beyond the one differentiated lever. Clock-pinned diagnostic, perf-state restored to `auto`.
