# Split-KV Economics Audit — Result

Date: 2026-06-21

Scope: `docs/split-kv-economics-audit-scope-20260621.md`. Follow-on to
`docs/b4-split-kv-combine-tax-result-20260621.md` (`COMBINE_TAX_DOMINATES + NO_POLICY_CLEARS_GATE`).

## Final verdict: **`SPLIT_KV_ECONOMICS_AUDIT_READY`**

A durable, reusable **split-KV economics audit** now exists, reproduces the banked B4 combine-tax conclusion from
the measured artifacts (no remeasure), and is wired into the evaluator binding contract so that **every future
split-KV decode-attention candidate must report tile/combine economics before any W==D promotion work**. No kernel
was built; no default changed.

> The previous audit did not miss a basic bug; it stopped at kernel quality and graph integration. B4 exposed the
> next lifecycle layer: split-KV reduction economics. From now on, every split-KV decode-attention candidate must
> report tile/combine split, combine fraction, effective bandwidth, workgroup count, and Amdahl projection before
> W==D promotion work.

## Why did earlier audits not catch this?

Each earlier audit layer caught a real failure mode and stopped exactly at the layer it was built to test:

- the **llama oracle / codegen-quality** audit proved tinygrad could not emit a llama-quality fused LDS/`v_dot2`
  tile (the 5–6× standalone gap);
- **B1** proved a llama-class tile **wins** when dispatched by tinygrad's HCQ (GPU-busy), and **B1→B2** proved raw
  per-call **dispatch overhead** eats that win unless the launches are graph-batched;
- **B2** proved one bound HCQ queue recovers the wall; **B3** proved an **owned** hand-AMDGCN tile beats
  `gqa_coop_vec` locally (2.35× GPU-busy); **B4** proved the owned `.co` can enter the TinyJit decode graph as
  `Ops.PROGRAM` nodes.

All of those are **kernel quality** and **graph integration** layers. None required the candidate to account for
the **split-KV reduction economics**. A Flash-Decoding tile splits the KV cache into `S` chunks and must MERGE the
per-split partials in a **separate combine kernel** — a flat latency/occupancy floor that the tile A/B never
measures. So a candidate could pass every layer above and still miss W==D for a reason none of them tested. B4 is
exactly that: it passed the owned local A/B and the graph-node capability, then missed W==D (`B4_WD_FAIL_INTEGRATION`)
because the combine gives back part of the tile win every layer. The blind spots, specifically:

- split-KV partial **merge is a separate lifecycle/economics tax**, distinct from the tile;
- **low combine bytes do not imply a cheap combine** — it can be latency/occupancy-bound;
- the combine can be the **binding W==D lever** even when the tile is a large local win;
- **Amdahl** (attention ≈17% of the decode step) can make a real local win non-promotable.

## What audit check is now permanent?

`extra/qk_split_kv_economics_audit.py` → `bench/qk-split-kv-economics-audit/latest.json`
(`split_kv_economics_audit_v1`, harness-contract-stamped **CONFORMS 13/13**). For each context it reports, from the
measured tile/combine attribution + routed W==D (it **reuses** the B4 data, it does not remeasure):

- `tile_us`, `combine_us`, `total_us`, `combine_fraction`;
- `combine_bytes_est`, `combine_effective_gbps` and `% of HBM peak` (latency- vs bandwidth-bound);
- `tile_workgroups`, `combine_workgroups`, and an occupancy proxy (workgroups vs the 96 CUs);
- the per-context optimal split `S` (min total attention µs) and the operative S (the S the W==D was measured at);
- an **Amdahl projection** of whole-decode W==D for **measured / half / free** combine;
- a **classification**.

It is enforced as `split_kv_economics_contract_v1` in `bench/qk-decode-eval/binding_templates.json` (referenced by
the `north_star_flash_attn_tile_v0` and `decode_attention_llama_flash_tile_owned_amdgcn_v0` split-KV bindings), and
banked as a principle in `structure/Development/performance-primitive-research-principles.md`
(§ "Split-KV Reduction Economics Are Part Of The Decode Primitive"). The tool also runs `--live` (regenerate the
attribution first) and general (`--attribution/--wd/--candidate`) for any future candidate.

## How does B4 classify under it?

**`COMBINE_TAX_DOMINATES`** — reproducing the banked B4 conclusion from the measured artifacts:

| ctx | operative S | tile µs | combine µs | combine % | eff GB/s | combine wg | W==D measured | half-combine | free-combine |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 48 | 15.9 | 12.6 | 44% | 64.5 | 32 | −0.14% (off) | — | — |
| 1024 | 48 | 23.3 | 12.6 | 35% | 64.7 | 32 | +0.20% | +1.74% | +3.33% |
| 2048 | 48 | 36.7 | 12.6 | 26% | 64.5 | 32 | +1.84% | +3.39% | +4.98% |
| 4096 | 48 | 62.5 | 12.6 | 17% | 64.5 | 32 | **+5.41%** | **+6.97%** | **+8.58%** |

- The combine is **latency-bound, not bandwidth-bound** (~64 GB/s ≈ **6.7% of HBM peak**, only **32 workgroups on
  96 CUs** → occupancy proxy 0.33) → **fixable** with more parallelism / fusion.
- The combine is a **flat floor in context**, scaling only with `S` (~12.6 µs at S=48 every ctx); as a share of
  attention it is large short (44%/35%) and shrinks long (17% @ctx4096).
- The Amdahl projection (anchored on the measured routed delta + the per-layer combine cost × 36 layers) shows a
  cheaper combine moving ctx4096 from **+5.41% measured → ~+7.0% (half) → ~+8.6% (free)** — i.e. a cheaper combine
  is projected to clear the **+7%@ctx4096** gate. **Amdahl co-limits** the long-context ceiling (even a free combine
  caps attention's ~17% share).

So B4's W==D miss is a **split-KV reduction-economics** problem, not a kernel-quality or graph-integration problem —
the next attention-specific lever is a **cheaper combine**, not another tile.

## What must future split-KV candidates report?

Per `split_kv_economics_contract_v1`, any candidate whose dataflow does Flash-Decoding KV-splits + a separate
combine MUST declare/produce, before any W==D promotion work:
`tile_us_by_ctx`, `combine_us_by_ctx`, `combine_fraction_by_ctx`, `combine_bytes_est`, `combine_effective_gbps`,
`split_count_by_ctx`, `tile_workgroups_by_ctx`, `combine_workgroups_by_ctx`, `amdahl_projection` — and be
**classified** by the audit. The four classes and what they mean for the next action:

- `COMBINE_TAX_DOMINATES` — a cheaper/fused combine is projected to clear the gate → **build a cheaper combine** (B5);
- `COMBINE_SMALL_AMDAHL_LIMIT` — even a free combine cannot clear it → attention's Amdahl share is the ceiling;
  **do not** build a cheaper combine, attack the FFN/GEMV share instead;
- `POLICY_ONLY` — a ctx-gated opt-in already clears the gate → the lever is a routing policy;
- `MEASUREMENT_UNSTABLE` — no trustworthy W==D anchor (or signal inside its noise band) → **tighten the harness first**.

A split-KV candidate that reports a tile A/B win but omits the economics fields is **not W==D-ready** — the audit
flags it incomplete. This closes the gap that let B4 reach a W==D attempt on a tile A/B alone.

## What is the next action after the audit?

The audit verdict for B4 is `COMBINE_TAX_DOMINATES`, which **justifies** (but this task does not build) the next
bounded scope:

```text
Route B B5: cheaper split-KV combine for the existing B4 route.
  target: owned_flash_combine from ~12-16 us to <= ~8 us (ideally ~5 us) at useful S
  allowed: more-parallel reduction over Hq x Hd; cooperative reduction that raises occupancy;
           bounded fused/streamed merge that removes the partial write->read round-trip or the 2nd launch
  gate:    re-run extra/qk_b4_decode_eval.py; pass W==D only if ctx4096 >= +7% (no ctx512/ctx1024 regress) or ctx1024 >= +5%
  re-audit: extra/qk_split_kv_economics_audit.py must re-classify after B5 (expect the combine fraction to drop)
  non-goals: no new attention tile, no Route-A codegen, no KV repack, no default promotion without the W==D gate
```

If B5 is not funded, the state is: B4 infrastructure win banked, B4 W==D promotion failed (Amdahl + combine tax),
bounded attention lane rests; the audit guarantees the next candidate is measured honestly.

## Acceptance / verification

- `extra/qk_split_kv_economics_audit.py` (default) → reproduces the B4 table above and classifies
  **`COMBINE_TAX_DOMINATES`**; artifact `CONFORMS 13/13`.
- no-W==D-anchor run → **`MEASUREMENT_UNSTABLE`** (honest: the Amdahl projection cannot be computed without a
  measured W==D anchor) — proves the classifier does not fabricate a ceiling.
- `binding_templates.json` / `candidates.json` validate as JSON; `split_kv_economics_contract_v1` references the two
  split-KV bindings and the B4 candidate.
- No `tinygrad/` change, no model route/default change, no remeasure of the underlying GPU data, no closed-lane reopen.

## Boundaries honored

Audit/tooling/docs only. No new combine kernel, no tile optimization, no Route-A codegen, no default change, no
broad-benchmark rerun (the audit reads the existing measured artifacts). `gqa_coop_vec` comparator SSOT. This task
ends with the audit/tooling/doc contract ready for the next B5 cheaper-combine scope — it does not build it.
