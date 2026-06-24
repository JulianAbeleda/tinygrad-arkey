# Split-KV Economics Audit — Scope

Date: 2026-06-21

Follow-on to `docs/b4-split-kv-combine-tax-result-20260621.md` (`COMBINE_TAX_DOMINATES + NO_POLICY_CLEARS_GATE`).
That task *measured* the B4 combine tax once. This task makes the lesson **permanent**: a durable, reusable
**split-KV economics audit** that every future split-KV decode-attention candidate must pass before any W==D
promotion work.

## Why this audit layer is needed (the gap the previous audits left)

The decode-attention lifecycle ladder already had several audit layers, and each caught a real failure mode:

| earlier audit | what it caught |
|---|---|
| llama oracle / codegen-quality | tinygrad cannot emit a llama-quality fused LDS/`v_dot2` tile (the 5–6× standalone gap) |
| B1 HCQ de-risk | a llama-class kernel **does** win when dispatched by tinygrad's runtime (GPU-busy) |
| B1 → B2 wall caveat | raw per-call **dispatch overhead** eats the GPU win; graph batching is required |
| B2 graph integration | folding launches into one bound HCQ queue recovers the wall |
| B3 owned tile | an **owned** hand-AMDGCN tile beats `gqa_coop_vec` locally (2.35× GPU-busy) |
| B4 graph-node | the owned `.co` can enter the TinyJit decode graph as `Ops.PROGRAM` nodes |

None of these layers required the candidate to account for the **split-KV reduction economics**. A flash-decode
tile that splits the KV cache into `S` chunks must MERGE the per-split partials in a separate **combine** kernel.
That combine is a **flat latency/occupancy floor** that the tile A/B never sees — so a candidate can pass every
layer above and still **fail W==D for a reason none of them measured**. B4 is exactly that: it passed the owned
local A/B and the graph-node capability, then missed W==D because the combine gives back part of the tile win.

What the previous audits did **not** require a candidate to report:

- split-KV partial **merge is a separate lifecycle/economics tax**, distinct from the tile;
- **low bytes do not imply a cheap combine** — the combine moves only ~0.8 MB yet costs ~12–16 µs because it is
  **latency/occupancy-bound** (~64 GB/s ≈ 6.7% of HBM peak, only 32 workgroups on 96 CUs);
- the combine can be the **binding W==D lever** even when the tile is a big local win;
- **Amdahl** can make a real local attention win **non-promotable** (attention is ~17% of the decode step).

## Objective

Create a durable audit that, for any split-KV decode-attention candidate, reports and classifies the
tile/combine economics **before** W==D promotion work — so a candidate cannot claim readiness on a tile A/B alone.

It must emit, per context:
- `tile_us`, `combine_us`, `total_us`, `combine_fraction`;
- `combine_bytes_est` and `combine_effective_gbps` (vs HBM peak → latency- vs bandwidth-bound);
- `tile_workgroups`, `combine_workgroups`, and an occupancy proxy (workgroups vs CU count);
- the per-context optimal split `S` (min total attention µs);
- an **Amdahl projection** of whole-decode W==D for **measured / half / free** combine;
- a **classification**: `COMBINE_TAX_DOMINATES` | `COMBINE_SMALL_AMDAHL_LIMIT` | `POLICY_ONLY` | `MEASUREMENT_UNSTABLE`.

## Method (reuse measured data; do not remeasure)

The B4 attribution (`bench/qk-decode-attention-route-b-b4-combine-tax/latest.json`, 44 ctx×S rows) and the routed
W==D (`.../policy_sweep.json`) are already measured. The audit **reads** them and **derives** the economics:

- **combine effective bandwidth** = `combine_bytes / combine_us` (latency- vs bandwidth-bound test vs 960 GB/s peak);
- **occupancy proxy** = workgroups / `CU_COUNT` (96 on gfx1100): combine = 32 wg → 0.33 (under-occupied);
  tile = `Hkv·S` (384 at S=48) → 4.0 (oversubscribed);
- **operative S** = the split the W==D was measured at (so the Amdahl anchor and the economics agree); the
  attribution's min-total-µs optimal S is also reported;
- **Amdahl projection** (additive, anchored on the measured routed delta `d`):
  `saved_meas = T_base · (d/100)/(1+d/100)`; making the combine cheaper by fraction `f` frees
  `N_LAYERS·combine_us·f` more per token; `delta(f) = saved(f)/(T_base − saved(f))·100` for `f ∈ {0, .5, 1}`.
  `N_LAYERS=36`, `T_base` from the canonical decode curve (`bench/qk-decode-runtime-overhead/result.json`).

The audit also supports a `--live` mode (regenerate the B4 attribution via `extra/qk_b4_combine_tax.py` first) and
a general mode (`--attribution`/`--wd`/`--candidate`) so any future candidate emitting the same attribution schema
flows through unchanged. No kernel is built; no default changes.

## Classification logic

| condition | verdict |
|---|---|
| no measured W==D anchor / signal inside its own noise band | `MEASUREMENT_UNSTABLE` |
| some context already clears the gate as measured | `POLICY_ONLY` (ctx-gated opt-in is the lever, not the combine) |
| a **free** combine clears the gate, combine is latency-bound, halving ≈ reaches the gate | `COMBINE_TAX_DOMINATES` (cheaper combine is the lever) |
| even a **free** combine does not clear the gate | `COMBINE_SMALL_AMDAHL_LIMIT` (Amdahl share is the ceiling) |

## Deliverables

- Audit tool: `extra/qk_split_kv_economics_audit.py` (reusable; default read-only, `--live`, general modes).
- Artifact: `bench/qk-split-kv-economics-audit/latest.json` (`split_kv_economics_audit_v1`, contract-stamped).
- Binding-template requirements (`bench/qk-decode-eval/binding_templates.json`): split-KV candidates must declare
  `tile_us_by_ctx`, `combine_us_by_ctx`, `combine_fraction_by_ctx`, `combine_bytes_est`, `combine_effective_gbps`,
  `split_count_by_ctx`, `tile_workgroups_by_ctx`, `combine_workgroups_by_ctx`, `amdahl_projection`.
- Result doc: `docs/split-kv-economics-audit-result-20260621.md`.
- Principle banked in `structure/Development/performance-primitive-research-principles.md`; handoff updated.

## Boundaries

No new combine kernel, no tile optimization, no Route-A codegen, no default change. Audit/tooling/docs only. Reuse
the measured B4 artifacts; remeasure only if a required field is missing. `gqa_coop_vec` comparator SSOT.
This task ends with the audit contract ready for the next **B5 cheaper-combine** scope; it does **not** build it.
