# 8B Remaining Architecture Understanding — Result (2026-06-22)

## Decision: **FUND_FUSED_ATTENTION_FIRST** (the single fused owned AMDGCN tile — the convergence substrate)

Audit/decision-only. No source implementation, no default change, no 14B/32B, no reopened bounded lanes.

## 1. Executive summary
The bounded 8B primitive ladder is exhausted; the two remaining levers are **architectural**: remove the
KV-copy tax (Lane 1) and close the attention tail (Lane 2). The audit's decisive finding is that **the two
lanes converge on one substrate — the single fused *owned* AMDGCN attention tile**:
- **Lane 1 (runtime KV)** removes the ~1.4 ms full-MAXC copy, with **measured** transfer (+1.5 ms / +8 tok/s
  via the MAXC-shrink test). But it **requires an opaque attention read** — the default functional
  `gqa_coop_vec` reduce over the mutated cache re-hits the read-after-write hazard, so Lane 1 cannot proceed
  without an opaque/pointer-length attention node.
- **Lane 2 (fused attention)** — the bounded *separate* tile+combine form is exhausted (+5.66 %@4096 < +7 %
  gate, because the combine overlaps off the critical path). But the **one untried lever** is a **single
  fused tile** (combine folded in, no part/meta HBM round-trip). The owned tile body already has the
  llama ingredients (`v_dot2` via `__builtin_amdgcn_fdot2`, LDS staging, online softmax+PV); B3/B4/B5 only
  ever emitted *two* graph nodes (tile → separate combine).

That single fused owned tile **is** the opaque attention read Lane 1 needs. Building it (a) tests the one
untested bounded attention lever and (b) unlocks the measured KV-copy removal. **Robust value:** best case it
clears +7 % (attention closes) *and* enables copy removal; worst case (attention rests ≤ +5.7 %) it still
serves as the opaque read to capture the measured +1.4 ms copy removal. The native-linearizer / renderer
routes are **unbounded core-tinygrad** — defer. **Promote `Q4K_GEMV_WARP` to default in parallel** — already
won, independent.

## 2. Current 8B closure table
| Lane | Status | Why |
|---|---|---|
| Weight GEMV | **Closed / won** | `Q4K_GEMV_WARP` lossless, W==D pass, default-eligible (owner decision pending). |
| FFN activation | **Closed** | silu fused into gate/up GEMV; old "activation" bucket was the KV copy. |
| Norm/Rope | **Closed** | genuine RMSNorm/qk-norm near parity or faster than llama. |
| Attention (bounded Route B/B5) | **Closed for bounded *separate* form** | +5.66 %@4096 < +7 %; combine overlaps off critical path. |
| KV-cache copy (local fix) | **JIT-blocked** | `.assign()` / slice-`.after()` fail under same-graph pure-function semantics. |
| KV opaque append (write) | **Microprobe PASS** | symbolic-offset in-place write + capture/replay with changing `start_pos`. |
| Same-graph KV read-after-write | **Blocked** | attention *functional reduce* over the mutated buffer re-hits the hazard. |

**No bounded model/primitive lever remains.** The remaining levers are the two architectures below.

## 3. Lane 1 — Runtime-managed / two-graph KV cache
**Verdict: `RUNTIME_KV_REQUIRES_ATTENTION_INTERFACE_REWRITE`** (`bench/.../runtime_kv.json`).

| design | what changes | why it might work | exact blocker/risk | recoverable | verdict |
|---|---|---|---|---:|---|
| A. separate append graph + attention graph | append realized before attention (gpt2 `.assign().realize()`) | separate graphs avoid the *intra*-graph read-after-write hazard | breaks the fused `@function` `_run`; 36 separate append realizes/token; cross-graph ordering without **host sync** is the open risk | ~1.4 ms | feasible but **model restructure** |
| B. runtime KV object + pointer/length | runtime object owns cache; opaque append writes, **opaque attention reads** | two opaque kernels (write→read same buffer) = clean buffer dependency, not a REDUCE-over-mutated-buffer hazard | **couples to an opaque attention read** (the owned tile); lifecycle (reset/batch/prompt/server) | ~1.4 ms | **scope-ready, couples to Lane 2** |
| C. state token in HCQ only | HCQ-level ordering token | ordering below the functional scheduler | the attention *read* is still a functional reduce → still materializes the copy; reduces to B | 0 | reduces to B |
| D. alias-aware same-graph mutation | scheduler rule for in-place read-after-write | default attention could read the mutated buffer | symbolic-**range** alias under symbolic `start_pos` → `_eval` wall + symbolic-size reject | ~1.4 ms | **UNBOUNDED (refuted)** |
| E. paged/block-table | paged KV + page-table attention | avoids fixed-MAXC; better serving | reopens attention layout (page-table-aware kernels) | ~1.4 ms | document-only |

**Answers:** two-graph decode is feasible but restructures the fused per-layer forward and risks host-sync
overhead eating the savings; the **runtime-KV object (B)** keeps one graph but **needs the opaque attention
read** (the owned tile already reads `cache_kv` natively). Correctness hazards: stale KV across prompts,
repeated generation, batch>1 (kernel is B==1), prefill→decode handoff, speculative/T>1 decode, server
concurrency — all manageable by a runtime KV object. **Blast radius: fork-local, NO core tinygrad change**
(uses `custom_kernel`); core change only for the refuted design D. The opaque append already exists
(`extra/qk_kv_cache_state_token.py`); the missing half is the opaque attention read.

**Performance model** (post-`Q4K_GEMV_WARP`):
| ctx | tok/s | token_ms | −KV copy → tok/s | attn gap (ms) | −both → tok/s | llama gpu / clean |
|---|---|---|---|---|---|---|
| 512 | 76.1 | 13.15 | **85.3** | 1.48 | 97.6 | 104.6 / 97.7 |
| 1024 | 74.0 | 13.51 | **82.7** | 1.64 | 95.7 | 100.3 / 97.4 |
| 2048 | 71.0 | 14.09 | **78.9** | 1.99 | 93.6 | 91.3 / — |
| 4096 | 67.0 | 14.93 | **74.0** | 2.58 | 91.5 | 77.7 / 92.4 |
KV copy ~1.4 ms **flat** (its relative value falls as ctx grows); attention gap **grows** with ctx.
Removing **both** ≈ llama parity at every ctx.

## 4. Lane 1 first gate (the follow-on, after the Lane 2 tile lands)
opaque append + owned-tile-reads-persistent-cache (no `assigned_kv` copy): no full-MAXC copy in the rendered
graph · append overhead < 0.25 ms/token · device-side ordering (no host sync) · byte-identical 64-token
generation · **W==D ≥ +5 %@ctx1024** · no stale-cache leakage across two prompts in one process · default-off flag.

## 5. Lane 2 — Codegen-level fused attention
**Verdict: `FUSED_ATTENTION_AMDGCN_SINGLE_TILE_GATE_READY`** (`bench/.../fused_attention_codegen.json`).

| route | target | why prior failed | what's different | first gate | risk | verdict |
|---|---|---|---|---|---|---|
| A. native linearizer coupled multi-reduce | LDS+`v_dot2` coupled softmax+PV | expressiveness is **not** the wall (coupled `(m,l,acc)` runs today) but the `.set/.after` idiom emits **scalar** code; tiled-GEMM ⟂ `.set/.after`; symbolic-count tiled matmul inexpressible | nothing concrete | — | high (core) | **UNBOUNDED_DEFER** |
| **B. AMDGCN single fused tile** | **ONE** kernel: QK+softmax+PV+`v_dot2`+LDS, **combine folded in** | **never tried** — B3/B4/B5 always emitted separate tile→combine | fold combine in → removes the part/meta HBM round-trip the +5.7 % ceiling never tested | single fused node, byte-correct, local A/B ≥1.5× vs `gqa_coop_vec`, W==D projection | **bounded, fork-local, HCQ-proven** | **GATE_READY** |
| C. renderer template | flash-decode schedule template | scoped only | nothing | — | highest (core) | **UNBOUNDED_DEFER** |
| D. improve split attention | more split/combine tuning | combine overlaps → cheaper combine doesn't transfer | — | — | — | REST (separate form exhausted) |
| E. do nothing / rest | — | — | — | — | — | premature (single-tile gate untried) |

**Why Route B didn't clear W==D:** combine **overlap / off-critical-path**, measured (combine 9.7→4.0 µs moved
whole-decode +0.25 % then ~0; the Amdahl/combine-tax projection was *refuted*). **Inexpressible object:** a
single kernel that is *both* LDS-tiled `v_dot2` *and* register-resident online-softmax `(m,l,acc)` across the
KV loop — tinygrad's two codegen paths (tiled-GEMM vs `.set/.after` scalar fusion) are mutually exclusive.
**Oracle:** llama's `flash_attn_tile` is **5.87/5.71/4.77×** faster than `gqa_coop_vec` standalone @ctx
512/1024/4096 (LDS `ds_load_b128` + 256–1024 `v_dot2` + softmax+PV in one kernel vs tinygrad's 6–8 scalar
HBM-roundtrip kernels). **Exceeding +5.7 %@4096:** the ceiling is the *separate-form* ceiling (combine
overlaps); a single fused tile removing the part/meta round-trip is the **one lever the overlap mechanism does
not rule out** — plausible but unproven. Blast radius: **fork-local** (`extra/qk_owned_flash_decode.hip` +
graph node + evaluator); routes A/C are core-tinygrad multi-week.

## 6. Lane 2 first gate (the recommended first funded work)
One Qwen3-8B/gfx1100 ctx1024 **single fused** flash-decode tile (combine folded into `owned_flash_tile_gqa`,
no part/meta round-trip), graph-node integrated, **default-off** · byte-correct vs `gqa_coop_vec`/numpy ·
local attention A/B **≥ 1.5×** vs `gqa_coop_vec` @ctx1024 · W==D projection before any in-model route ·
**hard stop:** if the single fused tile still saturates ≤ +5.7 %@4096 in W==D, the ceiling is structural →
`REST_DECODE`, but still pair the tile (as the opaque read) with the KV append to capture the measured
+1.4 ms copy removal.

## 7. Cross-lane ranking
| rank | lane | expected gain | confidence | impl size | risk | why now / not |
|---:|---|---|---|---|---|---|
| 1 | **Single fused owned AMDGCN tile** (Lane 2; unlocks Lane 1) | attention up to +7 %@4096 (uncertain) **+** KV-copy removal +1.4 ms/+8 tok/s@1024 (measured, via the tile as opaque read) | medium (attn) / high (copy) | medium, fork-local | medium | clearest bounded gate; captures the measured copy win as a fallback; tests the one untried attention lever |
| 2 | Runtime-KV copy removal (Lane 1) | +1.4 ms/+8 tok/s (measured), flat | high transfer, **blocked on rank 1** | medium-high | medium | can't go first (needs the opaque tile); fast follow-on |
| 3 | Native-linearizer / renderer fused-flash | full attention close | low | large, **core tinygrad** | high | unbounded; fund only after the escape-hatch proves the ceiling |
| ∥ | Promote/harden `Q4K_GEMV_WARP` | already won (+9.8 %@1024) | high | owner decision | low | independent; do in parallel regardless |

Ranking-rule check: rule 3 (no theoretical over measured) is honored — the recommendation rests on the
**measured** copy-removal transfer that the tile unlocks, not the uncertain attention gain. Rule 4
(promote-GEMV-if-both-unbounded) is *not* triggered (the single tile is bounded), but GEMV promotion is
recommended in parallel anyway.

## 8. Recommendation
**`FUND_FUSED_ATTENTION_FIRST`** — build the single fused owned AMDGCN tile (Lane 2 first gate) as the
gate-first, fork-local convergence substrate. Then, as the immediate follow-on, pair it with the proven
opaque KV append to capture the measured +1.4 ms copy removal (Lane 1). Promote `Q4K_GEMV_WARP` to default
(owner decision) in parallel. Do **not** fund the native-linearizer/renderer routes now (unbounded core
tinygrad); revisit only if the escape-hatch tile demonstrates a > +7 % attention ceiling worth the blast radius.

## 9. Required future docs if funded
- `docs/fused-flash-single-tile-{scope,result}-20260623.md` — the single fused tile + local A/B + W==D gate.
- `docs/runtime-kv-opaque-read-{scope,result}-20260623.md` — pairing the opaque append with the tile-as-opaque-read (Lane 1 follow-on).
- a `Q4K_GEMV_WARP` default-promotion owner-decision note (independent).

## 10. Artifacts and commands
- `bench/qk-8b-remaining-architecture-understanding/{runtime_kv,fused_attention_codegen,decision}.json`.
- Authority inputs (no new runs; this is an audit): `bench/qk-tinygrad-vs-llama-time-tax/latest.json`,
  `bench/qk-ffn-activation-gap-audit/latest.json`, `bench/qk-kv-cache-stateful-jit/*`, and the docs in §Required Reading.
- Repro of the perf model: `.venv/bin/python` over the two bench artifacts above (see this doc §3 table).

## 11. Working tree status
Audit-only: no source/default change; new `docs/8b-remaining-architecture-understanding-{scope,result}-20260622.md`
+ `bench/qk-8b-remaining-architecture-understanding/*.json` (the bench JSON carries projections; not committed
if it duplicates non-deterministic timing — here it is derived/static, so it is committable). `model.py` clean.
