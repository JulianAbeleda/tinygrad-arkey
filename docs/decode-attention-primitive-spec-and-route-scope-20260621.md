# Decode-Attention Primitive — Full Spec + Implementation-Route Decision (Scope)

Date: 2026-06-21

Scope/decision only — **no kernel built, no `tinygrad/` change, no model/default route, no closed lane reopened.**
This is the "next meaningful llama-facing work" after audit/cleanup: define the llama-style decode-attention primitive
as a **first-class lifecycle row** (full boundary), and **decide the implementation route** (native tinygrad codegen
vs AMDGCN/HSACO escape hatch). Authority/evidence: `fused-flash-concrete-gate-result-20260621.md` (the concrete gate
FAILED → bounded+concrete tinygrad codegen exhausted), `llama-flash-attn-tile-oracle-result-20260621.md` (llama is
5–6× standalone; source BOUNDED), `low-level-decode-attn-attribution-result-20260621.md` (`FIXABLE_CODEGEN`: scalar
no-`v_dot2`/no-LDS vs llama's LDS-staged `v_dot2` tile), `native-fused-flash-linearizer-scope-20260621.md`,
`gpu-low-level-control-tooling-reference-20260621.md` (the escape-hatch rule), `post-matmul-pv-decode-strategic-scope-20260621.md`,
`project-north-star-llama-and-lifecycle-search-20260620.md`, `structure/Development/performance-primitive-research-principles.md`.

## Decision: **`DECODE_ATTENTION_PRIMITIVE_SCOPE_READY — ROUTE_B_ESCAPE_HATCH_FIRST`**

The one runtime gap that matters is the **decode-attention kernel codegen/control surface** — llama owns a complete
fused LDS/`v_dot2` flash-decode primitive; tinygrad has the pieces but cannot emit it as one tile (proven: the concrete
fused-flash gate). Recommendation: **fund the primitive, implement it via the AMDGCN/HSACO escape hatch FIRST** (the
de-risked, oracle-proven, bounded path to a real W==D decode win, wrapped behind the existing lifecycle gates), and
treat **native tinygrad codegen as the north-star follow-on** that the escape-hatch win de-risks and justifies. If the
escape-hatch first gate fails, the evidence-backed fallback is `REST_DECODE` + tinygrad-v2/search consolidation.

---

## §0 Why this, why now (lifecycle-layer attribution)
Every other lifecycle layer is competitive or solved (prefill kernels opt-in llama-class; decode GEMV at/near parity +
q8 opt-in; HCQ/JIT runtime W==D-stable; the search/evaluator lifecycle is built — `decode_eval` + loop + templates +
guards). **The lone runtime-primitive gap is decode attention**, and it is specifically a **codegen/control-surface**
gap, not a knowledge gap: we know exactly what llama does (KV-split workgroups → GQA query-head packing → LDS-staged
K/V → `v_dot2` vector dot → register online-softmax+PV in one tile → efficient combine → graph-integrated route).
tinygrad's tiled-GEMM codegen and the `.set/.after` fusion idiom are **mutually exclusive** (the concrete gate's
finding), so the single fused LDS-tiled `v_dot2` tile is **inexpressible in current tinygrad codegen**. The bounded and
concrete-shape tinygrad lanes are exhausted; only a **deep codegen capability or a low-level escape hatch** closes it.

---

## §1 The primitive spec — first-class lifecycle row
Name (dataflow-complete, per the naming principle): **`decode_attention_llama_flash_tile`**
(*not* "flash" — the name carries the locality + instruction contract). Extends the existing binding
`bench/qk-decode-eval/binding_templates.json:north_star_flash_attn_tile_v0` (comparator `gqa_coop_vec`,
`decode_attention` role) from `PRUNE_NEEDS_TEMPLATE` to an **executable** candidate.

### Full primitive boundary (the row must specify ALL of these)
| facet | spec |
|---|---|
| math | exact GQA softmax(QKᵀ/√d)·V, T=1 decode, Hd=128, Hq=32, Hkv=8 (G=4), causal/start_pos |
| **work decomposition** | many **KV-split** parallel blocks (Flash-Decoding): `S` splits × Hkv head-groups → fill the GPU at T=1; combine across splits |
| **GQA packing** | the 4 query heads of a kv-group share one staged K/V (column-packed register accumulators), not serialized |
| **memory locality (LDS)** | K/V tiles staged into LDS once per block, reused across query-heads + output lanes (the locality primitive the scalar `flash_partial` lacks) |
| **instruction form** | `v_dot2_f32_f16` packed-fp16 dot for q·k and PV (not scalar `v_fmac`); `ds_load_b128` staged loads |
| **reduction strategy** | register-resident online softmax `(m, l, acc[D])` updated per key; per-split partials |
| **combine/fixup** | efficient LSE merge across the `S` splits (stream-k-style), negligible traffic (~1µs) |
| **activation/format** | fp16 K/V in, fp32 accumulate; no extra materialization of scores/prob to HBM |
| **integration boundary** | env-gated, default-off, shape-guarded decode route in the model; falls back to `gqa_coop_vec` on unsupported shapes |

### Gates (the lifecycle ladder — same as every candidate)
1. **correctness:** `rel_rmse ≤ 1e-3` vs numpy (greedy byte-exact or dNLL ≤ 0.01 in-model).
2. **local A/B (first gate):** ≥ **1.5×** vs `gqa_coop_vec` standalone @ctx1024 (clock-pinned GPU-time; the oracle says
   5–6× is the ceiling, so 1.5× is a conservative "it's working" bar; trending toward the oracle).
3. **W==D (promotion):** ≥ **5%@ctx1024 / ≥7%@ctx4096** whole-decode, no ctx512 regression, median-of-5 PROFILE-off.
4. **quality:** greedy byte-exact or dNLL ≤ 0.01.
5. **policy:** default-off, shape-guarded, one authority point, fallback tested.

### Machine-search knobs (the row's parameter space — the deep template layer)
The current templates are shallow (`FLASH_L` + `combine` only). The primitive's search space is the
**hardware-relevant** knob set, each a row dimension:
`kv_split_count (S)` · `K_tile_size` · `gqa_pack {serial|column-packed}` · `lds_layout {none|K|KV, pad}` ·
`vector_form {scalar|v_dot2|v_dot2+pk}` · `combine {per-split|stream-k}` · `occupancy {wg, VGPR/LDS budget}` ·
`split_threshold (ctx)`. These map onto whichever route is chosen (§2) and flow through `candidate_template_gen` →
`lifecycle_search_loop` → `decode_eval` unchanged.

### Artifact / refutation contract
Reuse the harness contract (`qk_harness_contract.stamp`): comparator `gqa_coop_vec` (the `DECODE_COMPARATOR` SSOT),
the T=1-parallelism fields (workgroups, S, LDS bytes, `v_dot2` count, VGPR/occupancy), local-A/B + W==D authority,
`Verdict` SSOT. Any failed gate → a refutation row (the search map).

---

## §2 Implementation-route decision
Two routes. The evidence already rules out "more bounded tinygrad graph tricks" (the concrete gate + every closed lane).

### Route A — native tinygrad codegen (own it as a pure tinygrad primitive)
- **Capability needed:** make the renderer/linearizer emit, from a fused UOp graph, a **single kernel** that LDS-stages
  K/V (`ds_load_b128`), runs `v_dot2_f32_f16` chains, and keeps register online-softmax `(m,l,acc)` — i.e. unify the
  two currently-exclusive paths (tiled-GEMM LDS/vectorization ⊕ `.set/.after` fusion).
- **First gate:** a renderer probe emits ONE concrete-shape LDS+`v_dot2` fused decode tile (ISA shows `ds_read` +
  `v_dot2`, 0 scalar V loads) that beats `gqa_coop_vec` standalone ≥1.5×@ctx1024. **This is exactly what the concrete
  fused-flash gate attempted and FAILED** — so Route A's real first gate is *building the codegen capability itself*.
- **Cost / risk:** **multi-week** `tinygrad/codegen/*` + `tinygrad/renderer/*` work; **highest risk** (the capability
  may need an LDS-tiling-of-hand-reductions feature tinygrad does not have); blast radius into core codegen.
- **Upside:** a genuinely **owned tinygrad-native** decode primitive — the purest north-star win; reusable beyond decode.

### Route B — AMDGCN/HSACO escape hatch wrapped into the lifecycle (DeepSeek-style)
- **Capability needed:** author a single LDS-tiled `v_dot2` flash-decode **kernel at the GCN/HSACO layer** (hand-asm,
  or port llama's `fattn-tile.cuh` — **BOUNDED**: no WMMA/`cp_async`, gfx1100 HSACO, ~700–900 lines, oracle Phase 0),
  compile to a `.co`, and launch it via tinygrad's **proven** raw-HCQ/Buffer bridge, wrapped as a `decode_eval`
  candidate behind the same gates + an env-gated default-off model route.
- **First gate (cheap, days):** value-correct (`rel_rmse ≤1e-3`) + local A/B ≥1.5×@ctx1024 vs `gqa_coop_vec` (the
  oracle already measured the *kernel* at 5–6×; this gate just wraps it into the launch path), then W==D ≥5%@1024.
- **Cost / risk:** **bounded** (the kernel design is oracle-proven; the bridge is proven). Risk = the GCN/HSACO author
  + launch integration (kernarg/occupancy/cache); **no `tinygrad/` core change**.
- **Promotability nuance (decide explicitly):**
  - a **vendored llama `.co`** = **non-promotable** (it is the reference oracle, not a tinygrad primitive) — useful
    only to *de-risk* (prove a llama-class decode runs + wins W==D under tinygrad's runtime).
  - a **hand-authored AMDGCN** tile = a **legitimate owned escape hatch**: the perf-principles' *DeepSeek-style* section
    explicitly sanctions "custom backend-specific assembly when the library path cannot express the needed schedule,"
    kept behind a measured lifecycle gate + fallback. **This is the promotable Route-B form.**

### Recommendation (evidence-backed): **Route B first, in two steps**
1. **De-risk (cheap):** wrap the **vendored** llama tile via the HCQ bridge as a non-promotable candidate, prove it
   lands **W==D ≥5%@1024 under tinygrad's runtime**. This converts "the kernel is 5–6× standalone" into "a llama-class
   decode actually wins whole-token under our runtime+lifecycle" — the decisive de-risk, days of work, zero core risk.
2. **Own it (promotable):** author the **hand-AMDGCN** LDS/`v_dot2` tile (the DeepSeek-style escape hatch), gate it
   through `decode_eval`/W==D, env-gated default-off, fallback to `gqa_coop_vec`. This is the shippable primitive.

**Route A (native codegen) becomes the justified north-star follow-on** once Step 1 proves the W==D win is real and
Step 2 yields the exact target kernel/ISA — you then know the codegen target *and* its W==D ceiling before committing
multi-week core work (the opposite of the concrete gate, which committed before knowing). **Why not Route A first:** the
concrete gate already proved the bounded/expressible tinygrad path can't emit the tile, so Route A *is* the multi-week
capability bet — funding it blind (before the escape hatch proves the W==D ceiling) is the exact "fund a blind month"
the strategy warns against.

---

## §3 Machine-search integration (the user's #3 — deepen the templates)
The search loop exists but its template space is shallow (`FLASH_L`/`combine`). The leap: `candidate_template_gen`
generates candidates over the §1 **knob set** (S, K-tile, GQA pack, LDS layout, vector form, combine, occupancy,
split-threshold), each binding to the chosen route's parameterized kernel, flowing through
`lifecycle_search_loop → decode_eval` with the §1 gates. Pruning rules already enforce closed-lane/default-promotion
guards. This is what turns "one hand-tile" into "a searched primitive" — the vLLM-style lifecycle around the
llama-style kernel (the north-star's combined target). **Build this AFTER Route-B Step 1 de-risks the win** (no point
searching a parameter space until one point on it clears W==D).

---

## §4 Execution plan + gate ladder
| phase | deliverable | gate | on fail |
|---|---|---|---|
| P0 | this spec as the executable `decode_attention_llama_flash_tile` binding + refutation contract | binding resolves; `decode_eval` candidate registered (`PRUNE_NEEDS_TEMPLATE`→executable) | — |
| P1 | Route-B Step 1: vendored-llama `.co` via HCQ bridge, **non-promotable de-risk** | correctness + **W==D ≥5%@1024** under tinygrad runtime | `REST_DECODE`+v2 (runtime can't host a W==D decode win) |
| P2 | Route-B Step 2: hand-AMDGCN LDS/`v_dot2` tile, env-gated default-off route | local ≥1.5× + W==D ≥5%@1024/≥7%@4096, byte-exact/dNLL, fallback tested | bank refutation; reconsider Route A vs REST |
| P3 | deepen machine-search templates over the §1 knobs | search finds ≥ the hand-tuned point; no JIT-bucket explosion | keep hand-tuned point; rest search |
| P4 (optional) | Route A: native codegen of the now-known target tile | renderer emits LDS+`v_dot2` fused tile ≥ Route-B perf | keep Route-B escape hatch; native codegen deferred |

**Fallback (project-level):** any phase's gate fails per its stop condition → `REST_DECODE` + tinygrad-v2/search
consolidation (decode capped at tinygrad's backend ceiling, with the escape-hatch evidence proving why).

---

## §5 What NOT to do (closed — do not reopen)
- **No bounded decode lanes:** `FLASH_L` sweep, fused softmax+V tail, matmul-PV (BLOCKED_BY_LAYOUT), scalar LDS+GQA
  tile, warp-cooperative tile, FFN micro-fusion — **all classified/refuted**; do not re-run.
- **No prefill kernels:** solved/opt-in; reopen only if the policy/VRAM **default** becomes the question.
- **No WMMA decode, no MMVQ reopen, no Path-A retry.**
- **No model/default route change** until P2's in-model W==D passes; default-off + shape-guarded when shipped.
- The llama oracle stays **non-promotable** (reference/target only); a *promotable* Route-B is the **hand-authored**
  AMDGCN tile, not the vendored `.co`.

## Acceptance gates (this scope)
| gate | result |
|---|---|
| G1 primitive spec names the full dataflow boundary (locality+instruction+decomposition+combine+integration) | PASS (§1) |
| G2 gates + search-knob space specified | PASS (§1) |
| G3 both implementation routes analyzed with first gate + cost + risk + evidence | PASS (§2) |
| G4 route recommendation is evidence-backed + names the de-risk step + the promotability nuance | PASS (§2) |
| G5 machine-search integration specified (deep knobs, ordering) | PASS (§3) |
| G6 execution gate-ladder + explicit fallback | PASS (§4) |
| G7 closed lanes + prefill + no-default-change boundaries restated | PASS (§5) |
| G8 no `tinygrad/`/model/kernel/default change; scope only | PASS |
| G9 policy guard passes | PASS (run pre-commit) |

## Boundary
Scope/decision only. No kernel built, no `tinygrad/` change, no model/default route, no benchmark rerun, no closed
lane reopened. Recommends funding the primitive via the **escape-hatch-first** route (de-risk vendored → own hand-AMDGCN
→ search → optional native codegen), with `REST_DECODE`+v2 as the explicit fallback. The decision to fund vs rest is the
owner's; this scope makes either choice evidence-backed and executable.
