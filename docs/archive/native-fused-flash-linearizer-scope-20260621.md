# Native Fused-Flash Linearizer Capability — Scope

Date: 2026-06-21

Scope-only (no `tinygrad/` change, no kernel build). Authority: `docs/llama-flash-attn-tile-oracle-result-20260621.md`
(llama `flash_attn_tile` is ~5–6× faster than `gqa_coop_vec` STANDALONE → native codegen targets the right layer).

## Decision: **`NATIVE_FLASH_LINEARIZER_SCOPE_READY`** — but the premise is CORRECTED

**Headline (decisive, overturns the framing):** the coupled online-softmax+V fused decode kernel — running max `m`,
running sum-exp `l`, running weighted-V `acc[D]`, coupled via `corr = exp(m_old − m_new)` — **already VERIFIES,
LOWERS, and RUNS value-correct in ONE kernel TODAY** on AMD, via the existing `UOp.set`/`.after`/register-array
idiom. **No `spec.py` REDUCE change, no new REDUCE op, no linearizer change is required.** (Empirically reproduced:
a single-kernel online-softmax @ V produced `[-0.4906,-0.2988,0.3412,-0.2674]` == numpy ref, `match True`.)

So the "compiler expressiveness wall" is **refuted**: `NEEDS_UOP_REDUCE_DESIGN_FIRST` and
`NEEDS_LINEARIZER_RANGE_MODEL_FIRST` are **not** the situation. The "6-kernel split / coupled reduces trip the
linearizer" note (`extra/qk_flash_decode.py:72-80`) and the `flash_fused_multireduce_linearizer_wall` refutation
described **idiom pitfalls**, not a hard wall. The scope is therefore a **bounded kernel-build** (Path A below),
**not a multi-week compiler project** — and its real risk is *performance*, not expressibility.

## Phase 1 — capability gap map (reframed: expressiveness EXISTS)

| llama requirement | tinygrad support TODAY | gap type | first file | risk |
|---|---|---|---|---|
| coupled `(m,l,acc)` online softmax in one range | **EXPRESSIBLE** via `placeholder(REG)` + `.set(val, end=j)` (proven) | **none (idiom)** | `extra/qk_flash_decode.py` builders | low |
| different range nests (q·k, softmax, V) in one kernel | expressible (chain scalar stores as `.after` deps of the finest-granularity store, single `END(j)`) | none (idiom) | `qk_flash_decode.py:189-190` pattern | low |
| GQA/query-column packing | coop already packs G=4 register accs | none | `qk_flash_decode.py:195-219` | low |
| many KV-split parallel blocks | coop has S-splits; grows with ctx | none (dataflow) | `qk_flash_decode.py:271` | low |
| register-resident online softmax | proven expressible | none | — | low |
| no WMMA | n/a (vector path) | none | — | none |
| **FAST in-kernel q·k** (vector FMA, scheduled to llama quality) | coop uses a SEPARATE matmul (fast); an in-kernel hand-rolled q·k (warp tile) is ~LATENCY-bound, FLAT ~163µs | **CODEGEN QUALITY (the real gap)** | `tinygrad/codegen/*` scheduling/regalloc | **HIGH / deep** |
| overall kernel scheduling/regalloc to match llama (5–6×) | tinygrad UOp codegen ≈ 5–6× slower than llama's hand-tuned kernel (oracle) | **CODEGEN QUALITY** | `tinygrad/codegen/late/*`, renderer | **HIGH / deep** |

**The three idiom pitfalls (sharp edges, NOT walls) — and the working pattern:**
1. **single-op REDUCE** (`tinygrad/uop/spec.py:163-165`, `op ∈ {ADD,MUL,MAX}`): online softmax can't be a high-level
   `Ops.REDUCE` → use the manual `.set`/register-array path (which is why the custom-kernel route exists; this is
   why `Tensor.scaled_dot_product_attention` does not auto-fuse).
2. **GROUP-shape-index trap** (`tinygrad/uop/ops.py:372`): `c.after(END(GROUP,...))[d]` raises "None input shape not
   supported for Ops.GROUP". **Fix:** make the `END` root a single STORE (chain scalar stores as `.after` deps of
   the d-range acc store) — exactly the gqa_coop pattern.
3. **two ENDs over one range** (`tinygrad/codegen/late/linearizer.py:81` assert): two `.set(...,end=j)` siblings
   over the same `j` create a CFGContext ordering contradiction. **Fix:** ONE `END(j)`, all accumulators grouped.
4. **same-slot intra-iteration RAW**: reading slot *k* (for a coupling term used by another accumulator) while also
   writing slot *k* in the same iteration folds the read to the post-store value (`corr` silently → 1). **Fix:**
   keep a **mirror slot** for the carried value read coupling terms from.

These four are the entire boundary. (1) is by design; (2)(3)(4) are the optional-hardening targets if the idiom is
to become *clean*, not enablers.

## Phase 2/3 — smallest increment + first executable gate

**The smallest increment is NOT a compiler change — it is a kernel built with the existing idiom (Path A).**

**Path A (zero-compiler, BOUNDED — the v0 first gate):** build `flash_fused_decode` in `extra/qk_flash_decode.py` =
**coop's optimized matmul q·k** (kept — it is near-optimal among tinygrad primitives) **+ ONE fused online-softmax+V
kernel** (the proven idiom: single `END(j)`, scalar `m/l/m_mirror` stores chained as `.after` deps of the d-range
`acc` store), replacing coop's `flash_max/prob/partial/gmax/den/combine` (5–6 kernels). First gate:
- **value-correct** vs numpy/coop (rel_rmse ≤ 1e-3) — **already met in the probe**;
- **local A/B + in-model W==D** vs `gqa_coop_vec` (the comparator), clock-pinned, GPU-time and W==D, no model route
  (env-gated default-off only if a W==D route is reached, and only after local A/B passes).

**Path B (compiler hardening, ~1–2 weeks, NOT v0):** fix the same-slot RAW fold + the line-81 single-END constraint
+ the GROUP-shape-index trap so the fused kernel can be written *naturally* (no mirror slot, multiple ENDs). Touches
`tinygrad/codegen/late/linearizer.py` and `tinygrad/uop/ops.py:372`. Only worth it if the idiom becomes a general
capability — **defer until Path A shows a fused kernel is even worth having.**

## Honest EV / risk (the load-bearing caveat)

Path A fuses **only the softmax+V** (it keeps coop's matmul q·k). coop already runs as a **batched JIT graph**, so
the win from fusing 5–6 graph kernels into one is **materialization + launch savings** — likely **marginal**
(coop's combine traffic is ~1µs; intermediates are ~1–2 MB), **NOT the 5–6× gap**. The **5–6× gap is the in-kernel
q·k + overall codegen QUALITY** (the warp tile's flat ~163µs floor with an in-kernel cooperative dot), which Path A
does **not** address and which the refuted-bounded-tile work showed tinygrad's UOp codegen does not match. So:
- If Path A **wins** (even marginally) in-model → a real, banked fusion gain, and the idiom is validated for deeper
  work.
- If Path A is **marginal/loses** → it confirms the gap is **codegen quality** (in-kernel q·k scheduling/regalloc to
  match llama), a **separate, deep** project (or rest) — and the bounded experiment cost ~a day to learn that.

This is why Path A is the right *first* gate: it is the cheapest way to learn whether kernel fusion (now proven
expressible) helps at all, before committing to deep codegen-quality work. **Do NOT skip to compiler hardening
(Path B) or a deep codegen-quality project until Path A's in-model A/B is measured.**

## Phase 5 — llama port vs profiling oracle for v0: **profiling oracle is enough**

The fused kernel is value-correct vs numpy (no byte-level oracle needed for correctness); the existing **profiling
oracle** (`docs/llama-flash-attn-tile-oracle-result-20260621.md`, llama 12.2µs vs coop 69.9µs @ctx1024) gives the
performance target. So **`NEEDS_LLAMA_SOURCE_PORT_ORACLE_FIRST` = NO.** The full llama source port (BOUNDED, audited)
is only needed if/when a fused kernel approaches llama and a byte-level comparison is wanted — a later step.

## Stop conditions

- If Path A's fused kernel **regresses or is within noise** of coop in-model → bank it; the lever is codegen quality
  (deep) → scope that separately or REST. Do **not** iterate bounded tile variants (closed lanes).
- If building the fused kernel requires touching `tinygrad/` to even compile → STOP (it shouldn't; the probe proved
  it compiles) and re-examine the idiom.
- If a W==D route is added, it is **env-gated default-off**, falls back to coop on unsupported shapes, and is added
  **only after** the local A/B passes ≥1.05× @ctx1024.
- No WMMA decode, no MMVQ, no bounded vector-tile, no closed-lane reopen, no performance claim until the emitted
  kernel passes local A/B, llama oracle stays non-default/non-promotable.

## Artifacts / decode_eval-lifecycle integration

- A new binding `flash_fused_decode_v0` (role `decode_attention`, comparator `gqa_coop_vec`, `local_ab_runner` =
  `extra/qk_flash_fused_decode_ab.py`, then a W==D runner if local passes) — clone of `north_star_flash_attn_tile_v0`.
- decode_eval candidate `flash_fused_decode` (family `attention_split` or a new `fused_flash`), `ab_script` runner →
  `FAIL_LOCAL_AB` / `LOCAL_PASS_WD_FAIL` / `PASS_PROMOTE` via the existing `classify` path. Generated candidate via
  the template generator. Runs through lifecycle-search exactly like every other candidate. Artifact carries the
  binding fields + the `(m,l,acc)` register layout + fused-vs-6-kernel intermediate-byte accounting.

## Files to touch (the NEXT project, Path A)

- `extra/qk_flash_decode.py` — add `flash_fused_decode` kernel builder (the proven single-`END(j)` mirror-slot
  idiom) as a new `FLASH_VARIANT`, default-off.
- `extra/qk_flash_fused_decode_ab.py` (new) — local A/B vs `gqa_coop_vec` (value-correct + GPU-time + W==D).
- `bench/qk-decode-eval/{candidates.json, binding_templates.json}`, the generator/loop metadata.
- **NO `tinygrad/` change for Path A.** (Path B, if ever, touches `tinygrad/codegen/late/linearizer.py:81` +
  `tinygrad/uop/ops.py:372` — deferred.)

## Acceptance / decision summary

Decision: **`NATIVE_FLASH_LINEARIZER_SCOPE_READY`** — corrected: the linearizer/expressiveness capability ALREADY
EXISTS (empirically proven); the scope is a **bounded kernel-build (Path A)** with the first gate = value-correct
(met) + local A/B + in-model W==D vs `gqa_coop_vec`. The deep 5–6× gap is **in-kernel-q·k codegen QUALITY**, flagged
and deferred (do not start it before Path A's A/B). `NEEDS_LLAMA_SOURCE_PORT_ORACLE_FIRST` = no (numpy + profiling
oracle suffice). `NEEDS_UOP_REDUCE_DESIGN_FIRST` / `NEEDS_LINEARIZER_RANGE_MODEL_FIRST` = refuted (expressible today).

## Boundary
Scope only. No `tinygrad/` change, no model route/default, no kernel built, no closed lane reopened, no performance
claim. The llama oracle stays a non-default, non-promotable reference.
