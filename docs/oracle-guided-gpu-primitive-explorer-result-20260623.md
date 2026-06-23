# Oracle-Guided GPU Primitive Explorer — Result (2026-06-23)

## 1. Verdict

**`ORACLE_GUIDED_GPU_PRIMITIVE_EXPLORER_SCOPED`** — the "Explore the GPU" system is consolidated into a registry +
shared spec + gate stack + ledger, mapped onto the existing (already-built) search backends. Per-component verdicts:

| component | verdict |
|---|---|
| inventory | `EXPLORER_INVENTORY_READY` |
| oracle registry | `ORACLE_REGISTRY_READY` (5 oracles) |
| search-spec schema | `SEARCH_SPEC_SCHEMA_READY` (reuses `qk_search_spec` + ledger schema) |
| unified runner | `EXPLORER_RUNNER_DESIGN_READY` (design only, not implemented) |
| decode backend | `DECODE_SEARCH_BACKEND_NEEDS_ADAPTER` (functional today; adapter to generic spec pending) |
| native-codegen backend | `NATIVE_CODEGEN_SEARCH_BACKEND_INTEGRATED` |
| prefill gate | `PREFILL_SEARCH_GATED_OFF_AT_REST` |
| cross-shape gate | `CROSS_SHAPE_SEARCH_NEEDS_TARGETS` |

This was a **scoping/consolidation** task: no defaults flipped, no model behaviour or kernels changed, no search runs.

## 2. Relation to existing autotuning systems

AutoTVM/Ansor, Triton autotune, OpenXLA persisted autotuning, Kernel Tuner, and vendor profilers all tune **kernel
speed** over a parameter grid. The differentiator here is the **lifecycle oracle**: a candidate is accepted only after
passing *graph route identity + ABI/materialization + ISA/resource + token correctness + whole-path (W==D /
whole-prefill) transfer* — not just a faster isolated kernel. This project does **not** claim to have invented
autotuning; it claims **oracle-guided lifecycle search for LLM inference primitives**. The repeated lesson that
motivates it (memory + this fork's history): isolated kernel wins routinely fail to transfer in-model, and flag-stack /
route confounds silently fake results — so acceptance must be lifecycle-complete and whole-path.

## 3. Oracle registry (`bench/qk-oracle-gpu-primitive-explorer/oracles.json`)

| oracle id | lane | status |
|---|---|---|
| `decode_whole_cache_owned_tile_8b_gfx1100` | decode | `ORACLE_FROZEN` / shipped default-on; search verdict `ORACLE_REMAINS_BEST` |
| `q4k_gemv_warp_8b_gfx1100` | decode/codegen | shipped weight-GEMV lever; 8B-speed lane closed; cross-shape-only knob |
| `prefill_graph_gemm_8b_gfx1100` | prefill | `PREFILL_FRONTIER_AUDIT_COMPLETE`; ~99.5 % of Tensile; knobs exhausted |
| `owned_attention_isa_template` | codegen | ISA reference/acceptance envelope (VGPR 60, no spill, dot+lds+cross-lane+vec) |
| `tensile_prefill_reference` | prefill/codegen | `AVAILABLE_VIA_FLAG` (`PREFILL_TENSILE_GEMM=1`), in-repo `.co` unverified |

Each entry is lifecycle-complete (source, code object, authority benchmark + numbers, correctness, route signature,
ISA facts, materialization/ABI, supported shapes, fallback, status). Two data corrections were folded in:
- **prefill authority** = post-kv_proj-fix **3554/3468/3221/2796 tok/s (~99.5 % Tensile)**; the doc-headline
  `1983 tok/s (~66 % llama)` is stale (older/nosync) and marked retracted.
- **decode authority** records both the gate's frozen no-warp W==D (90.6/89.3) and the canonical **warp-on**
  full-model (102.9/101.3/98.7/94.2), with the `Q4K_GEMV_WARP*` flag stack named — the same flag-stack distinction
  surfaced by the ctx-slope audit earlier today.

## 4. Search-spec schema (`search_spec_schema.json` + 4 examples)

Two-layer SSOT, **reused not reinvented**: spec layer = `extra/qk_search_spec.py`
(`SearchRow`/`Constraints`/`AcceptedPolicy`, with `Phase × Model × OpScope × SearchSpace × Objective` enums); memory
layer = the 15-field `bench/qk-project-search-ledger` entry. Examples written: decode-policy (live), native-codegen
microprimitive (live), prefill (placeholder, gated), cross-shape (placeholder, gated).

## 5. Backend integration status

- **Decode** (`decode_backend_integration.json`): functional + ledger-wired for the decode lane (harness-contract
  CONFORMS 13/13). Runner/executors are hard-wired to the decode attention tile and inline knob grids; driving them
  from the generic spec needs a `SearchRow → {env knobs, expected kernel symbol, oracle, reject envelope}` adapter +
  a per-lane gate registry. → `NEEDS_ADAPTER`.
- **Native-codegen** (`native_codegen_backend_integration.json`): executed and runnable; 5 ledger entries
  (`lane=codegen`). Authority is ISA + local correctness, never W==D. → `INTEGRATED`.

## 6. Decode search status

`DECODE_SEARCH_EXECUTED_ORACLE_REMAINS_BEST` (Mode A policy: 6 candidates, S48 control +0.1 %, S96 −1.1 %,
minctx1024 rejected `route_not_firing`). Mode B (generated tile variants, `QK_CAND_KERNEL`) is built and available but
not run to a winner. Gate stack: route-fire → E_49152 → buffer-identity → byte-identical tokens → ISA reject → W==D →
ctx512 regression. **The decode default is the policy optimum; nothing to promote.**

## 7. Prefill search gate

`PREFILL_SEARCH_GATED_OFF_AT_REST` — 4 of 6 search-justification criteria fail (kernel at Tensile parity; isolated
parity does not transfer in-model; tuning knobs exhausted; expected search gain ≈ 0). The real lever is the **in-model
integration penalty** (in-model gate/up ~22 TFLOPS vs isolated 63–78), with an ISA-exact micro-residual of **+23 %
VALU address arithmetic** vs Tensile — a deterministic addressing fix, not a search knob. **Caveat:** a fresh synced
per-role *in-model* breakdown was deferred; per-role attribution is the first non-search step.

## 8. Native-codegen microprimitive search status

`NATIVE_CODEGEN_MICROSEARCH_EXECUTED_TARGET_FOUND` — **2/4 targets native** (LDS staging + vector global loads
emittable); **`v_dot2` (fused fp16 dot) and cross-lane reduce (`ds_bpermute`) are confirmed renderer gaps** (fp16 MAC
lowers to LDS reduce; warp-axis sum uses an LDS tree, never `ds_bpermute`/`ds_swizzle`/`v_permlane`). This is the
machine-code-translation frontier: those two gaps are why the hand-written `.hip` owned tile still beats codegen.

## 9. Cross-shape gate

`CROSS_SHAPE_SEARCH_NEEDS_TARGETS` — single gfx1100 box; NVIDIA (`cuobjdump`/`nvdisasm`/SASS) and Intel
(`ocloc`/`iga`/Xe ISA) tooling absent; no alternate model present; no 14B/32B until the owner asks. Ready only after a
vendor backend or alternate model shape is provided.

## 10. What can be searched now

- **Decode policy (Mode A)** and **decode generated-tile (Mode B)** — bounded, gated, ledger-wired. (Mode A already
  concluded oracle-remains-best; Mode B is runnable but unrun.)
- **Native-codegen microprimitive expressibility** — runnable, non-promotion (ISA + local correctness).

## 11. What remains blocked

- **Prefill search** — gated off; needs per-role in-model attribution + a non-search integration-penalty fix first.
- **Cross-shape search** — needs an alternate GPU vendor backend or model target.
- **`v_dot2` / cross-lane codegen** — renderer gaps; closing them is a tinygrad-codegen capability task, not a search.
- **Generic spec-driven runner** — designed, not implemented; decode backend needs the SearchRow adapter.

## 12. Next implementation step

If the owner wants to *act*: implement the `SearchRow → decode-candidate` adapter + per-lane gate registry so the one
generic runner drives the decode backend (small, mechanical), **or** run the per-role in-model prefill attribution that
the prefill gate flags as the first non-search step. Otherwise this remains a consolidation artifact — the searchable
8B lanes (decode, codegen) have already been run and rest at oracle-best.

## 13. Files changed

New under `bench/qk-oracle-gpu-primitive-explorer/`: `inventory.json`, `oracles.json`, `search_spec_schema.json`,
`spec_decode_policy_example.json`, `spec_native_codegen_micro_example.json`, `spec_prefill_placeholder.json`,
`spec_cross_shape_placeholder.json`, `decode_backend_integration.json`, `native_codegen_backend_integration.json`,
`prefill_search_gate.json`, `cross_shape_search_gate.json`. New tools: `extra/qk_oracle_explorer_build.py` (JSON
synthesis). New docs: this result + `docs/oracle-guided-gpu-primitive-explorer-runner-design-20260623.md` + the scope.
**No `tinygrad/` or kernel changes; no defaults flipped; no search runs.**

## 14. Git status

Branch `qk-prefill-flag-leak-resolution`. Pre-existing uncommitted WIP **not made by this task** (recorded in
`inventory.json`, left untouched): `tinygrad/llm/model.py`, `extra/qk_owned_flash_decode.hip`,
`extra/qk_owned_flash_decode_graph_node.py`, `extra/qk_decode_search_gate.py`, `extra/qk_b4_combine_tax.py` (Mode-B
generated-tile WIP) + untracked `extra/qk_decode_mode_b_execute.py`, `extra/qk_prefill_search_execute.py`. This task's
deliverables are committed separately and do not include those WIP edits.

## 15. Learning layer: primitive-space proposer (addendum 2026-06-23)

A learned model/adapter sits **in front of** the explorer as a *primitive-space proposer*, not a kernel judge
(`PRIMITIVE_SPACE_PROPOSER_NOT_KERNEL_JUDGE`). It emits a bounded search spec (a `SearchRow`: lane, primitive,
hypothesis, knobs+bounds, required evidence, stop rules) — never source code, never a promotion decision. That spec
enters the runner **before candidate generation** (§3 of the runner design); everything downstream is unchanged and the
deterministic lifecycle gates (harness contract → route/materialization → ISA/resource → correctness →
W==D/whole-prefill) remain the only authority (`DETERMINISTIC_HARNESS_REMAINS_AUTHORITY`).

**LoRA/SFT first** (`LORA_FIRST_FOR_PRIMITIVE_SPACE_LEARNING`) because the task is structured, supervised
primitive-space generation with a programmatically scorable target. **RLVR is deferred**
(`RLVR_DEFERRED_UNTIL_SCHEMA_AND_REWARD_STABLE`) until the strict-JSON schema is format-stable, a deterministic reward
(with negative penalties for closed-lane reopens and missing evidence) is defined, rejection-sampling SFT has
plateaued, and the adapter beats deterministic baselines in shadow mode. Full plan + dataset/scorer/eval prerequisites:
`docs/primitive-space-learning-loop-lora-first-result-20260623.md`.
