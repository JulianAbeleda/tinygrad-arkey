# Post-Exhaustion Remaining Lanes Roadmap — Result (2026-06-23)

## 0. Verdict / recommended next action
**Recommended next action: run the ONE bounded small-ops/activation fusion gate** (Lane 1) — the only
immediately-actionable bounded lane. Everything else is closed, owner-gated, infrastructure, or deferred. This doc
is a sequencing synthesis (no implementation). Lane verdict labels:
`ATTENTION_CLOSED_MAINTENANCE_ONLY` · `GEMV_CLOSED_MAINTENANCE_ONLY` · `SMALL_OPS_NEEDS_FUSION_GATE_BEFORE_SEARCH`
(active fallback) · `RUNTIME_KV_DEFERRED_OWNER_DECISION` · `ISA_AUDIT_GUARD_ACTIVE` + `ISA_WRAPPER_AMD_ONLY_READY` ·
`MACHINE_SEARCH_NOT_READY` · `GENERALIZATION_DEFER_8B_FIRST` · `NATIVE_CODEGEN_DEFER_NO_WD_NEED`.

## 1. Current checkpoint
`POST_DEFAULT_AUDIT_COMPLETE` → `RUNTIME_KV_CORE_RUNTIME_BLOCKED_SMALL_OPS_NEXT` → `ISA_AUDIT_GENERAL_PRINCIPLE_CONFIRMED`
/ `AMD_ISA_AUDIT_READY` → `MACHINE_SEARCH_NOT_READY`. Owned AMDGCN attention default-on (validated 8B shape); Q4K
GEMV warp default-eligible; tinygrad ~88–89% of llama on Qwen3-8B-Q4_K_M decode; attention + FFN-GEMV at/near
parity; the residual delta is explained (KV materialization core-blocked; small-ops overlapped; ISA-audit ready;
search not ready).

## 2. Lane table
| # | lane | status | next action | verdict |
|---|---|---|---|---|
| 1 | Small-ops / activation fusion | **active bounded fallback** (transfer unknown — overlapped) | run ONE W==D-gated fusion | `SMALL_OPS_NEEDS_FUSION_GATE_BEFORE_SEARCH` |
| 2 | Runtime-KV core persistence | **biggest prize, core-runtime-blocked** | scope ONLY on explicit owner authorization | `RUNTIME_KV_DEFERRED_OWNER_DECISION` |
| 3 | ISA audit infrastructure | **ready (AMD)** | mandatory guard; optional vendor-neutral wrapper | `ISA_AUDIT_GUARD_ACTIVE` |
| 4 | Machine search | **not ready** | defer until a lane exposes a bounded knob | `MACHINE_SEARCH_NOT_READY` |
| 5 | Generalization / default hardening | strategic alternative | after 8B closed, if owner picks productization | `GENERALIZATION_DEFER_8B_FIRST` |
| 6 | Native tinygrad codegen learning | long-term | learn from owned tile/GEMV; not a near-term W==D lane | `NATIVE_CODEGEN_DEFER_NO_WD_NEED` |
| 7 | Attention / FFN GEMV | **closed** | regression guard only | `ATTENTION_CLOSED_MAINTENANCE_ONLY` / `GEMV_CLOSED_MAINTENANCE_ONLY` |

## 3. Immediate next action — Lane 1 (small-ops fusion gate)
The only bounded, immediately-actionable lane. Mission: `one confirmed kernel group → one fusion/removal → token
correctness → ISA/graph evidence → ≥1–2% W==D`. First candidate (per the corrected buckets): **silu(gate)·up into
the FFN-GEMV epilogue** (most llama-like). **Hard gate**: prove it is NOT mislabeled KV/cache work (rendered
source/AST), the old kernel group is removed (ISA-verified, no new spills), and it clears **≥1–2% W==D at
ctx1024/4096 with no ctx512 regression**. Artifacts under `bench/qk-small-ops-fusion-gate/`. **Stop rule**: if the
first fusion does not reach ≥1–2% W==D → `SMALL_OPS_OVERLAPPED_OR_LOW_RETURN`, close the lane, do NOT machine-search
small-ops. Scope: `docs/small-ops-activation-fusion-scope-20260623.md`.
**This roadmap recommends it as the next task but does not run it here** (synthesis-only task).

## 4. Runtime-KV core-work decision — `RUNTIME_KV_DEFERRED_OWNER_DECISION`
Biggest remaining lever (MAXC-shrink +11.8%@1536 / +12.9%@1280 → ~llama parity) but **not a kernel/model
optimization** — it needs a tinygrad core capability: *persistent mutable decode state without full-MAXC `.after()`
materialization*. Proven NOT ISA-blocked (`RUNTIME_KV_NOT_ISA_BLOCKED`; the append kernel is byte-correct
standalone) and NOT the owned tile / arg-patching (both fixed/proven) → it is `RUNTIME_GRAPH_LIFECYCLE_GAP`. **Do
not implement or scope core-runtime work without explicit owner authorization.** If authorized, the scope target is
`docs/runtime-kv-core-persistence-capability-scope-*.md` (design families: runtime-managed KV object · state-token
dependency primitive · bounded KV alias rule · two-graph decode split). Until then: **deferred, parked, evidence
preserved.**

## 5. Small-ops fusion gate — (see §3)
Active fallback; lower confidence than runtime-KV because the small-ops are heavily overlapped (GPU-busy 13.7ms ≫
wall 11.7ms), so wall transfer is unproven. The gate exists precisely to settle that with one experiment before any
expansion.

## 6. ISA audit guard policy — `ISA_AUDIT_GUARD_ACTIVE`
`extra/qk_amdgpu_isa_primitive_audit.py` is ready and is now the **mandatory guard** for every future candidate
lane: a candidate must carry code-object evidence (symbols, arch, VGPR/SGPR/LDS/scratch, instruction flags
`has_vector_dot`/`has_lds`/`has_cross_lane`/`has_vector_global_load`/`has_spill`) + a graph-lifecycle link + a W==D
artifact link, or an explicit tooling-limited note. It is a **disambiguation + false-win guard** (it proved
runtime-KV is lifecycle-blocked, not codegen). Optional infra follow-on: a vendor-neutral `extra/qk_isa_primitive_audit.py`
wrapper (AMD backend only now; NVIDIA `cuobjdump`/`nvdisasm` and Intel IGC/ocloc backends are scoped-only — tooling
absent on this host). Normalized contract realized: `bench/qk-isa-primitive-audit/owned_decode_attention.json`.

## 7. Machine-search readiness — `MACHINE_SEARCH_NOT_READY`
No lane exposes a bounded searchable knob: attention/GEMV closed (variants risk non-transfer), runtime-KV is
core-lifecycle-blocked (not a kernel-knob surface), small-ops has no proven transferable fusion yet. Search becomes
allowed only if: (a) the small-ops fusion gate passes (`MACHINE_SEARCH_READY_SMALL_OPS`), (b) a runtime-KV core
capability lands with tunable knobs (`MACHINE_SEARCH_READY_RUNTIME_KV`), or (c) a future audit finds a residual
kernel with a local correctness harness + verified ISA gap + W==D plausibility (`MACHINE_SEARCH_READY_RESIDUAL_KERNEL`).
The search loop and reject rules are pre-specified in the scope (token-correctness + route-fire + ISA + W==D gates).

## 8. Generalization decision — `GENERALIZATION_DEFER_8B_FIRST`
Strategic alternative (validate owned attention / Q4K warp on 14B/32B / other shapes; default hardening; packaging)
— **deferred** until the 8B bounded lanes are closed (i.e., after the small-ops gate resolves) and only if the owner
chooses productization over more 8B speed. **No 14B/32B until explicitly requested**; every new shape needs token
correctness + W==D.

## 9. Closed lanes — maintenance only
- **Attention** (`ATTENTION_CLOSED_MAINTENANCE_ONLY`): default-on, near llama parity, ISA-confirmed (v_dot2 / LDS /
  cross-lane / 56 VGPR / 0 spill). Allowed: regression tests, fallback correctness, docs, cross-shape (if
  generalization authorized). **Disallowed**: more tile variants, combine-only work, attention machine search.
- **FFN GEMV** (`GEMV_CLOSED_MAINTENANCE_ONLY`): llama parity. Allowed: regression tests, default decision/hardening,
  cross-shape. **Disallowed**: more schedule variants without a fresh residual gap.

## 10. Final decision matrix
| If… | Then… |
|---|---|
| small-ops fusion gate ≥1–2% W==D | scope a second fusion or a small fusion-boundary search |
| small-ops fusion gate fails | close small-ops (`SMALL_OPS_OVERLAPPED_OR_LOW_RETURN`); no machine search |
| owner authorizes core runtime | scope the runtime-KV core-persistence capability |
| owner does not authorize | keep runtime-KV deferred |
| no bounded 8B lane remains | move to generalization / default hardening |
| future audit finds a new residual kernel | require ISA audit + W==D gate before any search |

## 11. Final recommendation
1. **Now**: run the single small-ops fusion gate (silu→GEMV-epilogue), ISA- and W==D-gated. It is the last bounded
   8B experiment; it will either yield a small win or formally close the bounded 8B space.
2. **Owner decision**: authorize runtime-KV **core-runtime** work (biggest prize, ~+11% → parity) — separately
   scoped, not a kernel task.
3. **Standing**: ISA audit is the mandatory guard for all future candidates; machine search stays parked; closed
   lanes are regression-guard only; generalization is the strategic fallback once 8B is closed.

## 12. Files changed
Docs only: this roadmap result + `docs/README.md` + `structure/Development/session-handoff.md`. No source/default
changes, no new tooling, no machine search, no runtime-KV implementation, no 14B/32B.

## 13. Git status
Clean before this task. This task adds one result doc + two doc updates; no `tinygrad/` source or default changes.
