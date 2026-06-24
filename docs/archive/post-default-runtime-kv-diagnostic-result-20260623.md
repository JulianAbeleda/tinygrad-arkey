# Post-Default Runtime-KV Diagnostic + ISA Course — Result (2026-06-23)

## 1. Verdict: **RUNTIME_KV_CORE_RUNTIME_BLOCKED_SMALL_OPS_NEXT**

Runtime-KV is **impact-justified** (the full-MAXC KV materialization is on the W==D critical path — MAXC-shrink
**+11.8%@ctx1024**, reaching ~llama parity) but **core-runtime-blocked**: the copy-free opaque append still bakes
in the full model **even with the now-fixed fp16 owned tile**. The blocker is the TinyJit/`@function` persistence
lifecycle (`RUNTIME_GRAPH_LIFECYCLE_GAP`), not a bounded primitive. Pivot to small-ops/activation fusion
(overlapped → uncertain transfer); the **ISA audit tool is shipped**. No source/default changes (diagnostic
reverted).

## 2. Authority / config
HEAD `3ccea81d0`, gfx1100, Qwen3-8B-Q4_K_M. Owned attention default-on (fp16 cache), Q4K_GEMV_WARP env-forced.
Baseline confirmed: owned route fires, ctx1024 86 tok/s. (`authority.json`, `BASELINE_CONFIRMED`.)

## 3. Baseline confirmation
`BASELINE_CONFIRMED` — owned route fires (owned_flash nodes=2, fp16 cache), deterministic tokens, no gqa fallback.

## 4. MAXC-shrink A/B (`maxc_shrink_ab.json`, `MAXC_SHRINK_TRANSFERS`)
ctx1024, owned+Q4K, tight interleaved repeats:
| MAXC | tok/s | Δ vs 4608 |
|---|---|---|
| 4608 | 86.0 | — |
| 1536 | **96.2** | **+11.8%** |
| 1280 | **97.1** | **+12.9%** |
The cost is **MAXC-proportional** (the full-cache materialization; the owned-tile read is ctx-proportional and
unchanged). At MAXC=1280 tinygrad reaches **~97 tok/s = ~llama parity**.

## 5. E_49152 critical-path attribution (`e49152_critical_path.json`, `E49152_ON_CRITICAL_PATH`)
The materialization (`assigned_kv = cache.after(store)`, ~1.5ms/token GPU-busy) is the only MAXC-proportional
per-token cost → the MAXC-shrink delta IS it → **on the critical path**. Identical under gqa and owned (both use
the canonical `.after(store)`; FO2 removed the fp32→fp16 cast, not the materialization).

## 6. Opaque append with fixed fp16 owned tile (`opaque_append_fixed_tile.json`, `OPAQUE_APPEND_RUNTIME_GRAPH_BLOCKED`)
- **Standalone microbench: PASS** (persistence across replays, rel_rmse e-7).
- **Model-local: FAIL** — opaque append + owned tile + fp16 cache + @function bypass, real ctx1024 prefill,
  12-step decode → **151936 from step 1** (baseline materialized owned route = correct `[13,279,3974,…]`).
- **The fixed fp16 owned tile did NOT unblock it** → the append-NaN/persistence wall is **separate** from the tile
  dtype bug (and from GraphRunner arg-patching, both previously fixed/proven). The materialization is coupled to
  `@function` cross-replay persistence; removing it loses persistence.

## 7. Runtime-KV decision (`runtime_kv_diagnostic_decision.json`, `RUNTIME_KV_CORE_RUNTIME_BLOCKED`)
MAXC-shrink transfers (worth ~+11%) BUT opaque append fails at the full-model graph/persistence layer → per the
scope's decision rule, **core runtime blocker**. Detail: `docs/runtime-kv-core-runtime-blocker-result-20260623.md`.

## 8. Implementation scope
**Not written** — diagnostic did not pass (it is core-runtime-blocked, not implementable as a bounded route).

## 9. Core runtime blocker classification
`RUNTIME_GRAPH_LIFECYCLE_GAP` — removing the materialization without losing cross-replay persistence needs a
**core tinygrad capability** (runtime-owned mutable-across-replay cache the pure `@function` graph can read without
materializing). Out of bounded-primitive scope; scope separately only on owner request.

## 10. Small-ops fallback (`fallback_small_ops_decision.json`, `SMALL_OPS_FUSION_SCOPE_READY`)
Bounded fallback: fuse the unfused FFN-activation (~1.5ms) + norm/rope/small-reduces (~2.3ms) that llama fuses.
**Caveat: heavily overlapped** (GPU-busy 13.7 ≫ wall 11.7ms) → wall transfer uncertain. First gate = one fusion +
≥1–2% W==D. Scope: `docs/small-ops-activation-fusion-scope-20260623.md`.

## 11. ISA audit tool (`isa_primitive_audit_tool_result.json`, `ISA_PRIMITIVE_AUDIT_TOOL_READY`)
Shipped `extra/qk_amdgpu_isa_primitive_audit.py`: code object → clang-offload-bundler → llvm-objdump/readelf →
per-kernel VGPR/SGPR/scratch + flags (`has_v_dot2`/`has_lds`/`has_cross_lane`/`has_vector_global_load`/`has_spill`)
+ instruction counts. Auto-discovers owned `.co`; re-confirms owned tile (56 VGPR, 0 spill, v_dot2/LDS/cross-lane).
Minor partial: LDS-byte metadata parse misses some bundles (disasm flags reliable).

## 12. Holistic Exhaustion Ledger
| Lane | Current state | Remaining unknown | Exhaustion test | Final status |
|---|---|---|---|---|
| FFN Q4K GEMV | **parity** (7620 vs 7686µs) | residual schedule gap? | W==D + ISA + llama compare | **CLOSED** — warp-reduce ISA-confirmed; default-eligible |
| owned attention | **near parity** (663 vs 507µs) | route knobs material? | W==D + ISA + ctx sweep | **CLOSED** — default-on; +12.7–22.4% all ctx; ISA confirmed (v_dot2/LDS/cross-lane/0-spill) |
| KV materialization | **top residual** (~1.5ms) | critical path or overlap? | MAXC-shrink + dependency | **ON CRITICAL PATH (+11.8%) → IMPLEMENT-WORTHY but CORE-RUNTIME-BLOCKED** |
| small ops / activation | residual (~2.3+1.5ms GPU-busy) | real vs overlapped? | rendered-source + W==D | **SEARCHABLE but OVERLAPPED** — scope ready, transfer unproven |
| runtime graph | the KV tax | persistence model? | graph probes | **CORE WORK** — `RUNTIME_GRAPH_LIFECYCLE_GAP` (materialization == @function persistence) |
| ISA tooling | manual worked | reusable? | build partial tool | **READY** — `extra/qk_amdgpu_isa_primitive_audit.py` |
| llama delta | ~12–15% | exact cause? | per-lane ms reconciliation | **EXPLAINED** — KV-copy +1.3ms + small-ops +1.2ms + activation +1.0ms (GPU-busy); wall +1.73ms; attention+GEMV at parity |

**8B bounded-primitive exhaustion checkpoint**: attention + weight-GEMV are **closed** (at/near llama parity); the
llama delta is **explained**. The single biggest remaining lever (KV materialization, +11.8% → parity) is **core-
runtime-blocked** (needs a tinygrad persistence-without-materialization capability, not a bounded kernel). The only
remaining bounded lane (small-ops fusion) is **overlapped → uncertain transfer**. **Machine search is NOT yet
justified** — no clear bounded searchable knob remains; the next real wins require core-runtime work or speculative
codegen fusion. This is a clean "bounded 8B largely exhausted" state.

## 13. Artifacts and commands
- `bench/qk-post-default-runtime-kv-course/{authority,maxc_shrink_ab,e49152_critical_path,opaque_append_fixed_tile,runtime_kv_diagnostic_decision,isa_tooling_inventory,isa_primitive_audit_tool_result,fallback_small_ops_decision,next_course_decision}.json`.
- MAXC-shrink: `... DECODE_ATTN_AMDGCN_TILE=1 Q4K_GEMV_WARP=1 QK_MAXC={4608,1536,1280} QK_CK=1024 .venv/bin/python /tmp/maxc_wd.py`.
- Opaque re-test: re-apply the `RUNTIME_KV_CACHE` opaque route (reverted), real prefill ctx1024, 12-step decode.
- ISA tool: `PYTHONPATH=. .venv/bin/python extra/qk_amdgpu_isa_primitive_audit.py`.

## 14. Files changed
New: `extra/qk_amdgpu_isa_primitive_audit.py`; this result doc + core-runtime-blocker doc + small-ops fusion scope;
9 bench artifacts. Updated: `docs/README.md`, `structure/Development/session-handoff.md`. **No source/default
changes** (the RUNTIME_KV diagnostic route was re-applied then reverted; default decode byte-identical).

## 15. Git status
`model.py` clean (diagnostic reverted, default decode `[279,1156,22148,…]`). New tool + docs + artifacts only. No
14B/32B; no runtime-KV implementation; no new kernels; no default flips.
