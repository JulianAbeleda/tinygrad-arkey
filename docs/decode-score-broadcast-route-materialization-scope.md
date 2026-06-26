# Decode score-broadcast route materialization scope

Goal: make `DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE=1` model-route clean so it can reach bounded W==D.

Known good:

| Layer | Artifact | Status |
|---|---|---|
| Score reuse primitive | `bench/qk-decode-primitive-space/score_reuse_paths_latest.json` | pass |
| Standalone chain | `bench/qk-decode-primitive-space/score_broadcast_chain_latest.json` | pass |
| Direct eager route | `bench/qk-decode-primitive-space/score_broadcast_direct_latest.json` | pass |
| Direct constant-shape TinyJit route | `bench/qk-decode-primitive-space/score_broadcast_direct_latest.json` | pass |

Known bad:

| Layer | Artifact | Status |
|---|---|---|
| Full model route capture | `bench/qk-decode-primitive-space/score_broadcast_route_latest.json` | AMD MMU fault |

Hypotheses:

| Hypothesis | Test |
|---|---|
| Runtime symbolic `Tc_u` in custom kernels breaks variable-bound JIT | Minimal variable-bound TinyJit chain gate |
| Multi-consumer `state` lifetime breaks captured DAG | Reduce chain from one PV chunk to four PV chunks; reduced chunks are diagnostic-only |
| Combine kernel shape breaks only under captured DAG | Test direct variable-bound chain without full model |
| Full model capture adds materialization/aliasing not present in direct route | Compare direct variable-bound failure with model route failure |

Execution order:

1. Build `extra/qk_decode_physical_tile_score_broadcast_varjit_chain_gate.py`.
2. Test chunks `1,2,4` under variable-bound TinyJit.
3. If variable-bound chain fails, reduce whether failure is verifier, numeric, or runtime MMU.
4. If variable-bound chain passes, return to full model route gate and inspect model-only materialization.
5. Do not run W==D until `score_broadcast_route_latest.json` returns `SCORE_BROADCAST_ROUTE_CLEAN__WD_NEXT`.

Success condition:

`bench/qk-decode-primitive-space/score_broadcast_varjit_chain_latest.json` passes for four chunks and then the full model route gate passes with all four chunks.

Kill condition:

If even one chunk fails under variable-bound TinyJit after static `Smax`, classify this as a custom-kernel variable-bound callify/JIT bug and stop route work until that lower-level issue is fixed.

Current result:

| Artifact | Verdict | Meaning |
|---|---|---|
| `bench/qk-decode-primitive-space/score_broadcast_varjit_chain_latest.json` | `SCORE_BROADCAST_VARJIT_CHAIN_READY__ROUTE_NEXT` | The minimal variable-bound TinyJit chain now passes for 1, 2, and 4 PV chunks. |
| `bench/qk-decode-primitive-space/score_broadcast_direct_latest.json` | `SCORE_BROADCAST_DIRECT_READY__MODEL_CAPTURE_NEXT` | Direct variable-bound `flash_decode_attention_whole_cache` now passes. |
| `bench/qk-decode-primitive-space/score_broadcast_route_latest.json` | `SCORE_BROADCAST_ROUTE_FAIL` | Full model route capture still MMU-faults, even with `DECODE_ATTN_SCORE_BROADCAST_CHUNKS=1`. |

Guardrails:

| Guardrail | Meaning |
|---|---|
| `extra/qk_decode_eval.py` route-gate runner | W==D is blocked unless the route-gate script exits cleanly and records a passing artifact. |
| `DECODE_ATTN_SCORE_BROADCAST_CHUNKS=1/2/3` | Diagnostic-only liveness/materialization mode; requires `DECODE_ATTN_SCORE_BROADCAST_DIAGNOSTIC_CHUNKS=1` and is not full-width correct. |
| `DECODE_ATTN_SCORE_BROADCAST_CHUNKS=4` | Only chunk count eligible for route-clean-to-W==D promotion. |

Fixed failure signature:

`UOp verification failed ... Ops.BIND ... DEFINE_VAR ('start_pos', 0, 255)`

Fix applied:

Two compiler/scheduler changes make variable-bound custom-kernel chains work outside the full model:

| File | Change |
|---|---|
| `tinygrad/codegen/__init__.py` | Unbind runtime variables before program render so `Ops.BIND` does not reach the renderer. |
| `tinygrad/schedule/__init__.py` | Collect bound values from the full realized graph, not just top-level tensor inputs. |

Current interpretation:

The lower-level variable-bound custom-kernel problem is fixed for the minimal chain and direct route. The remaining blocker is full-model runtime memory safety: the reduced one-chunk model route still MMU-faults. That points to model-capture buffer lifetime/materialization around replacing the owned attention route with this chained custom-kernel route, not the physical score-broadcast primitive.

Decision:

Do not run W==D. Next work is a model-only materialization/lifetime audit around the generated attention replacement path.
