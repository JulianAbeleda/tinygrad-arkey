# Decode Machine-Search Readiness Package — Result (2026-06-23)

## Verdict: `DECODE_MACHINE_SEARCH_READINESS_PACKAGE_READY` + `ORACLE_FROZEN` + `SEARCH_RUNNER_READY` + `DECODE_SEARCH_NOT_WORTH_8B_SPEED_BUT_READY_FOR_GENERALIZATION`
Built the full safe, constrained decode machine-search framework, froze the current default as the immutable oracle,
and smoke-tested the runner (oracle PASS + deliberately-bad candidates rejected) — **without running a real search and
without changing any default**. Decode and prefill behavior untouched.

## Frozen oracle (P0) — `bench/qk-decode-search-readiness/baseline_oracle.json`
The current buffer-identity whole-cache default, measured by the gate:
- **verdict PASS**; greedy tokens `[315, 24231, 6009, 979, 220, 576]` (the correctness reference);
- **W==D 90.6 tok/s @ctx512 (spread 0.5 %), 89.3 @ctx1024 (0.4 %)** — matches canonical `wd_child` (89.1), tight band;
- route-fire: `owned_flash_tile_gqa_whole` present, slice route absent;
- materialization: **E_49152 absent, buffer_identity_inputs True**;
- ISA: `AMD_ISA_PRIMITIVE_CONFIRMED` (VGPR 60, 0 spill, v_dot2/LDS/cross-lane);
- fallback proven (`DECODE_ATTN_KV_IDENTITY=0` → slice route + E_49152, independently flagged).

## Components built
| phase | artifact | status |
|---|---|---|
| P1 gate (correctness + W==D, cost-ordered, short-circuit) | `extra/qk_decode_search_gate.py` | matches canonical W==D (90.6/89.3) |
| P2 route-fire checker | `extra/qk_decode_route_fire_check.py` | candidate present / slice absent |
| P3 materialization + ABI checker | `extra/qk_decode_materialization_check.py` | oracle E_49152=False; slice route E_49152=True, `buffer_identity_inputs=False` |
| P4 ISA-reject schema | `isa_reject()` in the gate (wraps `qk_isa_primitive_audit`) | requires JSON per candidate; rejects on lost v_dot2/LDS/cross-lane/spill/VGPR>96 |
| P5 candidate + result schemas | `bench/.../candidate_schema.json`, `result_schema.json` | bounded knob env map + result fields |
| P6 reject rules (10, cost-ordered) | `bench/.../reject_rules.json` (encoded in gate) | correctness→route→E_49152→sliced-view→ISA→envelope→ctx512→local-not-W==D |
| P7 candidate runner (generate→evaluate→prune→rank→remember) | `extra/qk_decode_search_runner.py` | `SEARCH_RUNNER_READY` (smoke) |

## Smoke test (P7) — `bench/qk-decode-search-readiness/search_runner_smoke.json`
The runner correctly distinguishes good from bad (proves the gates work before any real search):
| candidate | knobs | verdict | smoke |
|---|---|---|---|
| `oracle_replay` | KV_IDENTITY=1 | **PASS** (Δ −0.1 % vs frozen oracle) | OK |
| `bad_slice_route` | KV_IDENTITY=0 | **REJECTED:route_not_firing** (candidate `_whole` kernel absent; also flagged E_49152 by P3) | OK |
| `bad_no_route` | DECODE_ATTN_AMDGCN_TILE=0 | **REJECTED:route_not_firing** | OK |

Leaderboard top = `oracle_replay` (the only non-regressing PASS). The bad candidates are rejected by the **cheap**
structural gates before W==D ever runs (cost-ordered short-circuit).

## Searchable knobs (the ONLY allowed space) and hard rejects
Encoded verbatim from the scope: owned-tile policy (S, min_ctx, combine), tile constants (TK/wg/vector/unroll),
whole-cache ABI (must keep **buffer identity** — principle #12), resource envelope (VGPR/LDS/no-spill), route policy,
Q4K warp (cross-shape only). The 10 hard rejects (correctness, route-fire, E_49152, sliced-view, v_dot2, LDS/cross-
lane, spill, envelope, ctx512, local-only-not-W==D) are a single cost-ordered function; **W==D (synced) is the only
promotion authority**.

## Intended-use statement
```text
Decode is NOT currently worth searching for 8B speed (already at/above llama.cpp parity).
It IS machine-search-READY — as a safe, constrained framework — for:
  - regression-safe variant exploration (any tweak is auto-gated against the frozen oracle),
  - cross-shape / cross-model generalization,
  - native-codegen microprimitive search,
  - future GPU / model portability.
A real search is a separate, explicitly-authorized step that CONSUMES this package.
```

## Files changed
New tools: `extra/qk_decode_search_gate.py`, `qk_decode_route_fire_check.py`, `qk_decode_materialization_check.py`,
`qk_decode_search_runner.py`. New artifacts under `bench/qk-decode-search-readiness/`:
`authority/baseline_oracle/candidate_schema/result_schema/reject_rules/search_runner_smoke.json`. This doc +
README/handoff. **No `tinygrad/` source, no decode/prefill behavior change, no default flips, no real search.**

## Git status
Clean before; adds 4 tools + 6 artifacts + result doc + doc updates. Oracle frozen; decode byte-identical.
