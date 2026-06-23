# Decode Machine Search Execution — Result (2026-06-23)

## 1. Verdict: `DECODE_SEARCH_EXECUTED_ORACLE_REMAINS_BEST`
Ran a real, bounded **Mode A policy search** over the readiness package. 6 candidates, **5 passed all cost-ordered
gates, 1 correctly rejected**, and **no candidate beat the frozen oracle outside spread** — the current default
(S=48, combine=base, whole-cache buffer-identity) is already optimal within the policy grid. This is an explicitly
valid result. **No default flipped, no decode behavior changed, no prefill touched.**

## 2. Harness authority compliance
| checklist | answer |
|---|---|
| Did every performance claim use clean W==D authority? | **Yes** — only `run_wd` (synced, PROFILE=0, .item()/token, 30 repeats) ranks/promotes |
| Were PROFILE/DEBUG/no-sync timings excluded from promotion? | **Yes** — none used here |
| Was correctness checked before speed? | **Yes** — byte-identical greedy gate runs before W==D (cost-ordered) |
| Were repeats/spread recorded? | **Yes** — median + spread_pct per ctx in every artifact (`repro_band`) |
| Was the oracle comparator current? | **Yes** — frozen oracle re-checked, within 3% band (`SEARCH_ORACLE_RECHECK_PASS`) |
| Was git/dirty state stamped? | **Yes** — `qk_harness_contract.provenance()` in every artifact |
| Were artifacts stamped with `qk_harness_contract`? | **Yes** — all 8 artifacts; `contract_audit` **CONFORMS 13/13** |
| Were local diagnostics separated from W==D? | **Yes** — only W==D; route/materialization/ISA are pass-gates, not timings |
| Were rejected candidates stopped at first failed gate? | **Yes** — the gate short-circuits in cost order |

## 3. Oracle recheck (Phase 0) — `SEARCH_ORACLE_RECHECK_PASS`
Frozen oracle W==D 90.6@512 / 89.3@1024; recheck 90.6@512 / 88.9@1024 (median within 3% band; token byte-identical;
route fires `owned_flash_tile_gqa_whole`; E_49152 absent; ISA `AMD_ISA_PRIMITIVE_CONFIRMED`). No drift.

## 4. Search mode and plan — `SEARCH_PLAN_READY`
**Mode A (policy)**, small grid: `DECODE_ATTN_AMDGCN_S ∈ {32,48,64,96}` × `combine ∈ {base, hd64}` + one
route-policy probe (`MIN_CTX=1024`), all on the buffer-identity route (`DECODE_ATTN_KV_IDENTITY=1`). Comparator =
frozen oracle. Contexts 512/1024 first pass (2048/4096 only for first-pass winners — none arose). No generated code
objects (policy knobs recompile the same kernel family).

## 5. Candidate manifest summary — `CANDIDATE_MANIFEST_READY` (6 candidates)
S32_base, S48_base_control (= oracle, harness control), S64_base, S96_base, S48_hd64 (cheaper combine),
minctx1024_probe (route-policy probe).

## 6. Reject summary
| reason | candidates |
|---|---|
| `route_not_firing` | minctx1024_probe — **correct**: MIN_CTX=1024 means the owned route does not fire at the ctx512 W==D point → falls to gqa, candidate kernel absent → rejected at the route-fire gate (before W==D). |

The probe rejecting exactly as designed is the framework working: a route-policy that disables the route at a tested
ctx is caught cheaply, never reaching W==D.

## 7. Leaderboard (Phase 4) — `SEARCH_ORACLE_REMAINS_BEST`
W==D median tok/s delta vs oracle @ctx1024 (all 5 PASS within the oracle's spread band → no winner):
| rank | candidate | Δ@ctx1024 | Δ@ctx512 |
|---|---|---:|---:|
| 1 | **S48_base_control** | **+0.1 %** | +0.2 % |
| 2 | S32_base | −0.2 % | +0.3 % |
| 3 | S64_base | −0.2 % | −0.3 % |
| 4 | S48_hd64 | −0.4 % | −0.1 % |
| 5 | S96_base | −1.1 % | −0.7 % |

The S=48 control reproducing the oracle (+0.1 %) confirms harness fidelity. Every variant is within noise except S96
(more splits → measurable −1.1 % overhead). **The default is already the policy optimum.**

## 8. Winner recheck — `NO_WINNER_RECHECK_NEEDED`
No candidate beat the oracle outside spread, so Phase 5 was not entered.

## 9. Recommendation
**Keep the default unchanged** — S=48 / combine=base is optimal within the policy grid; there is no policy win to
recommend. The framework is validated end-to-end on a real run (gates distinguish good/bad, harness CONFORMS 13/13,
oracle stable). Next bounded modes, when desired: **Mode B** (generated owned-tile tile-constant variants — requires
code-object hashing + ISA per variant) or **Mode C** (native-codegen microprimitive search for v_dot2/LDS/cross-lane,
no decode-speed promotion). Both consume this same package; neither is needed for 8B speed (decode is at/above llama).

## 10. Files changed
New: `extra/qk_decode_search_execute.py` (the Mode A executor) + this doc + 8 stamped artifacts under
`bench/qk-decode-machine-search/` (authority, oracle_recheck, search_plan, candidate_manifest, results.jsonl,
reject_summary, leaderboard, decision). **No `tinygrad/` source, no default flips, no prefill, no 14B/32B.**

## 11. Git status
Clean before; adds 1 tool + 1 doc + 8 artifacts. Oracle unchanged; decode byte-identical.
