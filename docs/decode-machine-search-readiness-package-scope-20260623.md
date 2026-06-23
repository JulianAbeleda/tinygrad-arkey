# Decode Machine-Search Readiness Package — Scope / Prompt (2026-06-23)

## Mission
Make decode **machine-search-READY** as a **safe, constrained search framework** — not "go search random decode
kernels." Decode is already at/above llama.cpp parity (buffer-identity whole-cache tile, 102–105 %). Therefore:

```text
Decode is NOT currently worth searching for 8B speed.
It IS worth making machine-search-ready for:
  - regression-safe variant exploration,
  - cross-shape / cross-model generalization,
  - native-codegen microprimitive search,
  - future GPU / model portability.
```

This task **builds the readiness package** (gates, checkers, schemas, a candidate runner) and **freezes the current
default as the oracle**. It does **NOT run a search** and does **NOT flip defaults**. A search is a separate,
explicitly-authorized follow-on that consumes this package.

## Why now / what already exists
Most readiness components exist from the decode campaign; this task **factors them into reusable, machine-readable
gates** and fills the gaps.

| requirement | status | what this task adds |
|---|---|---|
| correctness authority | mostly yes (`qk_decode_runtime_overhead.py` W==D; byte-identical token check) | one reusable **token + W==D gate** wrapper returning structured PASS/FAIL |
| route identity | yes (post-parity guard inspects captured-graph kernel names) | a standalone **machine-readable route-fire checker** |
| ISA audit | yes (`extra/qk_isa_primitive_audit.py`) | **require a JSON per candidate** + encode the ISA reject rules |
| candidate knobs | partial | **enumerate the safe bounded knobs** (schema + ranges) |
| artifact schema | partial (`candidates.json`) | **standard candidate + result JSON** schema |
| reject rules | known | **encode** them as a single reject function |
| baseline oracle | yes (buffer-identity default) | **freeze** the current default W==D + ISA + route + materialization as the immutable oracle |
| rollback / fallback | yes (`DECODE_ATTN_KV_IDENTITY=0`) | **verify** fallback in the harness on every run |
| search harness | not yet | **build the candidate runner** (generate → gate → ISA-reject → W==D → rank → remember) |

## Required reading
1. `docs/decode-campaign-final-synthesis-20260623.md`
2. `docs/owned-tile-buffer-identity-kv-read-result-20260623.md`
3. `docs/machine-code-translation-roadmap-result-20260623.md` (search-readiness matrix, ABI rule)
4. `bench/qk-decode-eval/HARNESS_GUIDE.md` (Measurement-Authority table — the SOP) + `extra/qk_harness_contract.py`
5. `structure/Development/performance-primitive-research-principles.md` (esp. "Harnesses Are Performance Primitives
   Too" and "Machine Search Is Generate, Evaluate, Prune, And Remember"; principle #12 buffer-identity ABI rule)
6. `bench/qk-post-parity-hardening/regression_guard.json` (the route-fire + E_49152 + ISA guard, already passing)
7. `bench/qk-machine-code-translation/search_readiness.json`

Inspect: `extra/qk_decode_runtime_overhead.py`, `extra/qk_isa_primitive_audit.py`, `extra/qk_decode_eval.py`,
`tinygrad/llm/model.py` (owned-route guard + flags), `extra/qk_owned_flash_decode_graph_node.py`,
`extra/qk_owned_flash_decode.hip`, `bench/qk-decode-eval/candidates.json`.

## Boundaries
- **Do NOT run a machine search** in this task — build/validate the framework only.
- Do NOT flip defaults; do NOT change decode behavior (the oracle must stay byte-identical).
- Do NOT touch prefill.
- Do NOT do 14B/32B (cross-shape is a *future consumer* of this package, not run here).
- Only **bounded** knobs are searchable (below). No free-form kernel generation.
- W==D is the only promotion authority; local kernel timing never promotes.

## Searchable knobs (the ONLY allowed search space)
Encode these as the candidate knob schema. Every knob has a closed, enumerated range.

| area | knobs | bounds |
|---|---|---|
| owned tile policy | split `S`, ctx threshold (`DECODE_ATTN_AMDGCN_MIN_CTX`), combine variant (`base`/`hd<CWD>`/`sr<CWD>x<CSR>`) | S ∈ {32,48,64,96}; min_ctx ∈ {256,512,1024}; combine ∈ registry |
| tile constants | `TK`, workgroup size, vector width, unroll | TK ∈ {8,16,32}; wg ∈ {64,128,256}; enumerated |
| whole-cache ABI | offset strategy, one-buffer vs K/V-split, layer-offset handling | {whole-cache (current), split-K/V}; must keep **buffer identity** (principle #12) |
| resource envelope | VGPR cap, LDS size, no-spill requirement | VGPR ≤ 64 target / hard-fail > envelope; LDS ≤ kernel design; **0 spill required** |
| route policy | ctx guards, shape guards, fallback thresholds | within validated shape (B=1, Hq32, Hkv8, Hd128, G4) only |
| Q4K warp | lane mapping, unroll, K-block | **ONLY** for cross-shape/generalization, NOT 8B speed |

## Hard reject rules (encode as one function, applied BEFORE W==D)
Reject a candidate if **any** holds — cheap checks first, W==D last:
1. token correctness fails (not byte-identical greedy to the oracle on the validated prompts);
2. route does not fire (the candidate kernel is absent from the captured graph);
3. `E_49152` (full-MAXC materialization) returns;
4. an input is a **sliced view** instead of buffer identity (ABI rule violated);
5. ISA loses `v_dot2`;
6. ISA loses LDS or cross-lane;
7. spills appear (`scratch/spill > 0`);
8. VGPR/LDS exceeds the envelope materially;
9. ctx512 regresses;
10. improvement appears only in local kernel timing but **not** in W==D.

## Phases

### P0 — Authority + freeze the baseline oracle
Record HEAD/git/GPU/arch/model/default flags. **Freeze the current default** as the immutable oracle:
- W==D tok/s @ctx 512/1024/2048/4096 (3 interleaved reps + spread), clean synced;
- byte-identical greedy token sequence (the correctness reference);
- ISA JSON (`AMD_ISA_PRIMITIVE_CONFIRMED`, VGPR/spill/v_dot2/LDS/cross-lane) for `owned_flash_tile_gqa_whole`;
- route-fire signature (kernel name present) and **E_49152 absent**;
- fallback proof (`DECODE_ATTN_KV_IDENTITY=0` → slice route + E_49152, byte-identical tokens).
Artifact: `bench/qk-decode-search-readiness/baseline_oracle.json`. Verdicts: `ORACLE_FROZEN` / `ORACLE_UNSTABLE_STOP`
(stop unless the oracle reproduces within its spread band).

### P1 — Correctness + W==D gate wrapper
Build `extra/qk_decode_search_gate.py`: given a candidate (env flags / knob dict), returns structured
`{token_byte_identical, wd_tok_s, wd_spread, ctx512_regression, delta_vs_oracle_pct}` using the **synced** authorities
(`qk_decode_runtime_overhead.py` for W==D; greedy token compare for correctness). Reuses the harness SOP (clean W==D,
auto clock or pinned, repeated + spread). Verdict: `GATE_WRAPPER_READY`.

### P2 — Route-fire checker (machine-readable)
Factor the post-parity guard's captured-graph inspection into `extra/qk_decode_route_fire_check.py`: returns
`{candidate_kernel_present, slice_route_absent, program_node_names}` from the TinyJit captured graph. Verdict:
`ROUTE_FIRE_CHECKER_READY`.

### P3 — Materialization checker
`extra/qk_decode_materialization_check.py`: returns `{E_49152_present, full_maxc_copy_kernels, buffer_identity_inputs}`
— the ABI-rule enforcer (sliced-view detection). Verdict: `MATERIALIZATION_CHECKER_READY`.

### P4 — ISA reject schema (require JSON per candidate)
Wrap `qk_isa_primitive_audit.py` so every candidate **must** emit an ISA JSON, and encode the ISA reject rules (5–8
above) as `isa_reject(json) -> reason|None`. Verdict: `ISA_REJECT_SCHEMA_READY`.

### P5 — Candidate + result JSON schema
Define the standard schemas (extend `candidates.json` conventions + `qk_harness_contract`'s 13-field contract):
- **candidate**: `{id, knobs:{...}, env:{...}, shape_guard, comparator:"oracle", provenance}`;
- **result**: `{id, reject_reason|null, token_byte_identical, isa:{...}, route_fire:bool, e49152:bool, wd:{ctx→tok_s,
  spread}, delta_vs_oracle_pct, verdict}`.
Artifacts: `bench/qk-decode-search-readiness/{candidate_schema,result_schema}.json`. Verdict: `SCHEMAS_READY`.

### P6 — Reject-rule encoding
Single `reject(candidate_result) -> reason|None` applying all 10 hard rejects in cost order (correctness → route →
materialization → ISA → envelope → ctx512 → W==D-transfer). Verdict: `REJECT_RULES_ENCODED`.

### P7 — Candidate runner (the search harness skeleton — built, not run)
`extra/qk_decode_search_runner.py`: **generate** (enumerate a knob grid) → **evaluate** (route-fire → materialization
→ ISA-reject → token-correct → W==D, short-circuiting on first reject) → **prune** (drop rejected) → **rank** (by W==D
delta vs oracle, with spread) → **remember** (append to a results JSONL + a leaderboard). Implements "Generate,
Evaluate, Prune, And Remember." **Smoke-test it on a 2–3 candidate grid that includes the oracle itself** (must
reproduce the oracle as the top non-regressing entry) and one deliberately-bad candidate (must be rejected) — but do
**NOT** run a real search. Verdict: `SEARCH_RUNNER_READY` / `SEARCH_RUNNER_SMOKE_FAIL_STOP`.

### P8 — Readiness verdict + intended-use statement
Write `docs/decode-machine-search-readiness-package-result-20260623.md`: the package contents, the frozen oracle, the
smoke-test result, and the explicit statement that decode is **not worth searching for 8B speed** but is
**search-ready** for the four future uses. Update README/handoff/`candidates.json` notes. Verdict:
`DECODE_MACHINE_SEARCH_READINESS_PACKAGE_READY`.

## Required artifacts
```text
bench/qk-decode-search-readiness/
  authority.json · baseline_oracle.json · candidate_schema.json · result_schema.json
  reject_rules.json · search_runner_smoke.json
extra/
  qk_decode_search_gate.py · qk_decode_route_fire_check.py · qk_decode_materialization_check.py
  qk_decode_search_runner.py
docs/decode-machine-search-readiness-package-result-20260623.md
```

## Stop rules
Stop and classify if: the oracle cannot be frozen reproducibly (within spread); a checker cannot distinguish the
oracle from a known-bad candidate; the smoke test fails to reproduce the oracle as top / reject the bad one; building
the runner requires running an actual search to validate; or any decode default would have to change.

## Final verdict labels
`DECODE_MACHINE_SEARCH_READINESS_PACKAGE_READY` · `ORACLE_FROZEN` · `SEARCH_RUNNER_READY` ·
`DECODE_SEARCH_NOT_WORTH_8B_SPEED_BUT_READY_FOR_GENERALIZATION` · `READINESS_BLOCKED_<reason>`.

## Claude prompt
You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`. Decode is at/above llama
parity. Read and execute `docs/decode-machine-search-readiness-package-scope-20260623.md` (+ the required reading).
Build the decode machine-search **readiness package** — a safe, constrained framework — and **freeze the current
default as the oracle**. Do **NOT** run a machine search, do **NOT** flip defaults, do **NOT** change decode behavior,
do **NOT** touch prefill or 14B/32B. Only the enumerated bounded knobs are searchable; encode the 10 hard reject
rules; require an ISA JSON per candidate; W==D is the only promotion authority. Phases P0–P8; smoke-test the runner on
the oracle + one bad candidate (do not run a real search). Final response must include: final verdict; oracle frozen
(W==D + ISA + route + materialization + fallback); each checker/gate built; the runner smoke-test result; the
intended-use statement (not for 8B speed; ready for regression-safe / cross-shape / native-codegen / portability);
files changed; git status.
