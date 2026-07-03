# Pure Machine Search Artifact Cache Scope

Date: 2026-06-30

Status: future infrastructure scope. This prevents regenerating TG/PMS static artifacts every time while keeping
correctness and speed gates honest.

## Problem

The pure-search stack now has deterministic artifacts:

```text
profile -> quant semantics -> target features -> route manifest -> topology grammar
  -> template/candidate generation -> evaluator -> ledger
```

It is wasteful and noisy to regenerate the static parts each run. But it is unsafe to blindly cache correctness and
speed results because those depend on code, runtime, driver, GPU state, and route behavior.

So the cache must distinguish:

```text
static generation artifacts:
  safe to reuse by exact input fingerprint

correctness / route artifacts:
  reusable only when route/code/model/runtime fingerprints match

speed / W==D artifacts:
  reusable only as historical evidence unless exact runtime fingerprint matches and the caller accepts cached speed
```

## Source Citations

Load these before implementing:

| source | role |
|---|---|
| `docs/pure-machine-search-remaining-hot-kernels-scope-20260630.md` | PMS route/evaluator/ledger scope |
| `docs/pure-machine-search-true-generation-agnostic-scope-20260630.md` | TG topology/profile generalization scope |
| `extra/qk_route_manifest.py` | route manifest source |
| `bench/qk-search-spaces/default_route_manifest.json` | route manifest artifact |
| `bench/qk-search-spaces/search_profiles.json` | profile-driven candidate space |
| `bench/qk-search-spaces/profiles/*.json` | profile descriptors |
| `bench/qk-search-spaces/quant_semantics.json` | quant semantics |
| `bench/qk-search-spaces/targets/*.json` | target feature descriptors |
| `bench/qk-search-spaces/topology_grammar_v1.json` | topology grammar |
| `bench/qk-search-spaces/pms_r3_candidate_generator_check.json` | manifest-driven generator check |
| `bench/qk-lanemap-template-ir/latest.json` | TG1 template IR proof |
| `bench/qk-new-profile-search/qwen3_8b_q6k_ffn_down_gfx1100/latest.json` | TG7 new-profile search result |
| `bench/qk-candidate-evaluator/*/latest.json` | evaluator artifacts |
| `bench/qk-project-search-ledger/ledger.jsonl` | durable decisions |

## Cache Classes

### Class A: Static Deterministic Artifacts

Examples:

```text
profile regeneration
quant semantics audit
target feature descriptors
route manifest dump
topology grammar enumeration
template roundtrip by spec identity
candidate set generation
ledger row synthesis before benchmark
```

Cache rule:

```text
reuse if artifact.cache_key == hash(normalized inputs + code fingerprints)
```

No GPU is required to validate freshness.

### Class B: Correctness / Route Artifacts

Examples:

```text
token/logit equivalence
route attribution
no hidden fallback
kernel name counts
```

Cache rule:

```text
reuse only if artifact.cache_key matches AND route_code/runtime/model fingerprints match.
otherwise rerun.
```

These can be cached for local iteration, but promotion should normally rerun them.

### Class C: Speed / W==D / Whole-Prefill Artifacts

Examples:

```text
decode tok/s
whole-prefill tok/s
GPU timing
PMC counters
wall-share attribution
```

Cache rule:

```text
default: historical evidence only.
promotion: rerun unless --accept-cached-speed and exact runtime fingerprint match.
```

These artifacts are useful for selecting the next phase, but stale speed must never silently promote a route.

## Fingerprint Model

Add a shared helper:

```text
extra/qk_artifact_cache.py
```

Minimum API:

```python
def file_sha256(path: str) -> str: ...
def json_sha256(obj: object) -> str: ...
def code_fingerprint(paths: list[str]) -> dict: ...
def runtime_fingerprint() -> dict: ...
def build_cache_key(kind: str, inputs: dict, code_paths: list[str], runtime: bool = False) -> str: ...
def load_if_fresh(path: str, expected_key: str) -> dict | None: ...
def write_artifact(path: str, payload: dict, cache_meta: dict) -> None: ...
```

Every cacheable artifact must include:

```json
{
  "cache": {
    "schema": "qk_artifact_cache_v1",
    "class": "A_static | B_correctness | C_speed",
    "cache_key": "...",
    "inputs_hash": "...",
    "code_hash": "...",
    "runtime_hash": "... or null",
    "profile_id": "...",
    "route_id": "... or null",
    "target_id": "...",
    "generated_at": "ISO8601",
    "validity": "fresh | stale | historical_only"
  }
}
```

Recommended runtime fingerprint fields:

```text
git_head
git_dirty_relevant_files
python_version
tinygrad relevant code hashes
gpu target id
driver/ROCm version if discoverable
device name
runtime env keys that affect route/codegen
authority harness path + hash
```

Do not include volatile values like timestamp in the cache key.

## Phase C0: Cache Inventory

Goal: classify existing PMS/TG artifacts into A/B/C and identify missing fingerprints.

Build:

```text
BoltBeam: `boltbeam artifact-cache-inventory /path/to/tinygrad --out ... --markdown ...`
```

Outputs:

```text
bench/qk-artifact-cache/inventory.json
bench/qk-artifact-cache/summary.md
```

Acceptance:

- Lists all PMS/TG artifacts under:
  - `bench/qk-search-spaces/`
  - `bench/qk-lanemap-template-*`
  - `bench/qk-topology-author/`
  - `bench/qk-quant-semantics-audit/`
  - `bench/qk-profile-opener/`
  - `bench/qk-target-features/`
  - `bench/qk-template-candidate-gate/`
  - `bench/qk-new-profile-search/`
  - `bench/qk-candidate-evaluator/`
- Assigns cache class A/B/C.
- Reports which artifacts already have enough provenance and which need wrapping.

Verdicts:

```text
C0_PASS_CACHE_INVENTORY_PINNED
C0_BLOCKED_ARTIFACTS_UNCLASSIFIED
```

## Phase C1: Static Cache Helper

Goal: implement the shared helper for Class A artifacts.

Build:

```text
extra/qk_artifact_cache.py
```

Acceptance:

- Can compute stable hashes for normalized JSON.
- Can compute code fingerprints for declared file paths.
- Can write/read cache metadata.
- Has negative tests showing stale when an input JSON or code file hash changes.

Suggested first test:

```text
cache key for:
  profile qwen3_8b_q4_k_m_gfx1100
  quant_semantics.json
  targets/amd_gfx1100.json
  topology_grammar_v1.json
  qk_topology_candidate_author.py
```

Verdicts:

```text
C1_PASS_STATIC_CACHE_HELPER
C1_BLOCKED_HASH_UNSTABLE
```

## Phase C2: Wire Static TG Artifacts

Goal: avoid regenerating TG static artifacts when inputs have not changed.

Wire cache into:

```text
extra/qk_topology_candidate_author.py
extra/qk_profile_opener.py
extra/qk_quant_semantics.py
extra/qk_target_features.py
extra/qk_lanemap_template.py
```

Do not change their verdict semantics. Add optional flags:

```text
--use-cache=1 default
--force-regenerate=1
--explain-cache=1
```

Acceptance:

- First run writes artifacts with cache metadata.
- Second run with unchanged inputs returns `CACHE_HIT` and byte-identical payload except timestamp/log fields.
- `--force-regenerate=1` recomputes and verifies same content.
- A changed grammar/profile/quant file forces cache miss.

Verdicts:

```text
C2_PASS_STATIC_TG_CACHE
C2_BLOCKED_STATIC_ARTIFACT_NOT_DETERMINISTIC
```

## Phase C3: Candidate Evaluator Cache Boundary

Goal: cache route/correctness artifacts safely, and prevent stale speed from being used for promotion.

Wire cache into:

```text
extra/qk_candidate_evaluator.py
```

Policy:

```text
--use-cache=1:
  may reuse Class A static artifacts.
  may reuse Class B correctness/route artifacts if exact fingerprints match and verdict is not for promotion.
  may read Class C speed artifacts only as historical evidence.

--accept-cached-speed=1:
  may reuse Class C only when runtime_hash is exact and caller explicitly accepts it.

promotion mode:
  rerun Class B and C unless --accept-cached-speed=1 is set.
```

Acceptance:

- Replaying known G3 pass can skip static candidate generation.
- Promotion cannot happen from stale speed by default.
- If a route/env/code fingerprint changes, evaluator refuses cached correctness.
- Summary says which parts were cache hits/misses/reruns.

Verdicts:

```text
C3_PASS_EVALUATOR_CACHE_BOUNDARY
C3_BLOCKED_STALE_SPEED_PROMOTION_RISK
```

## Phase C4: Cache-Aware CLI Workflow

Goal: make the normal workflow not regenerate everything.

Add command examples to a doc or script:

```text
# Open a profile and reuse static cache
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_profile_opener.py \
  --profile bench/qk-search-spaces/profiles/qwen3_8b_q4_k_m_gfx1100.json \
  --use-cache=1

# Generate/search topology candidates, cache-aware
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_topology_candidate_author.py \
  --profile qwen3_8b_q4_k_m_gfx1100 \
  --use-cache=1

# Evaluate candidate, static cache on, speed rerun by default
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_candidate_evaluator.py \
  --route decode_q4k_g3_generated \
  --use-cache=1
```

Output:

```text
docs/pure-machine-search-cache-workflow.md
```

Acceptance:

- The workflow explains when generation is skipped.
- It explains when correctness/speed are rerun.
- It explains how to force regeneration.

Verdicts:

```text
C4_PASS_CACHE_WORKFLOW_DOC
```

## Staleness Rules

Regenerate static candidate universe if any changes:

```text
profile descriptor
quant semantics
target features
topology grammar
template IR schema
route manifest
refuted-axis ledger/policy
candidate author code
template emitter code
```

Rerun correctness/route attribution if any changes:

```text
model route code
kernel emitter code
runtime route flags
model weights/profile
token/logit harness
target backend code
```

Rerun speed/W==D if any changes:

```text
any correctness-level input
driver/runtime
GPU target
benchmark harness
clock/power controls
route implementation
promotion threshold policy
```

## Non-Goals

- Do not cache away a promotion gate by default.
- Do not make cache hits hide missing route attribution.
- Do not make speed artifacts permanent truth.
- Do not require GPU for Class A static cache validation.
- Do not rewrite TG/PMS phase semantics; add cache metadata and skip logic only.

## Claude Execution Prompt

Use this prompt for the next agent:

```text
You are in /home/ubuntu/tinygrad-arkey on master. Read:

1. docs/pure-machine-search-artifact-cache-scope-20260630.md
2. docs/pure-machine-search-true-generation-agnostic-scope-20260630.md
3. docs/pure-machine-search-remaining-hot-kernels-scope-20260630.md
4. extra/qk_route_manifest.py
5. bench/qk-search-spaces/pms_r3_candidate_generator_check.json
6. bench/qk-new-profile-search/qwen3_8b_q6k_ffn_down_gfx1100/latest.json

Task: execute C0 and C1 only.

Rules:
- Audit first. No kernel changes, no default changes.
- C0: inventory existing PMS/TG artifacts and classify Class A/B/C.
- C1: implement extra/qk_artifact_cache.py with stable JSON/code/runtime fingerprint helpers and negative stale tests.
- Do not wire cache into evaluator or generators yet.
- Commit only if C0/C1 pass or a precise blocker artifact is written.

Expected artifacts:
- bench/qk-artifact-cache/inventory.json
- bench/qk-artifact-cache/summary.md
- extra/qk_artifact_cache.py

Acceptable verdicts:
- C0_PASS_CACHE_INVENTORY_PINNED + C1_PASS_STATIC_CACHE_HELPER
- C0_BLOCKED_ARTIFACTS_UNCLASSIFIED
- C1_BLOCKED_HASH_UNSTABLE
```

## End State

After the cache track, the normal loop should be:

```text
open profile
check cache keys
reuse static candidate universe
rerun only stale or promotion-critical gates
update ledger
```

This avoids asking Claude to regenerate the same profile/grammar/candidate set every time while preserving the integrity
of correctness and speed evidence.
