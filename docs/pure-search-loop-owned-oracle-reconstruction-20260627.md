# Pure-search loop — owned-oracle reconstruction upgrade (2026-06-27)

Reframes the loop from "X→Y→Z open-ended exploration" into **owned-kernel reconstruction**: the owned AMDGCN
decode-attention tile is the **oracle**, and every iteration is **delta closure** against it. "Search got deep but
the measured owned-vs-generated delta didn't move" is now a **tooling bug** (`SEARCH_SPACE_BUG`), not a reason to
try more.

## Why
Pure search = generated/codegen + BubbleBeam default path, not hand kernels
(`docs/pure-machine-search-roadmap.md:5-24`). Owned already solves the target shape (LDS K/V staging, `v_dot2`,
cross-lane reduction; `docs/pure-machine-search.md:44-49`). The first full X→Y→Z run exhausted Level-1 knobs and they
**did not move the slope delta** — because the knobs target *other* deltas (coalescing, tt-carry, cross-lane parity),
not the open `outer_b_serial_carry`. Open-ended lever search hid that; framing every candidate as targeting a named
delta exposes it.

## Blocker taxonomy (machine-readable: `bench/qk-search-spaces/owned_delta_taxonomy.json`)

| kind | meaning | action | classified by |
|---|---|---|---|
| `MISSING_PRIMITIVE` | owned uses a primitive the generated path can't express | add IR/lowering/axis; prove codegen expresses it | `isa_diff`, `isa_vectorization` |
| `PRIMITIVE_PLACEMENT_BUG` | primitive exists but wrong place/layout/vec, fails verifier, or not route-bound | fix lowering/scheduler/verifier/route; verify owned-equivalent position | `isa_diff`, `isa_vectorization`, `microgate` |
| `TIMING_TRIGGER` | semantics expressed but loses time | attribute to the exact trigger (occupancy/VGPR/LDS/scratch/waitcnt/coalescing/dep-chain/instr-sel/launch+combine); change it and **prove it moved** | `hotloop_schedule_diff`, `occupancy_guardrail`, `split_kv_economics` |
| `SEARCH_SPACE_BUG` | candidates loop but the named delta does NOT move | STOP searching; fix generator/search-space/metric | the loop (delta unchanged) |
| `INSTRUMENTATION_GAP` | auditor can't explain why owned still wins after visible diffs close | improve ISA/lifecycle/timing attribution until it names the hidden delta | meta |

## Encoded
- `.claude/loop.md` + `.claude/commands/pure-search-loop.md`: the `reconstruct_owned_kernel` state machine
  (compare → classify → handle-by-kind), W==D-only `PROMOTABLE`, generator-only authority, append-only JSONL ledger.
- `extra/qk_pure_search_next_candidate.py`: candidates now carry `targets_delta` + `blocker_kind` (verified: the
  topology candidate emits `targets_delta=split_kv_combine_lifecycle, blocker_kind=TIMING_TRIGGER, requires_wd=true`).
- `bench/qk-search-spaces/decode_attention_loop_search_space.json` (v3): every axis declares its target delta.

## Rules (now enforced by framing)
Owned is the oracle · every candidate targets a named delta · don't search knobs unless the auditor predicts which
delta they move · no-movement ⇒ `SEARCH_SPACE_BUG` · `PROMOTABLE` ⇒ W==D + token-match (isolated is diagnostic) ·
exhaustion ⇒ every delta closed / proven unrepresentable / named `INSTRUMENTATION_GAP` · generator-only authority ·
append-only ledger.

## Retroactive read of the first run
The 14 refuted knobs were really a `SEARCH_SPACE_BUG` for the slope: they targeted `lds_kv_staging` /
`cross_lane_reduce` / the tt-carry, **not** the open `outer_b_serial_carry`. The only candidates that target the
*open* deltas are `DECODE_OUTER_B_SPLIT` (refuted, occupancy) and the W==D-gated topology axis
(`split_kv_combine`, untried). That is the honest frontier.

## Closed-parity reframe + the owned-ASM oracle parity layer

The loop is now a **closed parity problem**: the primitive set is known from the owned ASM
(`extra/qk_owned_flash_decode.hip`, `DECODE_ATTN_AMDGCN_TILE=1`); the job is parity closure, not discovery.

**Tool:** `extra/qk_owned_oracle_parity_audit.py` → `bench/qk-owned-oracle-parity/latest.json`. It consumes the
existing instruments and emits a 7-layer owned-vs-generated matrix; each row has `owned_property`,
`owned_observation`, `generated_observation`, `status` (MATCH|MISMATCH|MISSING|UNKNOWN), `blocker_kind`,
`responsible_tool`, `required_action`, `candidate_axis`, `gate_to_close`.

**Current matrix (grounded in `isa_diff` owned_tile + hotloop/occupancy/transfer):** 6 MATCH, 5 MISMATCH, 2 UNKNOWN.

| layer.row | owned | generated | status |
|---|---|---|---|
| resource.vgpr | 64 | 88 | **MISMATCH** (generated uses 24 more — work-removal target, NO knob axis) |
| schedule.waitcnt | 21 | 50 | **MISMATCH** |
| schedule.shadow_fill | 0.2 | 3.75 | **MISMATCH** |
| lifecycle.split_kv_combine | — | COMBINE_TAX_DOMINATES | **MISMATCH** |
| wd_token.wd_tok_s | 103/94 | 33/6 (6.6%) | **MISMATCH** (the bottom line) |
| placement.load_vectorization / reduce_placement | — | not block-tile-captured / loop-size-confounded | **UNKNOWN** |
| primitive ×3, topology, lds, scratch | — | — | MATCH |

Verdict: `PARITY_OPEN__UNKNOWN_ROWS_PRESENT__IMPROVE_INSTRUMENTATION_BEFORE_SEARCH`.

**The loop is now parity-driven** (`.claude/loop.md`, `/pure-search-loop`): run the parity audit → if `unknown_rows`,
improve that instrument first (don't search) → else
`qk_pure_search_next_candidate.py --failed-rows <searchable_failed_rows>` so **no candidate runs unless it targets a
failed row** → gate → **verify the target row's counter moved toward owned** (else `SEARCH_SPACE_BUG`) → W==D for
promotion. The `vgpr` and placement rows have no searchable knob axis → they are the honest next capability work
(work-removal + instrumentation), not more search.

**Not executed** (per request: Codex reviews the e2e system first). Nothing here ran the GPU loop; the matrix is
from already-captured artifacts.
