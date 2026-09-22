# NV dense Flash ceiling exhaustion result

## Claim

The current wide-Flash construction has been exhausted through its existing
load, horizon, combine, ownership, and launch-overlap knobs.  The installed
d512 authority is **4060.523 us/token (246.274 tok/s)**.  The retained llama
authority is **4021.721 us/token (248.711 tok/s)**, leaving **38.802 us/token
(2.437 tok/s)** end to end.

This is not a hardware roofline claim.  It is a construction ceiling: the
remaining parity-sized pool is cold score service, and no tested spelling of
the current one-head/CTA direct-streaming emitter converts it.

## Booked changes

| construction | semantic gate | installed wall result | status |
|---|---|---:|---|
| score launch bound plus cap-33 graph seam | exact token stream | -15.954 us/token in its installed bracket | booked |
| automatic S6 through Tc=768, S8 afterward | exact; both graph pairs captured; transition crossed | -40.224 us/token, +2.427 tok/s in reverse crossing bracket | booked |
| register-broadcast combine weights on installed S8 | bit-exact | -9.396 us/token, +0.565 tok/s | booked for S8 |

The active-horizon policy has a real startup cost: it pre-captures S6 and S8
greedy ping-pong pairs once per model.  That compilation is moved out of the
steady token window; it is not erased from first-request latency.

The canonical d512 S6 window measures 4060.523 us/token / 246.274 tok/s.  The
shorter transition authority measures 4050.937 us/token / 246.856 tok/s.  The
canonical number remains the endpoint because it uses the retained 24 x 9
d512 protocol.

## Closed tests

| theory | causal result | translation result | conclusion |
|---|---|---|---|
| vector fp32 Q loads | request decomposition is correct | +69.752 us/token, -4.123 tok/s in matched A/C/A | closed |
| forced V preload cadence | changes the intended schedule | slower cold at equal bytes/sectors | closed |
| shared-to-register combine at S6 | isolated and explicit-geometry arms pass | installed distinct-S6/S8 capture regresses 8.993 us/token | conversion-closed |
| normalized grouped ownership | QG2/S12 exact; QG4/S24 differs by one fp16 word | QG2 slightly slower; QG4 only about 0.34% faster and non-exact | topology-only closed |
| actual cross-head K/V sharing | QG2 is bit-exact and halves the global K/V-loading warps | score service regresses 3.520→4.224 us/layer | construction closed at primitive |
| reuse-class FFN-down admission | bit-exact; post-FFN/full-entry Flash penalty falls from +0.464/+0.608 to -0.016/-0.016 us/layer | wall regresses 69.877 us/token, -4.151 tok/s | mechanism proven, construction closed at wall |
| ordinary PDL / launch-ahead | legal schedule changed | about 1.5 us primitive movement, token wall failed | closed |
| score-to-combine overlap | timestamp accounting | zero device overlap on installed graph | no idle pool to hide |

The S6 combine false lead is important.  Its explicit geometry bracket reached
4039.546 us/token, but that harness made both prewarmed variants S6 through a
model-wide base geometry.  Production correctly owns distinct S6 and S8 graph
pairs.  Under that installed topology the candidate regressed, so 247.553
tok/s is not an endpoint and is not booked.

## Current Flash body ledger

The cap-33 graph tracker emits the repeating 33/66/132/185 batch signature.
Across 35 steady replays, node sum and device union are both about 3925 us and
device overlap is effectively zero.  Named Flash rows are:

| 36 calls/token | tinygrad installed S6 | llama PDL-off | remaining debt |
|---|---:|---:|---:|
| score | 194.048 us | 162.948 us | 31.100 us |
| combine | 48.448 us | 37.057 us | 11.391 us |
| score + combine | 242.496 us | 200.005 us | 42.491 us |

The 42.491-us gross Flash debt is not additive to the 38.802-us endpoint gap;
tinygrad retains small lifecycle wins elsewhere.  Matching the complete llama
Flash body would imply about 4018.032 us/token / 248.878 tok/s.  Matching score
alone would imply about 4029.423 us/token / 248.174 tok/s.  These are accounting
ceilings, not achieved claims.

## What remains open

The remaining score debt is equal-byte cold service.  K/V requests are already
coalesced and the launch-bound schedule has captured the useful compiler
cadence.  The two previously open branches have now been tested.  Explicit
shared-memory K/V reuse loses to its barriers and shared traffic.  Reuse-class
admission proves that producer cache displacement is turnable, but a second
pointer/ABI plus streaming-load construction overcharges the production graph
by more than it recovers.

The surviving form of the cache branch is narrower: attach a per-access cache
semantic to the existing weight pointer, or use a native persisting-window
mechanism, while preserving installed kernel ABI, producer service, and graph
identity.  An algorithmic branch must avoid materializing a full K/V tile
through shared memory.  Neither is a booked recovery pool.

## Evidence

- `docs/task_workflow/evidence/nv-flash-ceiling-exhaustion-20260827/wide-q-matched-aca-r9.json`
- `docs/task_workflow/evidence/nv-flash-ceiling-exhaustion-20260827/combine-register-installed-aca-r9.json`
- `docs/task_workflow/evidence/nv-flash-ceiling-exhaustion-20260827/horizon-selector-automatic-reverse-r9.json`
- `docs/task_workflow/evidence/nv-flash-ceiling-exhaustion-20260827/horizon-selector-canonical-d512-r9.json`
- `docs/task_workflow/evidence/nv-flash-ceiling-exhaustion-20260827/combine-register-s6-wall-r9.json`
- `docs/task_workflow/evidence/nv-flash-ceiling-exhaustion-20260827/combine-register-s6-installed-r9.json`
- `docs/task_workflow/evidence/nv-flash-ceiling-exhaustion-20260827/wide-qgroup-microgate-r9.json`
- `docs/task_workflow/evidence/nv-flash-ceiling-exhaustion-20260827/installed-s6.profile.jsonl`
- `docs/task_workflow/evidence/nv-flash-two-causal-branches-20260827/cross-head-kv-share-r2.json`
- `docs/task_workflow/evidence/nv-flash-two-causal-branches-20260827/reuse-class-entry-control.json`
- `docs/task_workflow/evidence/nv-flash-two-causal-branches-20260827/reuse-class-entry-candidate.json`
- `docs/task_workflow/evidence/nv-flash-two-causal-branches-20260827/q6-reuse-class-wall-r1.json`
