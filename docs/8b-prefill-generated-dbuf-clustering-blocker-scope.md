# 8B Prefill Generated DBUF Clustering Blocker Scope

Date: 2026-07-09.

## Current Blocker

The composed generated prefill route is now route-bound and correct, but it is not fast:

```text
pipe roles:       attn_qo, attn_kv, ffn_down
LDS/DBUF role:    ffn_gate_up
route id:         prefill_wmma_pipe_lds_dbuf_primitive_generated
whole smoke:      about 205 tok/s
stored Path1:     about 218 tok/s
hand authority:   about 4413 tok/s
```

The blocker is no longer hidden fallback, route identity, or cross-role LDS contamination. Those were fixed by the
composed route id, fail-closed binding gate, per-role route map, and role-scoped local-stage keys.

The current blocker is narrower:

```text
generated DBUF can either:
  A. preserve the D3 next-slot staging/cadence shape,
or
  B. reduce LDS reloads through phase-scoped fragment residency / WMMA clustering,

but the tested path does not yet do both at once.
```

That means the generated route still pays too much transport/wait overhead per unit of WMMA work.

## Evidence

Known-good binding and correctness:

| Layer | Status |
|---|---|
| Composed route identity | `prefill_wmma_pipe_lds_dbuf_primitive_generated` resolves correctly. |
| Whole-prefill binding | `prefill_route_binding_gate` passes and records role routes. |
| Pipe roles | `attn_qo`, `attn_kv`, `ffn_down` sampled correctness pass. |
| LDS/DBUF role | `ffn_gate_up` sampled correctness passes. |
| Role-scoped DBUF rewrite | Pipe roles are no longer contaminated by global LDS/DBUF env flags. |

Density comparison from the current lifecycle trace:

| Route | inst/WMMA | wait/WMMA | global_b128/WMMA | ds_store_b128/WMMA | ds_load_b128/WMMA | max WMMA cluster | D3 next-slot cadence |
|---|---:|---:|---:|---:|---:|---:|---|
| generated active DBUF `2x2` baseline | 39.062 | 3.312 | 2.0 | 2.0 | 4.0 | 1 | false |
| generated phase-scoped K-major probe | lower, 554 inst total | 2.875 | unchanged in probe | unchanged in probe | 2.0 | 3 | false |
| hand LDS2 oracle | 9.547 | 0.406 | 1.0 | 1.0 | 2.0 | 4 | true |

The phase-scoped K-major probe proved the right pressure point:

- `ds_load_b128` dropped from `64` to `32` over the same `16` WMMAs.
- `ds_load_b128/WMMA` moved from `4.0` to `2.0`.
- max WMMA cluster moved from `1` to `3`.
- unpinned bounded matrix TFLOPS moved to about `12.12`.

But it did not transfer to e2e:

- whole-prefill stayed around `205 tok/s`,
- the structural verdict downgraded because `D3_cadence.ok=false`,
- `body_has_next_slot_work=false`.

## Root Thesis

The hand LDS2 oracle is not fast because it is merely "hand asm." It is fast because its lifecycle simultaneously has:

```text
1. packed b128 global -> LDS transport,
2. DBUF next-slot work in the body, not only a prologue,
3. LDS fragment reuse across adjacent WMMAs,
4. fewer waits by clustering several WMMAs after each LDS load group.
```

The generated route has item 1. It partially proves item 3 under K-major phase. It does not yet preserve item 2 while
achieving item 3, so item 4 remains weak.

## Non-Goals

- Do not reopen the gfx1100 `4x4` path; that is parked on register budget.
- Do not add a full hand-authored kernel schedule.
- Do not clone `build_gemm_lds2` into a route-local full instruction list.
- Do not tune whole-prefill blindly before a bounded trace shows the combined D3 plus fragment-residency property.
- Do not build a new harness; use `kernel_lifecycle_trace.py`, `hand_vs_generated_shape_matrix.py`,
  `prefill_pipe_mvp_artifact.py`, and `prefill_whole_synced.py`.

## Definition Of Unblocked

The blocker is removed only when one generated opt-in route satisfies all of these in order:

| Gate | Requirement | Why |
|---|---|---|
| G0. Correctness | `ffn_gate_up` sampled correctness passes with finite nonzero output. | Prevents structural-only wins. |
| G1. Packed transport | scalar LDS fallback remains zero and b128 global/LDS chain remains visible. | Keeps the existing working transport. |
| G2. D3 cadence | `D3_cadence.ok=true` and `body_has_next_slot_work=true`. | Proves next-slot DBUF work is in the loop body. |
| G3. Fragment residency | `ds_load_b128/WMMA <= 2.0` on the bounded `2x2` trace. | Matches the hand oracle's LDS reload density. |
| G4. WMMA clustering | max WMMA cluster is at least `3`, target `4`. | Proves waits are amortized over compute. |
| G5. Bounded movement | bounded generated matrix timing improves over baseline DBUF. | Shows the structural change matters. |
| G6. E2E transfer | whole-prefill smoke moves above stored Path1 or records the next named bottleneck. | Decides whether this primitive is enough. |

## Phase Plan

### P0. Freeze The Known-Good Baselines

Record these as the comparison rows:

```text
baseline generated DBUF:
  D3=false, ds_load_b128/WMMA=4.0, max_cluster=1, about 7.9 TFLOPS bounded

phase-scoped K-major probe:
  D3=false, ds_load_b128/WMMA=2.0, max_cluster=3, about 12.1 TFLOPS bounded

hand LDS2 oracle:
  D3=true, ds_load_b128/WMMA=2.0, max_cluster=4
```

Done when the scope, trace output, or artifact contains all three rows.

### P1. Find The Conflict Between D3 And K-Major Phase

Question:

```text
Why does enabling phase-scoped K-major residency remove/reorder the future-slot work that D3 expects?
```

Work:

- Diff the baseline and K-major phase streams with `kernel_lifecycle_trace.py --full-rows`.
- Compare regions between WMMAs:
  - global load placement,
  - `ds_store_b128` placement,
  - barrier placement,
  - `ds_load_b128` grouping,
  - WMMA order.
- Identify whether K-major phase is:
  - moving next-slot global/LDS work back to the prologue,
  - merging slot identities too aggressively,
  - changing the proof key so future-slot work is no longer recognized,
  - or forcing a wait/barrier placement that prevents D3 cadence.

Done when the doc names the exact first point where the two traces diverge.

### P2. Smallest Combined Probe

Build the smallest opt-in probe that keeps both properties:

```text
preserve:
  D3 next-slot staging identity and body placement

add:
  phase-scoped LDS fragment residency / K-major grouping
```

Preferred implementation order:

1. make the K-major grouping key include the D3 slot/stage owner;
2. prevent the K-major transform from collapsing current-slot and future-slot LDS descriptors;
3. only if needed, add a post-D3 clustering pass that groups LDS loads and WMMAs without moving global/LDS stores.

Done when the bounded trace passes G0-G4.

### P3. Bounded Timing

Run the existing matrix harness on the combined probe:

```bash
DEV=AMD:ISA ... PYTHONPATH=. \
python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py \
  --shapes 2,2 --m 512 --n 5120 --k 5120 --loc 2 --unr 2 --skip-hand --json
```

Done when generated DBUF improves over the baseline DBUF row without correctness loss.

### P4. Route Transfer

Enable the combined probe only for the composed generated route and run:

```bash
PYTHONPATH=. python3 extra/qk/prefill_pipe_mvp_artifact.py \
  --lds-primitive --lds-sample-correctness --sample-cols 16 \
  --measure-per-role-timing --compact
```

Then run the fail-closed whole-prefill smoke with:

```text
--require-route prefill_wmma_pipe_lds_dbuf_primitive_generated
```

Done when:

- all four role sampled correctness checks pass,
- route binding passes,
- whole-prefill either moves above Path1 or records a named residual bottleneck.

## Parallel Work

| Lane | Parallel? | Output |
|---|---:|---|
| A. Trace divergence | yes | baseline vs K-major first divergent region. |
| B. Proof-key audit | yes | whether D3 slot identity is lost in K-major phase keys. |
| C. Regression gates | yes | unit/trace assertion for G2-G4. |
| D. Combined probe implementation | after A/B | opt-in flag path preserving D3 and residency. |
| E. E2E transfer | after D | per-role and whole-prefill artifacts. |

## Stop Condition

Call this path blocked only if:

- the exact D3/K-major conflict has been named,
- at least one combined probe preserves D3 but fails residency,
- at least one combined probe preserves residency but fails D3,
- and the trace proves the two requirements are structurally incompatible in the current postrange representation.

Until then, the next action is not broad scheduler tuning. It is the D3 plus phase-scoped residency conflict probe.

## Existing Probe Result - 2026-07-09

We do already have the required probe substrate. The relevant tools are:

- `extra/qk/prefill/kernel_lifecycle_trace.py --full-rows`
- `extra/qk/prefill/hand_vs_generated_shape_matrix.py`
- `extra/qk/prefill/prefill_stage_owner_audit.py`
- `extra/qk/prefill/wmma_frag_key_audit.py`

The smallest useful test was not a new harness. It was the existing K-major D3A marker path:

```text
PREFILL_WMMA_KMAJOR_PHASE=1
PREFILL_WMMA_AB_PROOF_KEY=1
PREFILL_WMMA_AB_PHASE_SCOPED_KEY=1
PREFILL_WMMA_AB_PROOF_FROM_LDS_DESC=1
PREFILL_DBUF_D3A_POST=1
PREFILL_DBUF_D3A_AUDIT=1
PREFILL_WMMA_KMAJOR_D3A_MARKER=1
```

Probe table:

| variant | inst | wait/WMMA | global/WMMA | store/WMMA | load/WMMA | barriers | D3 | max cluster | bounded TFLOPS | status |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| baseline DBUF | 625 | 3.312 | 2.000 | 2.000 | 4.000 | 2 | false | 1 | prior about 7.9 | ok |
| K-major only | 554 | 2.875 | 2.000 | 2.000 | 2.000 | 2 | false | 3 | prior about 12.1 | ok |
| D3A only | 769 | 5.562 | 3.500 | 3.500 | 4.000 | 2 | true | 1 | not promoted | structurally heavy |
| K-major + D3 marker, A+B | 656 | 4.188 | 3.125 | 3.125 | 2.000 | 2 | true | 2 | 9.09 | ok |
| K-major + D3 marker + stage steal | 680 | 4.562 | 3.125 | 3.125 | 2.000 | 17 | true | 3 | 10.47 | ok |
| K-major + D3 marker, B only | 611 | 3.625 | 2.562 | 2.562 | 2.000 | 2 | true | 2 | 7.38 | ok |
| K-major + D3 marker, A only | 602 | 3.625 | 2.562 | 2.562 | 2.000 | 2 | true | 2 | 6.43 | ok |

Interpretation:

- The original framing was slightly wrong: K-major did not break a working D3 cadence on this active `2x2` trace.
  Baseline DBUF also has `D3=false`; all global/LDS stores are still prologue-heavy.
- The existing K-major D3A marker proves the combined structural property is possible:
  `D3=true`, `ds_load_b128/WMMA=2.0`, and max cluster `2-3`.
- The combined marker is still too expensive because it duplicates/moves extra global and LDS store work:
  `global_b128/WMMA` and `ds_store_b128/WMMA` rise from `2.0` to `3.125`.
- Stage steal preserves cluster `3`, but adds 17 barriers and is not currently the lean primitive.
- A-only/B-only are structurally cheaper but slower in the bounded timing.

Updated blocker:

```text
The blocker is not "find a probe" and not "prove D3 plus residency is possible."
Both are done.

The blocker is to make the combined path lean:
  keep K-major reuse:        ds_load_b128/WMMA <= 2.0
  keep D3 body staging:      D3=true, body_has_next_slot_work=true
  avoid duplicate staging:   global_b128/WMMA and ds_store_b128/WMMA near 2.0, not 3.125+
  avoid barrier explosion:   keep barriers near 2, not 17
```

Next implementation target:

```text
teach `_dbuf_d3a_probe_marker` / stage stealing to move or reuse the already-planned next-slot stage work,
not emit extra duplicate stage stores.
```

The first code probe should therefore be a deduplicating D3A marker:

1. identify the existing prologue stage store that corresponds to the K-major carrier's future-slot window,
2. emit the moved copy in the body,
3. suppress or mark the original prologue copy as stolen for the same stage key,
4. keep the K-major pack cache and phase-scoped reuse key unchanged.

Done signal for the next patch:

```text
D3=true
ds_load_b128/WMMA <= 2.0
max_cluster >= 3
global_b128/WMMA <= 2.25
ds_store_b128/WMMA <= 2.25
barriers <= 3
bounded TFLOPS > K-major+D3-marker current 10.47
```

### Suppression Audit

Existing broad suppression was tested:

```text
PREFILL_WMMA_KMAJOR_STAGE_STEAL=1
PREFILL_WMMA_KMAJOR_STAGE_STEAL_MEMO=1
PREFILL_WMMA_KMAJOR_STAGE_STEAL_SUPPRESS=1
```

It reduced traffic but corrupted output:

| variant | inst | wait/WMMA | global/WMMA | store/WMMA | load/WMMA | D3 | max cluster | result |
|---|---:|---:|---:|---:|---:|---|---:|---|
| stage steal, no suppress | 680 | 4.562 | 3.125 | 3.125 | 2.000 | true | 3 | correct, `10.47 TFLOPS` |
| stage steal, broad suppress | 542 | 3.062 | 1.625 | 1.625 | 2.000 | true | 3 | wrong, `rr=nan` |

Audit finding:

- Stage stealing records 24 stolen stage windows.
- By the time original prologue stores lower, their stage key is gone; the audit reports `key=None` and only an
  absolute LDS slot remains.
- Broad suppression therefore matches by absolute slot and suppresses phase-0/prologue producers that are still needed
  before the first WMMA.
- A stricter exact-key suppression was briefly tested locally, but it had no effect because the lowered stores no longer
  carry the exact key. It was not kept.

Updated implementation requirement:

```text
suppression must be epoch-aware:
  suppress only the prologue/body store instance that is actually re-emitted for the same producer epoch,
  never suppress the phase-0 producer for an LDS slot merely because a later phase reuses the same slot.
```

The next patch should preserve a stage-store epoch/key through lowering, or attach enough metadata to the stolen-store
set to distinguish:

```text
(lds slot, producer epoch before first WMMA)
from
(same lds slot, producer epoch moved into body for later K phase)
```

## Fix Scope - 2026-07-09

The P8 failure should be fixed at the owned-stage lifecycle layer, not by adding another renderer-side deletion rule.

The primitive plan is:

```text
postrange owner metadata
  -> owned-stage prologue/body/tail materializer
  -> K-major fragment reuse on the materialized stream
  -> P8 phase-cluster gate
```

Do not use these as fixes:

```text
PREFILL_WMMA_KMAJOR_STAGE_KEY_SUPPRESS late deletion
slot-only suppression
renderer moved-store memoization
waitcnt-only tuning before the lifecycle shape changes
```

### P0. Owner Metadata Probe

Status: first slice implemented.

Flag:

```text
PREFILL_DBUF_OWNED_AB_STAGE_META=1
```

Result:

```text
postrange audit sees:
  A owned_stage=A_IDENTITY, nbuf=2, reduce range present
  B owned_stage=B_IDENTITY, nbuf=2, reduce range present
```

This is behavior-neutral. It only gives later lowering an explicit owner object instead of making the renderer infer
ownership from final LDS addresses.

Verification:

```bash
PYTHONPATH=. pytest -q test/unit/test_prefill_stage_owner_audit.py test/unit/test_prefill_kernel_lifecycle_trace.py
```

Result:

```text
40 passed
```

### P1. Rotate Metadata Probe

Add an opt-in rotate tag without changing codegen:

```text
PREFILL_DBUF_OWNED_AB_STAGE_META=1
PREFILL_DBUF_OWNED_B_STAGE_EMIT=rotate
```

Required audit result:

```text
B owned_stage=B_ROTATE
lifecycle=prologue_body_tail
rotation=kr_mod_nbuf
```

This must still fail closed before materialization if no owner-aware lowering is installed.

Status: complete for B.

Observed with:

```bash
PYTHONPATH=. \
PREFILL_DBUF_OWNED_AB_STAGE_META=1 \
PREFILL_DBUF_OWNED_B_STAGE_EMIT=rotate \
PREFILL_DBUF_ROTATED_STAGE_LOWERING_AUDIT=1 \
python3 extra/qk/prefill/prefill_stage_owner_audit.py \
  --shape 2,2 --m 512 --n 5120 --k 5120 --loc 2 --unr 2 \
  --boundary postrange --json
```

The postrange audit shows:

```text
B owned_stage=B_ROTATE
lifecycle=prologue_body_tail
rotation=kr_mod_nbuf
```

Full lowering with the same flags fails at the intended boundary:

```text
PREFILL_DBUF owned B rotate lowering reached rangeify hook,
but prologue/body/tail materializer is not implemented
```

That is the correct fail-closed behavior. It proves the owner metadata reaches the lowering hook without silently
falling back to old generic staging.

### P2. B-Only Owned Materializer

Implement the smallest real lifecycle rewrite for B only:

```python
prologue:
  produce_B(slot=0, epoch=k0)
  barrier()

body:
  consume_B(slot=k % 2, epoch=k)
  produce_B(slot=(k + 1) % 2, epoch=k + 1)
  barrier()

tail:
  consume_B(slot=last % 2, epoch=last)
```

Safety requirements:

```text
no late suppression
no same-epoch STAGE index rewrite
each consume has exactly one prior producer
barrier separates producer from consumer
phase 0 producer stays prologue-owned
```

Pass gate:

```text
correct 2x2 bounded result
global_b128/WMMA and ds_store_b128/WMMA do not rise above base
D3 body staging appears
```

Initial implementation boundary tested:

```text
tinygrad/schedule/rangeify.py::_prefill_dbuf_owned_b_stage_lowering
```

Inputs available there:

```text
Ops.STAGE with owned_stage=B_ROTATE
role/lds_buffer_id/nbuf/tile_count/tile_elems
reduce slot carrier from prefill_dbuf_reduce_range(...)
the consumer idx used by generic STAGE lowering
```

The materializer must construct new graph ownership. It must not:

```text
rewrite the same STAGE idx to k+1 without prologue/tail guards
emit generic STAGE and then suppress stores later
use final LDS slot equality as an epoch proof
```

P2 boundary correction:

The rangeify hook is the right fail-closed guard, but it is too late to be the lifecycle constructor. At that point the
audit row is:

```text
role=B
owned_stage=B_ROTATE
stage_dtype=dtypes.half.vec(128)
stage_src_ops=[GEP, ADD]
idx_op=ADD
stage_ranges=[REDUCE, GLOBAL]
idx_ranges=[WARP, LOCAL]
```

That proves owner identity survives to rangeify, but rangeify only owns `STAGE -> local buffer/store/barrier` lowering.
It does not own the surrounding WMMA consume schedule, so it cannot safely express:

```text
consume(k)
produce(k+1)
barrier
consume(k+1)
```

without either reintroducing all-before generic staging or creating a same-stage dependency cycle.

Therefore the B-only materializer must move one level earlier:

```text
postrange owned-stage rewrite before generic Ops.STAGE lowering
```

The rangeify hook should remain as:

```text
guard: fail if B_ROTATE reaches generic STAGE lowering unmaterialized
audit: record owner fields and exact lowered boundary
```

P2 probe result:

```text
PREFILL_DBUF_OWNED_B_STAGE_ROTATE_MATERIALIZE=1
```

Attempted implementation:

```text
reuse B tile-key layout
emit current slot only for kr==0
emit future slot for kr+1 when kr < K-1
guard future value with valid(kr < K-1)
drop the large GLOBAL tile loop for this owned materializer
```

What it proved:

```text
full lowering can represent the guarded future-store graph
rangeify no longer sees an unmaterialized B_ROTATE STAGE
```

Why it is not the fix:

```text
native ISA trace fails during allocation:
  NotImplementedError: Inc 0: no spills

full-boundary audit also shows the materialized graph has only 8 WMMA where the bounded 2x2 target expects 16.
```

Pressure-reduction test:

```text
PREFILL_LDS_PACK_CARRIER=1
```

Result:

```text
still fails native ISA allocation with NotImplementedError: Inc 0: no spills
```

Conclusion:

```text
The simple guarded future-store materializer is not viable. It creates too much live producer pressure and perturbs the
WMMA grouping before the K-major phase can recover it.
```

Next viable P2 shape:

```text
build the materializer at the K-major phase boundary, where B fragments are already grouped into the WMMA cluster,
or introduce an explicit stage pseudo that survives until K-major lowering.
```

Do not promote `PREFILL_DBUF_OWNED_B_STAGE_ROTATE_MATERIALIZE`; it is a default-off failed probe.

### P3. A+B Owned Materializer

Extend P2 to A and B only after B-only is correct.

Pass gate:

```text
D3=true
body_has_next_slot_work=true
ds_load_b128/WMMA <= 2.0
global_b128/WMMA <= 2.25
ds_store_b128/WMMA <= 2.25
max WMMA cluster >= 3
barriers <= 3 on bounded trace
```

### P4. Bounded Timing

Run the existing bounded matrix harness.

Promotion gate:

```text
candidate TFLOPS > K-major base
candidate is correct
candidate passes P8 phase-cluster quality
```

### P5. Route Transfer

Only after P4:

```text
enable for composed generated S10 route
run per-role correctness/timing
run whole-prefill fail-closed smoke
```

Success is:

```text
whole prefill beats stored Path1 or records the next named bottleneck
no route silently falls back
```
