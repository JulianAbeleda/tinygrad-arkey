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

## Small-Test Matrix - 2026-07-10

Bounded target:

```text
shape=2x2
M=512, N=5120, K=5120
loc=2, unr=2
target=AMD:ISA:gfx1100
```

### Structural Trace Results

| Variant | Correct compile | WMMA | inst | global/WMMA | store/WMMA | load/WMMA | wait/WMMA | barriers | max burst | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline DBUF | yes | 16 | 625 | 2.000 | 2.000 | 4.000 | 3.312 | 2 | 1 | correct but no reuse/cluster |
| K-major only | yes | 16 | 537 | 2.000 | 2.000 | 2.000 | 2.875 | 2 | 3 | best current generated signal |
| K-major + D3 marker | yes | 16 | 645 | 3.125 | 3.125 | 2.000 | 4.188 | 2 | 2 | additive over-stage |
| K-major phase steal, no suppress | yes | 16 | 775 | 4.250 | 4.250 | 2.000 | 6.250 | 20 | 3 | too much duplicate work/barriers |
| K-major phase steal + memo | yes | 16 | 669 | 3.125 | 3.125 | 2.000 | 4.562 | 17 | 3 | still duplicate/barrier heavy |
| K-major pipeline epochs | compiles | 16 | 532 | 1.625 | 1.625 | 2.000 | 3.125 | 17 | 3 | smaller but wrong output |
| postrange rotate materializer | no | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | no-spill |

### Correctness/Timing Smoke

| Variant | Status | TFLOPS |
|---|---|---:|
| K-major only | ok | 12.24 |
| K-major + D3 marker | ok | 9.18 |
| K-major phase steal + memo | ok | 8.84 |
| K-major pipeline epochs | wrong, `rr=nan` | 0.0 |
| stage steal broad suppress | wrong, `rr=nan` | 0.0 |
| stage steal epoch suppress | wrong, `rr=nan` | 0.0 |
| stage-key suppress B phase 1 | wrong, `rr=1.4e+00` | 0.0 |
| postrange rotate materializer | native allocation fails, `Inc 0: no spills` | n/a |

### Explicit Stage-Pseudo Feasibility

This route was tested without adding a new opcode by exporting the existing postrange owner records into the DBUF
lifecycle checker.

Input owner records:

```text
A owner: lds_buffer_id=990, nbuf=2, reduce=(0, AxisType.REDUCE)
B owner: lds_buffer_id=991, nbuf=2, reduce=(0, AxisType.REDUCE)
```

Logical event export:

```text
event_count=20
producer_count=8
consumer_count=8
check_events(require_p5=False): ok
check_events(require_p5=True): fails only because pseudo wait events are not represented
```

Interpretation:

```text
The owner metadata is sufficient to build a correct logical prologue/body/tail pseudo lifecycle.
The missing piece is a lowering representation that carries wait/barrier placement and materializes the pseudo late
enough to avoid the postrange materializer's register-pressure failure.
```

### Decision

Current-head small tests eliminate:

```text
late suppression
epoch suppression
postrange guarded future-store materialization
additive D3/stage stealing as a performance route
```

The only route with a positive small-test signal is:

```text
explicit owned-stage pseudo surviving until K-major/isel lowering
```

K-major-only should remain the current best generated baseline, but it does not solve D3/body next-slot ownership.

## S9/S10 Amortization Math - 2026-07-10

The useful-work formula is:

```text
1 fp16 RDNA3 WMMA = 16 * 16 * 16 FMAs = 8192 FLOPs
useful_flops      = wmma_count * 8192
flops_per_overhead(kind) = useful_flops / count(kind)
```

Commands:

```bash
DEV=AMD:ISA PYTHONPATH=. python3 extra/qk/prefill/kernel_lifecycle_trace.py --kind hand-lds2 \
  --m 512 --n 5120 --k 5120 --wm 2 --wn 2 --waves-m 1 --waves-n 1 --bk 64 --dbuf 1 \
  --target AMD:ISA:gfx1100 --json

DEV=AMD:ISA AMD_ISA_WMMA_B128_FRAG=1 AMD_ISA_REG_ACCUM=1 AMD_ISA_WAITCNT_TARGETED=1 \
PREFILL_TC_LOCAL_STAGE=both PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE_POST=1 PREFILL_LDS_PACK_WITHLOCAL_B128=1 PREFILL_DBUF=1 \
PREFILL_DBUF_LDS_CONST_IMM=1 PREFILL_DBUF_LDS_INDEX_SPLIT=1 PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 \
PREFILL_DBUF_DIRECT_B128_CHAIN=1 PREFILL_DBUF_LDS_ADDR_USE_DEP=1 REGALLOC_ADDR_REMAT=1 \
PYTHONPATH=. python3 extra/qk/prefill/kernel_lifecycle_trace.py --active-generated --kind generated \
  --shapes 2,2 --m 512 --n 5120 --k 5120 --loc 2 --unr 2 --target AMD:ISA:gfx1100 --json

DEV=AMD:ISA AMD_ISA_WMMA_B128_FRAG=1 AMD_ISA_REG_ACCUM=1 AMD_ISA_WAITCNT_TARGETED=1 \
PREFILL_TC_LOCAL_STAGE=both PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE_POST=1 PREFILL_LDS_PACK_WITHLOCAL_B128=1 PREFILL_DBUF=1 \
PREFILL_DBUF_LDS_CONST_IMM=1 PREFILL_DBUF_LDS_INDEX_SPLIT=1 PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 \
PREFILL_DBUF_DIRECT_B128_CHAIN=1 PREFILL_DBUF_LDS_ADDR_USE_DEP=1 REGALLOC_ADDR_REMAT=1 \
PREFILL_WMMA_KMAJOR_PHASE=1 PREFILL_WMMA_AB_PROOF_KEY=1 PREFILL_WMMA_AB_PHASE_SCOPED_KEY=1 \
PREFILL_WMMA_AB_PROOF_FROM_LDS_DESC=1 PYTHONPATH=. \
python3 extra/qk/prefill/kernel_lifecycle_trace.py --active-generated --kind generated \
  --shapes 2,2 --m 512 --n 5120 --k 5120 --loc 2 --unr 2 --target AMD:ISA:gfx1100 --json
```

| Route | P8 | WMMA | useful FLOPs | waits/WMMA | max burst | ds_load/WMMA | inst/WMMA | FLOPs/wait | FLOPs/DS load | FLOPs/inst |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S9 hand LDS2 `2x2` | pass, `hand_lds2_quality` | 64 | 524288 | 0.406 | 4 | 2.0 | 9.547 | 20165 | 4096 | 858 |
| S10 generated DBUF `2x2` | fail, wait amortization | 16 | 131072 | 3.312 | 1 | 4.0 | 39.062 | 2473 | 2048 | 210 |
| S10 K-major `2x2` | fail, not DBUF-like | 16 | 131072 | 2.875 | 3 | 2.0 | 34.625 | 2849 | 4096 | 237 |

Conclusion:

```text
K-major proves the formula moves in the right direction:
  ds_load/WMMA improves 4.0 -> 2.0
  max burst improves 1 -> 3
  FLOPs/DS-load improves 2048 -> 4096

But it does not yet solve wait amortization:
  S9 hand 2x2:      ~20165 FLOPs/wait
  S10 DBUF:          ~2473 FLOPs/wait
  S10 K-major:       ~2849 FLOPs/wait
```

So the remaining primitive gap is not just LDS load reuse. The generated route must also move waits from per-WMMA shape
toward phase-cluster shape.

### Small Test: Clustered LGKM Wait Coalescing

Probe:

```text
AMD_ISA_WMMA_CLUSTER_LGKM_WAIT=1
```

Meaning:

```text
For WMMA consumers only, coalesce a targeted LDS wait into `lgkmcnt(0)`.
This drains all outstanding LDS loads at the first WMMA in a group, so later WMMAs in the same group may avoid their
own waits. It is correctness-conservative, not an aggressive wait deletion.
```

Results:

| Route | Correct | TFLOPS | waits/WMMA | max burst | ds_load/WMMA | inst/WMMA | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| baseline DBUF + clustered LGKM wait | yes | 8.21 | 3.312 | 1 | 4.0 | 39.062 | no structural movement |
| K-major + clustered LGKM wait | yes | 11.88 | 2.562 | 4 | 2.0 | 34.312 | structural movement, slower than K-major-only |

Comparison to prior K-major-only:

```text
waits:      46 -> 41
wait/WMMA:  2.875 -> 2.562
max burst:  3 -> 4
TFLOPS:     12.24 -> 11.88
```

Decision:

```text
Do not promote clustered LGKM wait coalescing as a standalone performance fix.
It proves the target shape is reachable only after K-major grouping, but the larger drain hurts enough that timing does
not improve. The next primitive must move/cluster LDS loads and WMMAs together, not merely coalesce waits after the
current stream has already chosen its load placement.
```

### Small Test: Dependency-Only Clustered LDS Consume

Probe:

```text
PREFILL_WMMA_CLUSTERED_LDS_CONSUME=1
```

Attempted design:

```text
Inside `_try_wmma_kmajor_phase`, materialize all A/B packs for a phase first and add them as dependencies to the phase's
WMMA nodes. This tries to force "load all fragments, then emit WMMAs" without changing the fragment planner.
```

Results:

| Route | Correct | TFLOPS | waits/WMMA | max burst | ds_load/WMMA | inst/WMMA | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| K-major + dependency-only clustered consume | structural only | n/a | 2.812 | 3 | 2.0 | 34.562 | no material P8 movement |
| K-major + dependency-only clustered consume + clustered LGKM wait | yes | 11.55 | 2.562 | 4 | 2.0 | 34.312 | same shape as wait coalescing, slower |

Decision:

```text
Dependency-only preloading is not the primitive.
It does not change the final DS-load/WMMA/wait structure enough for P8, because the underlying fragment planner still
emits/reuses fragment packs in a shape the wait pass treats as per-WMMA-ish. The real primitive needs ownership of the
cluster plan itself: choose the WMMA group, choose the resident fragment VGPRs for that group, emit the LDS loads for
that group, then emit the WMMAs before those VGPRs are reused.
```

## Math-Driven Existing-Flag Ladder - 2026-07-10

After correcting the amortization inference, the test ladder is:

```text
T1. Does larger generated window size improve ops/wait with existing K-major?
T2. Does conservative clustered LGKM wait improve the larger windows?
T3. Does 4x4 prove the target math, and is it correct?
T4. If 4x4 proves math but is wrong, design a legal cluster-window primitive that gets 4x4-like math inside active
    legal shapes.
```

### T1. K-major Shape Sweep

Flags:

```text
PREFILL_WMMA_KMAJOR_PHASE=1
PREFILL_WMMA_AB_PROOF_KEY=1
PREFILL_WMMA_AB_PHASE_SCOPED_KEY=1
PREFILL_WMMA_AB_PROOF_FROM_LDS_DESC=1
```

| Shape | Correct | TFLOPS | WMMA | waits | waits/WMMA | ops/wait | FLOPs/wait | max burst | ds_load/WMMA | inst/WMMA |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `2x2` | yes | 11.68 | 16 | 46 | 2.875 | 0.348 | 2849 | 3 | 2.0 | 34.625 |
| `4x2` | yes | 9.05 | 32 | 70 | 2.188 | 0.457 | 3745 | 4 | 1.5 | 29.188 |
| `2x4` | yes | 8.24 | 32 | 70 | 2.188 | 0.457 | 3745 | 5 | 1.5 | 29.188 |

Readout:

```text
Larger legal generated windows improve amortization counters but reduce timing.
They are not enough: ops/wait is still far below S9 hand 2x2's 2.46.
```

### T2. K-major + Clustered LGKM Wait Shape Sweep

Additional flag:

```text
AMD_ISA_WMMA_CLUSTER_LGKM_WAIT=1
```

| Shape | Correct | TFLOPS | WMMA | waits | waits/WMMA | ops/wait | max burst | ds_load/WMMA |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `2x2` | yes | 11.79 | 16 | 41 | 2.562 | 0.390 | 4 | 2.0 |
| `4x2` | yes | 9.16 | 32 | 57 | 1.781 | 0.561 | 8 | 1.5 |
| `2x4` | yes | 7.90 | 32 | 57 | 1.781 | 0.561 | 8 | 1.5 |

Readout:

```text
Clustered wait helps structure only when the window is already larger.
It still does not cross P8, and timing remains worse than 2x2 K-major.
```

### T3. 4x4 As Math Oracle, Not A Valid Route

`4x4` is parked for correctness/resource reasons, but it is useful as a math oracle.

| Shape | Cluster wait | Correct | WMMA | waits | waits/WMMA | ops/wait | max burst | global/store/load per WMMA | inst/WMMA |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `4x4` | no | wrong, `rr=nan` | 64 | 88 | 1.375 | 0.727 | 13 | 1.0 / 1.0 / 1.0 | 23.312 |
| `4x4` | yes | wrong, `rr=nan` | 64 | 73 | 1.141 | 0.877 | 16 | 1.0 / 1.0 / 1.0 | 23.078 |

Readout:

```text
4x4 moves strongly toward the target math:
  WMMA count = 64
  max burst = 13-16
  waits/WMMA = 1.141-1.375
  global/store/load per WMMA = 1.0/1.0/1.0

But it is not correct on GPU, so it remains a math oracle only.
Do not unpark 4x4 as a route.
```

### Current Test Conclusion

The math path is now clear:

```text
Need 4x4-like scheduling-window amortization without relying on the current 4x4 generated route.
```

That means the next primitive should create a larger matrix-op cluster inside legal active shapes:

```text
cluster window over legal 2x2/4x2/2x4 execution
  -> choose resident fragments
  -> emit cluster loads
  -> one/few waits
  -> emit a 4+ WMMA burst
  -> preserve correctness and register pressure
```

Do not spend more time on:

```text
standalone wait coalescing,
dependency-only preloading,
or 4x4 route promotion.
```

## Next Test And Implementation List - 2026-07-10

The corrected S9/S10 math changes the priority. The instruction gap is mostly a density/window artifact:

```text
S9 hand LDS2:      611 total inst - 64 WMMA = 547 non-matrix inst
S10 generated:     554 total inst - 16 WMMA = 538 non-matrix inst
```

So the next work should not chase generic bookkeeping instruction deletion first. The real gap is wait amortization and
matrix-op density over one lifecycle window:

```text
S9 hand LDS2:      64 WMMA / 26 waits = 2.46 ops/wait
S10 generated:     16 WMMA / 46 waits = 0.35 ops/wait
```

### Things To Test, In Order

| Test | Purpose | Done Criteria | Current Readout |
|---|---|---|---|
| T0. Reconfirm S9 hand oracle | Keep the target math honest. | `64 WMMA`, `26 waits`, `max burst=4`, `global/store/load=1/1/2 per WMMA`. | Done. Still passes P8 as `hand_lds2_quality`. |
| T1. Reconfirm generated K-major base | Separate load-density wins from DBUF cadence. | `ds_load/WMMA=2.0`, but D3 false and wait-heavy. | Done. `16 WMMA`, `46 waits`, `max burst=3`, `D3=false`. |
| T2. Reconfirm D3 marker | Test whether D3 plus reuse exists at all. | `D3=true`, `ds_load/WMMA=2.0`, no correctness loss. | Done. Exists, but heavy: `global/store=2.5625 per WMMA`, `max burst=2`, `58 waits`. |
| T3. D3 marker + clustered wait | See if wait coalescing rescues the combined path. | Fewer waits and `max burst>=3` without traffic increase. | Done. Not enough: `55 waits`, `max burst=2`, traffic still duplicated. |
| T4. Lean owner-stage probe | Move/reuse the producer instead of emitting duplicate D3 marker stores. | `D3=true`, `global/store<=2.25 per WMMA`, `ds_load/WMMA<=2.0`, barriers <= 3. | Next implementation test. |
| T5. Cluster planner audit | Prove whether current final stream contains legal 4-WMMA clusters before lowering changes. | Emit candidate windows with required A/B loads, overwrite hazards, waits, and expected ops/wait. | Needed before broad rewrite. |
| T6. Real matrix-consumer cluster lowering | If T5 says legal windows exist, emit one resident-fragment cluster. | `max burst>=4`, `waits/WMMA<=1.0`, correctness pass. | Not started. |
| T7. Bounded timing | Verify structural movement translates to speed. | Candidate beats K-major base bounded TFLOPS. | After T4-T6. |
| T8. S10 route transfer | Verify it matters e2e. | Per-role correctness passes; whole-prefill beats Path1 or records next bottleneck. | After T7. |

### Fresh Small-Test Results

Fresh trace runs on 2026-07-10 confirm the current stop point:

| Variant | D3 | WMMA | waits | waits/WMMA | max burst | global/WMMA | store/WMMA | load/WMMA | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| S9 hand LDS2 oracle | true | 64 | 26 | 0.406 | 4 | 1.0 | 1.0 | 2.0 | target shape |
| generated K-major base | false | 16 | 46 | 2.875 | 3 | 2.0 | 2.0 | 2.0 | reuse but no DBUF cadence |
| K-major + D3 marker | true | 16 | 58 | 3.625 | 2 | 2.5625 | 2.5625 | 2.0 | cadence exists but duplicate/heavy |
| K-major + D3 marker + clustered wait | true | 16 | 55 | 3.438 | 2 | 2.5625 | 2.5625 | 2.0 | wait flag cannot rescue heavy lifecycle |

Conclusion:

```text
The immediate primitive is a lean owner-stage body producer, not a wait-only pass.
After that, add a cluster planner/lowering that increases WMMA work per lifecycle window.
```

### T5 Probe: Matrix Consumer Cluster Audit

The tracer now exports:

```text
matrix_consumer_cluster_audit
```

It checks whether the final stream already has target-sized WMMA windows where:

```text
all required A/B LDS loads occur before the first WMMA,
all A/B loads are covered by a reaching LDS store,
and later WMMAs in the window do not need intervening waits.
```

Fresh generated K-major `2x2` result:

| Metric | Value |
|---|---:|
| target cluster | 4 WMMA |
| WMMA count | 16 |
| reaching-def WMMA rows | 16 |
| candidate 4-WMMA windows | 13 |
| already cluster-like windows | 0 |

Best windows show useful partial structure:

```text
start=4..7,  waits_in_window=2, unique_lds_loads=4, loads_before_first=true, covered=true
start=8..11, waits_in_window=2, unique_lds_loads=4, loads_before_first=true, covered=true
start=12..15, waits_in_window=3, unique_lds_loads=4, loads_before_first=true, covered=true
```

Readout:

```text
The data dependencies for 4-WMMA windows are close, but the current lowered stream still inserts waits inside those
windows. Since no already-cluster-like 4-WMMA window exists, a late wait-only pass is not the primitive. The fix must
own the matrix-consumer cluster at the K-major/lowering boundary: choose the resident fragment group, emit its LDS
loads, emit one readiness wait, then emit the 4 WMMAs before fragment reuse.
```

### T6 Needle Test: Does Wait Movement Alone Move Timing?

Small bounded test on `2x2`, `m=512,n=5120,k=5120`, with clock pin enabled:

| Variant | Correct | TFLOPS samples | waits | Readout |
|---|---|---:|---:|---|
| K-major base | yes | `11.54`, `6.94`, `6.69` | 46 | noisy baseline |
| K-major + conservative clustered LGKM wait | yes | `11.36`, `6.14`, `7.96` | 41 | removes 5 waits, timing not decisively better |
| unsafe skip later WMMA LDS waits | no, `rr=1.1e+00` | `0.0` | 55 | not viable; downstream drains increase waits |

Readout:

```text
The conservative wait coalescer moves structure in the right direction and stays correct, but bounded timing is too noisy
and not decisively better. The unsafe deletion probe proves the obvious shortcut is wrong. Therefore the "will it move"
test does not justify a wait-only implementation; it supports the stricter primitive: legal cluster lowering must keep
the fragment-load dependencies intact while increasing the amount of WMMA work between waits.
```

The next code should be small and fail-closed:

```python
for each phase cluster:
  identify existing stage producer for (role, slot, epoch)
  if producer is needed by an earlier load:
    keep prologue producer
  if future epoch producer can be moved into body:
    move/reuse it with explicit owner metadata
  never suppress by LDS slot alone
  emit trace row: producer_epoch, consumer_epoch, slot, barrier_between
```

Stop condition for T4:

```text
If a lean owner-stage probe cannot get global/store below 2.25 per WMMA without breaking correctness,
then the current postrange representation is still too late and the primitive must move earlier than final LDS lowering.
```

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
