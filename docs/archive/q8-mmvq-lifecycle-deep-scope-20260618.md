# q8/MMVQ lifecycle deep scope — what it means and what would have to be true

This scopes the only meaningful remaining **decode** research arc after the per-role delta audit:
`decode_q4k_ffn_q8_sidechannel`. It is not a dot4 task and not a scheduler task. It is an activation/dataflow
lifecycle task.

## What "q8/MMVQ lifecycle" means

llama's fast Q4_K decode primitive is not just a kernel that calls `dot4`. The complete primitive is:

```text
fp activation after RMSNorm/apply
+ q8_1 activation representation produced cheaply enough
+ q8 scales/layout live long enough to feed the right linears
+ packed Q4_K weight decode
+ native signed dot4 (`sudot4`)
+ qsum/min/scale affine correction
+ row/thread scheduler and in-kernel reduction
+ model route + quality gate
```

tinygrad already has pieces of this:

- signed dot4 lowering is fixed and value-tested;
- the int-dot Q4_K kernel works;
- graph reuse can common one q8 pack across gate+up;
- shipped coop kernels fixed several coalescing problems.

But the whole primitive does not win because q8 activation production is too expensive when done as a separate
pack. The lifecycle problem is: **where is q8 born, how much does it cost, how many consumers reuse it, and does
the model path actually route through that representation?**

## Why the obvious versions are closed

| attempted path | result | reason |
|---|---|---|
| dot4 intrinsic only | refuted | the dot is not the primitive; pack/dequant/dataflow dominate |
| whole-linear sudot4 with separate q8 pack | refuted | one shared pack for gate+up still loses, 0.94-0.96x vs fp coop |
| duplicated q8 pack per linear | refuted | worse than shared pack |
| graph-commoned q8 pack | works technically, still loses | TinyJit commons the pack, but reuse ceiling is only 2 |
| fused standalone pack kernel | insufficient | 8-12us plausible floor; break-even is <=4.8us effective |
| fp-codegen tuning | refuted | handwritten fp is ~49%, tinygrad fp is already ~48%; ALU ceiling |
| 128-thread scheduler alone | refuted as route | scheduler does not remove q8 pack cost |

The crucial measured facts:

- Only `ffn_gate` and `ffn_up` share the same Q4_K activation: reuse ceiling = 2.
- Current q8 pack anatomy is ~29.7us across 4 kernels.
- Separate-kernel floor is ~7us; even ideal fused standalone pack is above the <=4.8us target.
- Perfect producer-side fold has best-case decode EV only ~+3-4%.
- The path is lossy and requires a dNLL gate before routing.

## The only reopening: producer-side q8 side-channel

The one plausible route is to make the producer of the FFN activation also emit q8:

```text
ffn_norm / RMSNorm apply
  -> fp normalized activation for existing fp consumers/fallbacks
  -> q8_1 packed activation for Q4_K ffn_gate/up
  -> per-32 q8 scales
```

This can win only if the q8 side-channel is nearly free because it rides on a pass over values already resident
from RMSNorm/apply. It cannot be a separate pack kernel.

Important mismatch:

- RMSNorm reduction is per-row sum of squares.
- q8 scale reduction is per-32 max(abs(x)).

So the q8 side-channel cannot simply reuse RMSNorm's reduction. It needs extra per-32 reduction work over the
normalized values, ideally inside the same custom producer kernel.

## Build shape if funded

This is the minimum real primitive:

```text
custom RMSNorm/apply producer
+ multi-output stores:
   1. fp output, same semantics as current path
   2. q8 packed int8/uint32 words, llama-compatible layout
   3. per-32 q8 scales
+ Q4_K ffn_gate/up int-dot consumers
+ fallback to current fp path for unsupported shapes/devices
+ dNLL quality gate
```

It touches a hot shared op, so it must support:

- decode shapes first;
- residual/add input semantics exactly as current RMSNorm path expects;
- fp fallback path;
- unsupported shape fallback;
- no decode regression when disabled;
- no accidental prefill route unless explicitly scoped.

## Phase plan

### Q8L-0 — producer contract audit

Goal: prove the exact producer/consumer contract before writing a kernel.

Map:
- where `ffn_norm(h)` is produced in the model graph;
- exact tensor shape, dtype, contiguity, and lifetime at decode;
- which linears consume it and in what order;
- whether attn_norm or other RMSNorm uses must remain untouched;
- current RMSNorm/apply program count and device time;
- required q8 layout and scale layout for the existing sudot4 consumer.

Gate:
- one producer feeds exactly `ffn_gate` and `ffn_up`;
- no hidden third consumer needed for correctness;
- no model route requires q8 outside that pair.

Kill if:
- the producer cannot be isolated to FFN norm without broad model refactor;
- the layout would require extra transposes/copies;
- q8 consumers cannot share one side-channel buffer.

### Q8L-1 — cost model and lower bound

Goal: decide whether the side-channel can plausibly hit <=4.8us before implementation.

Compute:
- bytes read/written by current RMSNorm/apply;
- extra q8 writes: packed q8 + scales;
- per-32 max work over normalized fp values;
- expected VGPR/LDS pressure;
- expected occupancy change;
- worst-case extra global traffic.

Gate:
- predicted incremental producer cost <=4.8us for the gate+up pair;
- predicted full decode EV >=3%;
- no extra kernel launches.

Kill if:
- the cost model requires a separate q8 pass;
- extra output traffic or register pressure makes <=4.8us implausible;
- expected EV falls below sub-gate maintenance cost.

### Q8L-2 — custom-kernel expressibility spike

Goal: prove tinygrad can express the producer without routing it.

Build only a minimal harness:
- one row or one small batch of RMSNorm/apply input;
- output fp normalized values;
- output q8 packed values;
- output q8 scales;
- compare fp output to current RMSNorm/apply;
- compare q8 output to existing pack oracle.

Gate:
- one kernel;
- no dense fallback;
- no extra realize/copy;
- q8 scales and packed words correct;
- compile time and source size sane.

Kill if:
- multi-output custom kernel plumbing fails;
- per-32 reduction cannot be represented cleanly;
- compile/runtime shape specialization explodes.

### Q8L-3 — isolated producer+consumer benchmark

Goal: measure the actual primitive cost, not producer alone.

Benchmark:

```text
current:
  RMSNorm/apply -> fp coop gate -> fp coop up

candidate:
  fused RMSNorm/apply+q8 side-channel -> sudot4 gate -> sudot4 up
```

Measure:
- device time for the whole pair;
- producer overhead vs current RMSNorm/apply;
- gate/up time;
- kernel count;
- correctness vs fp reference;
- rel error distribution.

Gate:
- paired candidate >=1.15x vs current fp coop pair;
- producer overhead <=4.8us effective;
- no more kernels than current path plus unavoidable consumers;
- no correctness anomaly beyond expected q8 quantization.

Kill if:
- candidate <1.15x isolated;
- q8 producer is correct but overhead >4.8us;
- extra memory traffic or register pressure erases sudot4 savings.

### Q8L-4 — one-block integration

Goal: test transfer before a full route.

Integrate behind an explicit flag in a one-block harness:

```text
Q8_FFN_SIDECHANNEL=1
```

Include:
- FFN norm;
- gate/up;
- SwiGLU;
- down projection using existing path;
- residual boundary;
- realistic decode shape.

Gate:
- one-block speedup implies >=3% full-decode potential;
- no dense fallback;
- fallback path unchanged;
- output error distribution understood.

Kill if:
- isolated win does not transfer;
- integration forces broad graph changes;
- down/SwiGLU/residual boundaries introduce extra copies.

### Q8L-5 — quality gate

Goal: decide if lossy q8 is acceptable.

Run only if Q8L-3/Q8L-4 pass speed gates.

Gate:
- teacher-forced dNLL <=0.01 over at least two windows;
- token smoke acceptable;
- compare OFF=current fp path vs ON=q8 side-channel only.

Kill if:
- dNLL exceeds threshold;
- degradation is shape/context dependent in a way that cannot be bounded.

### Q8L-6 — in-model route candidate

Goal: only after all prior gates, test real decode.

Run:
- W==D decode at ctx 128/512/1024/4096;
- byte/quality accepted;
- no ctx regression;
- disabled flag path identical to current default;
- unsupported devices/shapes fall back.

Candidate gate:
- >=5% W==D decode to justify route;
- or explicitly bank as sub-gate if +3-4% and maintenance cost is accepted.

Default gate:
- speed gate passes;
- quality gate passes;
- fallback tests pass;
- docs and machine-search row updated.

## Expected outcomes

Best case:
- producer-side q8 costs <=4.8us effective;
- gate/up pair improves >=1.15x;
- full decode gains ~3-4%;
- quality passes;
- route remains opt-in or sub-gate unless project accepts low-EV maintenance.

Most likely:
- producer-side q8 is expressible but overhead/register pressure exceeds the target;
- result becomes a durable refutation of the only remaining decode MMVQ lifecycle route.

Worst case:
- multi-output RMSNorm custom kernel is not cleanly expressible in tinygrad today;
- the blocker becomes a codegen/custom-kernel capability, not an MMVQ question.

## What this would and would not prove

Would prove:
- whether tinygrad can produce q8 cheaply enough at the activation producer;
- whether llama's q8/MMVQ lifecycle can transfer to tinygrad decode for gate/up;
- whether the last decode frontier is buildable or truly deferred behind codegen.

Would not prove:
- full llama parity. Gate/up best-case EV is ~3-4%, not the whole ~30% gap.
- prefill improvement. This is a batch-1 decode activation lifecycle arc.
- dot4 value. Dot4 already works; the question is whether the full primitive can afford q8.

## Decision

Do not build this as a casual kernel tweak. Fund it only if the goal is to close the remaining decode research
question, not because it is likely to be the highest-EV performance work. The highest-EV open performance arc
remains prefill fp16 WMMA LDS-tiling / BLAS boundary; q8/MMVQ lifecycle is the highest-value **decode explanation**
arc.
