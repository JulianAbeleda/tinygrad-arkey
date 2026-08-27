# Current dense lifecycle ledger versus llama and the roofline

## Decision

The remaining dense-decode loss is approximately accounted. Tinygrad is not
broadly slower than llama: it wins several fused lifecycle regions, then gives
those wins back primarily in flash attention, with smaller debt in residual
4096-wide norms, Q/O projections, and the vocabulary main body.

This is a lifecycle ledger. Llama's separate activation quantization, RoPE,
cache-store, and sampling work must be charged when tinygrad owns the same work
inside a fused GPU body. Concurrent command residence is not independently
recoverable service time.

## Endpoint authority

| implementation | latency | throughput |
|---|---:|---:|
| tinygrad, installed Flash ceiling endpoint | **4.060523 ms/token** | **246.274 tok/s** |
| llama official authority | **4.021721 ms/token** | **248.711 tok/s** |
| tinygrad debt | **38.802 us/token** | **2.437 tok/s** |

The load-schedule promotion is a paired compiler/capture contract.  Against
its enclosed cap-32 control it books 15.954 us/token (0.953 tok/s); relative to
the prior retained endpoint, the new conservative midpoint advances 10.700
us/token (0.640 tok/s).  Candidate-arm drift is preserved in the evidence and
the midpoint—not the 4.074658-ms fast arm—is used here.

The last full pre-load-schedule tinygrad device node sum is 3,954.656 us. The
retained like-for-like llama PDL-off node sum is 3,878.254 us, leaving 76.402
us of pre-promotion device-service debt.  The causal load-schedule profile
recovers 17.024 us in node sum and 16.250 us in device union; applying that
measured delta yields roughly 59--60 us of residual device debt, consistent
with the new 62.081-us endpoint gap.  A fresh full post-promotion profile is
still required before replacing the retained per-row timestamps wholesale.

The profiled tinygrad wall field is not an endpoint authority. Profiling adds a
large host/instrumentation term; only its device rows, node sum, union, and
route census are used here.

## Corrected lifecycle comparison

| lifecycle region | tinygrad | charged llama | TG - llama | current reading |
|---|---:|---:|---:|---|
| flash score | 194.048 us | 162.948 us | **+31.100 us** | installed S6 production row |
| flash combine | 48.448 us | 37.057 us | **+11.391 us unmatched row difference** | matched NCU: tinygrad 1.888 us vs llama 2.528 us; row is not a recovery pool |
| native 4096 norms | 231.680 us | 203.778 us | **+27.902 us** | lower-bound residual debt; shared provider owns some fused work |
| Q projection bodies | 296.352 us | 272.609 us including llama Q quant | **+23.743 us** before provider allocation | small genuine debt |
| O projections | 304.256 us | 284.993 us including llama O quant | **+19.263 us** | small genuine debt |
| vocabulary main GEMV | 313.824 us | approximately 300 us | **about +14 us** | main-body service debt only |
| gate/up | 1,271.872 us | 1,291.116 us including llama quant | **-19.244 us** | tinygrad faster |
| down projections | 840.512 us | 855.817 us | **-15.305 us** | tinygrad faster |
| Q norm + RoPE | 67.424 us | 72.641 us | **-5.217 us** | tinygrad faster |
| K norm + RoPE/cache | 68.416 us | 102.817 us | **-34.401 us** | tinygrad faster |
| K/V projection bodies | 232.608 us | 263.712 us including llama quant | **-31.104 us** before provider allocation | tinygrad faster before shared-provider charge; tied after conservative allocation |
| shared-Q8 provider | 31.264 us | shared across Q/K/V accounting | not independently additive | amortized norm/quant producer |

The table is intentionally approximate rather than falsely exact. The
shared-Q8 provider performs both norm and quantization work for Q, K, and V;
assigning all of it to any one row would double-count it. The vocabulary GPU
comparison also omits part of llama's D2H/host sampler lifecycle, so only the
roughly 14-us main-GEMV difference is treated as clean GPU debt.

The reconciliation is nevertheless strong:

- gross tinygrad losing territory is about 140 us after the booked Flash score recovery;
- tinygrad's fusion and lifecycle wins offset roughly 80 us;
- the resulting approximately 60-us net agrees with the post-promotion
  62.081-us endpoint difference and the converted device-union estimate.

## Exact weight and service-rate ledger

Every dense token streams approximately 4.671 GB of packed projection and
vocabulary weights.

| weight family | bytes/token | current time | payload rate | fraction of fitted 1.75-TB/s asymptote |
|---|---:|---:|---:|---:|
| gate/up | 2,038 MB | 1,271.872 us | **1.602 TB/s** | 92% |
| down | 1,253 MB | 840.512 us | **1.491 TB/s** | 85% |
| vocabulary | 511 MB | 313.824 us | **1.628 TB/s** | 93% |
| Q | 340 MB | 296.352 us | **1.147 TB/s** | 66% |
| O | 340 MB | 304.256 us | **1.117 TB/s** | 64% |
| K/V | 189 MB | 232.608 us | **0.813 TB/s** | 46% |
| **total** | **4,671 MB** | **3,259.424 us** | **1.433 TB/s** | **82%** |

The low aggregate Q/O/K/V rates are not evidence of a proportional recovery
pool. The measured size-aware model is:

```text
body time = about 3.27 us of physical stream/launch ramp
          + payload / about 1.75 TB/s asymptotic stream rate
```

Applying that model to the current physical launch populations gives:

| family | current | size-aware reconstruction | current minus model |
|---|---:|---:|---:|
| gate/up | 1,272.6 us | about 1,282.5 us | -9.9 us |
| down | 853.6 us | about 833.6 us | +20.0 us |
| vocabulary | 313.9 us | about 295.0 us | +18.9 us |
| Q | 297.2 us | about 311.9 us | -14.7 us |
| O | 305.0 us | about 311.9 us | -6.9 us |
| K/V | 233.6 us | about 258.6 us | -25.0 us |

Across the weight population, the current implementation is already about
18 us faster than this older size-aware reconstruction. The apparent Q/O/K/V
bandwidth shortfall is predominantly the latency ramp of small independent
streams. Wider loads or generic dequant scheduling cannot claim the distance
to the long-body rate unless they remove that physical ramp.

## Where tinygrad should be faster

Tinygrad should win where fused ownership removes separate lifecycle work:

1. Gate/up, after charging llama's activation quantization.
2. Down projection, where the installed Q4/Q6 bodies are already slightly
   faster in aggregate.
3. Q/K norm, RoPE, and cache storage, which tinygrad owns inside fused
   completion bodies.
4. The complete K/V lifecycle, through paired projection, direct cache output,
   and an amortized shared-Q8 provider.
5. Ancillary command mass generally: tinygrad performs less separate
   quantization, transform, and cache-store work.

These wins are real but total roughly 80 us against the charged llama
lifecycle. They do not compensate for the flash deficit.

## Where llama is genuinely faster

### Flash attention: about 73 us gross

```text
tinygrad score + combine = 272.928 us
llama score + combine    = 200.005 us
gross difference         =  72.923 us
```

> **Historical pre-promotion checkpoint.** The 4.094502-ms / 244.230-tok/s
> row below was superseded by the installed Flash ceiling authority at the top
> of this document: 4.060523 ms/token / 246.274 tok/s. It must not be used for
> current projections.

The extent-derived wide-load route was installed for NV sm_120 G4 fp16-KV
decode: S=MAXC/128 gives S8 on the official MAXC1024 graph. Its reps-9 reverse
bracket recovered 81.468 us/token with canonical token hashes, and that stage's
default-path endpoint was 4.094502 ms/token (244.230 tok/s). The remaining flash
debt is now about the same size as the complete endpoint gap. This is exposure,
not a claim that all residual flash service will translate independently.

The priority-1 cache-policy follow-up is now wall-resolved. An aggregate
36-layer/72-MiB primitive showed that evict-first one-use streams can preserve
the K/V-sized footprint exactly, but production conversion did not follow.
Q/K/V-only streaming recovered just 6.329 us/token (+0.377 tok/s), below the
50-us booking bar; applying the policy to every installed dense quantized
weight consumer regressed 23.588 us/token (-1.391 tok/s). The broad policy
barely moved the production flash row and slowed large gate/up and FFN-down
consumers by discarding useful intra-kernel packed-weight reuse. No endpoint is
booked. Reopen this row only with line/reuse-aware streaming or a K/V residency
mechanism that does not change projection service.

### Native 4096 norms: at least 28 us gross

The native promotion removed the large legacy penalty. The residual service
debt remains real, but one-warp, merge/broadcast, geometry, and input-retention
constructions are wall-closed. A new mechanism is required.

### Q/O: about 45 us gross before provider allocation

Llama's Q8/DP4A representation executes fewer instructions and consequently
services the same material weight bytes faster. Tinygrad reproduces that body
mechanism, but a standalone Q8 producer costs more than a single consumer
saves. Q can amortize its provider over Q/K/V; O cannot. Current Q/O are also
already at or ahead of the size-aware stream/ramp reconstruction.

This is a real comparison advantage for llama, but not an independently
claimable 45-us exact recovery pool. Translation requires removing or
amortizing a producer, eliminating a physical stream, or explicitly changing
the numerical/output contract.

### Vocabulary: about 14--19 us gross

The main Q6 body is a large streaming row and llama sustains a slightly higher
rate. The full lifecycle is not like-for-like because llama's GPU ledger omits
some host sampling work while tinygrad runs native GPU argmax. Only the main
GEMV difference is clean debt.

## Current optimization frame

The remaining endpoint is approximately:

```text
flash topology/reduction boundary        about 73 us gross
residual native norm service             about 28 us gross
Q/O representation advantage             about 43 us gross
vocabulary service rate                  about 14 us gross
                                         ------------------
gross losing territory                  about 158 us

less tinygrad fused-lifecycle wins       about 80 us
                                         ------------------
net measured debt                       about 73--76 us
```

The uniform-rate approximately-270-tok/s construction is not a claimable
roofline. It assigns every small stream the long-body rate and thereby erases
the measured ramp. The size-aware exact-streaming reconstruction returns
approximately the current weight time.

Residual flash is still the largest single losing region and is now roughly
equal to the endpoint gap. The next discriminator should explain why the wide
S8 score remains about 60 us/token behind llama under production conditioning,
while charging exact bytes, numerical order, output ownership, and token wall.

The priority-1 conditioning discriminator has since run. The exact installed
score is 4.536 us/layer hot and 6.184889 us/layer in production. Llama is
3.808--3.872 us/layer hot and 4.526 us/layer in production. The prior
"effectively tied" statement mixed tinygrad hot with llama production and is
withdrawn. Immediate Q/K/V producers add only 0.008 us/layer. Crossing the 96-MiB L2
capacity boundary and then running the exact local producer prefix reproduces
1.144 of the 1.644-us/layer production residual, or 41.184 us/token. This
identifies cache/working-set conditioning as the dominant mechanism, but books
zero recovery until a production residency/eviction policy passes token wall.

Aligned boundary accounting decomposes the 59.708-us/token production score
gap into 23.904--26.208 us/token of hot-body debt and 33.500--35.804 us/token
of kernel-to-production conversion debt. Thus conversion remains the majority
at 56--60%, but the hot bodies are not tied.

The matched llama conditioner closes the next branch. The identical 96-MiB
read stream costs llama S6 0.640 us/layer and tinygrad's installed S8 1.296
us/layer. Llama S8 costs 0.608 us/layer, ruling out six-versus-eight split
geometry as the explanation. Llama therefore experiences the same L2-capacity
knee but is about 2x less cold-sensitive. Matching that sensitivity exposes
23.616 us/token, a ceiling of about 245.65 tok/s, with zero recovery booked.
The open discriminator is now K/V allocation/stride/address color versus
load-address service for equal bytes, measured hot and cold with counters.

The full discriminator is now closed. The primary structural contributor is
excess physical horizon work: installed S8 reads and executes the empty upper
partitions at d512. An exact S6/768 closed lease cuts DRAM/L2/L1 bytes and
instructions by about 25% and passes a 144-token reverse wall bracket at
-9.484 us/token (+0.566 tok/s), with identical token hashes. Gated loads,
separate K/V, and address-color variants are no-gos. Because the recovery is
below the 50-us booking threshold and S6 becomes invalid when Tc reaches 769,
the then-current endpoint remained 4.094502 ms/token / 244.230 tok/s. This row
is superseded by the installed 246.274-tok/s authority above. The generic selector
has now been tested below and closes this translation path.

The graph-bucketing investment is now conversion-closed. Full-window S5 and
late-S6 candidates passed at -5.408 and -20.299 us/token respectively, while
S7 regressed 3.797 us/token. Those primitives suggested S6 through Tc 768 and
S8 afterward, but the typed selector does not convert. Cold execution hits a
deterministic lazy-S8 capture stall at the transition. Prewarming both S6 and
S8 greedy ping-pong pairs removes the stall, preserves the token stream, and
then regresses 10.676 us/token (-0.634 tok/s) versus the control midpoint. The
earlier +0.445 tok/s full-continuation estimate is not booked or actionable.

The exact pre-Flash hop ledger now locates the production conditioning event.
Gate/up alone leaves the reheated score at 4.768 us/layer with effectively no
target DRAM reads. Adding the prior layer's FFN-down moves the score to 5.600
us/layer, creates about 3.692 MB of target DRAM reads, and drops the L2 read
hit rate to 78.79%. The shared provider is neutral; Q adds 0.272 us/layer;
paired K/V and the completion hops are neutral or slightly reheating. The full
entry chain is 1.200 us/layer, or 43.200 us/token, above hot.

The cause and charge therefore occur at different hops: FFN-down crosses the
L2 capacity neighborhood, while the next Flash score pays for the displaced
historical K/V. Matching llama's retained S8 conditioning penalty leaves a
narrower 21.312-us/token excess-cold-sensitivity ceiling, worth at most about
245.51 tok/s from the current endpoint. Zero is booked. The next admissible
construction is line/reuse-aware FFN-down streaming or K/V protection, not a
repeat of the failed whole-dense eviction policy.

The matched llama production-order replay now separates residency from paid
service. Llama's K/V target begins refetching about 3.18 MB from DRAM after the
FFN prefix, yet that hop adds only 0.128 us. Q later adds 0.832 us with almost
no additional target DRAM traffic, and Q completion returns 0.192 us. A pure
capacity sweep is flat, crosses a sharp knee between 90 and 92 MiB, and then
plateaus through 108 MiB. This is ordinary cache replacement, not an explicit
one-to-one clear. Against the exact full prefix, tinygrad's remaining excess is
0.464 us/layer, or 16.704 us/token, for a 245.230-tok/s ceiling. It is not
booked: the llama small completion hops are modeled, and the next construction
must lower both target DRAM reads and paid target time.

Direct NCU on llama's actual PDL-off graph confirms that it reaches Flash cold:
3.166 MB of DRAM reads and a 75.58% L2 hit rate. Its corrected full-prefix
replay reproduces that state and 850,944 dynamic instructions. Llama still
falls 16.9--18.9% from hot to production; tinygrad falls 36.35%. Matching only
that conversion difference is a 246.245--246.384-tok/s ceiling, with zero
booked.

The first cold-body compiler test is now mechanism-closed. Program-scoped fast
math removes 5.65% of matched S8 instructions and one register, but leaves
about 4.23 MB of cold DRAM traffic unchanged and long-scoreboard stalls
dominant. The cold NCU body improves only 0.47%, versus 4.05% hot. Fresh
production graph profiles nevertheless reproduce a 4.976-us/token score-body
reduction and a 5.504-us/token total-node reduction with effectively no
overlap. Two unprofiled token brackets cannot resolve that roughly 0.12%
endpoint signal: the tighter bracket's candidate flanks differ by 5.786 us.
This is classified as a proven body/graph win behind a resolution wall, not
booked recovery. Its exact-translation ceiling is 244.53--244.56 tok/s, and
promotion additionally requires a general dense-model numerical/quality
contract because fast math is not bit-exact floating-point execution.

## Complete Flash lifecycle authority

### Ceiling refresh

The earlier S8-only Flash accounting is superseded for the installed dense
d512 endpoint by the ceiling campaign in
`docs/task_workflow/output/nv-flash-ceiling-exhaustion-result.md`.

The current endpoint is 4060.523 us/token / 246.274 tok/s versus llama at
4021.721 us/token / 248.711 tok/s.  Automatic S6-through-Tc768 selection now
passes both the transition and canonical fixed-window gates when both S6/S8
ping-pong pairs are captured ahead of steady decode.  S8 register-broadcast
combine is booked; the same spelling is explicitly not booked on S6 because
it reverses under the real distinct-graph capture topology.

The installed S6 Flash body is 194.048 us score plus 48.448 us combine.  Llama
is 162.948 plus 37.057 us.  Device overlap is effectively zero, so ordinary
PDL cannot hide the 42.491-us gross body debt.  The normalized QG ownership
sweep does not open a lever: QG2 is exact but slower, while QG4 is non-exact
for a sub-microsecond isolated movement.  The remaining open construction
class is equal-byte cold score service through real K/V sharing or line-aware
producer/cache admission.

The score row is no longer treated as an isolated black box. The complete
producer-to-O lifecycle, including graph horizon selection, Q/K/V readiness,
KV cache ownership, score geometry, partial ABI, combine, scheduling/PDL,
working-set conditioning, and every closed construction is recorded in
`docs/task_workflow/output/nv-flash-complete-lifecycle.md`.

Its accounting conclusion is narrower than the old gross 72.923-us
score/combine difference.  The installed S6 route reduces that gross Flash
debt to 42.491 us, while tinygrad remains faster in parts of the fused Q/K
completion and KV-store lifecycle.  Score-to-combine and combine-to-O device
overlap is effectively zero.  The active-horizon construction is booked; the
next work must derive a new equal-horizon cold-service mechanism rather than
repeat a closed geometry or PDL spelling.

## Evidence

- `docs/task_workflow/evidence/nv-ranked-wall-tests-20260826/post-landing-endpoint.json`
- `docs/task_workflow/evidence/nv-ranked-wall-tests-20260826/post-landing-ledger.json`
- `docs/task_workflow/evidence/nv-flash-causal-reopen/post-wide-installed-endpoint-r9.json`
- `docs/task_workflow/evidence/nv-flash-causal-reopen/post-wide-installed-ledger.json`
- `docs/task_workflow/evidence/nv-token-lifecycle-vs-llama-20260825/lifecycle-roles.json`
- `docs/task_workflow/output/nv-genuine-llama-gap-audit-result.md`
- `docs/task_workflow/output/nv-exact-streaming-service-rate-campaign.md`
- `docs/task_workflow/output/nv-byte-and-topology-wall-audit-20260825.md`
- `docs/task_workflow/output/nv-qo-service-rate-admissibility-result.md`
- `docs/task_workflow/output/nv-ranked-wall-tests-result.md`
- `docs/task_workflow/output/nv-flash-wide-production-conditioning-result.md`
- `docs/task_workflow/output/nv-flash-active-horizon-result.md`
- `docs/task_workflow/output/nv-flash-active-horizon-selector-result.md`
- `docs/task_workflow/output/nv-flash-entry-hop-ledger-result.md`
- `docs/task_workflow/output/nv-flash-entry-hop-vs-llama-result.md`
- `docs/task_workflow/output/nv-flash-kernel-to-production-conversion-result.md`
- `docs/task_workflow/output/nv-flash-fast-math-result.md`
- `docs/task_workflow/output/nv-flash-complete-lifecycle.md`
