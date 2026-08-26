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
| tinygrad, observed after extent-derived wide-flash promotion | **4.094502 ms/token** | **244.230 tok/s** |
| llama official authority | **4.021721 ms/token** | **248.711 tok/s** |
| tinygrad debt | **72.781 us/token** | **4.481 tok/s** |

The post-promotion tinygrad device node sum is 3,954.656 us. The retained
like-for-like llama PDL-off node sum is 3,878.254 us, leaving 76.402 us of
device-service debt. That is within about 4 us of the unprofiled endpoint gap.
The current loss therefore does not require a large unidentified host or
overlap term to balance.

The profiled tinygrad wall field is not an endpoint authority. Profiling adds a
large host/instrumentation term; only its device rows, node sum, union, and
route census are used here.

## Corrected lifecycle comparison

| lifecycle region | tinygrad | charged llama | TG - llama | current reading |
|---|---:|---:|---:|---|
| flash score | 222.656 us | 162.948 us | **+59.708 us** | residual production-conditioning debt |
| flash combine | 50.272 us | 37.057 us | **+13.215 us** | small residual body debt |
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

- gross tinygrad losing territory is about 158 us;
- tinygrad's fusion and lifecycle wins offset roughly 80 us;
- the resulting approximately 78-us net agrees with the measured 76.402-us
  device-node difference and 72.781-us endpoint difference.

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

The extent-derived wide-load route is now installed for NV sm_120 G4 fp16-KV
decode: S=MAXC/128 gives S8 on the official MAXC1024 graph. Its reps-9 reverse
bracket recovered 81.468 us/token with canonical token hashes, and the fresh
default-path endpoint is 4.094502 ms/token (244.230 tok/s). The remaining flash
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
score is 4.536 us/layer hot, effectively tied with llama's 4.526-us production
row. Immediate Q/K/V producers add only 0.008 us/layer. Crossing the 96-MiB L2
capacity boundary and then running the exact local producer prefix reproduces
1.144 of the 1.644-us/layer production residual, or 41.184 us/token. This
identifies cache/working-set conditioning as the dominant mechanism, but books
zero recovery until a production residency/eviction policy passes token wall.

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
the endpoint remains 4.094502 ms/token / 244.230 tok/s. Reopening requires a
generic active-horizon graph-bucket selector, not a fixed d512 split literal.

The graph-bucketing investment bracket is also closed. Full-window S5 and
late-S6 candidates passed at -5.408 and -20.299 us/token respectively, while
S7 regressed 3.797 us/token. Together with the prior early-S6 pass, the
measured policy is S6 through Tc 768 and S8 afterward, not a distinct graph for
every 128-token bucket. At the current endpoint this is worth about +0.57 tok/s
while the S6 graph is active and an estimated +0.445 tok/s when amortized over
a complete 512-token continuation to 1024. This remains unbooked until the
typed graph selector itself passes production wall; the existing alternate
Flash TinyJit selection provides the implementation substrate.

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
