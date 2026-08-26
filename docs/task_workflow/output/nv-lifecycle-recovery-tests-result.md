# Dense lifecycle recovery tests

## Outcome

Every ranked lifecycle row now has a causal or full-wall discriminator. The
initial native-lowering bracket for the 4096-wide attention, FFN, and output
norms reported 51.557 us/token with exact tokens; later reps=9 and profile
reconciliation supersede that estimate with a 12.317 us first-step booking,
followed by a separately qualified 62.995 us attention-completion win. Q/K native head
norms are neutral. Llama PDL is causally worth 125.196 us/token on the current
binary, proving that its overlap is real but accounts for only about half of
the current tinygrad-versus-llama wall gap.

## Recovery table

| rank | lever/test | measured result | token-wall recovery | disposition |
|---:|---|---|---:|---|
| 1 | llama PDL on/off/on | 4,139.424 us off versus 4,014.229 us on midpoint | 125.196 us in llama | causal mechanism pass |
| 2 | 4096-wide native norms | first bracket 51.557 us; confirmation/profile 12.317/12.250 us | **12.317 us first step** | promoted selectively |
| 3 | Q/O topology/rate | exact triple producer previously booked; known unroll and aggregation translations exhausted | 16.119 us booked | keep installed; distinct mechanism required |
| 4 | flash score | isolated tinygrad body faster; geometry and readiness wall tests neutral | 0 new | arithmetic rewrite closed; topology-only reopen |
| 5 | flash combine | 128-lane candidate reached wall and lost by 1.034 us | 0 | clean no-go |
| 6 | Q/K native head norms | 4.342436 / 4.343554 / 4.341088 ms | **-1.792 us** | clean no-go for native construction |
| 7 | K/V completion/ownership | pair/sink/triple campaign; current full triple books 16.119 us | 16.119 us booked | keep installed; tested topologies closed |
| 8 | vocabulary tail | native one-CTA argmax reverse bracket | 56.386 us booked | keep installed |

Previously booked rows are not added again to the current endpoint. They are
shown to distinguish successful mechanisms already present in production from
new recoverable headroom.

## Llama overlap causality

The current build was run with the runtime-supported `GGML_CUDA_PDL=1`, then
`=0`, then `=1`, using the same model, depth, generation count and r7 benchmark.

| arm | official latency | official throughput | settled median throughput |
|---|---:|---:|---:|
| PDL on A | 4,016.287 us/token | 249.035 tok/s | 250.401 tok/s |
| PDL off | 4,139.424 us/token | 241.634 tok/s | 243.080 tok/s |
| PDL on C | 4,012.170 us/token | 249.291 tok/s | 250.681 tok/s |

PDL therefore recovers 125.196 us/token against the on-arm midpoint, or about
7.53 tok/s at llama's endpoint. The matched profiled device spans are
4,018.852 us PDL-off and 3,896.695 us PDL-on, a 122.157 us reduction that
closely reproduces the unprofiled wall effect.

An anchor-start partition locates the change rather than pretending raw
concurrent kernel durations are independent service times:

| interval family across layers | PDL-off minus PDL-on |
|---|---:|
| Q start -> O start | +138.963 us |
| O start -> gate/up start | -41.192 us |
| gate/up start -> down start | -73.778 us |
| down start -> next-layer Q start | +94.968 us |

Positive rows shrink under PDL; negative rows expand because dependency wait
and residency move into later command intervals. The net device-span recovery,
not the positive rows alone, is admissible. This establishes the principle:
llama starts dependent work early and absorbs the wait while other useful work
is resident. Tinygrad cannot recover this by simply adding queues or broad PDL;
its prior exact broad-readiness tests were wall-neutral. A transferable version
must change resource compatibility or the physical producer/consumer body.

## New 4096-norm pass

The candidate marks only the attention, FFN and output 4096-wide RMSNorm sites
for the scheduler-owned native lowering. Q/K head norms remain on the installed
fused norm+RoPE/cache path.

| arm | median ms/token | timed tokens | token hash |
|---|---:|---:|---|
| control A | 4.381121 | 112 | identical |
| candidate | 4.311209 | 112 | identical |
| control C | 4.344413 | 112 | identical |

This first bracket was later found to overestimate the primitive-only win.
Matched profiling measured a 12.250 us device-union reduction and the reverse
candidate/control/candidate confirmation booked 12.317 us/token. The subsequent
attention completion is documented in `nv-native-4096-norm-promotion-result.md`;
it recovers another 62.995 us/token and crosses 240.

## Q/K head-norm closure

The independent Q/K native-lowering bracket preserved token identity but lost
1.792 us/token to the control midpoint. This closes only that native
construction. It also proves the earlier PDL-off role delta is not a separately
collectible 55 us pool: tinygrad's installed fused norm+RoPE/cache ownership
must remain intact.

## Corrected next action

Selectively promote native lowering only for 4096-wide attention, FFN and
output norms on NV sm_120. Do not use the existing all-norm promotion switch as
written, because it would also enable the now-negative Q/K sites. After adding
the selective policy, require:

1. route census proving the 4096 sites changed and Q/K fusion did not;
2. exact full-token hash qualification;
3. reps>=9 control/candidate/control bounded pre-cliff wall;
4. a fresh lifecycle ledger and tinygrad-versus-llama endpoint.

## Evidence

- `docs/task_workflow/evidence/nv-lifecycle-recovery-tests-20260826/llama-pdl-ab/`
- `docs/task_workflow/evidence/nv-lifecycle-recovery-tests-20260826/norm4096/`
- `docs/task_workflow/evidence/nv-lifecycle-recovery-tests-20260826/head-norm/`
- `docs/task_workflow/evidence/nv-token-lifecycle-vs-llama-20260825/llama-dag.json`
- `docs/task_workflow/output/nv-attention-token-lifecycle-reopen-result-20260824.md`
- `docs/task_workflow/output/nv-flash-vocab-result-20260824.md`
- `docs/task_workflow/output/nv-full-qkv-producer-substrate-campaign.md`
