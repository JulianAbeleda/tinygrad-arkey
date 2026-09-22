# Native NV decode parity implementation campaign

Date: 2026-08-05
Branch: `nvidia-bringup-20260731`
Workload: Qwen3-8B-Q4_K_M, RTX 5090 / driver 595.84
Status: ACTIVE IMPLEMENTATION SCOPE

## 1. Objective and terminal condition

Reach same-session native `DEV=NV` decode parity with llama.cpp at d512 while
preserving the model/output contract and tinygrad's maintainable primitive
route. The terminal gate is one same-session reverse bracket with:

- native NV midpoint `>= 1.00x` llama throughput at d512;
- full-logit agreement under the campaign's recorded tolerance and identical
  generated tokens;
- no debug/profiler timing in the wall authority;
- no unreviewed target promotion or environment-only result;
- d2048 and d4096 regression rows before any default flips;
- every accepted recovery assigned once to a disjoint implementation A/B.

If d512 parity is reached by interacting changes, the complete composed arm,
not the sum of isolated arms, is the authority.

## 2. Fixed authority and physical equation

The current same-session authority is:

| system | ms/token |
| --- | ---: |
| llama.cpp | 3.966140 |
| tinygrad native NV | 5.612310 |
| gap | **1.646170** |

The gap is already physically closed:

```text
(1108.082 support critical path
 +302.788 quantized cores
 -8.111 llama internal gaps
 -1.143 profile-to-device bridge)
+239.804933 outside-device delta
+4.749067 outer bridge
=1646.170000 us/token
```

The implementation campaign does not search for another bucket. It converts
these locations into correctness-passing native wall recovery.

## 3. Evidence and booking rules

Evidence classes, strongest first:

1. same-session native-NV composed real-token reverse A/B with full logits;
2. same-session native-NV family real-token reverse A/B with full logits;
3. included-cost native-NV primitive/graph microgate;
4. CUDA llama-kernel family substitution with exact outputs;
5. calibrated interval attribution, Shapley ownership, node sums, or ceilings.

Only classes 1-2 debit the parity remainder. Class 3 admits implementation to
a token test. Classes 4-5 choose direction and never debit native recovery.

```text
accepted_remainder_us = 1646.170
  - composed_native_recovery_us
```

Individual arm deltas are not added after a composed arm exists. Shapley rows,
llama hidden work, profiler node sums, independent role medians, and
profile-to-wall bridges are never recovery credits.

Every wall arm records clocks/thermal settling, command/environment, model
hash, first token, token stream, logits contract, program census, raw samples,
median, adjacent controls, and GPU lock provenance.

## 4. Ranked execution program

### P0 - correctness oracle and baseline lock

Expose a closed-default diagnostic full-logits output from the production
sampled graph. It may add a diagnostic output/tap but must not alter ordinary
sampling, feedback, buffer ownership, or the default graph.

Gate P0:

- baseline native token/logit pins reproduce;
- diagnostic OFF is byte-identical in topology and source;
- diagnostic ON exposes the same final logits used by sampling;
- repeated outputs are deterministic within the declared tolerance;
- one harness can gate all later family and composed A/Bs.

No production promotion occurs before P0 passes.

### P1 - productionize proven predispatch recovery

Integrate the two already-positive host constructions:

1. cache the settled structural descriptor while still discovering and
   realizing current tensors, binding fresh concrete buffers/variables, and
   failing closed on identity/rank/dtype/device mismatch;
2. reuse one private alias-safe feedback shadow per captured input slot while
   preserving every defensive copy and dependency edge.

Prior combined diagnostic: `-69.1655 us/token`, exact token stream.

Gate P1:

- P0 full logits pass;
- combined native reverse A/B reproduces `>= 40 us` recovery;
- cache hit/miss and changed-allocation tests pass;
- no stale concrete buffer or variable binding;
- no new graph programs or device-window regression;
- d2048/d4096 no regression.

Only the combined arm is booked. The individual 65.536 and 28.372 us point
estimates overlap and are not added.

### P2 - boundary-free scheduler-native fusion/dataflow substrate

This is the highest-leverage controllable substrate. It must solve the shared
cause behind norm and projection epilogue failures rather than adding another
opaque custom boundary.

Required capability:

- view-preserving lazy producer consumption;
- ordinary scheduler UOps or an equivalent path visible to HCQ graph capture;
- no unconditional `CONTIGUOUS`/copy at semantic boundaries;
- reusable output-buffer contract;
- fail-closed fallback for unsupported shapes/dtypes/routes;
- default unchanged until fixed-depth admission.

P2a RMSNorm:

- ordinary two-program device span authority: 4.250 us;
- existing one-body observation: about 3.07 us;
- current Path-3 failure: +110 programs/token and -0.9 to -1.3% wall.

Gate P2a: one graph-replayed program, zero new materializations, exact logits,
all applicable norm calls converted, d512 wall win, then d2048/d4096.

P2b projection epilogues, one disjoint family per arm:

1. FFN activation cast into fused gate/up;
2. FFN-down output cast + residual;
3. attention-O output cast + residual;
4. block-output contiguous into the next consumer/norm.

The native serialized ownership surface is 240.762 us, but it is a ceiling,
not promised recovery. Each family must shrink topology and pass a real-token
A/B. Repeating M4 custom transport, llama adapters, or KV-store fusion is
forbidden because those exact constructions already lost or were neutral.

### P3 - shared/amortized Q8 quantized consumers

The generic signed-int8x4 provider is available and reaches sm_120 DP4A. The
independent producer+consumer candidate lost Gate 1 by 1.172-1.352 us, so the
next construction must amortize production and match consumer layout.

P3a attention:

- prove exact common-producer/dataflow reuse across Q/V/K;
- quantize the shared activation once where legal;
- retain packed Q8 and scales across admitted consumers;
- implement native Q6 V/K consumers without partial-output + sum;
- include producer, layout, scale and consumer cost in every microgate.

Gate P3a-1: included-cost graph beats current partial4+sum with numerical
contract satisfied. Gate P3a-2: all 18 Q6 attention roles pass full-logit/token
native A/B and recover at least 50 us. The CUDA diagnostic range of 179-184 us
is direction, not the native gate.

P3b FFN-down:

- reuse P3a provider/layout rather than fork a new substrate;
- implement Q4 down first (CUDA causal signal 65.8-66.1 us);
- diagnose Q6 divergence at the known 2-to-4 replacement boundary using P0
  logits before scaling the family;
- never extrapolate the valid two-call 46.801 us local signal.

### P4 - concurrency and critical-path exposure

Exact llama-hidden support ownership is 445.954 us. This track runs in
parallel because it is likely parity-required but has a construction blocker.

P4a CUDA mechanism control:

- use the existing programmatic CUDA graph, which already co-schedules truly
  independent nodes;
- expose real Q/V/K or support/MMQ independence without changing values;
- do not rebuild the redundant explicit multi-stream capture lowerer;
- do not claim planner-alias removal as the root fix (measured CP impact about
  38.6 us).

P4b native RM/HCQ construction:

- reproduce an independently scheduled compute channel/queue using the
  driver's accepted RM sequence or a maintainable native abstraction;
- start with two independent decode-sized 2-5 us kernels;
- record exact failing RM operation if construction fails;
- a construction failure is not a hardware no-concurrency verdict.

Gate P4-1: exact numerics and at least 5% interval-union saving on the probe.
Gate P4-2: dependency-correct native token schedule and at least 50 us wall
recovery. Gate P4-3: composed wall under real bandwidth contention; an
unbounded-resource DAG simulation cannot promote.

### P5 - sampler, vocab, feedback and remaining host path

Only after P1-P4 gates settle:

- fuse the sampler tail and token feedback through ordinary graph UOps;
- keep feedback device-resident while preserving the required host token;
- remove the defensive D2D feedback copy only with an explicit alias/lifetime
  proof and A/B;
- optimize vocab only through an included-cost full-tail arm.

The support ownership is 71.215 us and the independently medianed vocab-core
direction is about 10.8 us. Treat them as separate evidence, not a composed
forecast.

### P6 - RoPE/KV tail

The remaining ownership is 33.543 us. Existing KV-store fusion was wall
neutral and the current fused Q6 compound role is only 27.402 us total. This
track is admitted only if the composed remainder remains above parity after
P1-P5.

## 5. Explicitly closed work

Do not repeat without new evidence:

- incremental flash tile/value/geometry sweeps;
- attention-Q or attention-O llama substitution adapters;
- native gate/up fusion (already fused and competitive);
- KV-store fusion;
- forced graph-group collapse (+378 us regression);
- inter-group launch-gap work (2.4 us measured);
- planner-alias removal as a parity-scale root (38.6 us CP impact);
- DP4A access without included Q8 production;
- RMSNorm body tuning before graph transport/materialization is fixed;
- opaque custom epilogues that add adapters, copies, or programs.

Flash's 247.989 us critical ownership is addressed first through P4 overlap.
Matched flash-body parity remains unproven, but current evidence does not
require a faster llama flash body to explain the wall gap.

## 6. Composition protocol

After any two P1-P5 families pass independently:

1. build A / family-X / family-Y / X+Y / A reverse brackets;
2. require full logits and identical tokens for every arm;
3. measure program census and device/outside windows;
4. use the composed delta, never `delta_X + delta_Y`;
5. update the residual ledger with interaction loss/gain;
6. revert or close any component whose composed contribution disappears.

Parity is published only from a final all-accepted-features arm against llama
in the same flocked session.

## 7. Artifact contract

Each phase produces:

- `docs/task_workflow/input/*-scope-or-record-20260805.md`;
- compact machine JSON under `docs/task_workflow/output/`;
- raw timing payload under `/tmp` with SHA256 in the record;
- focused hermetic tests;
- source/SASS/topology pins where applicable;
- a promotion/default decision of PASS, CLOSED, or BLOCKED_CONSTRUCTION.

No raw profiler database, API environment capture, cubin, or compiled probe is
committed. GPU work uses `flock /tmp/gpu-bench.lock`.

## 8. Ownership and integration

- Track H: P0/P1 full-logit oracle and host/runtime recovery.
- Track F: P2 boundary-free norm/epilogue substrate.
- Track Q: P3 shared-Q8 attention and FFN-down substrate.
- Coordinator: P4 concurrency, cross-track composition, P5/P6 admission,
  master residual, commit and push.

Tracks may not edit another track's primary implementation files without
coordinator handoff. Existing user modifications in `docs/README.md`,
`docs/beating-llama-first-principles-20260731.md`, and
`docs/what-makes-inference-fast.md` remain untouched.

## 9. Campaign stop conditions

Complete only when one condition holds:

1. **PARITY PASS:** same-session native d512 ratio `>=1.00`, correctness and
   depth gates pass, and the composed route is committed/pushed; or
2. **PROVEN BLOCKED:** every P1-P4 admissible construction is closed by a
   measured gate or a precise repeated construction blocker, the remaining
   maximum compatible recovery cannot reach parity, and the blocker is
   documented without a hardware impossibility claim.

Difficulty, a single failed implementation, or an ownership number without a
wall A/B is not a stop condition.
