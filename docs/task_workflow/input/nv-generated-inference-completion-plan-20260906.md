# NVIDIA generated inference completion plan

Date: 2026-09-06

## Goal

Run dense Qwen3-8B Q4_K_M decode and prefill through tinygrad/BoltBeam-generated
NV programs, with llama binaries used only as arithmetic and performance
oracles. The ordinary selected graph must contain no native-precompiled llama
programs. It must consume canonical packed GGUF weights, own all transient
storage, avoid hot-path weight copies or expansion, preserve full-logit and
recurrent correctness, expose explicit rollback, and reproduce llama parity or
better through the qualified context bands up to 4096 tokens.

## Frozen starting point

- Ordinary native pp512 is 34.890318 ms; fresh llama is 38.8192285 ms. The
  tinygrad graph/lifecycle can therefore beat the reference.
- The refreshed generated current252 authority is 47.777403 ms (shared-arm midpoint) with 252 generated projections and `--share-q-q4v`; the older frozen baseline was 50.746286 ms with 252 generated projections,
  generated vocabulary, canonical weights, zero V/down overlays, finite logits,
  token 198, and exact 20-cycle output replay.
- The generated pp512 gap is 11.9270575 ms to llama and 15.855968 ms to the
  ordinary native route. The latter is the substrate replacement budget.
- Generated decode has exact renderer/source ownership and effective d512
  endpoint parity (+0.113% latency versus bracketing llama controls), plus
  functional context coverage through 4096. Replication, cold-start accounting,
  and matched endpoint timing at the other context bands remain open.
- Gate/up main plus fixup is faithfully replayable at 271.616012 us per role.
  The 72-role generated lifecycle is 19.348799 ms versus 14.154342 ms native.
- Rejected gate mechanisms remain closed: intermediate unrolls, restrict,
  unsliced fixup, simple double buffering, fragment/shared-load reorder,
  accumulator interleave, boundary quantum 1, and paired K64 publication.

## Execution DAG

### M0: corpus authority and integration map

Inventory every retained winning generated primitive before designing another
kernel. Join the BoltBeam authority ledger, target-promotion policies, emitter
bindings, production resolvers, selected graph call sites, and current runtime
census by route and role. Classify each route separately as reproducible,
promoted, statically reachable, and observed at runtime. Historical performance
evidence remains valid only for its exact shape, ABI, graph, and measurement
protocol.

Exit gate: every claimed winner has one explicit disposition: already selected,
compatible but disconnected, closed by policy, superseded by correctness or
performance evidence, or incompatible with the current ABI. No new substrate
work begins for a role with a compatible disconnected winner.

### M1: restore compatible winners and build the exact role-delta ledger

Build one fresh-process route comparator that can independently choose native or
generated implementations for gate/up, Q, K, V by quant type, O, down by quant
type, Flash, vocabulary, and support epilogues without changing the graph API.
Every arm must use the same prompt, output contract, warmup, synchronization,
and R9 protocol. Record PROGRAM construction provenance and source transport.

First attach each compatible disconnected winner at the existing production
choke point and rerun its primitive, population, replay, and matched wall gates.
Only after those restorations, reconcile the native-to-generated difference.

Exit gate: restored winners and single-role substitutions plus the all-generated
arm reconcile the ordinary-native-to-current-generated difference within
control drift. No unassigned bucket may be called kernel or lifecycle debt.

### M2: missing generated packed-tile substrate

Implement a compiler-owned packed Q4/Q6 tensor-core tile contract rather than
continuing textual scalar schedule toggles. The contract must express:

- cooperative packed-weight and Q8 tile loading;
- packed scale/min reconstruction into tensor-core operands;
- multiple K panels with load/consume overlap;
- register-bounded accumulator groups;
- direct output or deterministic Stream-K partial publication;
- role-specific output strides and epilogues;
- generated SOURCE-to-cubin provenance.

The native algorithm is an oracle for geometry, useful work, and scheduling
signals. Its source or cubin is not a production dependency.

Exit gate: one real gate/up role is full-output correct, uses canonical packed
weights, has no copy/expansion, has no local spill, and beats the current
271.616012 us complete main/fixup boundary in two independent interleaved runs.

### M3: dense prefill roles

Apply M2 in measured-debt order from M1. Each role advances through one role,
full population, full model, and rollback brackets. Gate/up currently has a
known 5.194457 ms population deficit. Existing Q4-down and Q6-V measurements
remain inputs, but are re-run through the common comparator before promotion.

Exit gate: all 252 projections are generated and the generated dense-region
wall is no slower than the native dense-region control beyond measured noise.

### M4: generated attention and support lifecycle

Replace the native prefill Flash binary with a generated kernel using the exact
36-call logical contract. First close the remaining support semantic census by
source operation and buffer role. Implement only a support fusion or layout
change whose independently measured exposure exceeds noise.

Exit gate: zero native-precompiled PROGRAMs, zero unresolved support intervals,
full logits within `rtol=0.02, atol=0.5`, token 198, stable recurrent replay,
and generated pp512 median no slower than fresh llama.

### M5: ordinary selection and ownership

Replace the monolithic native/generated switch with qualified per-role leases.
Generated routes become ordinary only after their population and model gates
pass. Keep one explicit rollback for each role family. Test two models, two
captures, buffer non-aliasing, and failure on unsupported shape/device.

Exit gate: a clean process with no research enable variables selects only
generated programs and reproduces the qualified pp512 result.

### M6: decode closure

Repeat the promoted generated d512 bracket independently, measure cold compile,
first request, and steady replay separately, and run matched llama/tinygrad/llama
endpoint brackets at contexts 128, 256, 512, 1024, 2048, and 4096. Retain exact
program ownership and token/logit checks wherever prompts are shared.

Exit gate: parity or better at every qualified band, no native program, bounded
memory, no shadow captures, and an explicit rollback.

### M7: prefill context/depth closure and final promotion

Measure generated prefill at 128, 256, 512, 1024, 2048, and 4096 where memory
permits, and at layers 1, 6, 12, 18, 24, 30, and 36. Introduce measured geometry
bands only when one geometry fails a specific band. Run independent final
llama/tinygrad/llama replications and update the decode/prefill ledger.

Exit gate: generated ordinary decode and prefill meet or beat llama across the
declared bands with correctness, ownership, memory, cold-start, and rollback
evidence. Only then is the campaign complete.

## Operating rules

1. Rank work from matched end-to-end or complete-lifecycle deltas.
2. Change one mechanism per isolated experiment.
3. Reject on correctness before timing.
4. Require both minimum and median wins beyond noise before composition.
5. Commit and push every validated substrate, passing promotion, or durable
   negative result. Do not commit dead runtime switches.
6. Preserve raw samples, source/binary hashes, launch geometry, resource use,
   program census, and GPU state.
7. Never substitute the faster native default for generated completion.

## Immediate work

M0 is active. Generate the static corpus integration inventory, verify it against
the live program census, then attach the first compatible winner that is absent
from the selected graph. The initial audit finds that Q6 prefill stream-K down
already reaches the explicit current252 graph through a legacy research-module
binding, but the call path does not consume its route authority and ordinary
selection still chooses the native llama stack. Resolve these partial bindings
across the corpus before changing dispatch. Build new substrate only after all
compatible retained winners have been exhausted.
