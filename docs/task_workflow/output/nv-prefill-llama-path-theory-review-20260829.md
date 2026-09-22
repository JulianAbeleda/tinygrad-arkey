# NVIDIA pp512 llama-path theory review

Date: 2026-08-29. Review status: **test-then-invest recommendation**.

## Executive decision

The current authority is the H0.2 tinygrad route at **67.338268 ms median**
versus pinned llama.cpp at **35.019399 ms median**, a **32.318869 ms** gap.
The tinygrad result passes token 198, exact deep-20 replay, and the exact
198-role route census with zero unknowns. It is not the 84.025 ms FP16-overlay
route analyzed by the original lifecycle audit.

A packed Q4/Q6 x Q8 INT8-MMA primitive triangle is still the best first test,
but for a different reason than the old audit gave. Tinygrad has already built
and admitted packed Q4 x Q8 signed-IMMA for 198 Q4 roles. The useful question
is no longer whether NVIDIA INT8 tensor cores or a Q4 contract exist. The
triangle must now answer three narrower questions:

1. Can packed Q4/Q6 remove the still-visible FP16 service in all 36 down roles
   and the 18 Q6 V roles?
2. On roles that are already packed, how much of llama's remaining advantage
   is in its kernel/ownership implementation rather than representation?
3. Does stream-K itself win against tinygrad's current direct-output geometry,
   or is it only one way llama publishes the same result?

The first investment candidate is therefore the exact down lifecycle, not a
generic new INT8 substrate. The second is a matched static/native attribution
control for the already-installed Q4 body. A broad stream-K port, another B
schedule sweep, production integration, and attention work should wait for
those results.

## Authority and time boundary

| Evidence | What it establishes | Temporal limitation |
| --- | --- | --- |
| `nv-prefill-h0.2-parity-ledger-20260829.{md,json}` | Current 67.338268 ms tinygrad authority; 35.019399 ms llama authority; correctness and exact 198-role census | This is the performance and route authority for this review |
| `nv-prefill-exact-cross-runtime-accounting.md` and exact trace artifacts | A fully classified 69.492 ms tinygrad trace and 35.019 ms llama settled authority; traced regional debts; zero unknown launches | The tinygrad trace predates the 67.338 ms H0.2 wall and cannot be scaled onto it; its region debts are localization, not causal recovery |
| `nv-llama-prefill-lifecycle-audit.md` | Pinned llama representation, Q8 producer, MMQ, stream-K/fixup, and final-row model; older FP16-overlay comparison | Its 84.025 ms tinygrad route predates the current packed route. Its claim that tinygrad streams FP16 weights applies only to roles still on overlays |
| `nv-prefill-q4-imma-gate-result-20260828.md` | Pinned llama Q4 gate lifecycle and signed-IMMA evidence | Its statement that tinygrad lacks NVIDIA signed-int8 tensor-core lowering is obsolete. Current `tc.py`, CUDA lowering, and the 198-role route prove that substrate now exists |
| `nv-prefill-gate-up-service-audit-20260829.md` | Matched gate/up NCU bridge: identical useful IMMA count, similar L2/shared traffic, but current generated Q4 body is 1.87x slower with lower tensor/issue duty and more long-scoreboard pressure | It is a primitive/counter attribution, not a whole-wall rollback bracket |
| H0.2 B/C/D/F ledger | Current B variants STOP; C aggregate correctness but missing stage observability; D and F STOP | These STOPs must not be repackaged as fresh leading theories |
| D-L0 lifecycle pilot | Observer overhead 0.4779%; nine timed submissions; zero allocation, copyin, copyout, graph-copy, or materialization events | It falsifies recurring lifecycle allocation/copy cost. It does not measure Q6 arithmetic service |

The exact-accounting tinygrad role debts remain useful upper bounds in its
traced execution: down 12.067 ms, gate/up 13.570 ms combined, V 5.327 ms,
Q/O 4.058 ms combined, vocabulary 2.608 ms, Flash 1.671 ms, and K 0.959 ms.
They are not additive recovery forecasts for 67.338 ms.

## What is proven on the pinned llama side

The following are direct trace, source-map, source, or binary facts for the
pinned llama build and settled pp512 repetition:

- The settled unprofiled median is 35.019399 ms after excluding only the first
  graph-setup repetition by a declared lifecycle-state rule. The selected
  profiled wall closes exactly as 32.683341 ms device union, 0.201442 ms idle,
  and 2.545300 ms boundary residual.
- All 1,186 launches are classified with zero unknowns. The role parser checks
  source-defined graph order plus template, grid, block, graph-node sequence,
  and the 36-layer census.
- Full-batch Q, K, V, O, gate, up, and down projections consume resident Q4_K
  or Q6_K weights through Q8_1 activation records and signed INT8 tensor-core
  MMA. The Q4 gate binary evidence is signed `IMMA.16832.S8.S8`, not HMMA or
  DP4A. The older blanket count of 512 instructions was corrected by the Q4
  gate result: the captured `bool=false` specialization contains 256. Signed
  IMMA use is proven; a single instruction count must not be generalized
  across all specializations without the exact cubin.
- `quantize_mmq_q8_1` reads float4 values, warp-reduces absolute maxima and,
  for the layouts that need it, sums, rounds to signed int8, and publishes
  packed Q8 values plus layout-specific scale/sum metadata.
- The captured full-batch MMQ mains launch 170 CTAs of 256 threads. The source
  planner chooses an SM-sized stream-K grid when ordinary tiling is inefficient,
  partitions a continuous output-tile/K work space, writes incomplete tiles to
  scratch, and conditionally launches a fixup. The exact trace has 249 dense
  fixups, one for every full-batch dense main.
- The MMQ launch path uses ordinary global/shared accesses and barriers. The
  pinned MMQ mechanism does not require the repository's separate cp.async
  path or TMA.
- Full M=512 FFNs execute for layers 0-34. After layer 35 O, three gather/add
  launches retain one requested row; layer 35 then runs an M=1 norm, fused
  gate/up MMVQ, down MMVQ, final norm, and Q6 vocabulary MMVQ.
- The pinned Flash path is a tiled main plus stream-K reduction/fixup. Its
  traced active time is 1.657447 ms. This is not the failed tinygrad F vector
  topology.

## What remains inferred on the llama side

- The isolated value of stream-K versus an equally tuned direct-output MMQ is
  not measured. A 170-CTA grid and fixup are proven; their causal speedup is
  not.
- The older analytic packed-byte rates are not physical DRAM measurements.
  They support representation as the old FP16 route's dominant difference,
  but they do not explain the already-packed current Q4 route.
- Regional active-time differences do not assign recoverable current wall
  time. Only a current matched population test followed by a whole-model
  rollback bracket can do that.
- Final-row pruning is structurally proven in llama, but its tinygrad recovery
  is still an estimate until the current route is bracketed.
- The source absence of cp.async/TMA plus the trace is sufficient to reject
  them as required llama mechanisms. It does not prove that no unrelated
  compiler-generated load instruction anywhere in the binary has async-like
  behavior.

## Ownership and lifecycle comparison

| Dimension | Pinned llama pp512 | Current tinygrad 67.338 ms route | Review consequence |
| --- | --- | --- | --- |
| Weight representation | All full-batch dense roles retain Q4_K/Q6_K packed weights | 198 Q4 roles use canonical packed Q4_K weights; 54 V/down roles retain FP16 overlays: 18 Q6 V plus all 36 down roles | The old representation thesis survives mainly in those 54 roles, not across the whole graph |
| Activation provider | Production input is F32; each projection creates its required Q8_1 D4/DS4/D2S6 record | Each admitted Q4 projection starts from FP16 and uses a native compact Q8 producer with int8 values plus FP32 K32 scales and sums; overlay roles use FP16 directly | Provider cost and provider input dtype must be explicit in any comparison |
| CTA output ownership | Nominal 128x128 MMQ tile, eight warps, with the continuous tile/K space partitioned over 170 main CTAs | Gate/up and Q/O use 128x128x64 direct-output tiles; gate/up launch 384 CTAs and Q/O 128. K and serialized Q4 V use 64x32x64 direct-output geometry with 256 CTAs. No current main uses partial ownership | The old 32-CTA K underfill description is stale for current K and Q4 V |
| K partitioning | Stream-K splits K when selected by the planner; partial tiles are fixed up | Every current admitted Q4 CTA owns full K for its output tile. Research Q6 bindings are also direct-output and full-K | Stream-K is a hypothesis to test against current geometry, not a feature to assume |
| Integer accumulation and correction | Signed-int8 MMA produces integer subtotals; Q8 and weight scales plus Q4 minimum or Q6 scale grammar update FP32 sums | Q4 uses one K32 integer dot plus typed FP32 scale/minimum correction. Q6 research uses separate low/high K16 subtotals for the two Q6 scales in a K32 IMMA step. Metadata half round-trips are part of the current contract | Mathematical grammar is already expressible; performance, coverage, and ownership are the open issues |
| Publication/fixup | Complete tiles write FP32 output; incomplete tiles write scratch; a conditional fixup adds partials into FP32 output | Current Q4 routes allocate graph-owned Q8/output buffers and publish FP32 directly. H0.2 has zero old fixups and zero partial workspace. Q4 V uses a serialized direct-output cubin | Llama fixup cost must be included, but copying its fixup is not automatically an improvement |
| Final-row pruning | Gather occurs after final attention/O; final FFN and vocabulary are M=1 | H0.2 executes 72 gate/up mains, proving no final-row prune in the authority. A default-off terminal hook exists but is not part of the authority | This remains a small independent whole-graph test |
| Whole-graph admission | Production Qwen graph, one settled graph replay, 35 full FFNs plus M=1 tail | Default-off research composition with exact environment, compiler identities, safe cut, 198 Q8 producers, 198 mains, canonical weights, zero unknowns, and deep replay | Standalone wins are not admission. Population and whole-graph gates remain mandatory |

## Theories that are already closed or demoted

- **Recurring allocation/copy/materialization cost: STOP.** D-L0 observes all
  nine timed submissions with low perturbation and sees zero such events.
- **Generic overlap, launch-count reduction, or queue overlap as the primary
  gap: STOP.** The traced device is busy, idle debt is about 0.1 ms, and llama
  wins while launching more kernels.
- **cp.async or TMA as the leading mechanism: STOP.** They are not required by
  the fast pinned llama MMQ path.
- **The current B fragment/metadata/double-buffer schedule family: STOP.** The
  variants do not clear the 10 us noise floor or move the required counters.
  Theory 2 below is an attribution control between generated, matched static,
  and llama ownership, not another B parameter sweep.
- **The current F vector Flash topology: STOP.** It is correct but about 65x
  slower per call. No attention investment is ranked in this review.
- **Q4-down stage observability as a performance result: not established.** C
  proves aggregate numerical correctness for 18 roles, but unobservable
  producer/weight/dot/correction/epilogue stages cannot identify a removable
  cost. A complete standalone lifecycle timing can still identify a down
  lever without claiming stage localization.
- **Missing NVIDIA signed-int8 tensor-core substrate: obsolete.** That was true
  for the 20260828 Q4 gate. Current tinygrad has the signed-int8 descriptor,
  CUDA rendering, Q4/Q6 typed providers, correction contracts, and executed
  signed-IMMA routes.
- **Historical 32-CTA K/V underfill as a current global diagnosis: obsolete.**
  Current K and Q4 V use 256 direct-output CTAs. Only remaining FP16 V and a
  matched stream-K comparison keep topology open.

## Required primitive triangle

The triangle should be a fixture protocol, not a production route. Its first
population is one canonical role for each distinct shape/format, in this
order: Q4 down, Q6 down, current Q4 gate, Q6 V, Q4 V, Q/O, and K. A test may
stop after any format/shape gives a decisive result.

### Arm A: current installed tinygrad

Use the exact H0.2-installed asset for roles it admits, including its native
compact-Q8 producer and direct FP32 publication. For Q6 V and all down roles,
Arm A is the actual current FP16-overlay projection. Do not substitute a
logical route spec or an uninstalled Q6 research binding for the installed
arm. Freeze compiler identity, launch, canonical weight base, and generated
source/cubin identity in the record.

### Arm B: matched native/static control

Use this arm only where it resolves an attribution question. For an
already-packed role, implement the same Q4/Q6 correction grammar, same Q8
record ABI, same direct-output geometry, same output dtype, and the same
full-K ownership as Arm A in a static/native CUDA body. It answers whether
compiler lowering and instruction scheduling are leaving a large service gap.
For a remaining overlay role, Arm B is a direct packed Q4/Q6 body using the
existing typed grammar; it tests representation coverage before any model
integration. It must not introduce stream-K and then call itself a matched
static control.

### Arm C: llama-extracted lifecycle

Extract the pinned `quantize_mmq_q8_1` specialization, exact Q4_K or Q6_K MMQ
main, its planner decision for the fixture, and the conditional fixup. Retain
the pinned 170-CTA decision only when the planner selects it. Freeze source
commit, specialization, grid/block/shared/register facts, and packed layout.
This arm ends only when the final FP32 output is published.

### Accounting rules that prevent an apples-to-oranges result

- Use a captured production activation as the mathematical source. Keep a
  resident FP16 copy for tinygrad-native providers and an exact resident F32
  cast for llama-native providers. Report the provider input dtype and bytes;
  do not silently charge an F32-to-F16 or F16-to-F32 setup to one arm.
- Keep canonical packed weights resident outside the timed interval. Keep the
  current FP16 overlay resident for Arm A where that is the installed model
  representation. One-time GGUF extraction, dequantization, compilation,
  graph capture, pool growth, and fixture copies are outside every interval.
- Publish two separate ledgers. `provider-inclusive` starts before the arm's
  production activation provider and ends at complete FP32 output, including
  any fixup. `main-only` starts from a prebuilt, format-correct Q8 record and
  is diagnostic only. Never add a main-only delta to a provider-inclusive
  recovery claim.
- Preallocate scratch for all arms. Include every scratch write, fixup kernel,
  final add, and output cast required by that arm. A direct-output arm has no
  synthetic fixup charge; a stream-K arm cannot omit its real fixup.
- Use identical M, N, K, canonical weight tensor, output layout, cache order,
  synchronization boundary, hot R9, and rotated-cold R9. Do not mix llama's
  layer-35 M=1 tail into an M=512 projection comparison.
- Correctness is relative to an independent per-arm semantic oracle: FP16
  overlay semantics for the current FP16 arm and the declared Q8 quantized
  semantics for packed arms. Also report every packed arm's error against a
  common FP32 dequantized reference. Bit identity between FP16 and Q8 is not a
  valid requirement.

Passing fixture construction, SASS identity, readonly inputs, complete output,
or exact integer-dot canary only validates substrate. A lever is identified
only when the complete provider-inclusive lifecycle wins a declared timing
gate.

## Ranked live theories

### 1. Complete packed down lifecycles are the largest untested wall lever

**Exact causal claim.** Replacing the 18 Q4 and 18 Q6 full-batch down FP16
overlay calls with complete Q4/Q6 x Q8 signed-IMMA lifecycles will remove a
material part of the current wall because down remains the largest traced
service debt and none of those 36 roles is in the H0.2 198-role packed census.

**Why supported.** The exact trace localizes 12.067 ms of down active-time debt.
The current census leaves all 36 down overlays live. Llama executes 35 packed
full-batch downs plus one M=1 down. The Q4 aggregate numerical path and Q6 typed
K16-pair correction contract already exist. D-L0 removes recurring
allocation/copy/materialization as the explanation, concentrating the test on
provider plus arithmetic plus publication service.

**Smallest cheap discriminator.** Run the provider-inclusive triangle at
`(M,N,K)=(512,4096,12288)` on one canonical Q4 down and one canonical Q6 down.
No production stage ABI and no model attachment are required.

**Fixture/population.** Two captured real down activations, their exact GGUF
weights, current resident FP16 overlays, canonical Q4/Q6 storage, and
independent FP32/Q8 references. If either format passes, expand only that
format to its exact 18-role population.

**Correctness/census gate.** Finite full FP32 output, zero unwritten sentinel,
readonly activation/weight, exact shape and weight identity, format-correct
Q8 oracle within a declared error budget, common-reference error reported,
and 18/18 unique canonical weights for each promoted format. Population replay
must preserve token 198 and exact deep-20.

**Timing/counter gate.** Hot and rotated-cold synchronized R9 over the full
provider/main/fixup lifecycle. Require both median and minimum to improve by at
least 20% versus the installed per-format control and by more than 10 us/call.
Capture tensor duty, issue duty, eligible warps, long scoreboard, total
instructions, L2 bytes, and spills for attribution.

**PASS signal.** At least one format wins its complete lifecycle and its
18-role population, with a population-weighted measured saving of at least
2 ms. A combined Q4+Q6 result plausibly worth 4 ms or more is strong
authorization for whole-graph integration.

**STOP signal.** Correct main-only IMMA without a provider-inclusive win;
improvement at or below noise; spill/local traffic; incomplete population;
or any token/deep/census failure. Stage observability alone is not PASS.

**Blast radius if PASS.** Medium for one format because typed providers and
bindings exist; high if both formats require new role-specific geometry or a
new publication scheme. Whole-model activation-semantic qualification remains
mandatory.

**Dependencies.** Triangle fixture, existing Q4/Q6 typed contracts, canonical
down captures, and an independent oracle. No production integration,
lifecycle observer project, or stream-K substrate is a prerequisite.

### 2. The installed Q4 body has a kernel-implementation service gap beyond representation

**Exact causal claim.** On already-packed Q4 roles, a material share of the
llama gap comes from the current generated body's instruction/latency and CTA
ownership implementation, not from packed representation or extra matrix
work. A matched native/static direct-output body can determine whether the
recoverable layer is tinygrad lowering or llama's different MMQ organization.

**Why supported.** The matched gate/up bridge executes exactly 6,291,456 IMMA
instructions in both current tinygrad and llama and requests similar L2/shared
bytes. Tinygrad is 409.312 us versus 219.200 us, with tensor active 14.65%
versus 31.71%, issue active 37.44% versus 52.38%, 24.4% more total
instructions, fewer eligible warps, and 65% more long-scoreboard pressure.
The traced gate/up plus Q/O debt is large enough to matter. Existing B
fragment/metadata/double-buffer variations did not move the required signals,
so this is an attribution test, not another schedule variant.

**Smallest cheap discriminator.** Run all three arms on one canonical Q4 gate
at `(512,12288,4096)`. Arm B must preserve Arm A's 128x128x64, 384-CTA,
direct-output ownership and correction grammar. If Arm B separates from Arm A,
repeat on one Q/O role at `(512,4096,4096)` before any compiler work.

**Fixture/population.** One real gate activation/weight first; then 72
gate/up and 72 Q/O roles only for a passing mechanism. K and V are excluded
from this packet because their geometry is different.

**Correctness/census gate.** Generated and matched-static outputs must be
bit-identical or meet the exact typed-grammar tolerance with zero sentinels,
readonly inputs, identical Q8 record, same canonical weight, no spill, and
exact IMMA work. The llama arm uses its own Q8 semantic oracle. Population
requires 144/144 mains and producers with unique canonical weights.

**Timing/counter gate.** Matched hot/cold R9 plus one NCU pass. Require a
greater-than-10-us and at least 15% provider-inclusive separation, together
with one causal counter movement: total instructions down by at least 10%,
tensor or issue duty up by at least 5 percentage points, or long-scoreboard
pressure down by at least 15%, without higher L2 bytes or spill.

**PASS signal.** If Arm B beats Arm A, compiler/lowering organization is the
lever. If Arm B is near Arm A but Arm C remains materially faster, llama's
different ownership/kernel organization is the lever and only then should a
stream-K or persistent-CTA prototype be considered. Either conclusion must
win the exact role population before integration.

**STOP signal.** All matched arms converge; the separation disappears when
the producer and publication are included; counter changes do not clear the
declared thresholds; or only the already-falsified B variants reproduce the
gain.

**Blast radius if PASS.** Medium if a renderer/lowering instruction excess is
isolated. High if only llama ownership wins because that implies a new K-work
partition/publication primitive. Plausible wall leverage is roughly 5-12 ms,
but only a population bracket may assign it.

**Dependencies.** Triangle Arm B, frozen generated source/cubin identity, and
the existing matched NCU bridge. No model edit is required for the first test.

### 3. The remaining Q6 V half, not current K, is the live small-N topology lever

**Exact causal claim.** The 18 Q6 attention-V projections still on the FP16
overlay lose both packed representation and effective small-N work
distribution. A direct packed 256-CTA Q6 body may be sufficient; if it is not,
llama's 170-CTA stream-K lifecycle may show an additional ownership benefit.

**Why supported.** V has 5.327 ms of traced active debt. H0.2 packs only the
18 Q4 V tensors and leaves the 18 Q6 V tensors in the 54-overlay remainder.
Current K and Q4 V already use 256 direct-output CTAs, so the historical
32-CTA K/V diagnosis cannot justify a broad K rewrite. An existing Q6 V
research binding provides the exact 64x32x64, 256-CTA direct-output arm, while
llama supplies the distinct stream-K arm.

**Smallest cheap discriminator.** At `(512,1024,4096)`, compare installed Q6 V
FP16, the existing direct packed Q6 research binding, and llama-extracted Q6
MMQ with its real fixup. Add one current Q4 V control only to decide whether
stream-K helps after representation is already packed.

**Fixture/population.** One canonical Q6 V plus one Q4 V control; expand to the
exact 18 Q6 V population only after a provider-inclusive win.

**Correctness/census gate.** Full output, finite/no sentinel, readonly inputs,
correct K16-pair scale ownership, no spills, exact unique-weight census, token
198, and deep-20 exact replay after population attachment.

**Timing/counter gate.** Hot/cold R9 including producer and fixup. Require at
least 1.5x per-call speedup and more than 10 us/call versus installed Q6 V,
then at least 0.5 ms measured population improvement. Record achieved CTAs/SM,
tensor duty, issue duty, long scoreboard, instructions, L2 bytes, and fixup
share.

**PASS signal.** Direct packed 256 CTA wins: invest only in Q6 V admission.
Direct packed stalls but llama stream-K wins after fixup: authorize a
V-scoped ownership prototype. Q4 V also improves under stream-K: consider a
shared V topology, still not a generic all-role port.

**STOP signal.** Q6 packed provider-inclusive service does not beat FP16;
stream-K wins only main-only but loses after fixup; or the 18-role population
does not clear 0.5 ms.

**Blast radius if PASS.** Low-to-medium for direct Q6 V because the typed
binding exists. High for stream-K because current tinygrad has no partial
workspace/fixup in the admitted path. Plausible wall leverage is 1-4 ms.

**Dependencies.** Triangle fixture and existing Q6 V binding. Stream-K work is
dependent on the direct arm failing or the extracted arm showing a decisive
complete-lifecycle win.

### 4. A packed Q6 M=1 vocabulary head is a bounded independent tail win

**Exact causal claim.** The current M=1 vocabulary projection retains a much
slower service path than llama's Q6 MMVQ; routing only this head through a
resident packed Q6 vector kernel can recover a measurable tail cost without
changing full-batch projection ownership.

**Why supported.** The exact traced active times are 2.922 ms tinygrad and
0.313 ms llama, a 2.608 ms visible debt. The tail-work caveat is understood:
llama-bench stops after synchronized vocabulary while tinygrad additionally
runs a 0.009 ms argmax/token transfer. That tiny post-vocabulary mismatch does
not explain the projection debt.

**Smallest cheap discriminator.** Standalone triangle at
`(M,N,K)=(1,151936,4096)`: current installed head, a native/tinygrad packed Q6
MMVQ control, and the pinned llama MMVQ. Start from the same final-norm value
under each declared input dtype.

**Fixture/population.** One exact vocabulary weight and captured final-norm
activation. Population is one call, so a passing primitive can proceed
directly to a whole-tail graph bracket.

**Correctness/census gate.** Full 151,936-logit finite output, declared
Q6/Q8 error, same argmax token 198, readonly canonical weight, and exact one
head-main/one-provider census. Deep-20 follows before promotion.

**Timing/counter gate.** Synchronized hot/cold R9 for vocabulary projection
only, then whole-tail R9 with identical argmax/token work on both arms. Require
at least 1.5 ms primitive median recovery and at least 1.0 ms whole-wall
minimum and median recovery.

**PASS signal.** Both primitive and whole-tail thresholds pass with identical
token/deep result.

**STOP signal.** Only llama-bench's absent sampling work creates the apparent
win; the installed packed control cannot clear 0.5 ms whole wall; or full-logit
quality fails.

**Blast radius if PASS.** Low-to-medium: one terminal role and an existing
packed-vector problem class. Plausible wall leverage is 1-2.5 ms.

**Dependencies.** Exact tail fixture and equalized post-vocabulary boundary.
No full-batch MMQ or stream-K substrate is needed.

### 5. Final-row pruning is a small graph-liveness win available before broad kernel work

**Exact causal claim.** Gathering row 511 after final attention/O and before
the final FFN removes the 36th M=512 gate, up, and down work while preserving
the requested token, recovering roughly one millisecond on the current route.

**Why supported.** Llama's source and exact 1,186-launch trace prove 35 full
FFNs plus an M=1 tail. H0.2's 72 gate/up mains prove the authority is unpruned.
Tinygrad now contains a default-off terminal hook and a fail-closed gate, but
that hook has no H0.2 timing authority.

**Smallest cheap discriminator.** Use the existing research hook in a matched
whole-model control/candidate/control bracket. This is intentionally not a
primitive test, but it is small and isolated: no projection kernel changes.

**Fixture/population.** Exact Qwen3-8B pp512 token stream, current 198-role
route composition, row 511 only in the terminal block. Expected packed census
under the current Q4 V composition is 196 full-batch Q8/main calls rather than
198, plus explicit M=1 tail work; derive and freeze the final exact census
before running.

**Correctness/census gate.** Exact last-row logits, token 198, finite output,
exact deep-20, 35 M=512 FFNs, one M=1 FFN, no post-gather M=512 projection, and
the complete route/canonical-weight census with zero unknowns.

**Timing/counter gate.** Adjacent control/candidate/control synchronized R9.
Require candidate minimum and median below both controls by at least 0.75 ms,
with control drift below the declared noise ceiling.

**PASS signal.** Exact graph census and correctness plus the two-sided wall
threshold.

**STOP signal.** The hook silently falls back to dense work, fails composition
with Q4 V/Q/O, changes deep replay, or recovers less than 0.5 ms.

**Blast radius if PASS.** Low-to-medium graph plumbing with a narrow Qwen
requested-row admission rule. Plausible wall leverage is 0.75-1.5 ms.

**Dependencies.** Current route composition and a finalized expected census.
It does not depend on the primitive triangle, but should be timed after the
independent packets to avoid using a whole-graph change as primitive evidence.

## Why there is no sixth leading theory

The visible Flash debt in the exact trace is 1.671 ms, but the only current
tinygrad F vector candidate is already a decisive STOP. A llama-shaped tiled
Flash extraction would be a different topology, yet it has lower wall ceiling
and higher implementation blast than the down, packed-service, V, vocabulary,
and final-row tests above. It belongs after those packets, not in the first
investment queue. Generic overlap, cp.async, TMA, lifecycle work, and more B
schedule variants are closed rather than omitted.

## Serialized test order

1. Freeze the common triangle protocol and run only fixture/substrate checks.
   Do not interpret correctness, SASS capture, or stage visibility as a lever.
2. Run Theory 1 on one Q4 down and one Q6 down. Expand only a passing format to
   its exact 18-role population.
3. Run Theory 2 on one Q4 gate with the matched direct-output static control
   and llama extraction. Add Q/O only if the first result identifies a
   service mechanism; expand only then to gate/up and Q/O populations.
4. Run Theory 3 on one Q6 V plus one Q4 V control. Test stream-K only if the
   extracted complete lifecycle separates from direct packed ownership.
5. Run Theory 4 on the single vocabulary head and equalized tail boundary.
6. Run Theory 5 as the first whole-graph-only packet, with a frozen 196-role
   full-batch packed census plus explicit M=1 tail census.
7. Integrate exactly one population winner at a time. For each, run current
   control/candidate/rollback R9, token 198, deep-20, full role census, canonical
   weight identity, zero unknowns, and no duplicate timing charge before
   composing the next winner.

## Investment decision rule

Invest in implementation only when a provider-inclusive exact-shape test
clears its minimum and median gate, the same mechanism wins its complete real
role population, and the population-weighted measured saving is large enough
for its blast radius: at least 2 ms for a medium packed-role change, at least
4 ms for a high-blast ownership/stream-K substrate, or at least 0.75 ms for a
narrow graph/tail change. Correctness-only, counter-only, main-only, observer,
or stage-visibility PASS results validate substrate but do not authorize
production work. After integration, retain the change only if a fresh
control/candidate/rollback whole-model bracket passes both minimum and median,
token 198, exact deep-20, exact census, canonical weights, and zero unknowns;
otherwise book zero recovery and STOP that branch.
