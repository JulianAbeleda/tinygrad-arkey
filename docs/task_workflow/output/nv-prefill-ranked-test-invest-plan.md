# NVIDIA pp512 ranked test-to-invest plan

Status: scoped from the exact cross-runtime trace at commit `40dbc787b`.
This document ranks measured exposure separately from engineering order. It
does not convert traced percentages into an unprofiled recovery estimate.

## Decision

The packed compiler substrate is mostly present. NVIDIA can now consume
canonical Q4/Q6 weights, create compact Q8 activations, issue signed IMMA,
keep scale correction tile-local, preserve computed inputs through callify,
and replay the resulting graph without packed-weight copies. The admitted
Q4 graph already covers gate, up, K, Q, and O.

The remaining 34.697787-ms minimum wall gap is not primarily an idle-GPU or
host-submission problem. In the matched traces, tinygrad has only 0.105790 ms
more device idle than llama. The dominant remaining work is executed kernel
service.

There are three different kinds of work left:

1. **Substrate-ready service work:** gate/up, Q4 V, Q/O, and K can be tested
   using existing packed compiler assets.
2. **Small lifecycle work with a clear reference:** final-row pruning and the
   M=1 vocabulary path should be tested before another large compiler project.
3. **Substrate-first work:** Q6 down is arithmetically correct but loses in the
   model. It needs a better producer/K16/epilogue lifecycle before investment.

## Exact authority

| boundary | tinygrad | llama.cpp | measured gap |
|---|---:|---:|---:|
| unprofiled R9 minimum | 69.378154 ms | 34.680367 ms | 34.697787 ms |
| unprofiled settled median | 69.492178 ms | 35.019399 ms | 34.472779 ms |
| traced device union | 73.879808 ms | 32.683341 ms | 41.196467 ms |
| traced device idle | 0.307232 ms | 0.201442 ms | 0.105790 ms |

The unprofiled wall is the performance authority. The traced union explains
the instrumented runs only. It is useful for ranking exposure, not for
predicting a new token rate.

## Raw measured-debt ranking

This is the order of positive traced active-time debt. It is not yet an
investment order and it is not additive wall recovery.

| raw rank | region | tiny active | llama active | measured active debt | share of positive debt | present state |
|---:|---|---:|---:|---:|---:|---|
| 1 | gate + up | 25.768 ms | 12.198 ms | 13.570 ms | 32.3% | Q4 compiler substrate passes; service remains slow |
| 2 | down | 19.007 ms | 6.940 ms | 12.067 ms | 28.7% | 18 Q4 + 18 Q6; neither is in the winning graph |
| 3 | V | 6.381 ms | 1.054 ms | 5.327 ms | 12.7% | 18 Q4 + 18 Q6; both primitives pass, lifecycle does not yet pay |
| 4 | Q + O | 8.971 ms | 4.911 ms | 4.059 ms | 9.7% | compiler route passes; composed gain is small and requires a safe cut |
| 5 | vocabulary | 2.922 ms | 0.313 ms | 2.608 ms | 6.2% | M=1 Q6 asset likely reusable, prefill full-logit route unqualified |
| 6 | support | 3.119 ms | 1.379 ms | 1.739 ms | 4.1% | many small kernels; semantic source map incomplete |
| 7 | Flash | 3.328 ms | 1.657 ms | 1.671 ms | 4.0% | prefill attention topology/service gap |
| 8 | K | 2.211 ms | 1.251 ms | 0.959 ms | 2.3% | packed 256-CTA route passes; lower-priority residual service gap |

Tinygrad is already faster in normalization/conversion by 0.745287 ms and in
activation/multiply by 0.144963 ms in the traced executions. Those paths are
regression guards, not optimization targets.

## Engineering order

The work order differs from the raw ranking because early graph changes can
invalidate later population counts, and because some large debts require new
substrate.

| work rank | target | why it is here | substrate state | first decision |
|---:|---|---|---|---|
| 1 | final-layer requested-row prune | llama proves the graph law; it changes the last FFN population and should be settled before dense tuning | mostly ready | test graph spelling |
| 2 | vocabulary M=1 Q6 | large isolated tail debt; decode already has relevant packed Q6 assets | partial | qualify full-logit prefill contract |
| 3 | gate/up service | largest substrate-ready debt | ready | matched body and 72-role service audit |
| 4 | V, Q4 then Q6 | large debt and both primitives already exist; explains primitive-to-model mismatch | Q4 ready, Q6 arithmetic ready | separate format-specific lifecycle traces |
| 5 | Q/O service and queue policy | packed path is correct, but only a small composed wall gain survives | ready for kernel work; promotion substrate partial | geometry audit plus regenerated dependency cut |
| 6 | down, Q4 then Q6 | largest single debt, but format halves have different blockers | Q4 partial, Q6 substrate-first | Q4 role gate; Q6 lifecycle attribution |
| 7 | Flash prefill | real debt, but smaller than dense/tail and unrelated to decode Flash work | partial | matched score/reduction trace and oracle |
| 8 | K residual service | route already wins and has only 0.959 ms traced debt | ready | reuse findings from V/Q/O before a K-only campaign |
| 9 | support kernels | fragmented and not yet source-attributed | mapping missing | semantic naming/census only |

Queue-policy generalization and production-generic admission run across several
rows. They are promotion blockers, not reasons to delay isolated role tests.

## Global test -> test -> invest contract

Every target follows the same sequence.

### Test 1: exact primitive or graph-law discriminator

- Use canonical real model weights and legal activations.
- Require finite output, complete sentinel coverage, read-only inputs, and an
  independent oracle.
- Retain the exact binary/program identity, launch geometry, registers,
  shared memory, local traffic, and SASS census.
- For a graph-law change, compare the requested final hidden row, full logits,
  greedy token, and every state buffer that remains live.
- Keep the route default-off and isolated.

A Test-1 correctness failure is a stop. A performance failure is also a stop
for that spelling; it is not permission to integrate it and hope the graph
hides the cost.

### Test 2: exact population and lifecycle discriminator

- Run the complete real role population, not a scaled single-call estimate.
- Use fresh-process synchronized R9 and report minimum and median.
- Compare the exact call census, canonical packed-weight bases, compact-Q8
  records, copies, fixups, partial workspaces, and overlay population.
- Run same-input replay plus distinct-input freshness. Stateful attention arms
  require deep A/B/A replay with every KV slice checked.
- Trace the winning and control arms on their own clocks. Do not infer wall
  recovery from active-time percentages.

Only a correctness-qualified, repeatable population win advances to
investment.

### Invest: implement the smallest mechanism proven by the tests

- Change only the mechanism named by the discriminator.
- Preserve a direct rollback control in the same harness.
- Do not broaden model admission while the research route is being tuned.
- Re-run primitive correctness, population timing, and graph replay after each
  compiler, binding, or queue-policy change.

### Promotion gate

- Fresh candidate and control processes on the exact same commit.
- Synchronized R9 minimum and median both improve.
- Independent confirmation process.
- Full-vocabulary logits finite and within the declared tolerance; same token.
- Exact route census and zero unintended copies/expanded weights/workspace.
- Deep replay exactness for recurrent state.
- Exact graph-derived dependency policy; no stale digest reuse.
- Default remains off until every gate passes.

## Workstream 1: final-layer requested-row prune

### Measured fact

llama runs full M=512 FFNs for layers 0--34. After layer 35 attention it
gathers the requested row and runs the final FFN at M=1. Tinygrad runs the
layer-35 FFN at M=512.

Tinygrad's traced layer-35 full-FFN bodies are:

| body | active time |
|---|---:|
| gate | 0.355264 ms |
| up | 0.364928 ms |
| activation/multiply | 0.021216 ms |
| down | 0.528192 ms |
| total current FFN bodies | 1.269600 ms |

That 1.269600 ms is current traced exposure, not recoverable wall. The M=1
replacement is not free.

### Substrate state

- No new packed-fragment compiler substrate is required for the first test.
- The model needs a fail-closed requested-row graph spelling after final
  attention and before final FFN.
- M=1 gate/up/down may initially use existing correct vector routes. Packed M=1
  optimization is a second step, not a prerequisite for proving the graph law.

### Tests

1. Create an isolated arm: run full attention for layer 35, gather only the
   requested row, run its FFN at M=1, then execute the existing vocabulary
   path.
2. Compare the requested post-block hidden row, full vocabulary logits, token,
   and KV state with the unpruned control.
3. Require the census to show 35 M=512 FFNs and one M=1 FFN, with no hidden
   full-batch materialization after the gather.
4. Run fresh R9 and an independent confirmation.

### Invest/stop

- Invest in packed M=1 FFN routing only if the graph-law arm wins whole wall.
- Stop if gather/materialization plus M=1 service does not beat the exact
  unpruned control.

## Workstream 2: vocabulary M=1 Q6

### Measured fact

Both runtimes operate on one final row with shape `(1,151936,4096)`. Tinygrad
spends 2.921568 ms in the traced vocabulary projection; llama spends
0.313154 ms in its Q6 MMVQ plus 0.003040 ms in its vocabulary Q8 producer.

### Substrate state

- Relevant packed Q6 M=1 cubins and decode route logic already exist.
- Asset reuse is not yet proof that the prefill route is correct.
- The current prefill contract returns full logits and a token. A top-1-only
  decode kernel is not a legal replacement for callers requesting full logits.

### Tests

1. Full-logit gate: canonical Q6 vocabulary weight, one real final hidden row,
   packed M=1 output for all 151,936 logits, independent FP16/control compare,
   read-only and sentinel checks.
2. Exact lifecycle gate: producer + packed vocabulary main + existing argmax
   and token transfer, with the full-logit tensor still available.
3. Separate API gate for a greedy-only/top-1 route. Do not use that number as
   the full-logit result.
4. Compose with the final-row-pruned graph and rerun full-logit and R9 gates.

### Invest/stop

- Reuse the decode asset only after it passes the prefill full-logit contract.
- If full logits require a new packed output kernel, that kernel is the named
  missing substrate. Do not silently weaken the API to claim a win.

## Workstream 3: gate/up service

### Measured fact

Gate/up is the largest substrate-ready debt: 25.768 ms tinygrad active versus
12.198 ms llama active. The compiler route is already canonical packed Q4,
compact Q8, signed IMMA, tile-local correction, direct output, and zero packed
transport copies.

### What is not missing

- Q4 nibble decode and metadata ownership;
- compact-Q8 record ABI;
- signed-int8 tensor-core lowering;
- direct compiler-owned output;
- canonical packed-weight identity;
- callify/TinyJit computed-input ownership.

### Missing knowledge

The remaining question is executed service: issue schedule, fragment reuse,
barrier cadence, CTA ownership, register pressure, and cold packed-weight
delivery. Static IMMA/LDSM counts alone have already failed to explain the
llama lead.

### Tests

1. Capture the current generated gate and up programs and matched llama Q4
   MMQ roles on the same real weight/activation shapes.
2. Run hot and rotated-cold R9 for one projection and the exact 72-role
   population.
3. Use a CUDA-launchable matched cubin gate for NCU physical counters because
   external CUPTI does not observe native HCQ launches. Measure executed
   instructions, tensor duty, long scoreboard, L2/DRAM bytes, barriers,
   occupancy, and issue slots.
4. Sweep one variable at a time: output tile, K64/K128 service interval,
   CTA count/Stream-K ownership, warp ownership, metadata staging, and
   unroll/register point. Every candidate must pass the full real-output oracle.
5. Put only the best population winner into the whole graph and regenerate the
   queue cut before R9/deep replay.

### Invest/stop

- Invest in the compiler scheduler/topology only after a counter-backed
  candidate wins both cold role service and the 72-role proxy.
- Stop variants that win hot instruction count but lose rotated-cold service.
- Do not start a cp.async, TMA, or overlap project without a measured stall
  discriminator; llama reaches its result without those mechanisms.

## Workstream 4: V, split by format

### Measured fact

V carries 5.327464 ms of traced active debt. The model has 18 Q4 V weights and
18 Q6 V weights, so it must be treated as two populations.

### Q4 V substrate

- The qualified K geometry `(64,32,64)`, 128 threads, 256 CTAs has already
  passed a real Q4 V primitive at about the same isolated service as K.
- Missing work is model composition and proof that the K geometry transfers to
  V under Flash dependencies.

### Q6 V substrate

- Typed Q6 K16 fragment/correction and the 256-CTA V primitive pass.
- The prior Q6 V-only model bracket produced only a marginal signal and booked
  zero recovery.
- No new arithmetic substrate is required for the next test.

### Tests

1. Integrate Q4 V alone into the current gate/up+K+Q/O graph, regenerate the
   Flash dependency cut, and require exact 20-cycle records, outputs, KV, and
   logits.
2. Run an exact Q4-V-only trace and R9 bracket.
3. Repeat independently for Q6 V; do not combine formats until each passes.
4. If primitive time falls but token wall does not, compare producer, main,
   Flash readiness, queue placement, and cache intervals directly. This is the
   discriminator between a body problem and lifecycle absorption.
5. Only after both individual decisions, test the combined 36-V population.

### Invest/stop

- Q4 V is substrate-ready and should be tested before writing new kernels.
- Q6 V remains test-only until a repeated whole-wall win exceeds run noise.
- A primitive win without a model win does not advance.

## Workstream 5: Q/O service and queue ownership

### Measured fact

Q/O carries 4.058897 ms of traced active debt. Its separate 72-role packed
proxy wins strongly, but only 0.795 ms survived in the admitted composed graph.
Default two-queue ready placement was slightly faster and failed recurrent
correctness; the conservative graph-derived Flash dependency cut passed.

### Substrate state

- Packed Q4 compiler arithmetic is ready.
- Graph-owned buffers in the admitted combined harness are ready.
- The standalone Q/O binding still has mutable per-device pools and is
  harness-only.
- A production-generic dependency-identity policy is missing. The current cut
  is graph-digest-specific.

### Tests

1. Run matched Q and O separately through the same hot/cold/counter protocol as
   gate/up. Do not assume one geometry is best for both.
2. Test the occupancy-safe K/V-style geometry against the current Q/O geometry
   using full real outputs and the 72-role population.
3. Regenerate the graph dependency cut for every candidate graph.
4. Require default-ready, minimal-cut, conservative-cut, primary-only, and
   one-queue ablations. Admission requires the fastest arm that passes deep
   replay, not the fastest single R9 sample.
5. Generalize the safe cut from exact dependency identity only after the
   kernel result is stable.

### Invest/stop

- Kernel work can proceed now.
- Production promotion is blocked on dependency-identity admission and
  immutable/per-capture Q/O binding ownership.
- Any candidate that needs the unsafe default placement is rejected.

## Workstream 6: down, split by format

### Measured fact

Down is the largest single raw debt at 12.066558 ms. It contains 18 Q4 down
weights and 18 Q6 down weights. The formats must not be collapsed into one
campaign.

### Q4 down state

- The typed Q4/compact-Q8 compiler contract exists.
- The exact `(512,4096,12288)` Q4 down geometry, population, and residual
  epilogue are not qualified in the winning compiler graph.
- Therefore Q4 down is **partially substrate-ready**: start with a role gate,
  not a new compiler abstraction.

### Q4 down tests

1. Full real Q4 down oracle with direct compiler output and no global partial
   or expanded weight.
2. Geometry sweep appropriate to N=4096, K=12288; do not copy gate/up geometry
   without measurement.
3. Exact 18-role proxy including compact-Q8 producers.
4. Model arm with the existing rank-preserving residual lifecycle, followed by
   full logits, deep replay, and R9.

### Q6 down state

- Arithmetic substrate passes: typed K16 scales, paired masked IMMA, direct
  output, no global partial, no spills in the qualified primitive.
- Model lifecycle fails: down-only regressed by 1.611334 ms in the corrected
  bracket.
- Q6 down is therefore **substrate-first** for further investment.

### Q6 down tests before new implementation

1. Trace three matched boundaries separately: current FP16 down body; compact
   Q8 + Q6 main; and Q6 main + rank-preserving residual epilogue.
2. Attribute the regression to producer service, paired-K16 tensor work,
   output/epilogue materialization, queue placement, or cold-weight service.
3. Test sharing one compact-Q8 record with gate/up only where the activation is
   exactly identical; prove physical buffer identity and dependency order.
4. Test a compiler-owned residual epilogue without changing rank or adding a
   materialization.
5. Treat a one-IMMA K16 design as new substrate: it must expose two independent
   K16 subtotals exactly. A summed K32 dot is mathematically insufficient.

### Invest/stop

- Invest in Q4 down if its primitive and 18-role population pass.
- For Q6, invest only in the component named by the boundary trace.
- Do not re-integrate the current Q6 spelling; it is already a measured loser.

## Workstream 7: Flash prefill

### Measured fact

Flash carries 1.670649 ms of traced active debt. Cross-runtime overlap is tiny;
the result does not support an overlap-first campaign.

### Substrate state

- Decode Flash/O experiments are not direct evidence for pp512 prefill.
- The prefill route needs a matched score/reduction topology audit.
- A prefill Stream-K/fixup or better compiler-native attention topology may be
  required, but that is not yet proven.

### Tests

1. Extract exact attention shapes, causal mask, head grouping, score/reduction
   launches, and bytes from both traces.
2. Build a direct full-output oracle comparing current tiny prefill Flash with
   the exact llama math contract.
3. Measure score and reduction independently, hot and cold, with counters.
4. Test topology changes before fusion or overlap: CTA ownership, head/tile
   partition, reduction placement, and KV access order.
5. Integrate only a primitive/population winner and regenerate the queue cut.

### Invest/stop

- Invest only after the score or reduction discriminator identifies a
  dominant service cause.
- Do not transfer decode overlap/fusion conclusions to prefill without a new
  pp512 gate.

## Workstream 8: K residual service

### Measured fact

K carries 0.959419 ms of traced active debt. The current packed K route already
uses 256 CTAs, canonical Q4 weights, compact Q8, direct output, and passes the
model lifecycle.

### Plan

1. Reuse counter and geometry findings from Q4 V because the shapes and
   qualified primitive are shared.
2. Run a K-specific change only if V identifies a reusable service mechanism.
3. Preserve the exact 36-K replay and KV-state gate.

K is not the right first target while gate/up, down, V, and vocabulary remain
open.

## Workstream 9: support kernels

### Measured fact

Support carries 1.739349 ms of traced debt, but it is fragmented. Tinygrad's
largest hashed support family totals about 0.507 ms; no single dominant support
kernel is currently proven.

### Missing substrate

The missing piece is attribution, not a new GPU primitive. Each hashed HCQ
program must be mapped to a semantic source operation and matched to its llama
counterpart.

### Tests

1. Add stable semantic labels to every support program without changing its
   binary or graph placement.
2. Produce per-layer counts, bytes, dependencies, and exact active time.
3. Separate required math from transport/materialization and shape-only moves.
4. Rank only after the source map closes at 100%.
5. Test removal/fusion one family at a time with full replay and R9.

Do not start a broad support-fusion project from the aggregate 1.739-ms row.

## Cross-cutting substrate ledger

| substrate | state | blocks testing? | blocks production promotion? |
|---|---|---:|---:|
| typed Q4 packed fragments and K32 correction | pass | no | no for admitted shapes |
| typed Q6 K16 fragments and paired correction | pass arithmetically | no | yes for losing down lifecycle |
| compact Q8 producer | pass | no | no, except sharing/fusion experiments |
| computed PROGRAM input/output ownership | pass | no | no for qualified graph |
| canonical zero-copy packed weights | pass | no | no for qualified graph |
| immutable compiler asset + per-capture lazy buffers | pass for gate/up/K and combined wrapper | no | standalone Q/O binding still needs cleanup |
| graph-derived Flash dependency cut | exact current graph passes | no | yes; needs dependency-identity admission |
| pre-finalized typed kernel-input boundary | missing | no; compile-once PROGRAM is safe | yes for production-generic compiler routing |
| actual-HCQ physical counter bridge | missing | no; CUDA-launchable cubin gates can measure mechanisms | useful for direct in-graph counter proof |
| final requested-row prefill route | missing | yes for prune result | yes |
| packed Q6 M=1 full-logit prefill route | unqualified | yes for vocabulary result | yes |
| Q4 down role geometry/lifecycle | unqualified | yes for Q4 down result | yes |
| faster Q6 down lifecycle | missing | yes for Q6 down investment | yes |
| semantic support-kernel labels | incomplete | yes for support optimization | no for dense/tail tests |

## Exact execution queue

The next work should run in this order, with a ledger update after every stop or
pass:

1. Final-row prune Test 1 and Test 2.
2. Full-logit packed-Q6 vocabulary Test 1 and Test 2.
3. Gate/up matched service/counter audit, then the smallest winning schedule
   candidate through the 72-role proxy.
4. Q4 V model gate; Q6 V model gate; combined V only if both individual
   decisions permit it.
5. Q and O geometry/service audit under regenerated safe cuts.
6. Q4 down primitive and 18-role lifecycle.
7. Q6 down boundary attribution; implement only the proven missing component.
8. Prefill Flash score/reduction discriminator.
9. K follow-up only for mechanisms shared with V.
10. Support semantic map, then re-rank the remaining fragments.

After every accepted whole-model change, rerun the exact cross-runtime trace
and rebuild both rankings. The final-row prune in particular changes the dense
population, so old 36-layer gate/up/down totals must not remain the authority
after it is admitted.

## Stop rules

- No estimated token-rate claims.
- No scaling of traced region shares onto unprofiled wall.
- No booking a primitive win without a role-population and whole-model win.
- No combining Q4 and Q6 populations until each format passes separately.
- No use of unsafe default queue placement, even if its R9 sample is faster.
- No broad compiler abstraction to solve a shape-specific failure unless the
  smallest discriminator proves the abstraction is the wall.
- No optimization of norm/conversion or activation/multiply while tinygrad is
  already faster there.
- A failed spelling is retained as evidence and closed; a new spelling must
  name a genuinely different mechanism.

## Source authorities

## Execution-status addendum (2026-08-29)

The plan above is the original scope and ordering. Current execution status is:
Q4-V is retained default-off after its 18-role/deep20/matched-wall gate;
Q4-down is stopped because the authoritative matched A/B fails correctness and
is slower; and current Flash S6 spellings are stopped/not integrated because
the installed-population comparison rejects their service time. The matched
current best is the composed unroll4+Q4-V route at 67.153915 ms minimum /
67.235719 ms median.

The composed unroll4+Q4-V gate is now closed PASS: 67.153915 ms minimum /
67.235719 ms median, with exact deep20 replay, full logits/token, and the
198-role census. Its matched unroll4/no-Q4-V control is 69.165843 /
69.315714 ms. The isolated Q4-V bracket remains precursor evidence only.

- `docs/task_workflow/output/nv-prefill-exact-cross-runtime-accounting.md`
- `docs/task_workflow/output/nv-prefill-complete-lifecycle-ledger.md`
- `docs/task_workflow/output/nv-compiler-q4k-gkqo-combined-result.md`
- `docs/task_workflow/output/nv-prefill-compiler-q4k-kv-role-result-20260828.md`
- `docs/task_workflow/output/nv-compiler-q6k-imma-substrate-result.md`
- `docs/task_workflow/output/nv-compiler-q6k-model-lifecycle-result.md`
- `docs/task_workflow/evidence/nv-prefill-exact-cross-runtime-trace/`
