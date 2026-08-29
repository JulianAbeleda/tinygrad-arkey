# NVIDIA compiler-native Q4_K Q/O qualification

## Decision

**PRIMITIVE PASS. 72-ROLE PROXY PASS. DEFAULT-OFF WHOLE-MODEL PASS WITH AN
EXACT CAPTURED QUEUE CUT. DO NOT PROMOTE THE UNGUARDED READY-PLACEMENT PATH.**

The gate/up typed Q4_K/Q8_1 K64 compiler contract transfers correctly to the
Qwen3-8B attention-Q and attention-output shape `(M,N,K)=(512,4096,4096)`.
It produces a normal compiler-owned direct-output IMMA program over canonical
packed model weights.  The 72-real-weight population is approximately twice as
fast as its matched resident-FP16 population.

The separate default-off full-model arm is structurally clean and faster.
The initial failure was not packed arithmetic: default NV two-queue ready
placement intermittently violated repeated-A replay.  A captured fail-closed
queue cut that keeps Flash's direct dependency boundary on the primary queue
removes that drift while preserving the wall recovery.  This research cut is
graph-prefix/name-digest guarded and remains explicit/default-off.

No production route, default, core compiler file, scheduler file, or model
entry point was changed.

## Real Q and O Gate A

Both gates use the real block-0 GGUF weight, a deterministic nondegenerate
compact Q8 packet, and the compiler-emitted K64 program.  The full 2,097,152
outputs are compared against the qualified static Q4_K/Q8 IMMA oracle.

| check | attention Q | attention O |
|---|---:|---:|
| finite / written / nonzero | PASS | PASS |
| maximum absolute error | 0.00048828125 | 0.00048828125 |
| mean absolute error | 0.000009135 | 0.000011365 |
| `rtol=2e-5, atol=2e-3` | PASS | PASS |
| packed weight / Q8 record read-only | PASS | PASS |
| generated minimum | 232.357 us | 234.732 us |
| generated median | 238.639 us | 239.691 us |

The executed program has a `32 x 4` CTA grid, 256 threads/CTA, 21,504 B
shared memory, 222 registers, 32 static signed-S8 IMMA sites, two barriers,
and no local-memory loads or stores.  It allocates no expanded global weight,
group-partial tensor, or fixup workspace.

The synchronized one-call result is slower than the recorded resident-FP16
Q/O body (about 183.5 us/call).  As with gate/up, this one-call host bracket is
not the population authority because it charges synchronization and launch
service separately to every small role.

## 72-real-weight population

The captured population uses all 36 attention-Q and all 36 attention-output
canonical packed parameters.  Replay of the isolated population is exact;
representative real weights produce distinct finite outputs.

| arm | R9 minimum | per projection | reading |
|---|---:|---:|---|
| compiler Q4_K/Q8 K64 | 6.893014 ms | 95.736 us | candidate |
| resident FP16 | 13.883271 ms | 192.823 us | matched current control |
| llama Q/O estimate | about 4.428 ms | about 61.5 us | retained lifecycle audit |

The compiler population is `0.4965x` the FP16 wall, a 6.990-ms proxy
recovery.  It remains `1.5567x` llama's estimated per-call service.  This is a
real transferable primitive win, but it is not llama service parity.

The proxy has exactly 72 compiler calls, no expanded packed-weight allocation,
no partial workspace, and no fixup calls.

## Separate default-off model arm

The research arm uses `prefill_route_override`; ordinary model execution does
not see it.  Its captured census is:

| property | result |
|---|---:|
| compact Q8 producers | 72 |
| compiler-generated mains | 72 |
| canonical weight arguments / unique bases | 72 / 72 |
| Q/O FP16 overlays retained | 0 |
| expanded FP16 weight bytes in candidate mains | 0 |
| old fixups / partial workspace | 0 / 0 |

The O input still crosses an explicit FP32-to-FP16 materialization before the
FP16 compact-Q8 producer.  This is not a packed-weight transport copy: it
preserves the current prefill linear's FP16 numerical boundary for this gate.
It is charged in the whole-model wall and is not claimed to be unavoidable in
a future FP32-producer arm.

Fresh-process R9 walls:

| arm | minimum | median | reciprocal pp512 rate |
|---|---:|---:|---:|
| default-off packed Q/O + safe cut | 81.035840 ms | 81.452864 ms | 6,318 tok/s |
| fresh resident-FP16 control | 83.717813 ms | 83.885367 ms | 6,116 tok/s |

The measured recovery is 2.681973 ms, or about 3.31% prompt throughput.  It is
admissible for this separate research arm, but is not a combined-stack or
production booking.

Against the fresh FP16 artifact, the retained candidate output is finite,
chooses token 198 in both arms, and passes the existing full-logit
`rtol=0.02, atol=0.5` comparison (`max_abs=0.041781`,
`mean_abs=0.005252`).

## Replay root and guarded fix

The control returns bit-identical full logits for A, B, A.  The unguarded
candidate returns the same token but repeated-A logits vary intermittently.
The deepest initial localization found:

- maximum absolute drift: 0.0309 to 0.0549;
- mean absolute drift: 0.0046 to 0.0079;
- first changed packed stage: block-24 O; block-24 Q remained exact;
- first changed KV state: block 25; block-24 KV remained exact;
- explicitly zeroing all live KV caches before each A did not fix it;
- `HCQ_NUM_COMPUTE=1` made all 72 records, all 72 outputs, all 36 KV states,
  and final logits/token exact, but slowed the wall to 92.843 ms;
- a blunt primary-placement cut was exact but slowed the wall to 88.941 ms.

The direct primitive and isolated 72-call population replay exactly.  The
causal split is therefore the NV HCQ two-compute-queue placement boundary, not
weight mapping, Q8 cursor/output identity, KV reset, or IMMA arithmetic.

The conservative graph cut starts from the captured default auxiliary set and
removes every direct dependency of each Flash call.  It passed two fresh
A-B-A processes and a 20-cycle B,A stress: every cycle was bit-exact, as were
all retained packed records, compiler outputs, KV states, and final logits.
The unguarded default failed the same 20-cycle stress intermittently, with
some exact cycles and some drifting cycles (6/20 exact; maximum observed logit
drift about 0.0658).

Per-edge ablation strengthens the localization.  Pinning dependency positions
0, 1, or 3 alone failed.  Position 4 passed one cycle but failed the 20-cycle
stress.  Position 5 moves no default auxiliary calls and happened to pass its
single cycle, demonstrating why one replay is insufficient.  Position 2 alone
passed all 20 cycles.  The conservative all-direct-dependency cut remains the
authority rather than promoting that minimally sampled occurrence rule.

## Disposition

1. Preserve the Q/O primitive, population harness, default-off model arm, and
   captured safe-cut generator as passing substrate.
2. Keep ordinary ready placement unchanged; the cut must match the exact graph
   prefix digest and selected program identities or fail closed.
3. Treat 2.681973 ms as a Q/O-arm recovery only.  Do not add it to other role
   recoveries until a combined whole-model arm passes the same 20-cycle replay,
   full-logit/token comparison, structural census, and fresh R9 bracket.
4. A generic runtime promotion needs an independently proven dependency-aware
   rule for this boundary; this result does not authorize a graph-specific cut
   as a production default.

Primary evidence is indexed in
`docs/task_workflow/evidence/nv-compiler-q4k-qo-20260828/README.md`.
