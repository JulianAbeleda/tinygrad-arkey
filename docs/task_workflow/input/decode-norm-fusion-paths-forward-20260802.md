# Fused decode RMSNorm - paths forward (review scope)

Status: review scope for the M3 decision recorded in
`m3-fused-norm-measurement-record-20260802.md`. M3 is landed closed-default; the question is
which path reopens it, and what the campaign does in the meantime. The paths do not all
compete; the complementarity matrix is section 5.

## 1. Current design (as landed, measured facts)

M3 is one opaque kernel per decode norm (`decode_rmsnorm_1_4096` for attn/ffn, `_32_128` and
`_8_128` for q/k) replacing the generic reduce + epilogue pair, selected under the closed
`decode_norm_fusion` promotion record. Measured on NV sm_120, Qwen3-8B Q4_K, d512:

| class | baseline (M2) | fused (M2+M3) | delta |
| --- | ---: | ---: | ---: |
| norm family kernels/token | 361 (876.5us) | 288 (810.0us) | -73 kernels, -66.5us |
| decode kernels/token | 1021 | 1093 | +72 |
| decode kernel us/token | 6256 | 6398 | +142 |
| decode tok/s | 173.45 | 168.42 | -3% |

The family alone is a small paper win; the decode-level regression comes from what the fused
state adds around it, all verified in the census trace:

- 144 input-boundary copies: 108 x 4096-elem (`E_32_32_4_86a2`, 1.47us) + 36 x 1024-elem
  (`E_8_32_4_dd98`, 1.57us). The custom-kernel transport contiguous()s every non-identity
  input; the norm inputs are lazy producers (residual, qkv, embedding) with no buffer identity
  at trace time. A rank-3 pass-through was tried; the copy is not elided.
- 72 output materializations (`E_32_32_4_3b0f`, 1.54us), one per attn/ffn norm: the flat
  `(numel,)` kernel output reshaped to `(1,1,4096)` does not satisfy the downstream consumer's
  contiguity, so the scheduler realizes it. Trace order per attn/ffn norm:
  `copy -> decode_rmsnorm -> copy` (3 launches replacing the legacy 2).
- The fused kernels are launch-bound at 3.2-5.0us (4.96us median for `decode_rmsnorm_1_4096`);
  the design doc's llama-shaped 2.12us end-state is not reachable until per-kernel host
  overhead drops.

Tokens are byte-identical (sha `9d6b3787...` 3/3, first token `151936` 3/3), so correctness
is not the blocker; economics is.

## 2. Path 1 - copy-free opaque boundary (transport substrate)

Change the opaque-kernel transport so a consumer can declare "index my input by logical shape";
the boundary preserves the non-identity view instead of contiguous()'ing it, and the emitter
indexes the producer's logical dims so the scheduler resolves strides. Emitter-by-emitter opt-in
(new boundary mode, default unchanged).

Expected outcome for the norm family: 144 input copies (215.3us) and 72 output
materializations (110.9us) disappear; the family becomes ~144 fused kernels at ~594.7us vs
876.5us legacy (~-280us/token, -216 launches). Same mechanism removes the copy tax for the
flash q/k/v and gemv x views. Does NOT change the launch floor; fused kernels stay 3.2-5.0us
until Path 2.

Risk: every opted-in emitter must be audited for flat-index assumptions (q6k coop is flat;
flash tile reads cache with identity - unaffected); the "opaque = simple buffer" contract
becomes two modes that must not drift; pg3 pins for opted-in emitters move deliberately.

## 3. Path 2 - per-kernel launch/host overhead (B3)

The P2/P3 finding: per-kernel host cost is the dominant term for sub-10us kernels (the
launch-side counterpart of the measured ~1.5-5us floor). B3 batches/deletes the host work per
launch. Already scoped in `decode-gap-per-target-lever-scope-20260802.md`.

Expected outcome: lowers the floor for ALL ~1000 kernels/token, not just norms - the largest
single decode lever (order +10-20% wall time at the current census, to be measured). It also
makes the design doc's 2.12us norm shape reachable. Note: Path 2 ALONE does not make M3 land -
with cheaper launches, fused (3 launches) still loses to legacy (2 launches) until the copies
go. Path 2 and Path 1/3 multiply: fewer kernels x cheaper launches.

## 4. Path 3 - scheduler-native norm fusion (no opaque boundary)

Give the generic RMSNorm lowering an in-kernel reduction so one norm lowers to ONE scheduler
kernel: no boundary, no copies, no transport change. The machinery exists - M2's q6k coop
in-kernel merge uses the same staged-shfl + smem-barrier building blocks. This is the design
doc section 9 Q1's shape (b), the narrow generic chain-fusion option.

Expected outcome for the norm family: 145 kernels (144 fused + final norm), ~594.7us of kernel
time, no copy/materialization kernels, same ~-280us/token as Path 1's norm result - without
touching the transport or any other opaque consumer. Because it is the generic path, prefill
norms benefit too, which is the right outcome but widens the blast radius; admission must stay
closed-default per shape/target until measured. Reduce-order parity gate (decode sha) applies
as today.

Risk: generic-path blast radius (any model's RMSNorm), two lowering shapes that must not
drift, occupancy on small norms.

## 5. Complementarity

| | Path 1 (transport) | Path 2 (launch) | Path 3 (generic) | Path 4 (M4/M5) |
| --- | --- | --- | --- | --- |
| Path 1 | - | multiplies (fewer kernels x cheaper launches) | partial substitute for the norm family; Path 1 also fixes flash/gemv views, Path 3 also fixes prefill norms | orthogonal |
| Path 2 | multiplies | - | multiplies | orthogonal |
| Path 3 | partial substitute | multiplies | - | orthogonal |
| Path 4 | orthogonal | orthogonal | orthogonal | - |

The two genuinely competing choices are Path 1 vs Path 3 for the norm copies: both remove the
same 216 kernels from the norm path, and doing both is redundant on that family. They differ
only in where the win lands (transport benefits every opaque consumer; generic lowering
benefits every norm on every target). Path 2 does not compete with anything - it is a
multiplier. M4/M5 (q4k epilogue absorption, flash combine normalization) proceed regardless of
the norm decision.

## 6. Expected outcomes by choice

| choice | norm family | decode (est.) | reachable end-state |
| --- | ---: | ---: | --- |
| keep closed (today) | 361 kernels, 876.5us | 173.45 tok/s | norm story parked; ~0.4ms epilogue claim via M4/M5 |
| Path 1 alone | ~144 kernels, ~595us | +4-6% (est.; ~180-184) | norm win lands; flash/gemv copy tax also removed; launch floor unchanged |
| Path 2 alone | unchanged (361) | +10-20% (est., to measure) | everything faster; M3 still loses (3 vs 2 launches) |
| Path 3 alone | ~145 kernels, ~595us | +4-5% (est.) | norm win lands on the generic path, all targets; prefill norms too (gated) |
| Path 1 or 3 + Path 2 | ~145 kernels at ~2-2.5us | doc's original 0.9-1.0ms claim becomes the measured question | llama-shaped plumbing end-state |

All est. numbers are arithmetic from the measured census medians, not measured; each choice
must be verified at the campaign's fixed-depth protocol (d512/d2048/d4096, sha pins) before the
record flips.

## 7. Open questions for review

1. Path 1 vs Path 3: is the generic in-kernel norm (Path 3) preferred over the transport change
   (Path 1), given Path 1's blast radius across flash/gemv emitters - or is Path 1 worth doing
   anyway for the other consumers' copy tax?
2. If Path 1: should view-preservation be a NEW boundary mode (opt-in per consumer) rather than
   a change to `custom_kernel`'s default, to keep the existing emitters' flat-index contract?
3. If Path 3: is an in-kernel reduce in the GENERIC lowering acceptable for decode shapes only
   behind a closed gate, or does that create two norm lowerings that can drift?
4. The 72 output materializations (`E_32_32_4_3b0f`): are they a norm-boundary artifact (fixed
   by Path 1) or a separate contiguity bug in the flash/attention consumer worth fixing
   independently?
5. Sequencing: run M4/M5 now and let the norm paths land when measured, or treat the norm
   story as non-optional per design doc section 6 and sequence one of Path 1/3 before M4?

HARD STOP after this section. No implementation on any path until this scope is reviewed.

---

## 8. Correction and llama-informed priority (2026-08-02)

Reconciling this scope against the llama trace (`decode-gap-per-target-lever-scope-20260802.md`
section 1) changes two claims and the sequencing. This section supersedes sections 3 and 6
where they disagree.

### 8.1 Path 2 is a prefill lever, not a decode lever (correction)

Decode is 95% GPU-busy (5.83ms busy of 6.12ms wall at d512) and the flash-decode rollout is
already graph-replayed into 6 batches (`batched 32/64/128/256/512/29` = 1021 programs/token) -
the B3 replay mechanism is in place for decode. The per-kernel host-cost ceiling for decode is
therefore ~5%, not the "order +10-20%" stated in section 3. The 1.9x wall-busy evidence is
the PREFILL prime path (24.1ms busy / 44-46ms wall on the tuned schedule; the 1.35M to_mv
calls are from the pre-tuning 4.39s run and do not transfer - B3 scope section 1.1), where B3
remains open. Path 2 stays a live lever for prefill; it is demoted for decode.

### 8.2 llama's shape argues for Path 3 (generic norm), not Path 1

llama's graph has no opaque boundary and no norm copy tax: RMSNorm is one generic kernel per
norm (`rms_norm_f32`, 145 nodes, 1.3-3.4us class) - exactly the shape Path 3 (scheduler-native
in-kernel norm) produces, and exactly the 145-kernel end-state the design doc already targets.
llama does NOT keep a two-kernel reduce+epilogue pair, and it does not pay a contiguous copy to
read its norm input. The M3 opaque emitter was the wrong shape for the norm family
specifically: it introduced a toll booth llama does not have. Path 1 remains useful only for
consumers whose inputs are non-identity views (flash/gemv), which is a separate question.

### 8.3 llama-informed priority

llama's advantage decomposes as: plumbing +1.05ms (no separate add/silu kernels; fused w1w3),
GEMV bandwidth +0.44ms (Q6_K 1.4 TB/s vs our 0.82/0.2; k/v 1.04 vs 0.2), vocab +0.24ms (single
mmq 303.75us vs our ~540us chain), flash +0.17ms (3.17+3.35us vs 7.6+3.6us per layer). Two
asymmetries cut the other way: llama pays q8_1 quantization (217 nodes, 0.482ms) that we do
not, and llama's per-kernel times (1.3-3.4us) are the same league as ours (1.6-3.9us) - the gap
is count and bandwidth, not launch economics.

Priority, largest measured mass first, each additive and closed-gate:

1. M4/M5 epilogue absorption (the +1.05ms plumbing class; llama's "no separate add/silu"
   shape). Independent of the norm decision; this is the biggest single next lever.
2. Path 3 - generic in-kernel norm (the norm half of that class; llama's `rms_norm_f32` shape,
   no copies, all targets). Path 1 only if a later measurement shows flash/gemv view consumers
   also pay a material copy tax.
3. L2/L5 GEMV bandwidth (+0.44ms; llama mmq blocks are 128 threads vs our lanes=32, Q6_K at
   1.4 TB/s) - diagnostic microbench first, per the substrate trichotomy.
4. L4 vocab substrate fusion (+0.24ms; scalar reduce + scatter into the coop kernel).
5. Flash tile (+0.17ms; 7.6 vs 3.17us) - values/occupancy first.

Path 2 moves to the prefill campaign (B3), where its evidence lives. The campaign's stated
endpoint stays 195-210 tok/s at d512 (llama 245.6; like-for-like closer once llama's q8_1
asymmetry is excluded).

---

## 9. Reviewer amendment for feedback (2026-08-02)

Status: proposed correction, not implementation authority. This section is the canonical
review amendment and supersedes sections 1-8 wherever the accounting, mechanism ownership,
expected outcome, or sequencing differs. The HARD STOP remains in force.

### 9.1 Reconciled M2 -> M3 accounting

The name-level census explains the full `1021 -> 1093` kernel delta without assigning
unchanged q/k work to only one side of the comparison:

| changed component | M2 baseline | M2+M3 fused | delta |
| --- | ---: | ---: | ---: |
| replaced legacy norm work | 288 kernels, ~750.6us | 0 | -288 kernels |
| fused norm kernels | 0 | 144 kernels, ~594.7us | +144 kernels |
| norm input-boundary copies | 0 | 144 kernels, ~215.3us | +144 kernels |
| downstream output materializations | 0 | 72 kernels, ~110.9us | +72 kernels |
| **changed-component total** | **288 kernels, ~750.6us** | **360 kernels, ~920.9us** | **+72 kernels, ~+170.3us** |

The whole-decode trace measures `+72` kernels and `+142us`; the difference from the
per-name-median arithmetic is ordinary cross-run/kernel timing variation. The count identity
is exact: `1021 - 288 + 360 = 1093`.

The earlier `361 -> 288` family row mixed memberships. The 361-kernel legacy family also
contains 72 q/k-adjacent E_ kernels and the final norm that remain in the fused trace, while
the 288-kernel fused column counted only 144 fused norms plus 144 input copies. A like-for-like
copy-free norm comparison is therefore either:

- changed component: 288 legacy kernels / ~750.6us -> 144 fused kernels / ~594.7us; or
- full logical family: 361 legacy kernels / ~876.5us -> approximately 217 kernels / ~718us
  (144 fused norms + 72 retained q/k-adjacent kernels + final norm).

Both views imply approximately **-144 launches and -0.16ms of node-sum**, not -216 launches
and -0.28ms. Node-sum is not wall time; a first-order ~178 tok/s value is only directional and
is not licensed as a promotion forecast. Path 1 may recover additional non-norm copies, but
their count and time must be measured before a larger `+4-6%` claim is restored.

### 9.2 What the llama trace proves

llama proves the required externally visible topology: one RMSNorm kernel per norm, no
adjacent norm copies, 145 norm nodes including the final norm. It does **not** decide whether
tinygrad reaches that topology through a copy-free typed program boundary or through a
scheduler-native lowering; either mechanism can produce the same graph shape. The M3
non-landing proves the current opaque transport contract is uneconomic for this use, not that
opacity by itself is disqualified.

The reviewer preference is still Path 3 for the norm family, but only under the concrete
semantic-lowering contract below. Path 1 remains a separate transport proposal justified by
measured copy taxes, not by analogy to llama's internal implementation.

### 9.3 Path 3 candidate contract - semantic RMSNorm lowering

Tinygrad has no RMSNorm semantic/lowering today: `nn.RMSNorm` is the ordinary
`square -> mean -> rsqrt -> multiply` Tensor expression. Path 3 must not be implemented as a
decode/model-name special case in global `jit_lower`.

Proposed contract for review:

1. Represent RMSNorm once as a semantic operation carrying the reduction axis, epsilon,
   input/output dtype, and optional affine weight.
2. Preserve the current ordinary Tensor expression as the generic fallback for every
   unadmitted shape/target.
3. Add an admitted native lowering that produces one scheduler-owned kernel with the
   reduction result consumed by its epilogue in-kernel. Admission is closed-default by
   target, shape, dtype, and required shuffle/barrier capabilities; it is not keyed to a
   model name.
4. Keep decode and prefill promotion separate. A decode result does not authorize prefill;
   each target/shape class needs its own correctness, occupancy, and performance evidence.
5. Gate reduction order with isolation parity plus the fixed-depth token sha. The fallback
   and native lowering share semantic tests so they cannot silently become two definitions of
   RMSNorm.

A genuinely generic reduction-result-to-epilogue fusion facility remains a possible later
compiler project, but it has a broader reduction-level blast radius and is not implied by
this norm scope.

### 9.4 The 72 output materializations are an independent P0

The output copies have a narrower mechanism than logical input views. `KernelProgram`
allocates a flat output and execution returns an `AFTER` value; `_decode_rmsnorm` reshapes it.
`has_buffer_identity()` follows `RESHAPE` but not `AFTER`, while `UOp.custom_kernel` preserves
an exact `AFTER` but not `RESHAPE(AFTER)`. The downstream Q4K/Q6K routes additionally request
`.contiguous()` on their activation inputs.

Before changing the general opaque ABI, run a narrow output-identity P0:

- prove which of buffer identity, the explicit consumer `.contiguous()`, or both owns
  `E_32_32_4_3b0f`;
- add a unit graph probe for contiguous `RESHAPE(AFTER)` and a decode census asserting the
  exact 72-count delta;
- keep the default flat-buffer transport unchanged; and
- re-measure M3, but do not reopen it on this fix alone unless the full fixed-depth protocol
  beats M2.

Path 3 should naturally avoid this custom-output boundary if its scheduler-owned output has
the same identity behavior as the legacy epilogue, but the census must prove that rather than
assuming it.

### 9.5 Path 1 and Path 2 disposition

Path 1 becomes a separately justified, typed transport proposal. Before designing it, census
the existing flash/GEMV input materializations by producer, consumer, shape, count, and time,
and distinguish a contiguous view of an already produced buffer from a lazy producer that
requires real computation. If the measured tax warrants a build, add a new opt-in input ABI
mode to `KernelProgram`; do not change `custom_kernel`'s default flat-buffer contract.

Path 2/B3 leaves the decode matrix. B3 owns the measured prefill wait/submit problem. Decode's
5.83ms busy / 6.12ms wall bounds total non-GPU wall recovery at roughly 5% and shows that host
launch is already amortized by graph replay. A device-side CUDA-graph node/launch-floor
hypothesis would be a new lever requiring its own microbenchmark; it is not licensed by B3.

### 9.6 Amended sequencing and outcome language

1. Proceed with M4/M5 under their existing closed gates. Attribute approximately 0.4ms of
   node-sum opportunity to their GEMV/flash epilogue scope, not the full +1.05ms plumbing
   class.
2. Run the narrow 72-copy output-identity P0 and record its independent verdict.
3. Review the semantic RMSNorm contract above; only after approval, build Path 3 behind a
   closed target/shape gate and measure d512/d2048/d4096 plus sha pins.
4. Measure the non-norm flash/GEMV copy inventory before deciding whether Path 1 merits a
   transport campaign.
5. Continue the separately scoped GEMV-bandwidth, vocab, and flash work according to their
   measured masses.

The campaign's `195-210 tok/s` remains a **target**, not a forecast supported by this norm
scope. Recompute any end-state forecast after M4/M5 and the corrected ~0.16ms norm hypothesis
are measured in wall time; do not carry forward the old 0.9-1.0ms L1 arithmetic unchanged.

### 9.7 Questions for feedback

1. Accept the reconciled `-144 launches / ~-0.16ms node-sum` norm hypothesis as the planning
   basis?
2. Accept semantic RMSNorm + generic-expression fallback + closed native lowering as Path 3,
   and reject a decode-specialized global `jit_lower` rule?
3. Approve the 72-copy output-identity investigation as an independent P0?
4. Approve M4/M5 first with an approximately 0.4ms node-sum claim, while Path 3 remains behind
   its own review and measurement gate?
5. Require a named non-norm copy census before Path 1 receives an implementation scope?

---

## 10. Response to the reviewer amendment (2026-08-02)

Answers to 9.7, with the amendment's factual claims verified against the census files and the
transport code before writing this section.

### 10.1 Q1 - accepted, with the verification that makes it exact

Accepted as the planning basis. The whole-decode count identity `1021 - 288 + 360 = 1093`
holds exactly against the M2-on baseline, and the two census checks that could have broken it
both pass:

- The 72 retained q/k-adjacent kernels exist in the fused trace
  (`E_4_2_8_16_4` x36 + `E_2_8_16_4` x36), so the changed component is 288, not 361.
- The `E_32_32_4_02a9` +18 (54 -> 72) belongs to the M2 composition change, not M3: the
  all-fusion-off baseline has it at 54 with `r_32_32_4_2_8` x18 present, and the M2-on closed
  state has it at 72 with the merge kernel present - both at a constant 1021 kernels/token.

The time deltas are softer than the count identity: my runs give +141us (all-off baseline vs
fused) and +211us (M2-on closed vs fused) for the same pair of states, with the amendment's
+170.3us between them. That spread is median/cross-run variance, and it is why the -0.16ms
node-sum and the ~178 tok/s first-order value stay directional, not promotion language.

### 10.2 Q2 - accepted; the semantic-op precedent already exists

Accepted: semantic RMSNorm (reduction axis, eps, dtypes, optional affine weight) with the
ordinary expression preserved as the universal fallback, and a closed-default admitted native
lowering, not keyed to a model name. Verified: there is no RMSNorm semantic today (only
`role_metadata("rms_norm")` strings and the ordinary `square -> mean -> rsqrt -> multiply`
expression), so the reviewer's premise is correct.

The repo already has the mechanism to mirror: `ATTENTION` is lowered as a semantic at
`lower_attention_semantic` (rangeify) with an ordinary fallback, and promotion records follow
`boltbeam.route_policy.v1` with decode and prefill as separate records. The norm semantic
should use both: a `lower_rmsnorm_semantic`-style hook plus a separate
`norm_fusion`/`norm_lowering` record per path. Decode and prefill stay independent admission
classes, each with correctness, occupancy, and performance evidence. Reduction-order parity is
gated exactly as the amendment says (isolation parity + fixed-depth sha), which is also the
M2/M3 precedent.

### 10.3 Q3 - approved; the mechanism claim checks out

Approved as an independent P0. Verified in the code: `has_buffer_identity` follows `RESHAPE`
but not `AFTER` (uop/ops.py), and `UOp.custom_kernel` preserves an exact `AFTER` src but
contiguous()s `RESHAPE(AFTER)` - so the flat output's reshape is a candidate materialization
site, and the downstream Q4K/Q6K routes additionally request `.contiguous()` on activation
inputs. The 72 count is exactly attn+ffn (2 x 36), matching the trace-order signature
`copy -> decode_rmsnorm -> copy`; the q/k outputs reshape to `(1,32,1,128)`/`(1,8,1,128)` and
show no companion, which leans toward the consumer `.contiguous()` side of the P0's question
but does not decide it. If the P0 removes the 72 kernels (~110.9us node-sum), the changed
component becomes `-144` launches / ~-281us node-sum, which would be a materially better
planning basis; the P0 verdict lands before Path 3 sequencing is finalized. Agreed that this
fix alone must not reopen M3 without the full fixed-depth protocol beating M2.

### 10.4 Q4 - approved; claim stated as node-sum, not wall

Approved. This matches the design doc's own ordering (M4 = step 5, M5 = step 6) and the
measured class table: the +1.05ms plumbing class splits into the norm half (gated behind Path
3) and the GEMV/flash epilogue half, so M4/M5's claim is the approximately 0.4ms node-sum
epilogue-absorption mass, restated as node-sum and re-measured in wall after landing. The
norm story remains behind its own review and measurement gate.

### 10.5 Q5 - accepted; the census has concrete starting candidates

Accepted. The named census starts with three known classes rather than an open hunt:
`E_32_32_4_0a5e` (36x, combine-output normalization), the explicit per-call
`.contiguous()`/`.cast()` materializations in `decode_routes.py` (`_xv`, `x_vec`), and the
flash q/k/v view handling. The census must record producer, consumer, shape, count, and time
per class, and must distinguish a contiguous view of an already-produced buffer (fixable by an
opt-in input ABI mode) from a lazy producer requiring real computation (not a transport fix).
No Path 1 implementation scope before that census exists.

### 10.6 Corrections and notes on 9.1-9.6

1. Section 9.1's `+170.3us` sits inside the measured spread (+141 to +211us across the two
   baseline choices); the count identity is the hard claim and it is exact. No change needed.
2. Section 9.3's "no RMSNorm semantic today" is verified correct; the semantic-op precedent
   (attention) and the closed-record pattern (route_policy) are already in the repo and the
   contract should cite them.
3. Section 9.6's endpoint discipline is accepted: 195-210 tok/s is a target, and any end-state
   forecast is recomputed from wall measurements of M4/M5 plus the norm hypothesis, not carried
   forward from the old L1 arithmetic.

No further questions from this side. The amended scope (9.1-9.6) plus this response is the
working basis for M4/M5 sequencing and the Path 3 review.

HARD STOP. This amendment requests feedback only; it authorizes no implementation or route
promotion.
