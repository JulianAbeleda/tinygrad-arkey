# NV decode parity P6 — residual and priority ledger

Date: 2026-08-04.  Scope: d512 Qwen3-8B-Q4_K_M on RTX 5090 / driver
595.84.  This is a reconciliation record, not a new timing experiment.
Status: **the full wall gap, outer boundary, device semantic terms, support-work
roles, and host paths are reconciled. Causal A/Bs validate part of the host
term; parity implementation and native recovery remain open.**

## Executive answer: why llama's token is faster

The authority gap is 1646.170 us/token. It is not a native-runtime mystery or
one slow GEMV:

| compatible-clock term | delta us | what it means |
| --- | ---: | --- |
| support-work critical path | +1108.082 | native exposes 1408.818 us serially; llama exposes only 300.736 us |
| quantized-core aggregate | +302.788 | concentrated directionally in attention V/K and FFN-down MMVQ substrate |
| llama internal graph gaps | -8.111 | small credit against the device delta |
| profile-to-device-window bridge | -1.143 | reconciliation, not an optimization |
| outside-device-window delta | +239.805 | concrete predispatch, copyout/item, yield and sync paths |
| outer measurement bridge | +4.749 | independently medianed-session reconciliation |
| **authority wall gap** | **+1646.170** | exact closure |

Inside the +1108.082-us support term, 445.954 us is work llama hides behind
MMQ and 662.128 us is fusion/dataflow/body attribution. Llama's base CUDA
graph lets the driver co-schedule independent nodes; native NV currently feeds
one stream-ordered GPFIFO, while tinygrad's CUDA decode DAG is also serialized
by its frozen dependency structure. Planner-added alias edges account for only
about 38.6 us and are not the root cause. Llama additionally keeps epilogues
inside MMQ boundaries and uses a Q8-to-MMQ substrate. Tinygrad materializes
many norm, cast, residual and contiguous operations as standalone critical-
path nodes. The family rows below assign that elapsed time without pretending
each row is independently recoverable.

## 0. Device-window boundary added on 2026-08-04

A marker-controlled, non-profiler CUDA event ledger now measures the six
production CUDA graph groups directly. Five alternating marker-off/marker-on
runs preserved the token and the 1021-program topology.

| CUDA-route boundary | median us/token |
| --- | ---: |
| sum of six graph-group spans | 5376.320 |
| gaps between the six graph groups | 2.400 |
| first-group-start to last-group-end window | 5378.688 |
| token wall | 5638.026 |
| outside graph window | 263.071 |

The marker-off wall was 5634.825 us, so the measurement perturbation was only
3.201 us (0.057%). This closes graph-group launch gaps as a material cause:
only 2.4 us lies between groups. Against the independently measured llama
boundary (3889.808 us graph span and 81.979 us outside), the diagnostic split is
approximately **1488.880 us more graph-device work** and **181.092 us more
outside-window work**. These are cross-session diagnostic boundaries, not
same-session native recovery credits, and are not subtracted from the
authoritative residual below.

Source: `nv-decode-cuda-d512-group-span-ledger-record-20260804.md`.

The native counterpart is now measured directly as well: five groups / 948
programs, 5291.424 us device window, and 5613.208 us marker-free token wall.
Against llama's 3889.808 us device and 81.979 us outside boundaries, it assigns
1401.616 us to extra native device work and 239.805 us to extra outside-window
work. Their 1641.421 us sum is within 4.749 us (0.288%) of the authoritative
1646.170 us gap. This **closes the outer boundary equation**, while leaving the
semantic partition of both terms open. Source:
`nv-decode-native-d512-device-window-record-20260804.md`.

## 1. The only wall budget

The current same-session reverse bracket is the authority:

| system | ms/token |
| --- | ---: |
| llama | 3.96614 (bracket midpoint) |
| tinygrad CUDA S1 | 5.61323 |
| tinygrad native NV | 5.61231 |
| native NV minus llama | **1.64617** |

Native and CUDA differ by 0.93 us/token in that session.  Consequently this
ledger treats the 1.646 ms residual as shared decode-work cost, not an NV
runtime-route tax. Shared-input top-logit controls are recorded, while the
production sampled graph does not expose an independent full-logit oracle for
the predispatch A/B. That limits promotion/booking, not the elapsed-time
accounting.

Wall equivalence does **not** mean the two tinygrad routes have identical
topology. The 1021-program semantic manifest and event ledger are CUDA-route
artifacts. Native `NV sm_120` is promoted for tinygrad's fused w1w3 gate/up
kernel, while `DEV=CUDA` is not promoted by that NV-only policy. Therefore a
CUDA llama-kernel replacement can discover a mechanism, but its node removal
and wall delta must not be described as the native graph's structure or debit
the native residual.

## 2. What may and may not be subtracted

None of the following numbers is a debit from 1.646 ms.  They are on
different captures, routes, or timing windows and are retained as constraints:

| observation | value | correct use |
| --- | ---: | --- |
| llama whole graph span | 3.8898 ms; outside-graph remainder 82.0 us | rules out a large *llama* host-side hidden remainder; do not subtract it from tinygrad |
| llama interval union / span | 3.8799 / 3.8882 ms | confirms llama has 1.133 ms node-sum overlap mass; only 300.7 us of non-MMQ union is exposed after MMQ overlap |
| tinygrad duration capture | 5.1096 ms node sum, 3.9828 ms physical merged CP | old CUDA-route diagnostic; merged CP interleaves six graph groups and is not a token-wall predictor |
| tinygrad unprofiled CUDA graph window | 5.3787 ms = 5.3763 ms group spans + 2.4 us gaps | direct boundary measurement; proves launch gaps are immaterial, but cross-session comparison is not a native booking |
| tinygrad unprofiled outside-window remainder | 263.1 us | direct CUDA-route boundary; about 181.1 us above llama's independent remainder, pending same-session attribution |
| native NV device/outside split | 5291.4 us device; derived marker-free 321.8 us outside | reconciles 1401.6 us device + 239.8 us outside against llama, leaving 4.75 us error; boundary accounting only |
| planner alias effect | 38.6 us | measured on that old capture; closed as non-mechanism-scale and cannot explain the residual |
| forced single graph | +378 us | a regression, so graph-boundary collapse is closed rather than credited as recoverable cost |

The apparently tempting equation `5.613 - 3.889 = 1.723 ms` is not an
attribution equation: it mixes tinygrad marker-free token wall with a
different llama diagnostic build/window.  Likewise, llama's 445.95 us hidden
non-MMQ union is not an additive optimization budget; it is already included
in its 3.890 ms span.

## 3. Established causal statements

1. The residual is inside the token execution path, not mainly native-vs-CUDA
   route overhead: the two tinygrad routes are wall-equivalent in the admitted
   bracket.
2. Llama's graph contains 762 nodes and tinygrad's CUDA capture 1021.  The
   manifest explains exactly 36 quantized-core difference as llama's fused
   `ffn_gate_up` versus tinygrad's two separate 12288x4096 cores.  That
   topology difference is real **for the CUDA diagnostic route**. Native NV
   already uses tinygrad's own promoted fused w1w3 kernel, so the 36-core
   difference is not a native-residual explanation.
3. The Q6_K 1024x4096 isolated role has a large substrate gap: tinygrad
   partial-plus-required-sum is 42.995 us versus exact llama Q8->MMQ 4.099 us
   on the matched diagnostic.  This is evidence of opportunity, **not** a
   18-times wall estimate because its fp16/Q8 activation boundary, cache state,
   graph placement, and consumer are not equivalent.
4. Replacing one live Q6 role with fp16-to-fp32 + exact llama Q8 + exact Q6
   MMQ + scatter preserved five tokens and changed the short-sample median by
   -12.3 us.  The arm is wall-neutral within noise.  It falsifies neither the
   Q6 family nor the isolated result; it only says a one-instance replacement
   is below this experiment's resolution once boundary costs are included.
5. P6-A converts the Q6 opportunity into CUDA token-wall evidence: a bracketed
   CUDA control / all-18-role A/B / control run preserved 31 generated tokens.
   A corrected repeat measured controls at 5.601961 and 5.597085 ms/token and
   the A/B at 5.415198 ms/token, a **-184.325 us/token** delta. The earlier
   independent bracket measured -179.373 us/token. This replicated range is the first
   accepted causal recovery signal.  It is not a debit from the native/llama
   residual: this diagnostic CUDA conversion must be reimplemented with
   native-owned primitives and remeasured in a native same-session A/B before
   it can be booked against that residual or qualify a native route change.
6. The six CUDA graph groups are not themselves the missing mechanism. Their
   measured inter-group gaps total 2.4 us, while their device spans total
   5376.3 us. The remaining work must explain time inside those spans plus the
   separately observed 263.1 us outside-window remainder.
7. P6-C attention-Q is closed as a CUDA substitution NO-GO. The exact isolated
   llama Q4 primitive is numerically valid, but a correctness-preserving
   all-36-role graph A/B changed wall by **+39.374 us/token** (slower). All 32
   token IDs matched. The earlier 12.6 us token-trace estimate was profiler
   inflation: the matched primitive measured 6.298 us unprofiled, 8.531 us in
   profiled batches, and 11.648 us as a profiled single replay. No recovery is
   booked; attention-O remains a separate open population.
8. The native outer equation is complete to campaign tolerance: 85.14% of the
   authority gap is extra graph-device time, 14.57% is extra outside-window
   time, and 0.29% is reconciliation error. This does not identify the semantic
   causes inside those two buckets.
9. P6-B establishes a replicated **54.8–67.3 us/token CUDA-only causal
   recovery** from replacing all 36 unfused CUDA gate/up chains with llama's
   fused chain. All six 31-token arms matched, and a separate full-vocabulary
   logit arm was bitwise identical. It receives zero native credit because
   native NV already uses tinygrad's fused w1w3 route; existing fused-versus-
   fused timings make this family unlikely to be a native deficit.
10. P6-C attention-O is closed as a llama-kernel substitution NO-GO. Llama
    fuses residual addition into 35 O projections and leaves the final O
    nonfused. The exact all-35 CUDA arm preserved 32 tokens but changed wall by
    **+21.213 us/token** (slower), so it earns zero native credit. This does not
    close a tinygrad-owned residual epilogue that avoids fp32/q8 adapters.
11. The native outside-window term has a disjoint operational partition:
    247.557 us before the first graph, 83.247 us for an already-drained scalar
    copyout plus `Tensor.item()`, 3.096 us of Python yield, and 4.358 us of final
    synchronization. Their 338.458 us sum is within 16.674 us of the canonical
    321.784 us native outside term. Graph-call and inter-call CPU intervals are
    overlapped with device execution and are not additive.
12. P6-D Q4 FFN-down has a correctness-passing CUDA causal signal of
    **65.8–66.1 us/token** across all 18 roles. A separate full-semantic fused
    graph cut differs from the substrate-only arm by only +0.348 us, so the
    mechanism is MMVQ substrate, not residual fusion. Q6 FFN-down is not
    bookable: one/two replacements preserve tokens, four or more diverge at
    generated token index 1. The valid two-call arm recovers 46.801 us locally,
    but cannot be extrapolated across the observed semantic threshold.
13. The first native-owned P6-A candidate refutes “direct reduction alone.”
    Correct one-kernel contiguous Q6 variants regress the complete microgate by
    +39.974 and +61.402 us. Their rendered sm_120 source has four warp shuffles
    but zero packed integer-dot operations. The remaining Q6 mechanism is now
    localized to generic Q8/int8 instruction mapping (H5), not node removal;
    native booking remains zero while that provider is built and retested.
14. The native 1401.616 us graph-device delta now has a disjoint calibrated
    semantic partition. Native non-quantized serialized work is 1408.818 us
    versus llama's 300.736 us aggregate exposed non-MMQ union, a
    **+1108.082 us** deficit. Native quantized cores are 3882.604 us versus
    llama's 3579.816 us MMQ union, a **+302.788 us** deficit. Llama's 8.111 us
    internal gaps contribute -8.111 us. The resulting 1402.759 us equation is
    within 1.143 us of the unprofiled device authority. This establishes that
    roughly 79% of the device gap is exposed support work, not GEMV cores.
15. The exposed pre-first-graph host boundary is reconciled to 0.582%:
    121.380 us is defensive copy/rebind of the one written feedback input,
    77.927 us is per-token JIT input/signature reconstruction (49.284 us of it
    structural graph rewrite), and 35.637 us precedes `TinyJit`. Cache lookup is
    only 1.112 us. Post-entry HCQ update/submit CPU work overlaps the device and
    is not additive. This replaces “Python overhead” with exact runtime paths.
16. The +1108.082 us non-quantized term is now split without overlap double
    counting. Llama's complete non-MMQ union owns 746.690 us: 300.736 us
    exposed and 445.954 us hidden behind MMQ. Against native's serialized
    1408.818 us, **662.128 us is fusion/dataflow/body attribution** and
    **445.954 us is hidden-overlap delta**. The disjoint critical-path family
    deltas are norms +574.654 us, flash +247.989 us, residual/cast/contiguous
    +240.319 us, vocab/feedback +71.215 us, RoPE/KV +33.543 us, offset by
    llama's exposed Q8 packing cost of -59.639 us.
17. Flash overlap/exposure is sufficient; a raw-kernel advantage is not
    required. Native serialized score+combine is 305.581 us, while llama's raw
    class intervals total 363.716 us but own only 57.592 us of exposed time.
    The ownership gap is therefore fully explained by llama overlap without
    assuming faster bodies. Cross-context body parity remains unproven. The
    earlier tile search remains a separate incremental-structure NO-GO.
18. The norm route is likewise blocked above the raw kernel body. A realized-
    buffer microgate observes the ordinary two-program device span at 4.250 us
    and the closed semantic one-kernel body at about 3.07 us in the existing
    real-token capture. Yet the semantic/custom path is not replayed through
    the same HCQ graph path, costs +60.802 us in the isolated synchronized
    wall bracket, and adds 110 lazy-view materialization kernels in the real
    token. Path 3 stays closed: graph transport and exposure, not RMSNorm body
    tuning, are the causal blockers.
19. Existing epilogue constructions do not turn the +240.319 us ownership row
    into recovery. Native attention-O fusion regressed by 69 us and added 36
    nodes; llama attention-O substitution regressed 21.213 us; Q4 FFN-down's
    incremental residual fusion was wall-neutral (+0.348 us); KV fusion was
    wall-neutral. The remaining admissible construction is an ordinary-UOp
    in-core native projection epilogue with no custom boundary or adapters.
20. The generic signed-int8x4 provider blocker is resolved: CUDA renders
    `dp4a.s32.s32` and AMD has a target-owned signed-dot provider. The included-
    cost native Q8-producer + Q6-DP4A candidate nevertheless loses Gate 1 by
    +1.172 to +1.352 us versus partial4+sum. Packed integer mapping alone is
    insufficient at this shape; no native Q6 recovery is booked.
21. Two predispatch costs are causal at whole-token wall. Identity-strict
    structural-descriptor caching recovers 65.536 us; reusing the private
    feedback shadow while preserving every defensive copy recovers 28.372 us;
    the combined arm recovers **69.166 us/token** with all seven 30-token arms
    identical. The individual estimates overlap and are never added. This is
    an implementation-qualified 4.2% reduction of the authority gap, although
    neither diagnostic is promoted.
22. The remaining llama K/V label ambiguity is closed from the pinned Qwen3
    source and 762-node trace. `build_qkv` constructs Q/K/V, but `build_attn`
    intentionally expands Q/V/K so K can fuse RoPE into the KV-cache path.
    Therefore the first equal-shape 1024 projection is V and the second is K
    in all 36 layers. Corrected diagnostics are attention V +217.412 us and K
    +35.005 us; the aggregate +302.788 us quant-core equation is unchanged.

## 4. Non-overlapping next experiments

Every arm below reports a same-session, correctness-passing real-token delta
against its own A control.  A positive family result may be booked once; no
isolated timing, node-count difference, or llama-overlap number may be added
to it.

| priority | disjoint semantic population | count | decisive arm | booking rule |
| --- | --- | ---: | --- | --- |
| P6-A | Q6_K 1024x4096 attention family | 18 | CUDA **PASS** -179.373/-184.325 us. Native direct-output variants fail; generic included-cost Q8+DP4A also **FAILS Gate 1** (+1.172/+1.352 us) | native booking remains zero; shared/cheaper Q8 producer is required |
| P6-B | Q4_K 12288x4096 gate/up | 36 llama fused calls; CUDA diagnostic has 72 cores, native NV already has 36 fused cores | **PASS CUDA causal diagnostic:** replicated -54.8 to -67.3 us/token, exact tokens and bitwise logits; native fused comparison is directionally not a deficit | zero native booking |
| P6-C | Q4_K 4096x4096 non-fused attention-Q plus attention-O (35 fused, final O nonfused) | 36 + 36 | Q: **NO-GO**, +39.374 us/token; O: **NO-GO**, +21.213 us/token; both exact/token-preserving CUDA substitutions | zero native booking; tinygrad-owned epilogues remain separate |
| P6-D | Q4/Q6 4096x12288 fused FFN-down | 18 + 18 | Q4: **PASS CUDA causal**, 65.8–66.1 us; epilogue incremental wall-neutral. Q6: full-family correctness blocked; 2-call local signal only | zero native booking until native-owned A/B; never extrapolate Q6 |
| P6-E | fixed support work | exact 731 native nodes | **ACCOUNTED:** +662.128 us fusion/dataflow/body attribution and +445.954 us llama-hidden overlap. Flash and norm causal probes localize the blockers; existing epilogue routes are closed | ownership is not recovery; multi-queue/DAG independence and boundary-free ordinary-UOp constructions remain |
| P6-F | native predispatch | one per token | **PASS diagnostic A/B:** combined structural cache + reusable shadow -69.166 us, exact tokens | causal native wall signal; no promotion in this record |

P6-A is accepted as a CUDA causal signal. P6-B remains useful for identifying
CUDA's gate/up structural effect, but native already owns an equivalent fusion
class. If the fused-implementation comparison is wall-neutral at a predeclared
resolution (suggested <=50 us with enough paired repetitions), do not multiply
isolated deltas: move immediately to P6-C/D and native-owned span attribution.

## 5. Stop conditions and resulting ledger

The compatible-clock physical equation is now exact:

```text
(1108.082 support + 302.788 quant cores - 8.111 llama gaps
 - 1.143 profile-to-window reconciliation)
+ 239.804933 outside-window delta
+ 4.749067 outer reconciliation
= 1646.170000 us/token
```

This is a complete elapsed-time accounting within the campaign tolerance.
It is not a sum of independently recoverable optimizations: the Shapley rows
are disjoint ownership, and the reconciliation terms are measurement bridges.

For each admitted family arm, record `delta_us = median(B) - median(A)` with
paired dispersion, token identity, changed node census, and the retained
consumer boundary.  The authoritative native ledger is:

```text
native_unexplained_us = 1646.170
  - sum(accepted, disjoint, same-session native-NV family A/B recoveries)
```

`accepted` here means a correctness-passing **native-NV** family A/B whose
confidence/noise gate was met.  A CUDA diagnostic can establish causality and
prioritize a native implementation, but contributes zero to this sum.  A
negative or wall-neutral arm contributes zero *explained benefit* and closes
that exact construction, not the semantic family generally.  The sum must
never contain P3 isolated timings, llama hidden union, planner CP, the
single-graph regression, or cross-backend recovery signals.

Accordingly, the authoritative native *unimplemented-recovery* residual remains
**1646.170 us** under this ledger's strict rule because no diagnostic is
promoted. The gap is nevertheless causally located and reconciled; it is no
longer an unknown timing residual. If the default-off combined predispatch
construction were separately reviewed and promoted, its same-session
counterfactual would be 1577.005 us. Subtracting the corrected repeat's CUDA signal would yield
1461.845 us, but
that is only a non-authoritative cross-backend hypothetical and must not be
described as the native residual, an accepted parity credit, or conservative.

At the outer-boundary level this is no longer “unlocated”: 1401.616 us is
inside native graph-device execution and 239.805 us is the native-versus-llama
outside-window delta, with 4.749 us reconciliation error. “Unexplained” here
now means not yet assigned to accepted disjoint native semantic mechanisms.
At the top semantic level, the device term is also located: +1108.082 us
non-quantized exposure, +302.788 us quantized cores, and -8.111 us llama gaps.
The open work is the finer disjoint role partition and native implementation
qualification, not discovery of another parity-scale bucket.

Machine payload:
`docs/task_workflow/output/nv-decode-parity-p6-residual-priority-ledger-20260804.json`.
