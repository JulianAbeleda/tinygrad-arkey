# NV weighted inter-anchor causal gap scope (2026-08-20)

Date: 2026-08-20
Branch: `nvidia-bringup-20260731`
Scope authoring HEAD: `6570abc02`
Status: **measurement scope, not a performance claim.**

## 1. Exact question

Determine which concrete dependency segment makes llama.cpp decode faster than
tinygrad on Qwen3-8B-Q4_K_M / d512 / RTX 5090, and rank the corresponding
tinygrad changes by recoverable end-to-end wall time.

The question is not merely where tinygrad spends time. It is:

```text
Between the mandatory Q, O, gate/up, and down GEMV anchors,
which exposed dependency segment is shorter in llama, by how much,
and what implementation or dependency difference causes it?
```

The output must distinguish:

- total kernel work (`node_sum`),
- work hidden by overlap,
- work on the weighted dependency critical path,
- host/graph submission time,
- and wall sensitivity if a node, class, or edge is changed.

## 2. Why this scope is required

The existing audits establish several important facts:

- tinygrad's aggregate GEMV work is already faster than llama's MMQ work;
- llama has much more timestamp overlap;
- tinygrad has support work on its GEMV dependency spine;
- more queues, replay merge, early PDL placement, and coarse flash splits do
  not clear the promotion bar on the tested route;
- copy-free fp16 RMSNorm is a concrete current candidate.

They do not yet identify a weighted, matched llama support path. The current
`625.6 us` support ceiling compares tinygrad critical-path support against
llama's exposed non-MMQ timestamp union. Those are different quantities.
The llama graph record is structural and explicitly unweighted.

The existing clean same-session comparison also predates the current tinygrad
HEAD, while the current RMSNorm commit changes graph topology. Old node sums,
critical paths, and class rankings are context only until re-measured.

Therefore, do not use either of these statements as a conclusion:

```text
llama overlaps 1125 us, therefore tinygrad can recover 1125 us
tinygrad node_sum is lower, therefore kernel work cannot be the loss
```

llama may overlap support work that tinygrad already fused away. The useful
quantity is exposed dependency time between matched mandatory anchors.

## 3. Measurement invariants

All endpoint comparisons must use:

- the same RTX 5090 session under `flock /tmp/gpu-bench.lock`;
- the same Qwen3-8B-Q4_K_M file, d512 prompt, decode depth, and token count;
- production tinygrad `DEV=NV`, not `DEV=CUDA`;
- the pinned llama commit and exact command recorded in the result;
- byte-identical tinygrad token streams between control and candidate;
- control/candidate/control brackets with fresh processes;
- explicit git commit, dirty-state description, environment, and route flags.

Unprofiled wall and profiled topology are separate measurements. Never compare
an intrusive `PROFILE=1` tinygrad span directly with an nsys/CUPTI llama span.
Profiler-tax records must accompany any profiled route used for durations.

## 4. Common anchor model

Use the dependency chain already shown to exist in both implementations:

```text
Q GEMV -> O GEMV -> gate/up GEMV -> down GEMV -> next-layer Q GEMV
```

K/V projections and KV-cache writes are not assumed to be on this path. The
existing traces show they are already off the tinygrad critical path, matching
llama structurally.

For every layer, measure these support segments:

| id | segment | expected contents |
| --- | --- | --- |
| S0 | prior down end -> Q start | residual, next attention RMSNorm, casts/providers |
| S1 | Q end -> O start | q norm, rope, flash score/combine, required joins |
| S2 | O end -> gate/up start | attention residual, FFN RMSNorm, casts/providers |
| S3 | gate/up end -> down start | GLU/activation and required reduction/plumbing |
| S4 | down end -> next Q start | FFN residual and next-layer input preparation |

For each segment, retain two values:

1. Timestamp exposure: elapsed device interval between anchor boundaries.
2. Weighted dependency cost: duration sum on the actual longest dependency
   path between those anchors.

Timestamp exposure answers what reached the wall. Weighted dependency cost
answers which operations and edges forced it there.

## 5. Phase A: current-HEAD RMSNorm verdict

RMSNorm remains the first experiment because it has a concrete mechanism,
repeats across layers, and preserves the fp16 token contract.

### A1. Arms

Run separate `DEV=NV` control/candidate/control brackets for:

| arm | sites |
| --- | --- |
| A | `ffn` |
| B | `attn` |
| C | `attn,ffn` |
| D | `attn,ffn,output` only if output SHA policy permits it |

The fp32 q/k/output warp-reduce variants remain excluded unless their token
contract is explicitly changed. Isolated CUDA body timing is supporting
evidence only; it is not a production wall verdict.

### A2. Required outputs

For each arm, record:

- median and all accepted/rejected wall samples;
- token-stream SHA;
- native RMSNorm invocation count by shape/site;
- input materialization/copy kernel count;
- total kernel count and graph-group count;
- node sum, kernel union, overlap mass, and host gap;
- RMSNorm body time and copy time by family;
- S0-S4 interval changes and weighted-path changes.

Do not hard-code the historical `144 copies/token` as the current baseline.
Measure the current control and candidate counts directly.

### A3. Interpretation

The isolated 1x4096 result predicts approximately `3 us` less body time per
eligible invocation. This is a node-mass prediction, not a wall prediction.
The prior estimate of roughly `216 us` across attention/FFN norms must be
reconciled against the current weighted critical path and alternate paths.

A positive wall result is attributed only after the topology record shows
whether it came from:

- removed input copies,
- the faster warp-reduce body,
- direct fp16 output eliminating a cast,
- or a changed dependency/scheduling shape.

Promotion gate: identical token SHA and greater than `50 us/token` median wall
improvement on the bracket. Results below the bar stay in the explanatory
ledger even when they are not promoted.

## 6. Phase B: matched weighted traces

### B1. Tinygrad trace

Capture a current-HEAD steady decode token and export:

- kernel start/end timestamps;
- logical kernel family and layer index;
- graph/JIT replay group;
- exact dependency edges after memory planning;
- queue assignment and cross-queue waits;
- anchor identity for Q, O, gate/up, and down.

If the capture requires `PROFILE=1`, collect a paired unprofiled route and do
not use replay-boundary gaps as device durations.

### B2. llama trace

Capture the same model/context in the same GPU session and map CUPTI kernels
back to the logical ggml graph nodes. The output must provide a duration for
each real logical node and preserve its dependency edges.

A structural `.dot` path alone is insufficient. The required artifact is a
weighted llama dependency DAG with timestamped MMQ anchors and support nodes.

### B3. Equivalence map

Create one common taxonomy without hiding fusion differences:

| logical role | tinygrad representation | llama representation |
| --- | --- | --- |
| normalization | separate, native, or fused epilogue | `rms_norm_f32` plus scale/mul as emitted |
| projection quantization | folded provider or separate kernel | `quantize_q8_1` or folded path |
| attention | score/combine programs | flash-attention program(s) |
| residual/activation | epilogue or elementwise kernel | add/mul/GLU kernel |
| output reduction | separate/fused reduce-output | corresponding in-kernel or following operation |

If work is absent because it is fused, mark it `FUSED_INTO:<anchor>`. If work
is present but hidden, mark it `OFF_PATH` or `OVERLAPPED`. Do not classify both
cases as zero exposed time without preserving the mechanism.

## 7. Phase C: wall-sensitivity ranking

Compute the weighted critical path for both graphs. Then calculate, for every
node, class, and candidate edge:

```text
node ceiling  = CP_original - CP_with_node_duration_zero
class ceiling = CP_original - CP_with_class_durations_zero
edge ceiling  = CP_original - CP_with_legal_edge_removed
```

Recompute the path after every change so alternate-path takeover is included.
Do not sum raw node durations and call the result recoverable wall.

For each candidate, publish:

| field | meaning |
| --- | --- |
| observed llama mechanism | fused, shorter body, off-path branch, overlap, or host behavior |
| tinygrad difference | exact node/body/edge that does not match |
| node mass | total work represented by the class |
| current CP mass | work currently on the weighted path |
| zero-cost ceiling | maximum wall reduction before alternate-path takeover |
| legal mechanism ceiling | reduction from the specific buildable change |
| measured A/B | current-HEAD endpoint result |
| confidence | measured, simulated, inferred, or unknown |

The ranked list, not node sum or overlap mass alone, decides which lever moves
the needle.

## 8. Required reconciliation checks

The final accounting must satisfy all of these:

```text
unprofiled wall = device union + host gap
node_sum - device union = overlap mass
weighted critical path <= device span
sum(S0..S4 + anchor exposure) reconciles with the per-layer device path
predicted candidate ceiling >= measured candidate wall gain
```

Any residual greater than measurement noise remains open and is not assigned
to "serialization" or "launch overhead" without a measured mechanism.

## 9. Decisions already closed unless new evidence invalidates them

- More than two compute GPFIFOs: current-DAG ideal ceiling below the promotion
  bar before real wait tax.
- Replay merge: measured negative on production wall.
- Early `launch_dependents` START placement: no new overlap beyond the landed
  QMD latch behavior.
- Coarse flash split S=4/S=2: measured substantially slower on `DEV=NV`.
- Vocab tail: retain in the explanation ledger, but its measured wall transfer
  is too small to explain the primary gap.

These may be reopened only if the current weighted trace shows that a changed
HEAD invalidated the earlier topology or ceiling.

## 10. Deliverables

1. Current-HEAD RMSNorm control/candidate/control wall and topology record.
2. Weighted tinygrad DAG for one steady token.
3. Weighted llama DAG for the matched steady token.
4. Per-layer S0-S4 inter-anchor comparison.
5. Node/class/edge wall-sensitivity table with alternate-path takeover.
6. A reconciled causal ledger whose rows sum to the measured wall gap.
7. A prioritized build list containing only legal mechanisms with a measured
   or simulated wall ceiling above the implementation cost.

## 11. Completion condition

This scope is complete when the explanation can answer, without relying on
aggregate overlap as a proxy:

```text
llama is faster by X us/token.
Y us comes from these shorter weighted support segments,
Z us comes from these fused/off-path mechanisms,
H us comes from host submission,
and the remaining residual is within measurement noise.
```

Until then, RMSNorm and other candidates are hypotheses with measured local
benefits, not yet complete explanations of llama's wall advantage.
