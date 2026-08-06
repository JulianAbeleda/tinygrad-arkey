# NV decode exposure / overlap / host forward scope

Date: 2026-08-05
Status: **scope decided; CPU dev landed; all GPU arms parked**
Authority: `nv-decode-final-composed-same-session-record-20260805.md`
Constraint in force: no GPU use. Every item below is either CPU-only or gated
behind a CPU-verified forecast that must pass before any token wall arm.

## Question and answer

What remains between native NV and llama.cpp at d512, and in what order should
we spend GPU time once the GPU is available? The composed same-session
authority is native `5.3242440 ms/token` (187.82 tok/s) versus llama
`4.0056768 ms/token` (249.65 tok/s), a gap of **1318.5672 us/token**
(`0.75235x`). The fixed accounting authority is `5.612310 / 3.966140
ms/token`, gap **1646.170000 us/token**, with 270.468 us booked
(P1+P2+P5+Q4-g12+max17), remainder **1375.702 us/token**.

The audit locates the gap with a PASS on where but a FAIL on independent
recoverability:

```text
device window  +1401.616 us (85.14%)
  support-work exposure   +1108.082   (hidden-overlap 445.954
                                       + fusion/dataflow 662.128)
  quant cores             +302.788
  llama internal gaps       -8.111
  profile/device bridge     -1.143
outside window   +239.805 us (14.57%)
  native outside 321.784 vs llama 81.979
outer reconciliation  +4.749
= 1646.170000 us/token
```

Family Shapley attribution of the support term (ownership, not savings):
norms +574.654, flash +247.989, residual/cast/contiguous +240.319,
vocab/feedback +71.215, RoPE/KV +33.543, llama Q8 pack -59.639
(sums to 1108.081). The 662.128-us component is **fusion/dataflow and
exposed-work attribution**, not raw implementation cost; no row here is
bookable without an exact-output native A/B.

## tok/s translation at the composed baseline

Marginal deltas from 187.82 tok/s at `5.3242440 ms/token` (each row assumes
only that bucket is recovered; not a stacking table):

| bucket | us/token | tok/s after | +delta |
| --- | ---: | ---: | ---: |
| host outside-window delta | 239.805 | 196.68 | +8.86 |
| predispatch combined A/B (causal, unbooked) | 69.166 | 190.29 | +2.47 |
| defensive copy/rebind feedback | 121.380 | 192.20 | +4.38 |
| JIT input/signature reconstruction | 77.927 | 190.61 | +2.79 |
| scalar copyout + `Tensor.item()` | 83.247 | 190.80 | +2.98 |
| hidden-overlap (support behind quant) | 445.954 | 204.99 | +17.17 |
| fusion/dataflow attribution | 662.128 | 214.49 | +26.67 |
| support-work exposure total | 1108.082 | 237.18 | +49.36 |
| quantized-core attribution | 302.788 | 199.15 | +11.33 |

The disjoint recovery stack is the audit equation itself: outside delta
239.805 + overlap 445.954 + fusion 662.128 + quant cores 302.788 - llama
gaps 8.111 - bridge 1.143 + outer 4.749 = 1646.170 us, parity by
construction against the fixed authority. The composed gap of 1318.567 us
implies a composed host residual of about 210.5 us after P1/P2/P5 host
recoveries; the outside term is the single largest *cheap* host target.

## The arithmetic that decides overlap: calibrated wait cost

The P4 dependency-coherent cut was CPU-eligible at the old wait model
(0.363 us/effective wait, 184.992 us raw, 171.486 us costed, 73 effective
waits) and wall-NO-GO (+10.474 us vs A1, +9.199 vs A2). Backing the true
per-wait cost out of the wall:

```text
naive full-propagation bound = (184.992 + 10.474) / 73 = 2.678 us/wait
                              (7.38x the model)
```

That naive bound assumes every wait fully propagates into the final span.
The CPU schedule absorbs some waits into the critical path, so the honest
calibration is the wait cost that makes the Q-cut forecast reproduce the wall
delta (-10.474 us) under the schedule itself. Bisecting on the redirect-on
authority DAG yields **3.1865 us/wait**. At that cost every known
occurrence-pinned cut fails:

| cut | raw saving | effective waits | costed at 3.187 | verdict |
| --- | ---: | ---: | ---: | ---: |
| redirect-on Q rope/reduction chain | 184.992 | 73 | -10.474 us | CPU_NO_GO |
| redirect-on K rope chain | 160.320 | 73 | +42.962 us | CPU_NO_GO |

The Q row reproduces the wall exactly by construction (the calibration anchor);
the K row is the honest forecast at the same wait cost and still misses the
+50 us promotion gate by 7 us. The quarantined 948-capture rows (K 175.008 /
105 waits, Q 194.912 / 105 waits) must not be applied to the redirect-on graph.
Cross-queue cuts are therefore closed as scoped for this d512 redirect-on
construction.

Overlap (445.954 us) therefore requires **in-graph co-scheduling without
cross-queue waits** -- llama's mechanism of driver co-scheduling independent
nodes on one CUDA graph -- not two-queue cuts. Native two-GPFIFO concurrency
exists (9.7% light-kernel overlap verified), but queue/wait economics are
negative in the real decode graph.

## Workstream 1: overlap (445.954 us, 17.17 tok/s)

1. **Wait-adjusted forecast gate (CPU, this turn).**
   `nv_wait_adjusted_cut_forecast.py` reruns every occurrence-pinned cut at
   the calibrated 2.678 us/wait and emits a per-cut verdict. Any future cut
   candidate must first clear +50 us at calibrated cost on a fresh
   duration-bearing DAG from the current closed model graph.
2. **Close two-queue cuts as scoped.** No candidate clears the gate; do not
   reopen without a new mechanism (see gate above).
3. **In-graph co-scheduling is the only remaining overlap mechanism.**
   Name it explicitly in the campaign ledger; its feasibility question is
   whether the capture graph has dependency-independent support nodes whose
   complementarity with quant nodes can be proven.
4. **Resource-join census augmentation (CPU-only enabling step).** Add the
   compiled grid/block/registers/static+dynamic shared/local tuple to a
   tinygrad capture via `route_b3_dag_attribution.attach_compiled_descriptors`
   and rerun `nv_overlap_resource_join.py`. Only a dependency-independent
   pair with a positive complementary CTA residency bound authorizes a native
   two-queue span A/B. The join currently fails closed
   (`INCONCLUSIVE_FAIL_CLOSED`) because llama's manifest lacks local memory
   and tinygrad's census lacks compiled resource tuples.

HARD STOP: no GPU arm for overlap until (a) a fresh calibrated forecast
passes +50 us or (b) the resource join names an independent pair with a
positive residency bound.

## Workstream 2: graph exposure / fusion (662.128 us, 26.67 tok/s)

The audit's one admissible construction is an **ordinary-UOp in-core native
projection epilogue with no custom boundary or adapters**. Every custom-boundary
epilogue measured to date is closed:

| epilogue | result | disposition |
| --- | --- | --- |
| attention-O custom | +69 us | NO-GO, 0 credit |
| llama-O custom | +21.2 us | NO-GO, 0 credit |
| KV custom | neutral | closed |
| FFN-down custom | neutral | closed |
| RMSNorm semantic wrapper | +60.802 us, 110 lazy-view kernels | NO-GO |

1. **Decisive missing experiment (GPU, parked):** exact-output native A/B
   per fusion/dataflow population. This is the largest unresolved causal
   ambiguity; nothing in the 662.128-us attribution is bookable before it.
2. **Boundary-free gate is already CPU-testable**
   (`nv_boundary_free_ordinary_uop_gate.py`): an epilogue qualifies only if
   it lowers as ordinary UOps in-core with no custom CUDA boundary or
   adapters. Use it as the acceptance predicate for any new projection
   epilogue.

HARD STOP: no projection-epilogue GPU arm without an exact-output contract
and the boundary-free gate passing.

## Workstream 3: host / outside window (239.805 us, 8.86 tok/s)

Native outside-window partition (321.784 us native total vs 81.979 llama):

| component | us/token |
| --- | ---: |
| preparation before first graph call | 247.557 |
| - defensive copy/rebind feedback | 121.380 |
| - JIT input/signature reconstruction | 77.927 (incl. 49.284 structural graph rewrite) |
| - pre-TinyJit | 35.637 |
| - cache | 1.112 |
| scalar copyout + `Tensor.item()` | 83.247 |
| Python yield tail | 3.096 |
| redundant sync after `next()` | 4.358 |
| disjoint sum | 338.458 (closes 321.784 within tolerance) |

1. **Full-logit predispatch oracle A/B (GPU, parked).** Book the causal
   69.166 us signal (`nv_predispatch_full_logits_qualification.py` seam;
   sampled-token A/B exists). Full-logit equality is the admission rule.
2. **JIT input/signature reconstruction caching (CPU dev, closed default).**
   Cache the structural graph rewrite (49.284 us) keyed by program identity;
   closed default, hermetic unit tests, no behavior change when off.
3. **Reduce defensive copy/rebind (121.380 us).** Requires the P5 ping-pong
   seam audit; parked pending GPU.
4. **Copyout/item alternatives.** Packed greedy argmax was NO-GO
   (71.874 -> 142.647 us); do not reopen without a different mechanism.

## Dev this turn (CPU-only)

- `extra/llm_research/decode/nv_wait_adjusted_cut_forecast.py`: calibrated
  forecast gate. Default mode bisects the wait cost so the Q-cut forecast
  reproduces the P4 wall delta under the schedule itself (3.1865 us/wait on
  the redirect-on authority DAG), then scores every cut at that cost with the
  +50 us promotion gate, per-cut verdict, and break-even/gate wait costs.
- Hermetic test pinning calibration arithmetic and the three known cuts.
- This scope document.

No production default changes, no GPU work, no behavior changes while the
GPU ban is in force.

## References

- `nv-decode-final-composed-same-session-record-20260805.md` (current authority)
- `nv-decode-final-accounting-audit-20260805.md` (location PASS / recoverability FAIL)
- `nv-decode-parity-campaign-reconciled-ledger-20260805.md` (booked recoveries)
- `nv-decode-p4-dependency-closed-cut-record-20260805.md` (wait-cost calibration)
- `nv-decode-native-d512-host-partition-record-20260804.md` (host partition)
- `nv-rank2-native-concurrency-construction-verdict-20260805.md` (two-GPFIFO construction PASS)
- `nv-overlap-resource-compatibility-ledger-20260805.md` (resource join fails closed)
- `nv-decode-parity-p6-residual-priority-ledger-20260804.md` (P6 rows, host items)
