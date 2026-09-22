# NV edge-aware PDL runtime-hook result

Date: 2026-08-21

Commit: `6570abc025514273faa100c66b979e531585a1e1`

Scope: `docs/task_workflow/input/nv-edge-aware-pdl-runtime-hook-scope-20260821.md`

Evidence: `docs/task_workflow/evidence/nv-edge-aware-pdl-runtime-hook-20260821/`

## 1. Decision

**Decision: stop PDL endpoint work and spend the gap budget on bounded fusion
and Q4 FFN-down (section 1 cell 2; section 9 cell 3).**

The edge-aware split launch/data dependency is real and works on this device,
but it does not convert to endpoint recovery. The best measured arm recovers
at most an unattributable amount, and every candidate S1 value sits inside
the control spread. There is no promotion, no route-default change, and no
production interface change in this packet.

The gate remains `NV_SPLIT_PHASE=1` with `off` as the closed default. The
carry-forward budget is the locked S1 gap (`observed +634.334 us`) and the
locked token-wall gap (`observed +717.505 us`) against llama, minus zero
attributable PDL recovery.

This is "no demonstrated recovery", not "PDL disproven". The endpoint gate's
150 us threshold sits below the measured control spread, which is reported
in section 5 and is the packet's primary measurement caveat.

## 2. Q1-Q8 verdicts

| question | verdict | evidence |
| --- | --- | --- |
| Q1. Scheduler arms the safe RAW chain | supported for the conservative rule; full chain partially armable | construction census armed 83 edges; every other row has a named reason (section 3) |
| Q2. Native QMD lowering matches CUDA semantics | supported, with one native inert placement | Stage 2 semantic gate passed; latch sweep 0-83; trigger-position discriminator present; native wait-placement inert |
| Q3. Real decode route launches ahead | supported | both probe edges positive overlap, wait-exit valid, tokens identical |
| Q4. Launch-ahead converts to wall recovery | refuted | no arm reaches the 150 us gate; all deltas inside control noise |
| Q5. Factor attribution | measured, not promoted | nominal rank: start > end on 2 queues, 2 queues > 1 queue, one graph worst; every delta inside noise |
| Q6. Native negative while CUDA positive | named-unavailable | no CUDA real-route endpoint; synthetic Stage 2 rows are positive on both backends |
| Q7. Residual after the best arm | observed with attribution caveat | best candidate 2259.5 us S1 vs 2349.75 us control median, +90.25 us nominal; not separable from noise; residual budget stays at the locked 634.334 us |
| Q8. Graph grouping matters | measured: not supported | one-graph candidate -298.4 us S1 vs own controls, but control spread 329.25 us |

## 3. Q1 - construction census

The live construction census behind `NV_SPLIT_PHASE=1` arms 83 edges across
the five first-cycle replay groups (`observed`). Every remaining typed row has
a named reason (`observed`):

| reason | rows |
| --- | ---: |
| candidate_armed | 83 |
| multi_producer_fallback | 390 |
| adjacency | 352 |
| multi_consumer_fallback | 246 |
| queue_split | 140 |
| alias_rejected | 45 |
| encoded_wait | 36 |

The 83 armed edges reuse the 8-latch pool: latch id 0 is armed 12 times and
id 7 eight times (`observed`). The full real-route arm with the pool at 8
keeps the token SHA identical to control, so latch wrap is numerically safe
on this route when every armed edge also carries its kernel-side half
(section 4). The conservative rule leaves 1,164 typed rows on the existing
full-completion path; their reasons are the named coverage limits, not
silent misses.

## 4. Q2 and Q3 - mechanism evidence

Stage 2 matched-grid semantics (`observed`):

- latch IDs 0-83 each support a checksum-correct producer/consumer pair with
  positive overlap on the matched grid; per-ID overlap medians are 99.744 to
  99.808 us;
- native trigger-position discriminator is 109.888 us (CUDA 103.136 us),
  so moving the trigger from end to start moves the consumer launch;
- native wait-placement discriminator is 0.032 us (CUDA 2.336 us): on native,
  entry versus prologue wait placement is inert on the matched grid;
- data-readiness wait exit lands 0.352-0.384 us after producer end;
- one producer latch waited on by two consumers is checksum-correct
  (supported); arming across a non-consecutive middle kernel is refuted; a
  two-producer latch is left named-unavailable per the Stage 2 promotion rule.

Stage 3 real-route capture (`observed`), probe edges:

| probe edge | overlap (consumer grid start before producer end) | wait exit after producer end |
| --- | ---: | ---: |
| graph 1, 5 -> 7 (`E_2` -> `E_1187_32_4`) | +0.25 us | +0.3 us |
| graph 4, 392 -> 393 (`r_128_16_8_1187` -> `r_16_8`) | +9.75 us | 0.0 us |

The candidate and both controls produced token id 13876 with identical
1-token SHA `51a6b5bb6e00...` (`observed`). All ten Stage 3 gates passed.

### 4.1 The kernel-side half is required

The first Stage 3 capture armed the QMD latch halves for all 83 edges but
injected the matching `griddepcontrol.wait` / `launch_dependents` halves for
only the probe programs. That half-armed route diverged: the candidate
produced token 34583 while both controls produced 13876 (`observed`). After
the policy was extended to emit both halves for every armed edge, the
candidate SHA matched control again. This is direct evidence that the QMD
launch gate alone is not a data-ready gate, and that the full pairing is
numerically safe on the real route.

## 5. Q4, Q5, Q7, Q8 - endpoint brackets

Seventeen fresh-process rows, 8 tokens each, control/candidate/control per
arm, serialized under `flock /tmp/gpu-bench.lock`. Every row shares the same
8-token SHA `323f407295a7...` (`observed`). S1 is the summed per-layer
O.start - Q.end exposure of the final token's DAG; wall includes the
`PROFILE=1` instrumentation tax and is not an unprofiled wall.

| arm | role | S1 us | wall us | union us | dead us |
| --- | --- | ---: | ---: | ---: | ---: |
| off | control | 2463.75 | 7455.56 | 4581.5 | 1506.25 |
| off | candidate | 2401.75 | 7301.73 | 4581.5 | 1440.75 |
| off | control | 2511.0 | 7402.85 | 4578.75 | 1555.75 |
| 1q end/entry | control | 1958.5 | 6946.22 | 4552.0 | 1035.25 |
| 1q end/entry | candidate | 2233.5 | 7119.83 | 4526.5 | 1327.0 |
| 1q end/entry | control | 2214.5 | 7240.96 | 4554.75 | 1293.25 |
| 2q end/entry | control | 2110.5 | 7264.35 | 4583.0 | 1152.75 |
| 2q end/entry | candidate | 2207.0 | 7163.56 | 4507.0 | 1321.75 |
| 2q end/entry | control | 2114.5 | 7134.78 | 4588.5 | 1154.0 |
| 2q start/entry | control | 2357.25 | 7264.31 | 4584.25 | 1397.0 |
| 2q start/entry | candidate | 2259.5 | 7240.93 | 4477.25 | 1378.75 |
| 2q start/entry | control | 2342.25 | 7348.38 | 4577.75 | 1388.75 |
| 2q end/prologue | control | 2369.0 | 7299.15 | 4581.5 | 1412.5 |
| 2q end/prologue | control | 2021.25 | 6982.57 | 4596.0 | 1058.0 |
| one graph | control | 1978.75 | 7007.54 | 4579.0 | 1025.75 |
| one graph | candidate | 2441.75 | 7256.69 | 4505.75 | 1556.0 |
| one graph | control | 2308.0 | 7355.13 | 4581.5 | 1350.5 |

Belief-flip deltas (`observed`, not attributed):

| arm | S1 delta (base - candidate) | wall delta (candidate - base) |
| --- | ---: | ---: |
| off (phantom, no mechanism change) | +85.63 us | -127.47 us |
| 1q end/entry | -147.0 us | +26.24 us |
| 2q end/entry | -94.5 us | -36.01 us |
| 2q start/entry | +90.25 us | -65.41 us |
| one graph | -298.38 us | +75.36 us |

The twelve control S1 values span 1958.5 to 2511.0 us, a 552.5 us range
(`observed`). Two identical prologue controls differ by 347.75 us and two
identical one-graph controls differ by 329.25 us (`observed`). Every candidate
S1 (2207.0 to 2441.75 us) lies inside that control range, so no arm is
separable from the control distribution and the 150 us belief-flip gate is
below the noise floor of this bracket design.

Q5 nominal ranking is start-trigger (+90.25 us) better than end-trigger
(-94.5 us) on two queues, two queues better than one (-147.0 us), and one
graph worst (-298.38 us). None of these is promoted: the off phantom
(+85.63 us) shows the same magnitude with no mechanism.

Q7 residual: best candidate 2259.5 us against a 2349.75 us control median
gives +90.25 us nominal, which cannot be attributed above the control spread.
The locked S1 gap (+634.334 us) therefore carries forward in full to the
8.2 Q4 FFN-down and 8.3 bounded-fusion budgets (`inferred` from the locked
ledger plus the measured zero-attributable recovery).

Q8: the one-graph arm used its own one-graph control as required. Its
-298.38 us delta is inside the 329.25 us control spread, so graph grouping
does not carry the recovery (`measured`, labeled not supported).

## 6. Named missing capabilities

The construction cannot currently express the full safe RAW chain. The named
missing pieces are:

- multi-producer latch merge: construction falls back on 390 rows; the
  Stage 2 two-producer probe stayed named-unavailable;
- non-consecutive same-queue arming: refuted on the matched grid (the latch
  does not cross an unarmed middle kernel), so adjacency remains fallback;
- queue-split (140) and encoded-wait (36) edges: fallback by design;
- alias adjudication needs retained buffer spans; 45 rows are rejected and
  the capture does not retain overlapping spans;
- wait-before-first-dependent-access placement: the renderer has no literal
  source anchor on the real route, and native wait placement measured inert
  on the matched grid anyway;
- CUDA real-route endpoint: not constructed, so Q6 stays named-unavailable.

## 7. Measurement-layer repairs made this packet

These are probe-tooling fixes, not production changes:

- `PROFILE=1` and `HCQ_GRAPH_PROFILE_JSON` are set before `tinygrad` import
  in the Stage 3 capture worker; the earlier order exported empty graph
  profiles;
- the Stage 3 policy now emits both kernel-side halves for every armed edge
  (17 structural rules, 166/166 live name occurrences covered) and times only
  the two unambiguous probe edges;
- policy rules match on the structural kernel name with the 64-hex process
  hash stripped, because Python hash randomization changes that suffix across
  fresh processes;
- the positive-overlap gate reads scratch `%globaltimer` values, not the
  HCQ profile start signals, because HCQ reuses the producer end signal as
  the consumer start for chained same-queue pairs.

## 8. Acceptance checks

- off path unchanged: every control and the off candidate share identical
  token SHA with no `NV_SPLIT_PHASE` set (`observed`);
- construction census names every armed and missed row (`observed`);
- synthetic native/CUDA semantics measured, not assumed (`observed`);
- real-route capture records wait-exit, not just grid start (`observed`);
- endpoint recovery read from control/candidate/control with identical
  tokens (`observed`);
- Q1-Q8 each have a verdict or a named unavailable reason (section 2);
- the decision is cell 2 of section 1 and is falsifiable by the retained
  evidence.

## 9. Retained evidence

- `stage1_reconciliation.json`, `phase_a_construction_census_v4.jsonl`
- `stage2_semantic_gate.json`, `stage2_latch_sweep.json`,
  `stage2_multi_consumer.json`, `stage2_non_consecutive_arming.json`,
  `stage2_two_producer_one_consumer.json`, `stage2_replay_flush.json`
- `stage3_selected_edges.json`, `stage3_all_armed_edges.json`,
  `stage3_capture.json`, per-arm `stage3_*.json` and `.profile.jsonl`
- `stage4_endpoint_merged.json`, `stage4_endpoint_rows.jsonl`, per-arm
  `stage4_*.json`

No benchmarks beyond this packet's bracketed measurements were presented, and
no number in this report crosses the measured body/grid boundary.
