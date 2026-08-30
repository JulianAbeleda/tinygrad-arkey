# NV split-phase PDL causal design review result

Date: 2026-08-20 (Phase D static-census addendum 2026-08-21)

Scope:
`docs/task_workflow/input/nv-split-phase-pdl-causal-design-review-scope-20260820.md`

Status: tested causal verdict plus design recommendation. No production
runtime, renderer, scheduler, or model files were changed. Probe-only changes
live under `extra/llm_research/decode/**`; all retained evidence lives under
`docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820/`.

Notation used throughout: `O` observed, `I` inferred from source or an
interval, `U` unmeasured.

## 1. Causal answer

The locked 717.505 us/token wall gap is still not closed by either candidate
diagnosis. What is observed is where it lives: llama overlaps 1128.020 us of
kernel residence mass against tinygrad's 5.996 us, and the decisive S1 segment
(Q end to O start) exposes 1152.250 us for tinygrad versus 517.916 us for
llama, a +634.334 us contribution (O). The split-phase mechanism itself is
real: the same-grid Phase C probe showed both CUDA PDL and native QMD release
a consumer grid before the producer finishes, then hold it at an in-kernel
data wait, and the CUDA trigger position moves the launch shadow by roughly
54 us (O). However, the current tinygrad PDL arm only armed 108 (1q) or 144
(2q) pairs, every one `gemv->support`, with zero `support->support` and zero
`support->gemv` edges, and its measured positive overlap count was zero in
both queue modes (O). It therefore does not reproduce llama's 761-edge
programmatic chain, and the earlier -11.641/-8.201 us endpoint result cannot
falsify split-phase dependencies (I). On the fusion side, the only measured
legal fold on this head is copy-free fp16 RMSNorm at +12.506 to +17.141 us,
the measured Q4 FFN-down body deficit is +30.016 us (corrected floor; see the
2026-08-21 addendum in section 4), and the residual/reduce
ceilings remain unmeasured simulations; the legal composition required to
close 717.505 us has not been demonstrated (O/I). The endpoint decision
between Direction A and Direction B is therefore `unmeasured`. The
Phase D static census now shows that the existing name-pinned arm cannot
express a llama-equivalent chain even when every unique program name is
admitted: on two queues it reaches at most 176 of llama's 328
`support->support` edges and zero of its 144 `support->gemv` /
`gemv->support` edges, and all 27 cross-group edges are unreachable. The
next step is an edge-aware runtime-hook construction scope, followed by the
Q4 FFN-down body bracket and a bounded-fusion bracket.

## 2. Corrections to the prior brief

The prior briefs
`docs/task_workflow/output/nv-ledger-roofline-pseudocode-brief-20260820.md`
and
`docs/task_workflow/output/nv-pdl-queue-theories-test-20260820.md`
contain these corrections:

1. **Sign convention.** The device-union identity is
   `delta_union = delta_node_sum - delta_overlap`, not
   `delta_node_sum + delta_overlap`:

   ```text
   delta_node_sum   = tinygrad - llama = -282.301 us
   delta_overlap    = tinygrad - llama = -1122.024 us
   delta_union      = (-282.301) - (-1122.024) = +839.723 us
   ```

   In advantage form, llama gains 1122.024 us from hiding and loses
   282.301 us by carrying more residence mass, for a net llama device win of
   839.723 us.

2. **`node_sum` is not work.** It is summed kernel residence-time mass:
   `sum(end_i - start_i)`. Under PDL a consumer interval can include time
   spent waiting at a grid dependency, so residence mass is not FLOPs, bytes,
   operations, or simultaneous useful execution. Only the identity
   `node_sum - union = overlap_mass` is exact.

3. **`overlap_mass` is not simultaneous useful traffic by itself.** The llama
   1128.020 us figure is interval overlap, not measured concurrent useful work.
   The Phase C probe demonstrates the distinction: in the matched CUDA
   `pdl_start` grid, overlap mass is 99.680 us while the measured in-kernel
   wait is 100.128 us, so nearly all of that synthetic overlap is launch-ahead
   plus dependency wait (O). The llama endpoint split remains unmeasured until
   a consumer wait-exit timestamp exists (U).

4. **Retract "Theory A is falsified."** The queue/PDL result called
   `correct-direction PDL` a loss, but Phase B now shows the arm armed only
   `gemv->support` pairs and no support-to-support chain. A negative result on
   a non-equivalent construction does not falsify the split-phase diagnosis.
   The prior brief's conclusion, "the remaining mechanism is not 'add more
   overlap'; it is to remove the separate support programs," is too strong
   and is downgraded to unmeasured until equivalent edge coverage and wait
   placement are tested.

5. **Do not award llama the folds tinygrad already has.** Tinygrad at
   `6570abc02` already folds the O residual, gate/up GLU, and down residual.
   Its S3 exposure is 0.000 us against llama's 15.937 us, a tinygrad win (O).
   The gap cannot be attributed to those fused epilogues.

6. **Llama "hides" support but does not delete it.** The 764.057 us behind
   MMQ anchors and 326.134 us of S1 self-overlap are residence-interval
   overlaps. They do not prove that the hidden bodies execute concurrently
   with useful anchor traffic, only that their grid lifetimes overlap (O for
   the intervals, U for useful-body concurrency).

7. **Critical-path comparison direction.** Llama's logical critical path
   4443.435 us exceeds its observed device span 3901.205 us because PDL
   launch-completion lets consumers start before producers logically finish.
   Tinygrad's span 4842.250 us exceeds its critical path 4249.216 us because
   off-path support fills otherwise serial gaps. Both are consistent, but they
   mean different things and should not be reported as the same inequality.

8. **Byte rooflines are estimates.** The ~5.04 vs ~4.70 GB/token figures are
   accounting estimates, not hardware DRAM counters. The S1 conclusion does
   not depend on them; it depends on the timestamp ledger.

## 3. First-layer launch, wait, and useful body

Three quantities are distinct: `grid-start` (kernel grid has begun),
`wait-exit` (consumer crossed its PDL dependency wait), and `useful-body`
(dependent work after wait-exit). Timestamps in the tables are microseconds
relative to the token start.

### 3.1 Tinygrad first layer

Control route, canonical current-HEAD capture. Serialized, so there is no
launch-ahead wait to hide: grid-start and useful-body are effectively the same
phase.

| interval | start | end | observed status |
| --- | ---: | ---: | --- |
| Q GEMV `q4k_..._4096_4096` | 20.000 | 28.750 | O grid start/end |
| q norm `reduce_output_rmsnorm_32_128` | 28.750 | 32.500 | O |
| q rope `E_16_32_4_2` | 32.500 | 35.250 | O |
| K GEMV `q4k_..._1024_4096` | 36.000 | 41.000 | O |
| V GEMV `q6k_v_..._1024_4096` | 41.000 | 45.500 | O |
| k norm `reduce_output_rmsnorm_8_128` | 45.500 | 48.000 | O |
| store/cast `E_8_8_16_2` | 48.000 | 49.750 | O |
| no kernel active | 49.750 | 53.750 | O dead device time |
| flash score | 53.750 | 62.000 | O |
| flash combine | 62.000 | 65.000 | O |
| O GEMV `..._epi_resadd_4096_4096` | 65.000 | 74.500 | O |

Between Q end 28.750 and O start 65.000 no anchor runs; every support kernel
occupies serial device time, and there is a 4 us hole at 49.750-53.750 (O).

The Phase B native-PDL candidate capture records zero positive overlap
over every armed pair in both queue modes (O). On the first layer the q norm
grid starts exactly at the Q GEMV end (23.250 in the capture), and the
median launch shadow is 9.750 us on 1q and 9.250 us on 2q, so the consumer
grid is consistently behind the producer end. The producer trigger is `end`
and the consumer wait is prepended at instruction zero, so the current arm
never produces launch-ahead on the real route (O). No in-kernel
`%globaltimer` wait-exit was collected on the real route, so the real-route
wait-exit and useful body remain `unmeasured` (U). The Phase C matched grid
provides the measured wait semantics in section 6.3.

### 3.2 Llama first layer

Canonical llama trace:

All llama source lines below are under
`/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/`.

| kernel | start | end | trigger | wait site | status |
| --- | ---: | ---: | --- | --- | --- |
| attention norm | 0.000 | 3.423 | `norm.cu:100` start | `norm.cu:130` after prologue | O start/end; I wait-exit |
| Q quant | 0.671 | 4.223 | `quantize.cu:9` start | `quantize.cu:34` after index math | O start/end; I wait-exit |
| Q MMVQ | 1.215 | 11.071 | `mmvq.cu:733` end | `mmvq.cu:510` after parameter setup | O start/end; I wait-exit |
| q norm | 11.231 | 12.895 | `norm.cu:100` start | `norm.cu:130` after prologue | O/I |
| q rope | 11.839 | 13.727 | `rope.cu:137` start | `rope.cu:152` after index math | O/I |
| K quant | 12.383 | 14.495 | `quantize.cu:9` | `quantize.cu:34` | O/I |
| K MMVQ | 12.863 | 18.207 | `mmvq.cu:733` | `mmvq.cu:510` | O/I |
| V quant | 18.303 | 19.743 | `quantize.cu:9` | `quantize.cu:34` | O/I |
| V MMVQ | 18.783 | 22.911 | `mmvq.cu:733` | `mmvq.cu:510` | O/I |
| k norm | 22.943 | 24.479 | `norm.cu:100` | `norm.cu:130` | O/I |
| k rope | 23.359 | 25.215 | `rope.cu:137` | `rope.cu:152` | O/I |
| k store | 23.711 | 25.695 | `cpy.cu:19` | `cpy.cu:40` | O/I |
| flash score | 25.567 | 30.079 | `fattn-common.cuh:731` | `fattn-common.cuh:679` | O/I |
| flash combine | 26.111 | 31.167 | `fattn-common.cuh:918` | `fattn-common.cuh:859` | O/I |
| O quant | 26.687 | 31.775 | `quantize.cu:9` | `quantize.cu:34` | O/I |
| O MMVQ | 27.487 | 38.943 | `mmvq.cu:733` | `mmvq.cu:510` | O/I |

The key example: O MMVQ's grid starts at 27.487 before flash combine
(ends 31.167) and O quant (ends 31.775) finish. Its 11.456 us timestamp
duration therefore includes wait time; it is not an isolated 11.456 us useful
MMVQ body (O for the interval, U for the split inside it). The llama trace has
no `%globaltimer` wait-exit records, so the useful-body column in the endpoint
ledger remains an inference from call-site placement, not a measurement.

## 4. H1-H8 verdicts

Verdicts are endpoint verdicts unless noted. "Source-confirmed" proves code
shape; it does not prove endpoint causation.

All llama source lines below are under
`/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/`; tinygrad lines are relative
to the repository root.

| H | endpoint verdict | synthetic/source sub-verdict |
| --- | --- | --- |
| H1 | `supported` | wait-exit instrumented; 91.9-95.4% shadow, 4.6-8.1% useful |
| H2 | `supported` | construction census observed |
| H3 | `unmeasured` | source-confirmed placement differs; synthetic delta ~1.7 us |
| H4 | `unmeasured` | supported on matched synthetic grid |
| H5 | `unmeasured` | source/observed granularity differs |
| H6 | `unmeasured` | scheduler shape source-confirmed; synthetic schedule change supported |
| H7 | `unmeasured` | opaque boundary source-confirmed |
| H8 | `unmeasured`; measured subset does not close the gap | see composition below |

### H1: residence-time accounting

The llama 1128.020 us overlap mass is interval overlap; timestamp-only overlap
is insufficient to call it simultaneous useful traffic (O). The Phase C
matched CUDA grid gives the required wait-exit instrument: `pdl_start` has
99.680 us overlap versus 100.128 us wait, and `pdl_end` has 47.616 us overlap
versus 48.096 us wait, so on that grid the overlap is almost entirely launch
shadow and dependency wait (O). The llama endpoint is now measured with an
instrumented llama build: across seven steady decode replays, 91.9-95.4% of
llama's overlap mass is dependency wait plus launch shadow and only 4.6-8.1%
is simultaneous useful execution (O). Applied to the 20260821 ledger's
authority overlap mass of 1133.255 us (its refinement of the 1128.020 us
figure above), that is ~52-91 us useful versus ~1042-1081 us shadow per
replay. Evidence: `nv-llama-useful-body-h1-result-20260821.md`,
`nv-llama-useful-body-h1-20260821/h1-reconciliation.json`;
`phase_c_driver_fixed.json`, CUDA rows; `nv_pdl_phase_c_cuda_probe.py:81` for
the matched-grid wait instrumentation.

### H2: sparse PDL mismatch

`supported`. The static census and Phase B agree exactly: 108 armed pairs on
1q and 144 on 2q, all `gemv->support`, zero `support->support` and zero
`support->gemv`; controls arm zero and the candidate positive-overlap count
is zero in both queue modes (O). Llama's chain has 328
`support->support`, 145 `support->anchor`, 144 `anchor->support`, 72
`gemv->support`, and 72 `support->gemv` programmatic-adjacent edges (O).
The tested arm did not reproduce llama's launch-ahead chain.
Evidence: `phase_a_pdl_edge_census_1q.json`,
`phase_a_pdl_edge_census_2q.json`, `phase_b_driver.json`.

Phase D widened the name filter to all 34 unique program names and still
could not close the coverage gap; see section 6.5. H2 therefore remains
`supported` as a construction limitation. It does not by itself prove the
endpoint direction; that remains `unmeasured`.

### H3: wait-placement mismatch

`unmeasured` at endpoint; `source-confirmed` and effectively no synthetic
effect. Tinygrad prepends `griddepcontrol.wait` as the first consumer
instruction (`tinygrad/renderer/cuda.py:30`); llama syncs after index math
(`mmvq.cu:510`, `rope.cu:152`, `quantize.cu:34`, `norm.cu:130`). The matched
Phase C `pdl_prologue` arm moved wait from 48.096 us to 46.400 us on CUDA, a
~1.7 us effect (O). Wait placement is not the observed driver of the gap at
this grid shape.

### H4: trigger/QMD mismatch

`unmeasured` at endpoint; mechanism `supported` on the matched grid. The
native QMD latch arms `enable_program_pre_exit=1`,
`pre_exit_at_last_cta_launch=1` (`tinygrad/runtime/ops_nv.py:49`), while the
renderer default emits the trigger at producer end
(`tinygrad/renderer/cuda.py:31`). Llama emits
`cudaTriggerProgrammaticLaunchCompletion` near kernel start in norm, rope,
quantize, and unary (`norm.cu:100`, `rope.cu:137`, `quantize.cu:9`,
`unary.cu:119`). Phase C matched arms: CUDA trigger-at-start shadow
386.016 us vs trigger-at-end 488.128 us; native start 394.240 us vs end
503.808 us; both trigger positions launch ahead and wait on the synthetic
grid (O). On the real route the current end-trigger plus entry-wait arm
produced zero positive overlap (Phase B), which shows that arming
`gemv->support` pairs is not sufficient for launch-ahead there, but does not
isolate coverage, wait placement, graph grouping, or per-pair dependency
shape as the suppressor. Native `pdl_start` shows an 8.8 us producer and
9.5 us wall drift versus `pdl_end`, treated as noise.

### H5: graph-granularity mismatch

`unmeasured` endpoint causal effect; the difference is observed. Tinygrad has
five replay groups of sizes 32/64/128/256/116 from the doubling batch split
(`tinygrad/engine/jit.py:244`, `tinygrad/engine/jit.py:333`); llama runs one
762-node graph. Phase A treats group boundaries as QMD-chain resets and Phase
B records five group mappings. The only prior replay-merge attempt
(`JIT_BATCH_SIZE=1024`) was measured +112.9 us slower on production wall and
was not a PDL-arm bracket (`nv-weighted-inter-anchor-causal-gap-result-20260820.md`
section 8), so no equivalent one-continuous-graph PDL construction ran (U).

### H6: one-phase scheduler

`unmeasured` endpoint effect; the scheduler shape is source-confirmed and the
synthetic schedule change is supported. `HCQGraph._resolve_deps` lowers
resources to `(queue, value)` signals with no separate launch-readiness/data-
readiness edge (`tinygrad/runtime/graph/hcq.py:382`). The current split phase
is a name filter in the renderer and runtime, not an edge in the scheduler
(`tinygrad/renderer/cuda.py:26`, `tinygrad/runtime/ops_nv.py:38`). Phase C
proves that adding a launch/data split changes the schedule: native
`pdl_start` overlap 117.472 us and in-kernel wait 117.824 us, versus ~0
overlap in `no_pdl` (O). Native `qmd_latch` launches ahead with wall
463.488 us, but it has no in-kernel `griddepcontrol.wait`, so that row is
launch-ahead without split-phase data readiness and is not llama's semantics.

### H7: opaque primitive boundary

`unmeasured` in this packet; the boundary is source-confirmed. The decode
GEMV is an opaque `UOp.custom_kernel` program (`tinygrad/tensor.py:316`), the
epilogues are manually enumerated in `Q4KGEMVEpilogue`
(`tinygrad/llm/decode_kernels.py:170`), and model admission is explicitly
decode-specific (`tinygrad/llm/model.py:648`). Existing typed views are
fail-closed per-slot machinery (`tinygrad/llm/kernel_program.py:283`). Prior
route records report M3/M4 non-landing because the opaque boundary
materializes per-input copies
(`tinygrad/llm/generated/decode-norm-fusion-route-policy.json`,
`tinygrad/llm/generated/decode-q4k-epilogue-fusion-route-policy.json`), but
the exact view-preserving/fused probe required for an endpoint-causal H7
verdict was not run in this review (U).

### H8: fusion-only sufficiency

`unmeasured` as a closed endpoint claim; the measured subset does not close
the gap. Recomputed composition, not raw ceiling addition:

| change | bookable evidence | number |
| --- | --- | ---: |
| copy-free fp16 RMSNorm | measured wall A/B | +12.506 to +17.141 us |
| Q4 FFN-down body | measured matched anchor rows | +30.016 us available (corrected 2026-08-21) |
| residual fold | simulated zero-cost CP ceiling | 148.736 us |
| output-reduction fold | simulated zero-cost CP ceiling | 203.680 us |

Residual + reduction ceilings total 352.416 us before alternate-path takeover.
Adding the RMSNorm zero-cost ceiling (234.528 us, not the measured 17 us)
would give 586.944 us; adding vocab (313.632 us) would give 900.576 us. That
naive sum exceeds the gap and is not admissible: ceilings overlap on the
critical path, alternate paths take over when support work is removed, and no
legal residual/reduce variant has been built or endpoint-measured. The only
measured fusion gain is 12.5-17.1 us. Tinygrad already wins the aggregate
anchor union by 54.867 us, so fixing Q4 FFN-down at the corrected 19.232 us
floor would make tinygrad win by roughly 85 us there while leaving the S1
exposure gap untouched (I).
H8 is therefore not a proven close.

**2026-08-21 addendum (Q4 FFN-down floor).** The
08-13 core correction pins llama's Q4 FFN-down at 19.232 us/node; the 11.776
us value quoted in the original H8 row is attention-O's value and is retired
here. Re-running the recomposed zero-cost composition with the corrected
floor is `nv-h8-fusion-composition-20260821.json`: the Q4 ceiling is 30.016
us (18 x (20.900 - 19.232)), and residual + reduce + vocab + Q4 compose to a
696.064 us ceiling, leaving 21.441 us of the 717.505 us wall gap even at
zero cost. Every measured row converts far below its ceiling, so the binding
limit remains wall conversion, not path interference.

## 5. Edge-coverage comparison

Comparable buckets, llama 762 nodes / 761 programmatic-adjacent edges versus
the tested tinygrad arm (596 nodes, 1230 edges):

| edge class | llama | tested 1q | tested 2q |
| --- | ---: | ---: | ---: |
| support -> support | 328 | 0 | 0 |
| support -> anchor | 145 | 0 | 0 |
| anchor -> support | 144 | 0 | 0 |
| gemv -> support | 72 | 108 | 144 |
| support -> gemv | 72 | 0 | 0 |
| total programmatic | 761 | 108 | 144 |

Tinygrad's census bucket maps every named q4k/q6k producer into `gemv`, so
the tested pairs are all `gemv->support`. The 108/144 counts were not a
prediction: Phase B recorded the exact same armed counts on device, with zero
armed controls and byte-identical tokens across control/candidate/control
(O). Five graph-group boundaries reset the native QMD chain, so pairs never
cross groups in the tested construction. The `gemv` bucket conflates llama's
anchor and K/V classes, so the support-to-support and support-to-gemv zeros
are the directly comparable rows; the 108/144 cannot be assigned back to
llama's `anchor->support` row one-to-one. The Phase D all-names census in
section 6.5 shows that lifting the name filter does not lift the structural
limits: adjacency, queue splits, encoded waits, and group boundaries still
block the chain.

## 6. Commands, controls, raw evidence, results

### 6.1 Phase A: ledger and static census

Commands:

```bash
.venv/bin/python extra/llm_research/decode/nv_rmsnorm_current_head_topology.py \
  --arm control --sites ffn \
  --out docs/task_workflow/output/nv-rmsnorm-phaseB-control-20260820.json

.venv/bin/python extra/llm_research/decode/nv_inter_anchor_analysis.py

.venv/bin/python extra/llm_research/decode/nv_pdl_phase_a_census.py \
  --control docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820/phase_a_control.json \
  --llama docs/task_workflow/output/nv-weighted-llama-real-edge-dag-20260820.json \
  --queues 2 \
  --out docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820/phase_a_pdl_edge_census_2q.json

.venv/bin/python extra/llm_research/decode/nv_pdl_phase_a_census.py \
  --control docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820/phase_a_control.json \
  --llama docs/task_workflow/output/nv-weighted-llama-real-edge-dag-20260820.json \
  --queues 1 \
  --out docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820/phase_a_pdl_edge_census_1q.json
```

Result: locked wall 4723.214 vs 4005.709 us, gap +717.505; device union
+839.723; S1 +634.334; interval identity closes exactly. Retained:
`phase_a_control.json`, `phase_a_ledger.json`, `phase_a_sensitivity.json`,
`phase_a_pdl_edge_census_1q.json`, `phase_a_pdl_edge_census_2q.json`.

### 6.2 Phase B: current construction census

Command:

```bash
.venv/bin/python extra/llm_research/decode/nv_pdl_phase_b_driver.py \
  --queues 2,1 \
  --out docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820/phase_b_driver.json \
  --evidence-dir docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820
```

Each child process is wrapped in `timeout 900 flock -w 120 /tmp/gpu-bench.lock`
and runs fresh. The candidate environment was:

```text
NV_PDL_PRODUCER_PROGRAMS=prefix:q4k_,prefix:q6k_
NV_PDL_CONSUMER_PROGRAMS=prefix:reduce_output_rmsnorm,prefix:E_,prefix:r_,prefix:flash_,prefix:rmsnorm_q8_1_llama_provider
NV_PDL_TRIGGER_POSITION=end
```

Results:

| bracket | controls armed | candidate armed | census |
| --- | ---: | ---: | --- |
| 1 queue | 0, 0 | 108 | matches expected 108 |
| 2 queues | 0, 0 | 144 | matches expected 144 |

All three runs in each bracket share token SHA
`323f407295a70421c78d02bb0954c20dffc76a0020ce52f7bdba0e0901ba8332`.
Every candidate row had `positive_overlap_count = 0`; median launch shadow
was 9.750 us (1q) and 9.250 us (2q), so no armed pair actually started its
consumer before the producer finished.
Profiled spans are instrumentation-taxed and are not endpoint walls:
1q control/candidate/control 5541.5/5608.5/5611.5 us; 2q
5991.25/6035.5/6075.0 us. All armed pairs were `gemv->support`, including
incidental pairs without a static data edge. Retained:
`phase_b_driver.json` and the six per-arm JSON/profile files.

### 6.3 Phase C: semantic discriminator

Command (driver must not itself hold the lock; each child takes it):

```bash
.venv/bin/python extra/llm_research/decode/nv_pdl_phase_c_driver.py \
  --backend both --reps 12 --warmup 1 \
  --workdir /tmp/nv-pdl-phase-c-fix \
  --evidence-dir docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820 \
  --out docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820/phase_c_driver_fixed.json
```

Same grid for every arm: 4096 producer blocks x 256 threads, one consumer
block x 256 threads, 100 us producer tail spin, full-coverage checksum.
Control/candidate/control ordering and warmup are recorded in the driver
payload. Two probe defects were fixed before acceptance: the shared CUDA
`cudaLaunchConfig_t.gridDim` was changed to one block for the consumer and
never restored, and timestamp/checksum/output resets were not stream-ordered
(`nv_pdl_phase_c_cuda_probe.py:149`, `nv_pdl_phase_c_cuda_probe.py:167`).
Medians in microseconds, 11 rows after warmup:

| arm | producer | trigger shadow | launch shadow | overlap | wait | wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA no_pdl | 500.384 | - | 501.024 | -0.608 | 0.064 | 527.264 |
| CUDA pdl_end | 488.416 | 488.128 | 440.544 | 47.616 | 48.096 | 514.592 |
| CUDA pdl_start | 486.048 | 386.016 | 386.368 | 99.680 | 100.128 | 511.936 |
| CUDA pdl_prologue | 486.528 | 486.240 | 440.640 | 45.920 | 46.400 | 512.960 |
| native no_pdl | 502.656 | - | 503.232 | -0.576 | 0.000 | 529.920 |
| native pdl_end | 503.808 | 503.808 | 404.032 | 99.776 | 100.160 | 529.184 |
| native pdl_start | 512.608 | 394.240 | 394.656 | 117.472 | 117.824 | 538.720 |
| native pdl_prologue | 503.712 | 503.712 | 403.904 | 99.808 | 100.160 | 529.824 |
| native qmd_latch | 503.360 | - | 403.712 | 99.808 | 0.000 | 463.488 |

All checksums passed in every arm. Driver gates:

```text
cuda_pdl_fired                            true
cuda_trigger_position_moves_launch        true
cuda_wait_placement_moves_wait            true
native_in_kernel_pdl_overlaps             true
native_qmd_latch_overlaps                 true
checksum.{all nine arms}                  true
```

Caveats: CUDA wait placement changed wait by only ~1.7 us; native `qmd_latch`
has no in-kernel `griddepcontrol.wait`, so its 463.488 us wall is launch-ahead
without data readiness and is not the same dependency semantics as llama;
native `pdl_start` producer/wall drift may be noise. This phase proves
mechanism existence on a matched synthetic grid, not an endpoint result.
The earlier `phase_c_driver.json` in the same evidence directory is the
pre-fix run and is superseded by `phase_c_driver_fixed.json`; do not cite the
superseded row.
Retained: `phase_c_driver_fixed.json`, `phase_c_cuda_compile.json`, and the
eleven per-arm JSON files.

### 6.4 Why the endpoint Phase D arm was not run

Phase D requires an equivalent mechanism: the same tinygrad kernels and
dependency DAG with llama-equivalent edge coverage and trigger/wait placement.
Phase B shows the tested arm lacks support-to-support and support-to-gemv
coverage and does not timestamp wait-exit. Building that equivalent arm
requires either a broader probe-only construction or a production change;
neither was silently substituted here. The endpoint split-phase claim is
therefore left `unmeasured`.

The static-census half of Phase D was run on 2026-08-21 and is recorded next
in section 6.5. The endpoint half still requires an equivalent arm and has
not run.

### 6.5 Phase D: broad static coverage census

Command:

```bash
.venv/bin/python extra/llm_research/decode/nv_pdl_phase_d_static_coverage.py \
  --control docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820/phase_a_control.json \
  --llama docs/task_workflow/output/nv-weighted-llama-real-edge-dag-20260820.json \
  --queues 1,2 \
  --out docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820/phase_d_static_coverage.json
```

This is GPU-free. It reuses the Phase A placement and wait-encoding replay,
then admits every unique program name as both producer and consumer. The
capture has 34 unique program names, not 596; repeated layers reuse the same
rendered name. A same-queue consecutive pair arms only when no encoded wait
remains and both names match. A static forward edge is armed only when its
producer and consumer are the same-queue consecutive pair and no wait breaks
the chain. The replay reproduces the Phase A controls exactly (108/144), and
the broad name filter blocks zero pairs.

Forward-static-edge coverage of the broad arm, in llama's
anchor/gemv/support vocabulary:

| edge class | llama | broad 1q | broad 2q | 2q shortfall |
| --- | ---: | ---: | ---: | ---: |
| support -> support | 328 | 112 | 176 | -152 |
| support -> anchor | 145 | 109 | 109 | -36 |
| anchor -> support | 144 | 72 | 108 | -36 |
| support -> gemv | 72 | 0 | 0 | -72 |
| gemv -> support | 72 | 0 | 0 | -72 |
| anchor -> anchor | 0 | 36 | 36 | +36 tinygrad-only |

Totals of armed real edges are 329 (1q) and 429 (2q) against llama's 761
programmatic edges, and the bucket totals are not one-to-one because the
`anchor->anchor` rows have no llama counterpart. In 1q all 591 same-queue
consecutive pairs arm; in 2q 586 consecutive pairs produce 518 arms and 68
encoded-wait breaks, 35 of which sit on forward static edges.

Why the remaining static edges cannot arm:

| block reason | 1q | 2q |
| --- | ---: | ---: |
| armed | 329 | 429 |
| not consecutive on the same queue | 874 | 632 |
| producer and consumer split across queues | 0 | 107 |
| encoded wait between consecutive pair | 0 | 35 |
| graph-group boundary | 27 | 27 |
| name filter | 0 | 0 |

The K/V transitions are unreachable in both modes: all 72 `support->gemv`
edges are non-adjacent in 1q and queue-split in 2q, and all but one
`gemv->support` edge is non-adjacent (one crosses groups). Two-queue
`support->support` misses are 242 adjacency, 35 encoded wait, 35 queue split,
and 12 group cut.

Verdict: the existing name-pinned mechanism cannot produce a faithful
llama-equivalent coverage census even with an unrestricted name filter. The
name filter is no longer the limiter; adjacency, queue placement, encoded
waits, and graph groups are. Experiment 8.1's first gate therefore fails.
This is a construction census, not an execution measurement; no endpoint PDL
claim changed. Retained: `phase_d_static_coverage.json`; script
`extra/llm_research/decode/nv_pdl_phase_d_static_coverage.py`.

## 7. Direction A and Direction B designs

### 7.1 Direction A: first-class split-phase dependencies

```text
# Program metadata
Program:
  launch_signal: LaunchEvent | None
  wait_points: list[WaitPoint]   # edge-scoped, not program-name-scoped

WaitPoint:
  edge: SplitEdge
  position: entry | before_first_dependent_access

SplitEdge(producer, consumer):
  launch_ready = producer grid launch event
  data_ready   = producer output tensor ready

# Graph edge representation
Dependency:
  launch_edge: LaunchEdge        # consumer may grid-launch
  data_edge:   DataEdge          # consumer may touch output bytes
  lowering:    serial_fallback   # both collapse to the old full wait

# HCQ/QMD lowering
lower(producer -> consumer):
  producer_qmd.write(
    arrive_at_latch_valid=1,
    enable_program_pre_exit=1,
    pre_exit_at_last_cta_launch=1)      # or in-kernel trigger
  consumer_qmd.write(
    wait_on_latch_valid=1,
    wait_on_latch_id=edge.latch_id)     # grid may start
  codegen consumer:
    if wait.position == before_first_dependent_access:
      emit prologue (index math, non-dependent setup)
      emit griddepcontrol.wait
    else:
      emit griddepcontrol.wait

# Codegen placement
producer trigger:
  start if producer has no dependent prologue, else last pre-launch point
consumer wait:
  after index/pointer arithmetic, immediately before first dependent load

# Correctness rules
for edge in graph:
  reject launch-ahead when producer/consumer buffers alias or overlap
  keep producer output live until every consumer data-wait has exited
  multiple consumers share the producer launch event but each owns a wait
  never reuse a latch id across replay groups or graph flushes
  if a producer has no legal launch point: emit serial_fallback

# Closed default
if device lacks PDL or edge fails an alias/liveness rule:
  emit the existing full data dependency; no launch-ahead
```

This is scheduler-owned and edge-aware; environment name prefixes remain
probes, not the production interface. The fallback must be the current
one-phase behavior, byte-identical when the feature is off.

### 7.2 Direction B: bounded decode fusion

Smallest legal sequence, in order:

| producer | consumer | removed boundary/bytes | required communication | why it generalizes or not |
| --- | --- | --- | --- | --- |
| Q/K/V q4k/q6k GEMV | `reduce_output_rmsnorm` + `E_*` q norm/rope/cast | one norm launch + cast/rope round trips | scalar RMS reduce within warp/CTA | decode-shape norm epilogue; needs a bounded, view-preserving epilogue |
| fused gate/up GEMV | FFN norm `r_16_256` + `E_32_32_4` | norm reduce plus fp16 transport | scalar RMS scale/affine applied at packed loads | existing M1 rms-affine spelling is research-only; currently loses to boundary copies |
| down GEMV | residual elementwise add/mul | one `E_32_32_4` program and its output round trip | none beyond in-kernel add | already exists for O/down; expand to admitted consumers |
| vocab GEMV | post-vocab norm/quant/reduce | reduce and provider launches | cross-warp reduce to one logits row | model tail, not general |
| none | Q4 FFN-down body | no fusion; body-speed parity | four-warp layout already landed | on-spine kernel lever, separate from S1 fusion |

Every candidate must preserve the token SHA and be endpoint-bracketed; the
legal ceiling is the measured bracket, not the zero-cost critical-path number.
The Q4 FFN-down row is an on-spine kernel lever, not S1 fusion.

## 8. Three cheapest remaining decisive experiments

### 8.1 Full-chain equivalent PDL bracket

The static census is complete: `phase_d_static_coverage.json` admits all 34
unique program names and cannot produce a faithful llama-equivalent census.
The name filter is no longer the limiter; non-adjacent placement, queue
splits, 35 encoded waits, and 27 group-boundary edges block the chain. The
result therefore hits the stop gate: the existing name-pinned mechanism
cannot express the full chain, and no production patch is authorized here.
The construction scope for an edge-aware runtime hook is now drafted at
`docs/task_workflow/input/nv-edge-aware-pdl-runtime-hook-scope-20260821.md`;
it names the scheduler edge model, codegen wait placement, QMD latch fields,
correctness pins, and the Q1-Q8 knowledge ledger that closes the decision.
Do not run the endpoint bracket against a non-equivalent construction and
report it as the split-phase verdict.

Belief-flip gates:

- support-to-support launch-ahead with positive data-wait overlap and endpoint
  recovery over ~150 us supports Direction A;
- faithful coverage with flat or negative wall supports Direction B;
- native negative while the matched CUDA grid positive isolates native
  lowering rather than disproving split-phase scheduling.

### 8.2 Q4 FFN-down body parity bracket

Take the existing four-warp Q4-down route, test the cheapest body changes
under `extra/llm_research/decode/**` only, and bracket endpoint wall. Expected
available recovery is ~30.016 us if the body reaches llama's corrected
19.232 us (the old 11.776 us value was attention-O's floor).
Gate: promote only if a fresh-process bracket clears the promotion gate with
an unchanged token SHA; otherwise keep the row as explanatory evidence.

### 8.3 Bounded S1 fusion bracket

Admit one bounded epilogue (residual or output-reduction) through the existing
`Q4KGEMVEpilogue`/typed-view machinery, again harness-only. Expected zero-cost
ceiling is 148.736 or 203.680 us with alternate-path takeover; the realistic
outcome is lower. Gate: a >50 us wall recovery with byte-identical tokens
supports bounded fusion through existing abstractions; a boundary-copy
regression repeats the M3/M4 result and argues for the boundary fix or a
bespoke primitive.

## 9. Ranked next actions

Immediate endpoint work:

1. The 8.1 static census is complete and hit the stop gate: the name-pinned
   interface cannot express llama-equivalent coverage. Write the edge-aware
   runtime-hook construction scope; it is drafted in
   `docs/task_workflow/input/nv-edge-aware-pdl-runtime-hook-scope-20260821.md`.
   Do not patch production until its staged gates pass.
2. Run experiment 8.2 now; Q4 FFN-down is measured, independent of the PDL
   construction question, and worth ~30.016 us at the corrected floor.
3. Run experiment 8.3 only after the 8.1 construction scope exists, because
   fusion effort before the split-phase verdict can build the wrong boundary.

Generic architecture work, deferred:

5. Add split launch/data readiness as scheduler edges with a closed serial
   fallback.
6. Generalize the bounded epilogue ABI so each fusion is not a new
   decode-specific spelling.
7. Revisit replay grouping only with a measured one-graph construction, not as
   an a priori design change.

## 10. Existing abstractions or bespoke kernels?

Neither the "existing abstractions only" nor the "bespoke kernels only"
position is supported by the current evidence. Existing abstractions already
express bounded epilogues (the landed O residual, GLU, and down residual
folds), native QMD PDL, and fail-closed typed views; those are why tinygrad
wins S3 and the anchor union. But the current fusion interface is per-shape
and the current PDL interface is name-pinned rather than edge-aware, and the
opaque custom-kernel boundary has historically materialized copies that
canceled M3/M4 gains. The measured route therefore points to a hybrid: a
scheduler-owned split-phase edge model and a general bounded epilogue ABI in
existing code, plus at most a small number of bespoke primitive bodies (the
Q4 FFN-down row being the concrete current example). This recommendation is
falsifiable by experiments 8.1-8.3: if full-chain PDL recovers the S1 gap, the
scheduler edge is the binding piece; if the fusion bracket recovers it through
typed views, the boundary generalization is; if both fail with faithful
constructions, the remaining win is bespoke kernel work.
