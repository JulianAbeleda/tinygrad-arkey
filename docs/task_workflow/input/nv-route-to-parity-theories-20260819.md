# NV route-to-parity theories (2026-08-19)

Date: 2026-08-19
Branch: `nvidia-bringup-20260731`, HEAD `d14e6964e`
Status: **prioritized theory list + test plan.** This is the list the task
asked for: falsifiable theories from the llama audit, each with a predicted
wall delta and a same-session test design. Only candidates that can plausibly
clear the `+50 us/token` promotion bar are scheduled; the rest are closed here.

## 1. Current anchor (what parity means)

| side | tok/s | wall us/token | source |
| --- | ---: | ---: | --- |
| tinygrad landed (2 GPFIFO + reuse lanes) | 212.12 | 4714 | reuse-lanes A/B, this session |
| llama same-session 08-17 | 246.37 | 4059 | exact wall account |
| llama fresh 08-19 (llama-bench tg128) | 254.4 | 3931 | first-principles re-audit |

Gap to close: **~655-783 us/token** depending on the llama anchor. The honest
target for the bar is `llama - 50 us`; every theory below is judged against
that, not against "some improvement".

## 2. The corrected loss decomposition (ground truth)

`wall = GPU busy (union) + host gap`, `GPU busy = node_sum - overlap`.

| term | llama | tinygrad | gap |
| --- | ---: | ---: | --- |
| node_sum | ~4774-5015 us | ~4519 us | tinygrad does LESS |
| overlap mass | ~946-1125 us | ~0-140 us | llama wins |
| GPU busy (union) | ~3835-3890 us | ~4519 us | +~630-684 us |
| host gap | ~168-212 us | ~269 us | +~57-100 us |
| wall | ~3931-4059 us | ~4714 us | +~655-783 us |

The loss is **serialization, not kernel work**. tinygrad's node_sum is already
below llama's; it is slower because it runs that work fully serialized while
llama pipelines its support kernels behind the GEMV anchor.

## 3. Theories

### T1. Early programmatic dependent launch (highest value)

Status (2026-08-19): **tested, NO-GO as a new lever.** The corrected
same-session microbench has valid checksums on all four arms. Moving
`griddepcontrol.launch_dependents` from END to START changes overlap by
~0.03-0.21 us (launch noise, both ~-0.4 us). The only real overlap is the
QMD latch (`arrive_at_latch` + `pre_exit_at_last_cta_launch`), which gives
+99.8 us and is the already-landed `HCQ_NV_READY_PLACEMENT` path.

Hypothesis: the native NV PDL releases the dependent grid at the **last CTA**
(`pre_exit_at_last_cta_launch` in `tinygrad/runtime/ops_nv.py`, and the
renderer's `griddepcontrol.launch_dependents` placed at the producer END).
llama's PDL fires `cudaTriggerProgrammaticLaunchCompletion` at kernel START, so
the consumer's launch + prologue overlaps the producer's full body. Matching
that early-trigger semantics is the difference between "final-wave overlap"
(measured wall-neutral) and llama's real pipeline overlap.

Predicted delta: unbounded-to-large. This is the mechanism that produces
llama's ~946-1125 us of overlap. The recoverable share on tinygrad's DAG is
capped by the separable shadow mass (~1040 us of non-GEMV kernels), minus
hardware contention, but it is the only single lever with >50 us headroom.

Test design:
1. Two-kernel microbench (producer ~100 us + consumer ~1 us) comparing
   `launch_dependents` at END vs START vs the QMD last-CTA latch, on DEV=NV.
2. If the microbench shows a real launch-shadow saving, run a full decode A/B
   with exact token sha under `flock /tmp/gpu-bench.lock`.

Scope: `tinygrad/renderer/cuda.py`, `tinygrad/runtime/ops_nv.py`, a new probe.

### T2. Cross-layer anchor+shadow restructuring

Hypothesis: the decode DAG has real schedule slack (the old 08-12 DAG had
~650 us between serialized span and critical path), but the landed 2-queue
substrate only captured ~140 us of it because of HBM contention and
cross-queue wait boundaries. Moving support kernels (norms, rope, flash
quantize for layer N+1) behind layer N's long GEMV anchor is mechanism C and
is the structural source of llama's `union < node_sum`.

Predicted delta: up to ~600 us if the current DAG still has the slack; zero if
the fusion/P1 work already collapsed the critical path. Must be bounded by a
fresh critical-path measurement first.

Test design:
1. Fresh DAG capture at landed HEAD -> `dag_critical_path_sim.py` to get the
   actual serialized span, critical path, and remaining slack.
2. If slack > ~150 us, build a representative anchor+shadow microbench and a
   legal decode A/B.

Scope: new probe + read-only DAG analysis; construction lands only if T2
clears the bar.

### T3. Flash coarse-split install on DEV=NV

Hypothesis: llama's source-derived d512 install is a 4-way KV split (isolated
body ~3.14 us) vs tinygrad's 48-way split (~4.16 us). The prior flash-shape
A/B measured on the wrong backend (DEV=CUDA, ~177 tok/s) and never tested the
legal `S=4` route on DEV=NV. A coarse split also cuts per-launch cold-L2/launch
overhead across 36 score launches.

Predicted delta: ~38 us body + launch overhead, ~50 us total. Borderline; only
promotable if it clears the bar or stacks with T1.

Test design: DEV=NV same-session A/B, `S=48` control vs `S=4`/`S=2` candidate,
exact token sha, `flock /tmp/gpu-bench.lock`.

Scope: `tinygrad/llm/flash_decode_attention.py`, `tinygrad/llm/decode_routes.py`,
a new probe.

### T4. Replay-merge re-test at landed HEAD

Hypothesis: the 5-graph-replay structure costs ~+100 us host gap vs llama's
single replay. The prior `JIT_BATCH_SIZE=1024` A/B was measured-negative
(+112.9 us GPU) before the overlap substrate landed, so it deserves one
re-check now that two GPFIFOs exist.

Predicted delta: uncertain; prior evidence negative. Low priority.

Test design: same wall A/B with `JIT_BATCH_SIZE` 32 vs 1024 at landed HEAD,
exact token sha.

## 4. Closed below the bar (not scheduled)

| row | honest wall value | verdict |
| --- | ---: | --- |
| vocab argmax codegen | ~5.6 us | hidden mass, below bar |
| reduce_output q/k geometry | ~0 us | bitwise-blocked |
| deeper compute queues (3+) | +26.4 us upper bound | wait tax erases it |
| host-gap submit-ahead | -6.2 us | measured flat |

## 5. Acceptance gate

Every claim of wall improvement must be a same-session `flock /tmp/gpu-bench.lock`
A/B on `DEV=NV` with byte-identical token sha, and must clear `+50 us/token`
against the landed control. Anything below is recorded and dropped.

## 6. Evidence

- `nv-overlap-substrate-reuse-lanes-20260819.json` (landed control 212.12 tok/s)
- `nv-queue-count-sweep-20260819.json` (queue-count closed)
- `nv-first-principles-reaudit-20260819.md` (corrected loss trace)
- `nv-us-vs-llama-side-by-side-20260818.md` (wall equation authority)
- `nv-internal-gap-first-principles-20260818.md` (replay/host-gap structure)
- `nv-replay-merge-retest-20260819.json` (T4 whole-token merge re-test, closed negative)

## 7. Measured results (2026-08-19, DEV=NV, same-session flocked A/B)

Fresh landed control: **210.8 tok/s / 4744.5 us/token**, token sha `1d299b89...`
(`nv-fresh-critical-path-audit-20260819.json`). Every theory below was tested
against that control.

| theory | result | verdict |
| --- | --- | --- |
| T1 early PDL trigger | `griddepcontrol.launch_dependents` at START vs END differs by 0.24 us (noise); both give ~-0.4 us overlap. Only the already-landed QMD latch gives +99.8 us overlap. | **NO-GO** (no new lever) |
| T2 cross-layer shadow | 3 us support kernel hides ~1-5 us behind a 100 us anchor on 2 GPFIFOs; this re-measures the landed substrate. Extrapolated 449-670 us is noise-scaling. | **NO-GO** (not a new needle) |
| T3 flash coarse split | S=4 +763 us slower, S=2 +1672 us slower; tokens bitwise identical to S=48. | **NO-GO** (S=48 already optimal) |
| T4 replay merge | re-run at landed HEAD, same-session: JIT_BATCH_SIZE=1024 is 27.4 us/token slower than JIT_BATCH_SIZE=32 (209.12 vs 210.33 tok/s), token sha identical. | **NO-GO** (re-confirmed negative) |

Conclusion: none of the four needle-moving candidates clears the `+50 us/token`
bar on top of the landed baseline. The remaining 685.6 us gap to llama's
same-session anchor is llama's own overlap of its unfused support mass
(946-1125 us), which tinygrad has already fused away (node_sum ~496 us below
llama), plus ~100 us host gap.

## 8. Graph-pipelining claim test (2026-08-19, decisive)

The claim "graph pipelining is the missing lever" was tested directly against
the live decode graph instead of being asserted from the llama trace.

1. Captured a fresh steady-token dependency DAG at HEAD (`HCQ_NUM_COMPUTE=1`):
   596 nodes in 5 groups, serialized node-sum 4646.3 us, observed overlap 0.0.
2. Computed the ideal 2-queue longest-tail schedule over that DAG: critical
   path 4187.0 us, 2-queue span 4188.7 us, saving 457.6 us (9.85%). This is an
   upper bound because it omits cross-queue wait tax and HBM contention.
3. Compared the ideal assignment to the landed ready-placement census: the
   landed scheduler already matches the ideal aux-node set in 4 of 5 groups
   (the fifth differs by 3 short nodes). Graph pipelining is already installed
   at essentially its optimal 2-queue placement.
4. Same-session wall A/B, token sha identical: serial 205.99 tok/s
   (4854.6 us/token), landed 210.97 tok/s (4740.1 us/token). Realized overlap
   is 114.5 us, i.e. 25.0% of the 457.6 us ideal saving.

Even perfect zero-tax pipelining would cap at ~4397 us/token (227.4 tok/s),
which is still ~338 us below llama's same-session 4059 us wall. The tested
verdict is **DISPROVEN as the missing lever**: graph pipelining contributes
about 115 us and is already landed; the rest of the gap is dependency-critical-
path time plus the host gap, not unexploited overlap.

Evidence: `docs/task_workflow/evidence/nv-graph-pipelining-claim-test-20260819.json`.

## 9. Critical-path reframing (2026-08-19)

The graph-pipelining test used the wrong unit. Wall is decided by the longest
dependency chain, not by node_sum or by overlap of already-independent nodes.
A fresh extraction of tinygrad's steady-token DAG gives:

| class on critical path | us |
| --- | ---: |
| gemv | 3259.07 |
| flash | 314.27 |
| reduce_output | 235.30 |
| rmsnorm | 154.66 |
| residual | 135.49 |
| scatter (vocab) | 55.65 |
| q8 provider | 32.58 |
| **critical path** | **4187.01** |

llama's same-model graph span is `3899.5 us` (kernel union `3891.3`), so its
critical path is at most `3899.5 us`. tinygrad's critical path alone is
**287.5 us longer than llama's entire span**. Queue overlap cannot shorten a
critical path; only relaxing dependency edges can.

The non-GEMV support on tinygrad's critical path is `927.9 us`; llama's
non-mmq exposed union is `302.4 us`. The `625.6 us` difference reproduces the
known `628.8 us` busy delta. The mechanism that still needs to be found is the
set of dependency edges that keeps llama's flash/reduce/residual support off
its mmq chain. The llama dependency DAG has been dumped to
`/home/ubuntu/env/llama.cpp/llama_decode.dot` for that edge-by-edge diff.

Evidence:
`docs/task_workflow/evidence/nv-critical-path-gap-20260819.json`.

### 9.1. Trace refutation (2026-08-19)

A full weighted trace of tinygrad's 4187.008 us critical path
(`docs/task_workflow/evidence/nv-tinygrad-critical-path-trace-20260819.json`)
refutes the earlier "move K/V and set_rows off the matmul spine" framing: those
kernels are already off the critical path. tinygrad's chain already carries only
Q, O, gate+up, and down matmuls, exactly like llama. The 287.5 us gap is entirely
support glue (flash 314, reduce_output 235, rmsnorm 155, residual 135, scatter 56,
q8_provider 33 us) sitting on that spine, versus llama's ~302 us of exposed
non-mmq work. The geometric ceiling is ~625 us if that glue could be hidden to
llama's exposure level, but no tinygrad code path has yet demonstrated a legal
way to hide flash/reduce/norm/residual behind the gemv chain.
