# Five-lever parity test record

Date: 2026-08-03 (measured same day, same RTX 5090 box, sequential flocked
sessions; no concurrent GPU work)
Status: measurement record for the five parity levers named in
`executable-taskgraph-ir-scope-20260803.md` section 3. Runs each lever's proof
experiment where it is runnable, anchors every number, and does not change any
implementation. The parity wall authority remains
`nv-decode-parity-final-20260802.md`.
Branch: tinygrad `nvidia-bringup-20260731`, HEAD `afc6609ba` (taskgraph scope
state-ranking revision). All numbers carry evidence class OBSERVED unless marked
INFERRED.

## 0. Session and environment

- Host: RTX 5090 / sm_120, driver 595.84, CUDA 13.2, nvcc 13.2, nsys 2026.1.3.
- Model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf` (identity sha256
  `b8ef0be84bfa0588efae9fb84a3b3e5b7beb53f5620ada7d8c48bd3a26633605`).
- Harness: `extra/llm_research/decode/decode_runtime_overhead.py` (W==D method;
  W = production generate item/token, D = same JIT with final sync).
- Fused prefill attention is OFF for every run (house convention; the NV fused
  prefill ABI is deterministically broken at HEAD, `b3-tuned-schedule-
  characterization-record-20260803.md`).
- Model routes: flash decode, w1w3 scalar fused QV landed, kv-store fusion
  closed-default, fp16 KV cache capability-enabled on NV.

## 1. Lever 1 (overlap): P0 status

P0 status: **Nsight CUDA node tracing is inapplicable to the executing DEV=NV
backend. Native HCQ timestamp/dependency tracing is available; the current
implementation uses one compute queue/GPFIFO.**

The scope's P0 protocol is `nsys --cuda-graph-trace=node` on our decode replay.
Three capture attempts on this host produced `.nsys-rep` files with no GPU
trace data (`nsys stats`: "does not contain GPU trace data"). The cause is a
profiler/backend mismatch, not missing tooling: Nsight Systems 2026.1.3 is
installed and functional (a native CUDA smoke test records API calls and kernel
events; existing llama reports carry CUDA graph/node data), but the DEV=NV
route never calls the CUDA launch API. `NVDevice` (`ops_nv.py:598`) drives the
hardware through native NVIDIA ioctls, QMD command construction, and direct
GPFIFO submission, so CUPTI sees nothing to trace. The native counterpart works:
`HCQ_GRAPH_PROFILE_JSON` exported per-node timestamps and dependencies
(`/tmp/hcq_graph_smoke.json`, anchored at
`docs/five-lever-test-20260803-hcq-graph-smoke.json`).

Fallback signal, same session family, d512 (OBSERVED arithmetic, since
corrected): the DEBUG=2 kernel-sum 5.981 ms exceeds the W wall 5.63 ms, which
initially read as ~6% overlap.

| quantity | value | source |
| --- | ---: | --- |
| kernels per token | 948 | `five-lever-test-20260803-d512-prime-trace.log` |
| node-sum (kernel tm sum) | 5.981 ms | same log (record: 6.02 ms) |
| W wall (production, no sync) | 5.63 ms | `five-lever-test-20260803-l4-fp16.json` |
| D wall (with final sync) | 7.19 ms | same file |

**Correction (same session, follow-up record):** this fallback is refuted.
`decode-replay-overlap-measurement-record-20260803.md` measures real GPU
timestamps via the HCQ profile path: replay overlap is 0.0% at d512 and d4096
(per-group span equals node-sum to the microsecond; artifacts
`five-lever-test-20260803-overlap-d512.json` / `-d4096.json`). The 0.35 ms gap
is per-kernel launch/sync overhead in the DEBUG=2 tm values (~0.65 us x 948
kernels), not GPU concurrency. llama's 22% overlap reference
(`nv-decode-gap-decomposition-record-20260803.md` section 3) is structurally
unreachable today: HCQGraph creates one compute queue per device
(`hcq.py:68`), assigns every kernel to it (`hcq.py:134`), and the NV queue
submits to a single compute GPFIFO (`ops_nv.py:205`, `ops_nv.py:633`).
`JIT_BATCH_SIZE=0` only removes graph-group boundaries; it produces one larger
command sequence on the same channel and cannot create compute-compute
overlap. Lever 1 therefore requires either multiple native NV compute
channels/GPFIFOs with semaphore dependencies, or routing through DEV=CUDA and
the existing CUDAGraph lowerer; the missing layer is native multi-compute-queue
execution, not executable-graph replay.

## 2. Lever 2 (GEMV per-kernel efficiency): microbenches reproduce

All seven CUDA microbenchmarks ran sequentially on the 5090 and reproduce their
recorded ceilings within ~0.3% (sources: `extra/llm_research/microbench/*.cu`,
README in the same directory):

| microbench | measured | recorded / ceiling |
| --- | ---: | ---: |
| dp4a nacc=8 | 952.3 G dp4a/s (7.6 INT8 TOPS) | 950.8 @ 32768 |
| dp4a nacc=16 | 951.5 G dp4a/s | NACC-invariant ~950-959 |
| dp4a nacc=32 | 958.2 G dp4a/s | 959.0 |
| L2 Q6K partial (installed) | 12.91 us / 0.27 TB/s | 12.92 / 0.27 |
| L2 Q6K partial (best row) | 7.39 us / 0.47 TB/s | 7.38 / 0.47 |
| flash tile anchor | 14.13 us/kernel | 13.98 (llama floor 3.2 us) |
| vocab coop (installed / nacc=4) | 331.2 us / 86.0% of 1792 GB/s | 330.7-331.4 |

The per-kernel ceilings in the lever-2 scope row hold; the 824 GB/s / 46%
route-level figure is unchanged by these peak-state measurements (they are not
wall A/Bs).

## 3. Lever 3 (kernel count): 948 kernels at HEAD

d512 prime-token census (artifacts `five-lever-test-20260803-l3-classes.json`,
`five-lever-test-20260803-l3-gemv-census.json`):

| class | kernels | sum | notes |
| --- | ---: | ---: | ---: |
| GEMV-class | 216 | 3.807 ms | q4k lanemap + q6k coop/partial, vocab excluded; recorded like-for-like cap 3.836 ms |
| flash | 72 | 0.367 ms | score + gmax combine |
| other | 655 | 1.424 ms | rmsnorm/residual/scatter/elementwise classes |
| total | 948 | 5.981 ms | |

The scope's 1021-kernel figure was an earlier session; this session's 948
matches the kv-store chain fusion record (948 -> 948 kernels). The gap to
llama's 762 nodes stands, as does llama's zero add/residual-kernel property.

## 4. Lever 4 (depth scaling): fp16-cache A/B vs fp32 control

Same harness, same session family, `--ckpts 512,4096`, nmeas 10 reps 2.
Artifacts `five-lever-test-20260803-l4-fp16.json` (route) and
`five-lever-test-20260803-l4-fp32.json` (control, `kv_cache_fp16_eligible`
patched to False before construction; realized cache dtype verified per arm:
`float16` vs `float`).

| arm | d512 W | d512 D | d4096 W | d4096 D | depth penalty (W) |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16 route | 177.72 | 139.05 | 156.57 | 124.89 | -11.9% |
| fp32 control | 177.04 | 139.23 | 152.21 | 121.73 | -14.0% |

The fp16 route is +2.9% at d4096 and reproduces the recorded d512 177.2 tok/s.
Our depth penalty is -11.9% (fp16) / -14.0% (fp32, matches the recorded
-14.2%) versus llama's -9.3%; fp16 closes about half the depth-penalty delta.

## 5. Lever 5 (prefill host, B3): HEAD baseline reproduces; pinned rows unreachable

The B3 probes (`/tmp/measure_warm_prefill.py`, `/tmp/probe_warm2.py`,
`/tmp/measure_busy_debug2.py`) run with the house fused-attention-off patch
(wrapper `/tmp/b3_runner.py`); bare runs crash on the documented
`PACKED_FRAGMENT_LOAD` UOp verification failure. Measured at HEAD
(`five-lever-test-20260803-l5-warm-prefill.json`, probe stdout):

| quantity | measured | record (HEAD rows) |
| --- | ---: | ---: |
| warm pp512 wall (pass2/pass3) | 158.4 / 156.8 ms | 161.31 / 160.70 ms |
| waits per pass | 10 | 10 |
| wait() time | 136.2 / 136.1 ms | 137.34 / 137.26 ms |

The pinned rows (warm wall 44-46 ms, busy 24.1 ms, ~1.9x) were measured on the
fused-prefill-attention-ON schedule and are unreachable at HEAD; the record's
section 6 already states this. The busy probe reads 0.0 ms under graph replay
(GlobalCounters limitation), so busy stays anchored to the record's DEBUG=2
138.0 ms row.

## 6. Verdict

Four levers have runnable tests and all reproduce their recorded numbers. Lever
1's P0 is answered by the follow-up probe: nsys CUDA tracing is inapplicable to
DEV=NV, and native HCQ timestamps show 0.0% replay overlap (llama's 22% is
structurally unreachable on the single compute queue/GPFIFO). The lever-1
implementation path is native multi-compute-queue execution or a DEV=CUDA
route. No implementation changed in this record.
