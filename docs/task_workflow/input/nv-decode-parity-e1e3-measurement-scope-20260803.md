# NV decode parity - E1/E2/E3 decisive experiments measurement scope

Date: 2026-08-03
Status: measurement scope, docs plus the named probes and harness runs only.
Authorized by `nv-decode-parity-external-review-amendment-20260803.md` section 7
(next authorized artifact: a measurement scope for E1-E3). Authorizes: (E1) the
llama `GGML_CUDA_GRAPH_OPT` A/B, (E2) the tinygrad dependency-DAG critical-path
simulation, and (E3) the CUDA-stream vs native-channel control, each under its
own protocol below. Does NOT authorize: any route record change, any promotion
to `dev`/`exp`/`master`, any decoder/HCQGraph implementation, or any composed
performance endpoint. Branch boundary: tinygrad `nvidia-bringup-20260731` at
`24deaffd7` (scope) + amendment at `39d10369b`.

The amendment's HARD STOPs remain in force: no declaring native overlap
impossible, no composing a parity endpoint, no promoting a route. The
experiments below only produce evidence; verdicts are recorded in measurement
records with the amendment's belief-flip criteria.

## 0. Validation status of the amendment (OBSERVED, 2026-08-03)

Two independent read-only validation passes (flash agents, no GPU) checked the
amendment's factual claims against the artifacts and source trees before this
scope was written:

- tinygrad side (4 claims): all TRUE. `five-lever-test-20260803-overlap-
  d512.json` has 5 groups `32/64/128/256/468 = 948` kernels, node-sum
  `5367.2 us`, span `5367.2 us`, 0.0% overlap; wall `5.6268 ms` at `177.72
  tok/s`; `5.3672/5.6268 = 95.4%`. Harness prompt is tokenizer-derived
  (`"the quick brown fox jumps. " * 800`, repeated to depth), not `[1]*depth`;
  `_host_residual` refuses W-D subtraction when `d_ms > w_ms`. The `824 GB/s =
  46%` row is derived from the 5.04 GB / 6.12 ms estimate; on the 5.63 ms wall
  the same byte estimate is `895 GB/s = 50%`.
- llama.cpp side (4 claims): 1, 2, 4 TRUE; 3 PARTIAL in a material way. The
  QKV fork/join exists exactly at `ggml-cuda.cu:4289/4381/4662`, gated on
  `GGML_CUDA_GRAPH_OPT=1` (`:4605`, default OFF, never set by llama-bench).
  The nsys trace `/tmp/llama_nsys_d512.nsys-rep` EXISTS and its captured
  environment contains NO `GGML_CUDA_GRAPH_OPT` (nor any `GGML_*` variable),
  so the QKV concurrent-stream discovery was INACTIVE during the trace that
  produced the 762-node / 22% span numbers.

Consequence, stated plainly: llama's ~22% replay-span reduction does NOT depend
on the gated QKV fan-out mechanism. It is base CUDA graph node scheduling
(nodes captured on multiple internal streams with event edges; the graph
launch stream is 43). This makes the provenance question sharper: E1 below
must also test whether `GGML_CUDA_GRAPH_OPT=1` adds anything on top of the
already-observed base overlap, and E3 must establish whether two CUDA streams
co-schedule on this device at all.

## 1. E1 - llama graph-optimization A/B (OBSERVED target)

Question: what does `GGML_CUDA_GRAPH_OPT` actually change on this box, and
which policy produced the authority rows (251.8 tok/s, 762-node trace)?

Protocol (one flocked GPU session; llama-bench CUDA build `ac4cddeb0`):

1. Arm 0: `GGML_CUDA_GRAPH_OPT=0 llama-bench -m Qwen3-8B-Q4_K_M.gguf -ngl 99
   -fa 1 -p 0 -n 10 -d 512 -r 5 -o json`; record raw wall samples, median,
   mean, environment (env | sort), and the llama.cpp commit.
2. Arm 1: same command with `GGML_CUDA_GRAPH_OPT=1`.
3. Trace each arm with `nsys profile --cuda-graph-trace=node` (same command,
   `-r 3`), export node-sum vs replay span per graph instance, node count,
   and the captured environment (to prove the arm's flag).
4. Compare arm 0 vs arm 1: tok/s wall delta, span delta, and node-sum delta.

Belief-flip criteria (amendment section 6, E1):

- If `=1` reduces replay span and wall by a parity-scale amount, the QKV
  overlap is an additional lever beyond base graph concurrency.
- If the span changes but wall does not, or wall delta < 0.2 ms, the 22%
  trace is not a parity-scale wall lever and QKV streams are not the
  explanation for the authority wall row.
- If authority rows and the 22% trace both ran with the default (opt OFF),
  the overlap decomposition stands as base CUDA graph scheduling, and the
  provenance question resolves without rebuilding the decomposition.

Deliverable: `nv-decode-parity-e1-llama-graphopt-measurement-record-20260803.md`
plus raw JSON per arm.

## 2. E2 - tinygrad dependency-DAG critical-path simulation (OBSERVED target)

Question: how much of llama's 1.116 ms missing-overlap term is even legal on
tinygrad's actual decode dependency DAG, before any implementation?

Protocol:

1. Capture one complete d512 decode token with the native HCQ dependency
   exporter: `PROFILE=1 HCQ_GRAPH_PROFILE_JSON=/tmp/nv_decode_d512_dag.json`
   on the decode harness (same command family as the overlap measurement
   record). Record per-node: kernel name, start/end timestamp, duration,
   dependency indices, semantic metadata, and graph-group membership.
2. Offline (CPU-only, hermetic): compute, per graph group and across groups:
   a. serialized span (node-sum, the current 5.367 ms reference);
   b. unlimited-resource critical path (longest dependency chain);
   c. deterministic list schedules on 2 and 3 queues (ready set, static
      priority by longest remaining tail), with per-node durations from the
      capture.
3. Report: how much span reduction each schedule achieves, which node classes
   overlap (GEMV behind GEMV? norm behind GEMV?), and whether any cross-group
   overlap is legal (cross-group dependencies explicit).

Belief-flip criteria (amendment section 6, E2):

- Reopen overlap as parity-scale if a realizable two/three-queue schedule
  saves ~0.8-1.1 ms on the captured DAG.
- Downgrade overlap if even the unlimited-resource critical path saves less
  than ~0.4 ms.

Deliverable: `nv-decode-parity-e2-dag-critical-path-measurement-record-20260803.md`
plus the captured DAG JSON and the simulator script under
`extra/llm_research/decode/`.

## 3. E3 - CUDA-stream vs corrected native-channel control (OBSERVED target)

Question: on this exact device/driver, do two CUDA streams co-schedule
independent kernels (span < node-sum) while the native shared-context
construction serialized? And does a corrected native independent-context
construction change the native answer?

Protocol:

1. CUDA leg: small standalone CUDA probe (no tinygrad) launching the same
   elementwise and partial-SM workload shapes as the E3-E5 probe on two and
   three CUDA streams, timed with CUDA events, same span/node-sum criterion.
   Reuse the existing microbench build tooling
   (`extra/llm_research/microbench/*.cu`).
2. Native leg: re-run the existing `nv_multi_queue_probe.py` E1-E5 baseline
   (regression check, engines `0,0,0`), then the corrected separate-ctxshare
   construction: per-channel `FERMI_CONTEXT_SHARE_A` with the channel bound
   and scheduled on its own context share (the setup gap recorded in
   `nv-multi-compute-queue-probe-measurement-record-20260803.md` section 5).
   Fixes to the probe are allowed under this scope; no decoder/HCQGraph
   changes.
3. Compare: CUDA overlap vs native shared-context serialization vs native
   independent-context.

Belief-flip criteria (amendment section 6, E3):

- CUDA overlap with native serialization proves a native RM
  construction/scheduling gap, not a hardware limitation.
- At least 5% native overlap with correct numerics reopens D2-D4.
- Zero overlap in a correctly validated native construction narrows the
  blocker to the native RM/driver path, but still does not imply the GPU
  cannot co-schedule through CUDA.

Deliverable: `nv-decode-parity-e3-stream-vs-native-measurement-record-20260803.md`
plus probe sources/artifacts.

## 4. Gates and conventions

- GPU sessions sequential, flocked (`flock /tmp/nv_gpu.lock`), same RTX 5090
  box, no concurrent GPU work.
- Evidence classes: OBSERVED (measured under session provenance) vs INFERRED
  (arithmetic/attribution), labeled per record.
- No implementation beyond the named probes; no decoder/HCQGraph/route
  changes; no composed endpoint forecast.
- Correctness pins (token sha `9d6b3787...`, first token `151936`) must be
  preserved by any probe run that exercises the model route; standalone
  device probes carry their own numeric checks.
- Never touch user files: `docs/README.md`,
  `docs/beating-llama-first-principles-20260731.md`,
  `docs/what-makes-inference-fast.md`,
  `extra/llm_research/microbench/*` binaries, `scratchpad/t6_metal_admission_probe.py`.
