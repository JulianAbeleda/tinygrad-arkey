# NV decode parity: llama d512 unprofiled graph-span measurement

Date: 2026-08-04
Status: **P4 PASS**
Evidence class: **OBSERVED diagnostic, unprofiled**

## Finding

For the pinned llama.cpp d512 route, the dependency-neutral CUDA-event
measurement observes:

| quantity | result |
| --- | ---: |
| marker-free wall | **3.971787 ms/token** |
| whole CUDA-graph replay span | **3.889808 ms/replay** |
| derived outside-graph remainder | **0.081979 ms/token (2.063%)** |

The graph contains 762 nodes in every sampled arm. The two full-density span
medians are 3.888912 and 3.890112 ms, a 0.001200 ms difference. This closes
the missing unprofiled llama graph-span number without using a profiler.

The 0.081979 ms remainder is `marker-free wall - measured replay span`. It
includes all benchmark/runtime work outside the graph timing window. It is
not a measurement of launch cost alone and must not be labeled as such.

## Construction and equivalence

The diagnostic was built in detached worktree `/tmp/llama-p4-span` at exact
llama commit `ac4cddeb0dbd778f650bf568f6f08344a06abe3a`. The 31-line patch is
`/tmp/llama-p4-span.patch`, SHA256
`c91ae9bfbabd20155120f64d2e7b1e76b71fdf60117d392c9f71f6482bdef2ce`.
It is disabled unless `ARKEY_P4_GRAPH_SPAN` is set.

The patch records a timing event immediately before `cudaGraphLaunch` and a
second timing event immediately after it on the same launch stream. It then
synchronizes only the stop event and reports elapsed time. Both events are
outside graph capture. `cudaGraphGetNodes` reports 762 nodes in all 207
sampled replays.

Kernel identity was checked more strongly than a whole-library hash. In the
same build directory and with identical flags, patched and unpatched
`.nv_fatbin` sections are byte-identical:

```text
patched   23759674d7794b386ef991cdee72b0b21a8a69e3a2146f658526ac6e59bbe131
unpatched 23759674d7794b386ef991cdee72b0b21a8a69e3a2146f658526ac6e59bbe131
```

The whole shared-library hashes differ, as expected, because the host-side
diagnostic differs. The canonical historical build's fatbin hash also differs
from this fresh build, so this record does not claim binary equivalence to the
older July binary. It proves patched versus unpatched equivalence within the
fresh pinned build. The canonical llama wall authority remains unmodified;
this build is diagnostic only.

Build manifest: CUDA compiler 13.2.86, g++ 13.3.0, CMake 3.28.3,
`CMAKE_BUILD_TYPE=Release`, `GGML_CUDA=ON`, architecture `120a`. Diagnostic
binary SHA256 is
`22ddbfb5d610d4ec9879bf22af5ed561d65a00c43b5dbbfa4a35efa0f28d1c96`.

## Protocol

All arms ran under `flock /tmp/gpu-bench.lock`, sequentially, with a bounded
timeout on the RTX 5090 (UUID
`GPU-c800ade9-21ea-2e55-f75c-6d7a458fb186`, driver 595.84):

```text
off_a -> density1_a -> density2_even -> density2_odd -> density1_b -> off_b
```

Common argv:

```text
llama-bench -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  -ngl 99 -fa 1 -p 0 -n 10 -d 512 -r 7 -o json
```

Model SHA256 is
`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`
(5,027,783,488 bytes). Analysis discards the first `samples_ns` repetition.
The first repetition maps to graph launch indices 0-8; span analysis therefore
retains launch indices >=9.

## Perturbation controls

Marker-free brackets are stable: 3.971918 versus 3.971073 ms/token, only
0.0213% drift. Relative to the pooled marker-free median:

| arm | wall ms/token | delta |
| --- | ---: | ---: |
| every replay, first | 3.982085 | +0.259% |
| alternating even | 3.977883 | +0.153% |
| alternating odd | 3.974792 | +0.076% |
| every replay, second | 3.984589 | +0.322% |

All controls are below the <=1% perturbation gate. Alternating even and odd
orders produce span medians of 3.889888 and 3.889312 ms. The result is
therefore not dependent on sampling parity or density at the precision needed
for the parity ledger.

## Implication

The prior profiled replay span near 3.888 ms was not materially inflated: the
unprofiled event result is 3.8898 ms. More importantly, only about 0.082 ms of
the measured llama token wall lies outside the whole graph replay window.
The remaining tinygrad-to-llama parity gap cannot plausibly be assigned to a
large hidden llama host-side remainder. Continue causal attribution inside
the graph—especially MMQ/dataflow and tinygrad's group/kernel topology.

Machine-readable result:
`docs/task_workflow/output/nv-decode-llama-d512-unprofiled-graph-span-20260804.json`.
Raw session artifacts and their hashes remain in `/tmp/p4_*` and
`/tmp/p4_artifact_sha256.txt`. No llama or tinygrad production code was
changed, committed, or pushed.
