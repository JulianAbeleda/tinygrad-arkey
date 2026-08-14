# NV overlap Layer-1 substrate test: does multi-stream move the decode wall?

Date: 2026-08-14. Target: RTX 5090, `DEV=CUDA` CUDAGraph route, sm_120.
Status: **measurement record. Substrate runs, but buys ~0 wall on the decode DAG.**

## 1. Question

The committed arithmetic split the 193 -> 245 gap into ~398 us kernel work, ~936 us
overlap, and ~211 us launch gaps. The ceiling test bounded overlap by the DAG critical path
(~219 tok/s), but nobody had run the actual decode graph through the multi-stream lowerer.
This record answers the substrate question directly: does `CUDA_GRAPH_STREAMS>1` move the
real decode wall?

## 2. Method

Same machine, same model, one flocked session, three fresh subprocesses:

```text
DEV=CUDA CUDA_GRAPH_STREAMS=1/2/3 python3 extra/llm_research/decode/decode_runtime_overhead.py \
  --model Qwen3-8B-Q4_K_M.gguf --ckpts 512 --max-context 4608 --nmeas 32 --reps 2
```

The harness is the canonical fixed-depth decode authority (`tinygrad.decode.fixed_depth.v2`);
W is the production generate path. The only knob changed between arms is the stream count.

## 3. Result

| streams | tok/s (W) | wall ms/token | token stream sha |
| --- | ---: | ---: | --- |
| 1 | 178.42 | 5.6048 | `5ede6924aaaa9acc69f9` |
| 2 | 178.98 | 5.5872 | `5ede6924aaaa9acc69f9` |
| 3 | 178.76 | 5.5941 | `5ede6924aaaa9acc69f9` |

Streams 2 and 3 run correctly (identical tokens, `rc=0`, no capture/assert failure), but the
wall is unchanged within noise (+0.3%). The multi-stream CUDAGraph lowerer is present and
numerically correct; the decode DAG has no exploitable independence for it to overlap.

This matches two prior findings that were never reconciled into one place:

- The B2 belief refinement (`70de5dc0f`): capture is redundant on this driver, the lever is
  DAG independence, not stream count.
- The node ledger (`nv-tinygrad-node-ledger-gap-record-20260813.md`): the `DEV=CUDA` decode
  token has `overlap_mass_us: 0.0` (span exceeds node-sum), i.e. a chain.

## 4. Why this reframes the 219 claim

The ceiling test's 219 figure used the 596-node `DEV=NV` pre-split DAG and applied a
multi-stream critical-path schedule to it. The route that actually has the multi-stream
lowerer (`DEV=CUDA`) runs a different, chain-like graph (and loses the promoted
`reduce-output` norm fusion, which is why its baseline is ~178 tok/s versus ~193 native).
Turning on streams does not reconstruct that 596-node independence: it buys ~0.

Therefore Layer 1 as a "flip a stream knob" item is closed at ~0 tok/s. The 219 bound is
not a substrate toggle; reaching it requires DAG-shape work (re-consolidate the six
ping-pong graphs, restore the fused-norm graph, and create GEMV-GEMV / support slack), which
is the critical-path work, not the overlap lowerer.

## 5. Verdict

- Hardware + lowerer substrate: **present and correct** (multi-stream capture runs end to end,
  identical numerics).
- Transferable wall on the current decode DAG: **~0 tok/s**.
- Next lever is DAG independence/critical-path shortening, not another stream or channel
  construction attempt.

## 6. Artifacts

- `/tmp/overlap_decode_s1.json`, `/tmp/overlap_decode_s2.json`, `/tmp/overlap_decode_s3.json`
- B2 microbench (substrate smoke, S=1/2/3 numerics-clean, wall-neutral):
  `extra/llm_research/microbench/cuda_graph_multi_stream_tg_probe.py`
