# NV overlap root cause: the memory planner serializes the decode DAG (2026-08-15)

Date: 2026-08-15
Branch: `nvidia-bringup-20260731` (HEAD `e184453d8`)
Status: **measured. Ground-truth runtime dependency capture plus wall A/B.**

This record corrects the overlap verdict in
`nv-overlap-route-b-head-wall-record-20260815.md` and
`nv-decode-dag-width-verdict-20260815.md`. Those records concluded the decode
DAG is width 4 (q/k/v siblings) and that multi-stream overlap is FLAT because
the only parallelism is bandwidth-bound. Both conclusions came from a
reconstructed DAG. This record measures the real graph instead.

## 1. The real runtime DAG is a chain, not width 4

`scratchpad/nv_decode_runtime_deps_probe.py` monkeypatches
`DepsTracker.access_resources` and `CUDAGraph.__init__` so it records, for every
programmatic graph node at capture time, the exact producer list returned by the
runtime dependency tracker. This is ground truth, not a reimplementation.

First decode graph (32 nodes), default memory planner:

| idx | kernel | actual predecessors |
| ---: | --- | --- |
| 0 | E_c9 (vocab root) | - |
| 1 | E_2 (vocab root) | - |
| 2 | E_16 (vocab) | [0] |
| 3 | E_1187 (vocab) | [1, 2] |
| 4 | r_32_32_4 (norm) | [3] |
| 5 | r_16_256 (norm) | [4] |
| 6 | E_32_32_4 (norm) | [5] |
| 7 | q GEMV | [6] |
| 8 | k GEMV | **[7]** |
| 9 | v GEMV | **[8]** |
| 10..31 | rope/kv/flash/output/FFN | single chain onward |

Every node after index 3 has exactly one predecessor. q/k/v are not siblings;
they are `q -> k -> v`, and the rest of the layer is one serial chain. This is
why the profiled decode token has `overlap_mass_us = 0.0`.

The earlier width-4 reconstruction missed these edges: it resolved buffers
before `ensure_allocated()`, so its `id(buf.base)` / range keys did not match the
runtime's keys and it omitted the WAR/WAW edges the memory planner adds.

## 2. The memory planner is what adds the chain

The same probe with `NO_MEMORY_PLANNER=1`:

| idx | kernel | actual predecessors |
| ---: | --- | --- |
| 6 | E_32_32_4 (norm) | [4, 5] |
| 7 | q GEMV | [6] |
| 8 | k GEMV | [6] |
| 9 | v GEMV | [6] |
| 14 | rope/kv | [7, 12] |
| 15 | rope/kv | [8, 13] |
| 17 | r_8_8_16_2_4 | [9, 15] |
| 25 | gate GEMV | [24] |
| 26 | up GEMV | [24] |

q/k/v and gate/up become true siblings. The full planner-on graph is width 1;
the planner-off graph has 9 fan-in/fan-out joins in the first 32 nodes alone and
many more across the 1021-node token. So the planner's liveness-based arena
reuse is aliasing independent fan-out buffers (q/k/v, gate/up) into one slot,
adding WAR/WAW edges that collapse the DAG into a chain.

## 3. Does breaking the chain move the wall? Yes, modestly

Canonical decode authority, fresh process per arm, depth 512, 32 tokens,
Qwen3-8B-Q4_K_M / RTX 5090:

| planner | streams | wall ms/token | tok/s |
| --- | ---: | ---: | ---: |
| on | 1 | 5.59 | 178.95 |
| off | 1 | 5.58 | 179.35 |
| off | 2 | 5.48 | 182.55 |
| off | 3 | 5.37 | 186.13 |
| off | 4 | 5.33 | 187.53 |
| off | 6 | 5.34 | 187.24 |

The same Route B wall probe used by the prior record measures 188.05 tok/s at
planner-off/4 streams, and its token sha is `ddf344135e...` - bitwise identical
to the planner-on serial arm. Breaking the chain is correctness-clean.

Plateau is ~187.5 tok/s at 4 streams: **+8.5 tok/s (+4.8%)**. The gain is real
but far below the old ~219-245 tok/s overlap claims.

## 4. Why the gain is only ~8.5 tok/s

- The only work the planner-off DAG exposes is q/k/v and gate/up fan-out. The
  rope/kv/flash/norm support is a fan-in on the critical path (each consumes a
  GEMV output), so it cannot hide behind an anchor.
- q/k/v and gate/up are GEMVs. Overlapping them does not stack 1:1 because they
  contend for the same HBM bandwidth; the measured saving (~220 us) is smaller
  than their serialized sum (~400 us q/k/v plus ~680 us gate/up).

This is the correct, measured reason the Route B wall A/B was FLAT: the
multi-stream lowerer was fed a planner-serialized chain. It was not bandwidth
physics, and it was not "the DAG has no independent work".

## 5. Verdict and next action

- Overlap is **not closed**. It was blocked by the memory planner aliasing
  independent live ranges.
- The honest, measured ceiling of this specific fix is ~187.5 tok/s.
- Landing it requires a **targeted memory-planning change**: keep independent
  fan-out live ranges (q/k/v, gate/up) in distinct arena slots instead of
  aliasing them, then enable the existing `CUDA_GRAPH_STREAMS>1` capture path.
  `NO_MEMORY_PLANNER=1` is only the probe; it is not a product change (it pins
  every buffer and spikes VRAM).
- This does not change the parity ledger: 220-240 still requires kernel work on
  the Q6 GEMV core (~240 us) and the flash-score floor (~90 us).

## Evidence

- runtime dep capture: `/tmp/nv_decode_runtime_deps_probe.json` (planner on),
  `/tmp/nv_decode_runtime_deps_probe_np.json` (planner off)
- wall A/B: `/tmp/np_wall_{on_s1,off_s1,off_s2,off_s3,off_s4,off_s6}.json`
- token-identity wall: `/tmp/route_b_wall_np_s4.json` (sha `ddf344135e...`)
- nsys multi-stream trace: `/tmp/tg_s2.sqlite` (streams=2 capture)
- prior nsys serial trace: `/tmp/tg_node_head_20260815.sqlite` (overlap 0.0)
- probe: `scratchpad/nv_decode_runtime_deps_probe.py`
