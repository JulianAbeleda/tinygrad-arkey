# K four-warp Q4_K gate result (2026-08-21)

## Verdict

The exact four-warp Q4_K candidate is **faster than the installed K GEMV** at
the production K shape, and the primitive gate is **PASS**.

| Quantity | Installed K | Four-warp K | Delta |
|---|---:|---:|---:|
| cudaEvent median (us/launch) | 3.4417 | 2.9599 | -0.4818 |
| Ratio | 1.000 | 0.8600 | -14.0% |
| ncu occupancy | 11.98% | 47.96% | +35.98 pp |
| ncu DRAM throughput | 22.17% | 25.95% | +3.78 pp |
| ncu duration (ns) | 6112 | 5216 | -896 |
| registers/thread | 61 | 34 | -27 |

All timing values above are observed from the standalone `cudaEvent` and `ncu`
brackets. The candidate is the exact Q4_K group-factorized fp16 consumer with
128 threads per row (four warps), no Q8 provider node, one contiguous
`fp32[1024]` output.

## Premise correction

The earlier "port four-warp to Q/O/K" direction was partly wrong. The
per-shape ledger already shows Q, O, V, gate/up, down Q6_K, and vocab at or
better than llama; four-warp Q4 at the 4096-row FFN shape was already measured
wall-neutral (+2.868 us exact, -0.208 us factorized). The only row-starved
anchor that still trails is the legacy K path at 1024 rows, one warp per row.
That is the single shape this gate targets.

## Method

The Python microgate (`q4k_k_four_warp_microgate.py`) confirms numerical
equivalence but is timing-blind here: tinygrad graph replay adds about 120 us of
host overhead on top of a roughly 3 us kernel, so both arms read ~122 us and the
host gate is uninformative. The measurement was therefore moved to a standalone
CUDA bracket (`q4k_k_four_warp_cuda_bracket.py`) that renders both production
CUDA sources through `CUDARenderer`, compiles with nvcc for `sm_120a`, and times
control/candidate/control with `cudaEvent`. `ncu` supplies occupancy and DRAM
counters.

Correctness was established independently in the Python gate:

- candidate vs installed max abs error: 2.146e-06
- candidate vs installed relative L2: 2.456e-07
- independent fp64 oracle max abs error: 3.576e-07 (tol 0.01936)

No production route, selector, or promotion imports the candidate.

## First-principles read

The K shape is row-starved: 1024 output rows at one warp per row leaves only
about 12% of the SM warps active. Four-warp ownership raises active warps to
about 48%, and the lower register count (34 vs 61) removes the old path's
occupancy pressure. DRAM throughput rises only modestly because this is a short
stream, so the win comes from recovering launch/occupancy latency rather than
from saturating DRAM.

## End-to-end impact (inferred, not yet measured)

The measured saving is about 0.48 us per K anchor. The model has one K anchor
per layer, so 36 layers imply roughly 17 us/token of recovered decode time if
this geometry replaces the legacy K route. That is a real, small, clean win; it
is **not** the ~240 tok/s parity answer by itself. Route-level qualification
(token-stream SHA plus a reverse wall bracket) is required before promotion, and
that has not been run here.

## Evidence

- `docs/task_workflow/evidence/nv-q4k-k-four-warp-20260821/result.json`
- `docs/task_workflow/evidence/nv-q4k-k-four-warp-20260821/cuda_bracket.json`
- `docs/task_workflow/evidence/nv-q4k-k-four-warp-20260821/sha256.txt`
