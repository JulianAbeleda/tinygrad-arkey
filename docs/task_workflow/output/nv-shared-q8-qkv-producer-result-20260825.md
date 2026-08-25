# NV shared-Q8 Q4/Q4/Q4 producer result

Date: 2026-08-25

Target: Qwen3-8B Q4_K_M dense decode on NV `sm_120`

## Outcome

The shared-Q8 Q+K/V triple-producer idea is numerically valid, but neither
currently expressible exact geometry passes the cold complete-span gate. No
production route is changed.

Both candidates reproduce all 6,144 fp32 Q/K/V output words bit-for-bit and
use the same weight payload as the installed Q producer plus paired K/V
producer. Both win in the L2-hot regime. Once the weights are cold, CTA
aggregation lowers the service rate enough to consume the launch saving.

The experiment reaches a precise compiler/geometry wall rather than an opaque
output or incomplete-population wall: the remaining full-parallelism spelling
needs a workgroup-uniform conditional containing the exact K/V four-warp merge
barrier. tinygrad's typed post-barrier region deliberately forbids inner
workgroup barriers because it cannot yet prove the predicate is uniform over
the workgroup.

## Exact geometries tested

The installed control is one 4,096-CTA Q body followed by one 1,024-CTA K/V
pair body. Every projection row uses four warps and the existing left-to-right
four-partial merge.

| geometry | CTAs | rows owned per CTA | output storage |
| --- | ---: | --- | --- |
| control | 4,096 + 1,024 | one Q, then one K+V | separate Q/K/V |
| collapsed | 1,024 | four Q + K + V | separate Q/K/V |
| balanced | 2,048 | two Q + one packed K-or-V | Q plus contiguous K/V |

The collapsed and balanced kernels each use 40 registers, do not spill, and
preserve the exact Q4 block loop, warp shuffle tree, and cross-warp merge for
every logical row. The balanced input/output packing is part of the isolated
gate; it is not installed model storage.

## Correctness and timing gates

All nine repeated measurements are medians of complete CUDA-event spans.
Pointer-rotated cold timing cycles through a weight population much larger
than L2 and does not place a cache-flush operation next to the measured launch.

| regime, us/group | control | collapsed | balanced |
| --- | ---: | ---: | ---: |
| hot | 7.166 | 6.342 | 5.976 |
| explicit-flush cold | 17.798 | 23.038 | 18.818 |
| pointer-rotated cold | 13.654 | 13.798 | 15.676 |

The hot gains are real: 0.823 us/group for collapsed and 1.190 us/group for
balanced. They do not survive the production-like cold regime. The clean
pointer-rotated result projects to losses of 1.296 us/token and 18.198
us/token across the nine eligible groups. This fails before model integration,
so there is no justification for an output-ownership lease, route census, or
full-token A/B/A.

## Service-rate ceiling

The measured payload is about 14.17 MB per Q4/Q4/Q4 group. Holding that byte
count and all other work fixed, the best-case arithmetic ceilings are small:

| hypothetical aggregate rate | body time | recovery versus balanced | nine-group token recovery |
| ---: | ---: | ---: | ---: |
| 1.135 TB/s (Q-control rate) | 12.48 us | 3.20 us/group | 28.8 us/token, about +1.6 tok/s |
| 1.47 TB/s (down rate) | 9.64 us | 6.04 us/group | 54.3 us/token, about +3.0 tok/s |
| 1.60 TB/s (gate/up rate) | 8.85 us | 6.82 us/group | 61.4 us/token, about +3.5 tok/s |

These are roofline ceilings, not forecasts: they assume the higher rate costs
no extra instructions, barriers, or output transport. They show why CTA
service-rate improvement is worth testing, but cannot by itself close a
roughly millisecond-scale endpoint gap.

## Counter accounting

Cold NCU replay observes effectively the same DRAM payload:

| body | DRAM bytes | kernel duration | DRAM peak fraction |
| --- | ---: | ---: | ---: |
| Q control | 9,448,704 | 8.320 us | 64.44% |
| K/V pair control | 4,729,088 | 5.888 us | 45.85% |
| collapsed | 14,170,112 | 13.376 us | 60.14% |
| balanced | 14,167,040 | 15.456 us | 52.04% |

The small byte differences are cache-sector replay variation, not a payload
reduction. NCU's kernel-only replay makes collapsed nominally faster than the
sum of the two separately profiled controls, while the independent
pointer-rotated complete span is flat-to-negative. That disagreement is why
the flush-only negative was not accepted at face value. The repaired
complete-span gate resolves it: collapsed does not recover wall time, and
balanced clearly reduces rate.

## Promoted wall and next rank

The only remaining exact single-launch topology that retains the control's
4,096-CTA Q population is:

- one Q row in every CTA;
- one packed K-or-V row in the first 2,048 CTAs;
- no K/V loads or arithmetic in the inactive 2,048 CTAs;
- the same four-warp K/V partial merge inside the active CTA branch.

CUDA permits a barrier in a branch when the predicate is uniform for the
whole workgroup, but tinygrad currently has no typed proof for that contract.
Its `PostBarrierRegion` correctly rejects all inner workgroup barriers. The
full-grid construction is therefore promoted as a compiler prerequisite, not
silently approximated with duplicate reads, changed reduction association, or
a second completion launch.

The exact-action queue is now:

1. add and independently validate a workgroup-uniform structured region that
   safely admits an inner barrier, then test the 4,096-CTA Q+K/V spelling;
2. if that prerequisite is not taken, return to single-body issue/dequant
   scheduling for rate-deficient Q/O/K/V bodies;
3. keep numerical byte reduction in its separate quality-evaluated lane.

No token-rate recovery is booked from this experiment. The installed endpoint
and prior full ledger remain authoritative.

Evidence:

- `docs/task_workflow/evidence/nv-shared-q8-qkv-producer-20260825/microgate-r9.json`
- `docs/task_workflow/evidence/nv-shared-q8-qkv-producer-20260825/cold-counters.json`
- `docs/task_workflow/evidence/nv-shared-q8-qkv-producer-20260825/rotated-cold-r9.json`
- `docs/task_workflow/evidence/nv-shared-q8-qkv-producer-20260825/verdict.json`

Verdict: `NO_GO_CURRENT_GRAMMARS_FULL_GRID_BLOCKED_BY_TYPED_UNIFORM_CONTROL_FLOW_WALL`.

## 2026-08-25 follow-up: typed full-grid geometry clears the wall

The compiler-side uniform-region prerequisite was implemented and the exact
full-grid producer was retested with caller-owned, separate Q/K/V outputs.
The isolated microgate remained bit-exact and recovered about 1.5--2.0 us per
producer group in pointer-rotated cold replay. The decisive production bracket
used 9 eligible blocks, depth 512, 32 tokens, and 7 accepted repetitions per
arm. All token hashes matched. The control midpoint was 4.380153 ms/token and
the candidate was 4.364033 ms/token: **16.119 us/token recovered**, or about
0.84 tok/s at this endpoint (228.3 -> 229.1 tok/s).

This is a promoted geometry win, not a byte win: DRAM payload is unchanged and
the end-to-end recovery is smaller than the isolated sum because the producer
overlaps the rest of the token path. The earlier packed-output result was not
accepted because output slicing introduced copy kernels; separate caller-owned
outputs remove that accounting defect.

Evidence: `full-split-output-smoke.json` and
`production-wall-triple-r7-split.json` in the producer evidence directory.
