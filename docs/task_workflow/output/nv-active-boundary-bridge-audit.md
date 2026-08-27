# Active-boundary bridge audit

## Verdict

The front-end cadence signal survives audit, but the original heterogeneous
numbers were invalid. The ABI-correct result is a 68.460--78.356-us advantage
for CUDA graph over native HCQ across the synthetic 208-call projection chain.
It remains a substrate qualification, not production wall recovery.

## Defect found and corrected

Three Q8-based captured kernels require one scalar argument with value `700`:

- `q4k_warp_coop_q8_dp4a_direct_4096_4096`;
- `q4k_q6k_warp_coop_q8_dp4a_pair_direct_1024_4096`;
- `q4k_warp_coop_q8_dp4a_pair_direct_1024_4096`.

The initial bridge passed only pointer arguments. This affected 36 of 208
calls and made the CUDA parameter block undersized. The harness now carries
captured scalars through both `NVProgram.fill_kernargs` and CUDA `encode_args`,
asserts that every captured family has one stable ABI, and asserts exact name
coverage against the physical-count manifest.

The four pre-fix heterogeneous JSON files are audit provenance only. They are
not valid recovery authorities.

## Corrected confirmations

The arms were run in both orders.

| ordering | native HCQ | CUDA graph | native minus graph |
|---|---:|---:|---:|
| CUDA then NV | 2674.572 us | 2606.112 us | **68.460 us** |
| NV then CUDA | 2681.716 us | 2603.360 us | **78.356 us** |

CUDA graph wins at every prefix (`16, 32, 64, 128, 208`) in both corrected
runs. At the full population, individual-sample standard deviations are about
2--4 us. The smallest cross-product difference between any native sample and
any CUDA sample is 58.476 us in the first ordering and 61.200 us in the second.
The separation is therefore not ordinary sample noise or arm order.

The independent single-cubin slope remains valid because that ABI has only
three pointers. It measures native HCQ at 1.5258 us/node and CUDA graph at
1.2941--1.2947 us/node.

## What the experiment proves

- Identical cubins can receive lower serial replay service through CUDA graph.
- The effect survives heterogeneous production binaries, launch geometries,
  buffer sizes, captured scalar ABI, and reversed process order.
- Ordinary CUDA launches are slower, so the advantage belongs to graph replay
  rather than the CUDA driver in general.

## What it does not prove

- The bridge does not reproduce the installed two-GPFIFO dependency DAG.
- It reuses one allocation per kernel family rather than distinct per-layer
  weights, although the same reuse and ordering apply to both arms.
- Native duration uses submit-to-synchronize CPU drain while CUDA uses device
  events. Prefix scaling removes most fixed-domain bias, but a production wall
  bracket remains mandatory.
- It does not identify whether QMD encoding, QMD reuse, pushbuffer density, or
  replay submission structure causes the advantage.

## Semantic oracle

The corrected bridge was rerun with a deterministic nonzero byte pattern for
every captured buffer. After identical prefix populations and replay counts,
all 46 logical buffer extents were copied back and SHA-256 compared across
native HCQ and CUDA graph. The result is **46/46 bitwise matches**.

The first oracle attempt falsely reported seven differences because it hashed
native allocator padding beyond the captured logical buffer size while CUDA
hashed only the logical allocation. A focused single-kernel check proved the
logical read-only bytes were unchanged. The oracle now hashes exactly each
captured ABI extent and passes with zero mismatches. The failed padded-extent
files remain uncommitted scratch evidence and are not authorities.

## Decision

Retain the construction as `SUBSTRATE_PASSES_MECHANISM_AND_SEMANTIC_GATES`. Do not book the
68.460--78.356 us. Before a native implementation, add cross-runtime output
validation and capture/compare the CUDA graph command stream with native
dependent-QMD chaining. After identifying a transferable property, qualify it
behind a gate on the real two-queue graph and require a reverse endpoint wall
bracket.

The corrected mechanical translation is 3982.167--3992.063 us/token, or
250.50--251.12 tok/s. It is an investment ceiling only.

## Persistent-replay follow-up

An additional audit corrected the remaining front-end asymmetry: the first
bridge rebuilt native QMD allocations per sample while reusing one CUDA graph
executable. CUDA's first graph replay is indeed expensive, but a bound native
queue with stable command-buffer and QMD addresses still measures 2686.277 us
against a paired CUDA graph at 2613.920 us. All 46 logical hashes match. The
surviving persistent-replay difference is **72.357 us**.

Native already encodes the dependent chain in a constant 20-word pushbuffer
and enables QMD prefetch. Disabling prefetch loses 127.230 us; moving completion
to the pushbuffer is slightly slower; removing the leading wait/barrier is
approximately neutral. The remaining cause therefore requires a CUDA graph
QMD/descriptor/USERD capture rather than another source-level queue guess.

## Resolution: internal grid membar

A matched 208-node no-op population retained 36.900 us of the gap, proving a
pure node-service component. Field isolation then found the transferable cause:
every native grid QMD requested `CWD_MEMBAR_TYPE_L1_SYSMEMBAR`. Clearing that
field only when the QMD has a same-queue dependent successor moved the real
population from 2686.277 to 2635.776 us while preserving all 46 logical buffer
hashes. The final QMD remains unchanged.

An adversarial 256-pair producer/consumer chain reported zero errors across 511
internal edges per replay. The production reps=9 reverse bracket preserved the
token-stream hash and recovered 67.530 us/token. This property is now the
Blackwell default with `NV_RELAX_INTERNAL_QMD_MEMBAR=0` as rollback.
