# NV packed Q4_K IMMA terminal wall

## Verdict

The generic signed-int8 IMMA substrate passes for direct int8 buffers, but the
current tensor-core optimizer cannot legally consume an ALU-produced fragment
from packed Q4_K bytes.  Production admission remains closed.

Two direct-packed constructions were tested at `(M,N,K)=(16,8,256)`:

1. canonical `uint32[36]` Q4_K words, with shift/mask byte extraction and
   nibble extraction kept lazy in the matmul sink;
2. the same canonical words viewed as `uint8[144]`, then direct payload slice
   and nibble extraction kept lazy in the matmul sink.

Both graphs had exactly one scheduled sink and rendered
`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`, with the packed buffer as a
kernel argument and no expanded-weight allocation.  Both returned an all-zero
int32 matrix for a nonzero legal fixture, while the independent NumPy oracle
returned nonzero group dots (for example 2051, 2748, 3657).  In contrast, the
new IMMA substrate is bit-exact when A and B are direct int8 buffers.

This isolates the failure below Q4 arithmetic and above instruction emission:
the generic TensorCore lane-map rewrite assumes load/reshape fragment leaves.
When the fragment values are produced by byte/nibble ALU nodes, the rewrite
still emits IMMA but loses their value mapping.  SASS presence is therefore not
sufficient evidence for packed-fragment correctness.

## Production-shape bridge result

Before the numerical failure was exposed, the same one-sink spelling executed
the full gate/up shape and allocated `[128,512,12288]` int32 group partials:
3.221 GB.  First compile plus execution was 0.492 s.  This is structurally and
numerically inadmissible, and even a corrected form would be far from the
required lifecycle because it writes one partial per 32-value group.

## Required next substrate change

Do not add more graph-level reshapes.  The next implementation must introduce a
typed packed-fragment boundary, either:

- a TensorCore operand-provider contract whose lane map accepts byte/nibble ALU
  producers and proves every register byte, or
- a research-only composite UOp/native emitter that owns packed Q4_K loads,
  nibble placement, eight IMMA instructions per 256-K block, scale/min
  correction, and guarded FP32 writeback.

The arithmetic contract is pinned by
`extra/llm_research/prefill/q4k_q8_imma_oracle.py`: unsigned Q4 nibbles, Q8
signed values, canonical six-bit scale/min unpack, and raw prequantized FP32
group sums rounded to FP16 for the min term.  A future emitter must pass that
oracle before production-shaped timing.
