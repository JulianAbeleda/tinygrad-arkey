# NV P3b FFN-down shared-Q8 successor: topology and ABI close gate

Date: 2026-08-05
Target: one-block research lease only; native `NV sm_120`
Status: **SUPERSEDED — existing-producer hypothesis closed; distinct cooperative producer later built and wall-rejected**

This record closes only reuse of the installed scalar producer. The explicitly
permitted distinct 32-row construction was subsequently built, passed its ABI,
resource, topology, and numerical gates, then regressed the included chain by
**+151.191835 us/replay**. The terminal result is recorded in
`nv-p3b-ffn-q8-cooperative-producer-static-record-20260805.md`; it supersedes
the construction-blocked next step below without changing this record's causal
finding.

## Hypothesis tested

Reuse the landed scalar W1/W3 producer, turn its
`z[r] = silu(gate[r]) * up[r]` values directly into llama-compatible Q8_1
packets, and feed a native Q4_K or Q6_K DP4A FFN-down consumer.  The desired
benefit is removal of both the standalone fp16 activation boundary and a
separate Q8 boundary.

The exact one-block shapes are `gate/up: 12288 x 4096` and
`down: 4096 x 12288`.  Q8_1 has 32 values per block, hence the proposed output
contains 384 independent packets: 3072 packed int8x4 words plus 384 `d|s`
half2 metadata words (13,824 bytes total).  The source value for every packet
must be `float32(silu(dot_g)*dot_u).cast(fp16).cast(fp32)`: the existing native
down route performs that fp16 prelude before consuming `z`, so omitting it is a
numerical-route change, not a fusion.

## CPU ABI result

The live llama Q8 oracle is already pinned in
`scratchpad/llama_cuda_quantized_live_oracle.py`: `quantize_row_q8_1_ref`
requires 32 rounded fp32 inputs, emits an int8 payload, and records both fp16
`d` and fp16 raw-input-sum `s`.  The existing native shared-Q8 attention helper
uses a **consumer-private structure-of-arrays layout** (all int8x4 words then
all metadata words), not a literal 36-byte contiguous `block_q8_1`; its Q4/Q6
emitters explicitly index that private layout.  It cannot be passed unchanged
to an FFN-down consumer or claimed llama ABI compatible.  A P3b route must
either define and test one exact private ABI end-to-end, or use literal llama
blocks and teach both Q4 and Q6 consumers that layout.  No such FFN consumer
exists today; current down primitives consume fp16 directly.

## Structural blocker

The landed scalar W1/W3 program in `decode_kernels.py` is one output row per
32-thread workgroup (`gidx0 = row`).  It finalizes and stores one `z[r]` after
its 4096-wide dot reduction.  An exact Q8_1 packet needs the max, rounded
quantization, and raw sum across **32 adjacent rows**.  Those rows are 32
separate workgroups under the landed map, and generic UOp/KernelProgram has no
grid-wide synchronization or cross-workgroup reduction that can make their
packet in the existing program.

The alternatives do not satisfy the stated topology:

1. Store `z` (fp16), then quantize it in a second program: preserves the
   activation boundary and adds a Q8 write/read.
2. Use atomics or a multi-pass max/sum: introduces additional global stages and
   cannot be called a direct producer-to-packet fusion.
3. Rebuild W1/W3 as a 32-output-row cooperative CTA: is a wholly new GEMV
   mapping, not reuse of the landed scalar producer.  It changes its reduction/
   scheduling basis and invalidates the established in-loop performance and
   token-safety evidence; the prior quad W1/W3 mapping already regressed in
   loop.

Therefore no candidate simultaneously (a) reuses the existing producer, (b)
eliminates the fp16/Q8 boundaries, (c) preserves the exact fp16-rounding/Q8_1
contract, and (d) shrinks program topology.  There is no safe Q4 or Q6
one-block GPU qualification to run under this hypothesis.

## Default and reopen rule

No model route, admission, generated policy, or default was changed.  The
existing W1/W3 fusion and direct fp16 down consumers remain byte-identical.

Reopen only with a CPU-tested **new** 32-output-row producer design that states
its packet ABI (including `d`, `s`, byte order, and fp16 round point), proves
the exact llama Q8 oracle on rounded `z`, and shows a program census that is
strictly smaller than landed W1/W3 + fp16 boundary + down for one Q4 block and
one Q6 block.  Only then may independent Q4 and Q6 full-logit/token gates run;
wall work remains after those gates.
