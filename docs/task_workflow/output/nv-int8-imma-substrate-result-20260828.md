# NV signed-int8 tensor-core substrate result

## Result

The smallest generic NVIDIA signed-int8 to int32 tensor-core substrate now
executes on RTX 5090 / sm_120.  It is research-only and does not alter model or
prefill routing.

Added substrate:

- CUDA `m16n8k32` signed-int8 input / int32 accumulator TensorCore descriptor;
- validated 32-lane map with 16 A bytes, 8 B bytes and 4 int32 accumulators per lane;
- CUDA and PTX spelling for `s32.s8.s8.s32`;
- architecture admission at sm_89 and newer (therefore sm_120), while sm_80
  remains closed;
- isolated executable and cubin-capture microgate.

The GPU microgate computed `(16x32 int8) @ (32x8 int8)` exactly against NumPy
int32: finite, maximum absolute error zero, output SHA-256
`a0bf57818cb866896f55081e78fe66886bdc9b4f7928eeef8be482f099b27cd2`.

Binary proof from `/tmp/nv-int8-imma-artifacts/int8_imma.cubin`:

- one `IMMA.16832.S8.S8`;
- zero HMMA and zero IDP.4A;
- 32 registers/thread;
- stack 0, shared 0, local 0.

The reproducible harness is
`extra/llm_research/prefill/nv_int8_imma_microgate.py`.  The host-synchronized
microsecond samples in this single-tile construction include Python scheduling
and dispatch and are not a kernel-performance claim.

## New precise wall

The generic IMMA instruction substrate is no longer the blocker.  The next wall
is the direct-packed Q4_K fragment emitter.  The existing
`emit_q4k_q8_mmq_kernel` still raises `NotImplementedError`; the current generic
Tensor contraction can consume int8 operands only after Q4_K has been expanded
to an int8 tensor.  Such materialization would surrender the 3.56x weight-byte
advantage and is therefore not an admissible Q4 prefill proof.

The next implementation must map packed Q4_K bytes and scale/min metadata
directly into the new IMMA A/B lane fragments, apply the Q8_1 scale and Q4_K min
correction, and retain output guards without creating a global expanded-weight
buffer.  Only then should stream-K and the production-sized gate/up lifecycle be
measured.  Production admission remains closed.

## Verification

`53 passed` across the new NV descriptor/render tests, fp8 admission regression,
Q4/Q8 MMQ logical spec, and arithmetic reference tests.
