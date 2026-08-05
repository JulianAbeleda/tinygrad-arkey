# Native-NV Q6_K exact warp32 and integer-MMA substrate record

Date: 2026-08-05  
Baseline: `a1a51c349`  
Status: **exact warp32 NO-GO; generic CUDA integer-MMA substrate missing; model route remains default-off**

## Finding

The distinct exact-fp16 map is correct but decisively slow. One physical warp
owns one `1024x4096` output row. Lanes `0..15` own the even Q6 scale group and
lanes `16..31` the adjacent odd group at the same position. Eight group-pair
steps cover every one of the 4,096 elements exactly once. This is neither the
installed 16-lane map nor the closed flat 128-thread/four-warp Q8 construction.

The independent `q6_k_reference` dequantization plus NumPy matvec oracle passes:

| row | installed | exact warp32 |
|---|---:|---:|
| max abs vs CPU | `2.95639e-5` | `1.85966e-5` |
| allowed atol | — | `0.02` |

After 1,000 alternating warmups, the included graph A/B/A is a large loss:

| arm | median us / graph replay |
|---|---:|
| installed partial4 + external sum midpoint | 67.166065 |
| exact warp32 direct output | 123.529102 |
| delta | **+56.363037** |

Seven 500-replay samples per arm were collected under `/tmp/gpu-bench.lock`.
The candidate is therefore closed before role integration or token-wall work.

## Production instruction and resource census

The smaller static body did not buy wall time:

| program | regs | PTX instructions | SASS instructions | PTX global loads | shuffles |
|---|---:|---:|---:|---:|---:|
| installed Q6 partial4 | 69 | 268 | 279 | 37 | 0 |
| installed external sum | 26 | 33 | 48 | 4 | 0 |
| exact warp32 | 50 | 174 | 175 | 23 | 5 |

The result falsifies “more exact lanes plus a shorter instruction body is
sufficient.” Physical traffic and ownership dominate: the warp map repeats
Q6 scale/dequant work across lanes and pays a five-shuffle reduction, while
the installed independent partial threads retain cheaper serialized ownership.

The observed llama d512 artifact must also constrain the next hypothesis. The
checked-in semantic-call manifest records `mul_mat_vec_q<type14,ncols=1>` from
`ggml-cuda/mmvq.cu`, launched as `grid=[1024,1,1]`, `block=[32,4,1]`, with 48
registers/thread and 384 bytes static shared memory. Direct disassembly of its
canonical extracted cubin entry contains 295 instructions, including six
`IDP.4A.S8.S8`, five `SHFL.BFLY`, three `LDS`, and no integer MMA. The matched
captured Q8+Q6 graph measured 4.09856 us.

This matters because `mmq.cuh` does contain an sm_120 integer-MMA path, but it
is not the path in the observed one-column d512 trace. A new trace may overturn
that fact; source capability alone does not.

## Generic CUDA integer-MMA substrate gate

The independent compiler probe passes
`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` on sm_120 with four A b32
registers, two B b32 registers, and four s32 accumulator registers. The GPU
and NVRTC surface therefore exist.

The generic tinygrad surface fails at three earlier points:

1. `tinygrad/codegen/opt/tc.py` has no CUDA `int8 -> int32` TensorCore descriptor.
2. `PTXRenderer.tensor_cores` admits only float/half CUDA descriptors.
3. An executable integer `render_wmma` probe fails at the float/half-only dtype
   map with `KeyError: dtypes.int`.

That is a localized substrate blocker, not a hardware verdict. The next
bounded substrate task is a generic CUDA s8-to-s32 descriptor, a mechanically
validated m16n8k32 fragment lane map, PTX register packing/lowering, and an
independent exact integer matrix value test. Only after that passes should a
Q6 packed-tile integration be timed.

## Admissibility verdict

Integer MMA remains **admissible as a research substrate**, despite Q8_1 not
being bitwise identical to the fp16 recurrence. The campaign's model authority
is exact tokens/argmax/top-10, aggregate relative L2 at most `1e-3`, and
perturbation below the minimum top-k margin. Shared-Q8 g12 already passed that
contract at `8.36963e-4` relative L2.

It earns no d512 parity credit today. The current exact warp32 primitive lost,
and the observed llama d512 Q6 mechanism is MMVQ/IDP.4A rather than integer
MMA. A future integer-MMA route must first pass exact Q6×Q8 format algebra,
then the declared full-logit contract, then an included-cost primitive A/B/A,
and finally settled token wall. No production emitter or selector was changed.

Artifacts:

- `extra/llm_research/decode/q6k_exact_warp32_microgate.py`
- `extra/llm_research/decode/q6k_nv_int_mma_substrate_gate.py`
- `test/unit/test_q6k_exact_warp32_mapping.py`
- `test/unit/test_q6k_nv_int_mma_substrate_gate.py`
- `docs/task_workflow/output/nv-q6k-exact-warp32-and-int-mma-gate-20260805.json`
- raw `/tmp/q6k_exact_warp32_full.json`, SHA256
  `67dbe0432303aeb31e7da30dd4001dfd3e0c53a8d3b1329a0960ae7b5c2365a0`
