# Native NV exact llama Q6 cubin bridge record

Date: 2026-08-05. Target: RTX 5090, native `DEV=NV`, sm_120, driver
595.84. This is a research diagnostic only. It adds no selector, route, or
production dependency on an external cubin.

## Verdict

**PASS.** Tinygrad's native NV runtime loaded and executed the pinned llama
CUDA Q6_K MMVQ machine-code symbol directly. The single guarded
`1024 x 4096` prepacked-Q8 case preserved both output guards and matched an
independent packed-Q6/Q8 decoder plus dense matvec oracle with maximum absolute
error `3.5762787e-6` and mean absolute error `7.605413e-7` (`atol=0.02`).
The completion-timed kernel event was `4.352 us`.

This proves the exact llama consumer can be used as a native-NV diagnostic
oracle. It does not prove a production win: this gate excludes the Q8 provider,
and the installed Q6 attention path includes its external sum. Included-cost
comparison is the next required gate.

## Exact CPU ABI proof before launch

The pinned cubin SHA-256 is
`57c0a9caeee467316fc295b9447a61873d5578eb2a5a28b36e58f237e605bd88`.
The selected symbol is the plain Q6_K, one-column, no-fusion MMVQ entry:

`_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj`

The raw CUDA ELF metadata establishes:

- target text: 5,120 bytes;
- target function-symbol index: 355;
- target `EIATTR_REGCOUNT`: 48, selected by that index rather than by record
  order in the multi-kernel cubin;
- target static shared section: 1,408 bytes;
- target stack bytes: zero;
- `EIATTR_PARAM_CBANK`: constant-section symbol index 906, parameter base
  `0x380` (896), parameter size `0x90` (144);
- `EIATTR_CBANK_PARAM_SIZE`: independently 144;
- target `.nv.constant0.<symbol>` extent: exactly `896 + 144 = 1040` bytes;
- 19 `EIATTR_KPARAM_INFO` rows, ordinals 0--18, with declared relative
  offsets 0--140 and exact sizes;
- relocation types: only `R_CUDA_64` (2), supported by `NVProgram`.

This resolves two unsafe assumptions before dispatch. The first PARAM_CBANK
word is a constant-section symbol index, not a byte offset. Also, the existing
multi-symbol `NVProgram` scan retains the last global register-count record
(`160`) unless the target record is selected explicitly. The bridge therefore
installs the exact 896-byte native prefix plus 144-byte parameter region,
sets CB0 extent to 1,040 bytes, sets the QMD register count to 48, and expands
the kernargs allocation before launch.

## Protocol and result

The parent acquired `/tmp/gpu-bench.lock` and launched one worker subprocess
with a 45-second timeout. The worker generated deterministic fp32 weights and
activation, quantized them with pinned llama CPU Q6_K and Q8_1 reference
quantizers, and constructed an independent numerical oracle by decoding both
packed payloads and applying a dense matvec. Q8 was prepacked; no provider was
included in this first ABI/correctness gate.

| item | result |
| --- | ---: |
| shape | `1024 x 4096` |
| launch | grid `(1024,1,1)`, block `(32,4,1)` |
| guards | exact |
| finite | true |
| maximum absolute error | `3.5762787e-6` |
| mean absolute error | `7.605413e-7` |
| tolerance | `0.02` |
| completion-timed kernel | `4.352 us` |
| verdict | **PASS** |

## Artifacts and boundaries

- `docs/task_workflow/output/nv-native-llama-q6-cubin-bridge-gate-20260805.json`
  is the anchored result.
- `extra/llm_research/decode/nv_llama_cubin_bridge.py` is the CPU inspector,
  ABI serializer, target-QMD adapter, timed-subprocess harness, and guarded
  worker.
- `extra/llm_research/decode/nv_eiattr_kparam_census.py` owns the raw
  EIATTR decoder and ordinal/offset packer.
- The external cubin remains a pinned local diagnostic input and is not copied
  into a production path.
- No timing is booked toward parity from this gate. A native included-cost
  provider plus exact consumer comparison must pass before any model lease.
