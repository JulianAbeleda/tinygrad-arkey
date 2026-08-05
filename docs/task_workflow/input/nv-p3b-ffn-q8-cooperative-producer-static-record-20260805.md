# NV P3b cooperative FFN W1/W3-to-Q8 producer: CPU/static record

Date: 2026-08-05
Target: Qwen dense Q4/Q4/Q4 FFN on NV sm_120 (`gate/up: 12288 x 4096`, `down: 4096 x 12288`)
Status: **WALL NO-GO; construction, ABI, resource, topology, and correctness pass, but included cost regresses +151.192 us**

## Construction

`extra/llm_research/decode/ffn_q8_cooperative_producer.py` adds a research-only producer. A 256-thread CTA owns 32 consecutive hidden rows: eight lanes reduce one W1/W3 output row, and the 32 lane-zero owners publish their fp16-rounded `silu(gate) * up` values into a 32-half LOCAL array. After the CTA barrier, the first warp emits exactly one Q8_1 packet.

This removes the global 12,288-element activation materialization. Its private shared-Q8 ABI is the already-used consumer layout: 3072 `uint32` int8x4 payload words followed by 384 `uint32` words with `fp16(d) | fp16(raw_input_sum)<<16`. The producer has 384 CTAs and 3,456 output words (13,824 bytes), one packet per CTA.

The inner Q4 work is two runtime LOOP axes (logical Q4 group-word and nibble); it does not expand the eight-group tree into static source. Each of eight row lanes owns two Q4_K blocks and performs the same 4,096 scalar products across both W1 and W3 weights.

## CPU oracle and declared numerical contract

`test/unit/test_ffn_q8_cooperative_producer.py` passes four CPU-only tests
(five pytest cases including the external-oracle availability branch).

1. The required round point is exact in the declared reference: fp32 `silu(gate) * up`, then one fp16 cast before packet quantization.
2. The packet payload and fp16 `d` bits match an independent call to llama's `quantize_row_q8_1_ref` from the pinned `libggml-base.so.0.14.0` on 12,288 rounded inputs. This also caught and fixed the critical distinction that quantization divides by fp32 `d`, while only stored metadata rounds `d` to fp16.
3. `s` is intentionally **not** the CPU model-file reference's `sum(q)*d`. It is the fp16 raw-input sum with CUDA's fixed shfl-down association `16,8,4,2,1`, matching the live CUDA-provider contract already used by the Q4/Q6 shared-Q8 consumers. This difference is declared rather than hidden.

This proves the packet boundary and the one-row direct Q4/Q8 FFN-down CPU oracle. The cooperative W1/W3 reduction uses eight lanes per output row, whereas the landed scalar producer uses 32. Therefore future qualification must explicitly use the approximate/shared-Q8 semantic contract; it cannot inherit the scalar producer's bitwise full-logit certificate.

## Topology gate

The installed chain has at least three physical stages: landed W1/W3 producer, materialized fp16 input boundary, and direct FFN-down consumer. The proposed chain now has two: cooperative Q8 producer and `q4k_q8_ffn_down_4096_12288`. The latter mirrors the established Q4/Q8 consumer's Q4 minimum correction, but uses the producer's `3072` payload / `384` metadata private ABI for `K=12288`. It is research-only and has no route hook. The program chain is therefore structurally 3 -> 2. This record does not claim a wall-time win.

## Offline static resource gate

CUDA 13.2 is present outside the default `PATH`. Both actual emitters were rendered and compiled offline through `CUDARenderer(..., use_nvcc=True)` with `/usr/local/cuda-13.2/bin` prepended, targeting `sm_120`; `cuobjdump --dump-resource-usage` reports:

| program | launch | registers/thread | stack | local | shared | conservative blocks/SM |
|---|---:|---:|---:|---:|---:|---:|
| `ffn_w1w3_q8_cooperative_12288_4096` | `384 x 256` | 48 | 0 | 0 | 1088 B | 5 |
| `q4k_q8_ffn_down_4096_12288` | `2048 x 16` | 48 | 0 | 0 | 0 B | 24 |

The provider's limit is registers: `floor(65536 / (48*256)) = 5`; thread capacity is six CTAs/SM (`1536/256`) and shared memory is nonbinding. The consumer is limited by the 24-block architectural cap. Both are above the required two-block/SM gate, with zero stack/local spill evidence. Rendered source retains `for` loops and the provider's `__syncthreads`; no generated global 12288-element activation store appears.

This cleared the offline static gate and admitted the later, serialized GPU
microgate recorded below. It was not itself wall evidence.

The primitive harness is `extra/llm_research/decode/nv_ffn_q8_cooperative_microgate.py`. Its control explicitly preserves the landed W1/W3 -> fp16 cast/contiguous boundary -> Q4-down chain; its candidate is provider -> Q4/Q8-down. Census proved the common scalar sink aside, control is exactly three physical chain programs and candidate exactly two, with no hidden candidate adapter.

The first correctness run caught a real high-group Q4 header bug before timing: the dynamic decoder advanced through header word 3 at nibble rather than byte stride, masking away the high minimum nibble. That construction produced 14.87% relative-L2 error and was rejected. After correction, offline resources remained 48 registers / zero local and stack, and the repeated GPU correctness gate passed:

- provider packets 0, 191, and 383: payload, `d`, and raw-sum `s` bitwise equal to the independent cooperative CPU oracle;
- consumer rows 0, 1, 2047, and 4095: maximum relative error `3.4024e-6` versus the independent scalar Q4/Q8 oracle;
- candidate versus the fp16 control primitive: relative L2 `0.00768269`, maximum absolute `1.924742`, reported under the declared approximate shared-Q8 contract.

Only then was reverse included-cost timing run (`200` replays, `7` reps):

| arm | median us/replay |
|---|---:|
| control A | 78.470150 |
| candidate B | 229.715160 |
| control C | 78.576500 |
| control midpoint | 78.523325 |

Candidate delta is **+151.191835 us**, approximately `2.93x` the installed chain. This is a decisive wall NO-GO despite the 3 -> 2 topology. The one-block full-model semantic gate was not run because the included-cost primitive gate failed. No default, route, promotion, commit, or push changed.

Reopen only with a distinct cooperative W1/W3 mapping whose included provider cost is no slower than the installed W1/W3 plus fp16 boundary it replaces; another ABI or launch-count spelling is insufficient.

No runtime route, policy, or promotion was changed.
