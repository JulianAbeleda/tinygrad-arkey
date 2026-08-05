# NV Q4_K dynamic four-warp ownership static record — 2026-08-05

## Result (chronological; final corrected verdict below)

The first fixed-bound construction proved the ownership claim but failed the
pre-GPU PTX gate. A runtime-bound construction then passed static resources,
and its corrected byte-stride retry passed an included native gate by
**56.957153 us/replay**. No token-wall credit is booked and no route was
changed; the final authority is the corrected-construction section below.

`extra/llm_research/decode/q4k_warp_cooperative_dynamic.py` is research-only. It uses one flat `LOCAL=128` launch (four warps per output row), Q8_1 packed activation ABI, and `int8x4_dot`/CUDA DP4A. For each Q4_K block a lane owns precisely two logical four-value words:

```
warp = lid // 32
lane = lid % 32
group = lane // 4
word = (lane % 4)*2 + {0, 1}
block = warp*4 + {0, 1, 2, 3}
```

The pure ownership enumeration has 1024 unique `(warp,lane,block,group,word)` records, and within every block it covers the exact `8 groups * 8 logical words` domain once. It is therefore not the rejected control-masked witness.

## sm_120 static gate

NVRTC PTX on the fixed Qwen production Q4 shape `(rows=4096, K=4096)` reports:

| body | local | `ld.global` / thread | body-wide static issue proxy | `dp4a` | PTX b32 registers | local spill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| installed Q4 G3 fp16 | 32 | 40 | 1280 | 0 | prior audit baseline | none observed |
| dynamic Q4/Q8 four-warp candidate | 128 | 36 | 4608 | 16 | 286 | none observed |

The candidate contains DP4A and uses dynamic group/word addresses; the failure is not the old eight-arm control masking. The compiler unrolled the four compile-time Q4_K blocks despite the author-level loop, leaving four copies of the packed fragment. Its body-wide issue proxy is `36*128 / (40*32) = 3.6x` the installed body, and 286 virtual b32 registers is pathological for a bandwidth GEMV even before ptxas allocation.

## Verdict

**STATIC NO-GO.** The four-warp/Q8 ownership idea is now represented faithfully enough to reject this construction, but it is not GPU-authorized. The included-cost microgate remains intentionally unrun.

The next distinct reopening condition is not another source-level loop spelling: it must make the per-warp four-Q4_K-block traversal survive lowering as one compact body (or use an explicit tile/loop primitive with measured PTX) while retaining `LOCAL=128`, DP4A, exact two-logical-word ownership, and a body-wide global-load proxy no greater than 1280. Without that, a hardware run would measure compiler unrolling and register pressure rather than the intended mapping.

## Compiler-substrate reopen: runtime scalar bound

The generic UOp/NV renderer **does** support a non-unrolled loop contract. A separate research emitter passes `UOp.variable("q4k_coop_blocks", 1, 4)` as the `AxisType.LOOP` range bound. The unbound `DEFINE_VAR` renders as one scalar kernel parameter, and NVRTC emits one loop body rather than four cloned bodies; no CUDA-string pragma or renderer change was used. Static result:

| extent form | scalar vars | `ld.global` / thread | `dp4a` | PTX b32 registers | local spill |
| --- | ---: | ---: | ---: | ---: | ---: |
| compile-time `4` | 0 | 36 | 16 | 286 | none |
| runtime `q4k_coop_blocks in [1,4]` | 1 | 9 | 12 | 102 | none |

This settles the capability question: the IR can express runtime scalar loop bounds and the NV renderer preserves them. The PTX virtual register namespace is intentionally not treated as physical allocation; the next section records the final ptxas resource result.

## Final physical resource audit and admission

The runtime-bound PTX was assembled offline with CUDA 13.2:

```
/usr/local/cuda-13.2/bin/ptxas -arch=sm_120a -v \
  -o /tmp/q4k_coop_runtime_resource.cubin /tmp/q4k_coop_runtime_resource.ptx
/usr/local/cuda-13.2/bin/cuobjdump --dump-resource-usage /tmp/q4k_coop_runtime_resource.cubin
```

`ptxas` reports **36 physical registers/thread**, `0 bytes stack frame`, `0 bytes spill stores`, and `0 bytes spill loads`. `cuobjdump` independently reports `REG:36 STACK:0 SHARED:0 LOCAL:0`. The earlier `%r<102>` is NVRTC virtual naming, not the allocated resource count.

Read-only CUDA device attributes for this RTX 5090 are warp size 32, max threads/SM 1536, registers/SM 65536, shared memory/SM 102400 bytes, and max blocks/SM 24. At `LOCAL=128` (4 warps), the hard-resource occupancy bounds are:

```
threads: floor(1536 / 128) = 12 blocks/SM
registers: floor(65536 / (36*128)) = 14 blocks/SM before allocation rounding
shared: unbounded for 0 B/block
block limit: 24 blocks/SM
```

Thus the conservative theoretical residency is **12 active blocks/SM = 48 active warps/SM**, comfortably above the required two-block gate. The runtime-bound body preserves 9 `ld.global`/thread, DP4A, and zero spills.

**Resource gate PASS; GPU microgate admitted.** This changes no route and makes no speed claim. The microgate must bind `q4k_coop_blocks=4`, verify the Q8 approximation contract against the existing Q8 consumer, and use an included-cost A/B/A timing bracket only after the shared GPU lock is released.

## One authorized included-cost microgate

The native-NV run acquired `/tmp/gpu-bench.lock` and stopped at the required correctness-first gate. The installed control matched its independent CPU fp16 reference (`4.7684e-06` max absolute error, tolerance `0.0981481`). The cooperative candidate missed the independent CPU Q8 reference (`0.9392953`, tolerance `0.0980610`).

Structural census was one installed Q4 program/control replay versus three included-cost candidate programs: the llama-Q8 provider, the four-warp Q4 partial kernel, and the four-partial sum. Because numerical Gate 1 failed, reverse A/B/A was not run; the artifact contains empty timing arms and books zero recovery.

The failure is understood rather than attributed to hardware: the dynamic header path extracted `w3` at a four-bit stride, but Q4_K groups 4..7 own one full header byte each. That made the high-group minimum nibble zero. The research emitter now uses byte-stride extraction, but that correction has only static/CPU reasoning behind it and was deliberately not given a second hardware attempt under this one-run authorization.

An offline post-fix resource rerun shows the corrected body remains admissible: 9 `ld.global`/thread, 12 PTX DP4A occurrences, 38 physical registers/thread, zero stack/spills/shared/local, and the same thread-limited 12-block/SM theoretical residency. This does not repair or replace the failed live numerical record.

**Live verdict: GATE1_NUMERIC_FAIL / NO-GO, zero credit, no integration.** Reopening requires a fresh explicit numerical authorization for the corrected byte-stride body, followed by the same correctness-first included-cost protocol. The physical resource PASS remains a property of the runtime-bound construction; it is not a wall-performance result.

## Corrected construction retry

The corrected byte-stride body received one fresh authorization after an exhaustive CPU packed-layout oracle covered all 16 Q4_K blocks, groups 0..7, logical words 0..7, and four nibbles/word. Every dynamic scale, minimum, and Q nibble matched the independent raw-byte layout.

The exact `(rows=4096, K=4096)` included-cost retry passed numerics:

| check | observed | tolerance |
| --- | ---: | ---: |
| installed control vs CPU fp16 reference, max abs | 0.0000047684 | 0.0981481 |
| cooperative candidate vs CPU Q8 reference, max abs | 0.00255585 | 0.0980610 |
| candidate Q8 vs control fp16, relative L2 | 0.00479908 | characterized, not a promotion contract |

The timed candidate includes all three programs: `q8_1_llama_provider_4096`, `q4k_warp_coop_q8_dp4a_partial_4096_4096`, and the `4096x4` partial sum. Reverse A/B/A, 200 replays and seven samples/arm:

| arm | median us/replay |
| --- | ---: |
| control A | 124.463500 |
| candidate B | 67.669830 |
| control C | 124.790465 |
| control A/C midpoint | 124.626983 |

The included-cost delta is **-56.957153 us/replay (-45.70%)**. This is a clear isolated Gate-1 win and reopens bounded model integration. It is not yet booked as token-wall recovery: the candidate changes the activation representation to Q8_1, adds two programs relative to the installed Q4 primitive, and still needs a real-role/full-logit semantic gate plus composed wall timing. Default routes remain unchanged.

**Final isolated verdict: PASS / INTEGRATION-ADMISSIBLE, zero token-wall credit pending composition.**
