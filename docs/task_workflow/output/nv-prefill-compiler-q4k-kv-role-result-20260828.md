# NVIDIA pp512 compiler-native packed K/V role gate

## Verdict

**PASS for Q4_K K, including a default-off whole-model integration.  Primitive
PASS only for the Q4_K half of V.  SUBSTRATE WALL for Q6_K V.**

The ordinary compiler-owned packed matmul can cover `(M,N,K) =
(512,1024,4096)` without a Stream-K sidecar.  The qualified geometry is a
`64x32x64` tile, four warps, and 256 CTAs.  It reads the canonical GGUF Q4_K
buffer directly, reads one compact Q8_1 record, performs signed-int8 IMMA, and
publishes one normal scheduler-owned FP32 output.  It creates no expanded
weight allocation and no partial/fixup buffer.

The old 128x128 spelling is retained only as a control.  Its 32 CTAs are
explicitly rejected as a production claim.  The intermediate 64x64 spelling
is also rejected by this strict gate: 128 CTAs still do not cover all 170 SMs.

A separate default-off K-only binding and narrow model hook were added after
the primitive passed.  Q4 V and Q6 V remain outside that route.  No core
compiler, scheduler, function, rangeify, or existing gate/up binding file was
changed by the K integration.

## Full-output and ownership gate

Fixture: real `blk.0.attn_k.weight`, all 524,288 outputs, deterministic legal
Q8_1 values/scales/raw sums, and the qualified static Q4_K/Q8_1 body as the
oracle.

| check | result |
| --- | ---: |
| finite outputs | 524,288 / 524,288 |
| nonzero outputs | 524,288 / 524,288 |
| NaN sentinel outputs left unwritten | 0 |
| maximum absolute difference | 0.000305176 |
| mean absolute difference | 0.000010437 |
| `allclose(rtol=2e-5, atol=2e-3)` | pass |
| ordinary Tensor matmul output equals direct compiler-binary output | exact |
| packed weight unchanged | pass |
| compact activation record unchanged | pass |
| exact candidate identity and K64 contract | pass |

The separate typed raw-fragment unit suite passes 7/7 and covers exact Q4
nibble selection across K32 boundaries.  The small rounding difference above
is the qualified FP32 accumulator association difference against the static
full-projection oracle; it is not a raw-fragment mismatch.

## Geometry sweep

Cold R9 values below are direct CUDA-event durations from the exact binary
emitted by tinygrad.  Every row passes the same full-output oracle, but only
grids with at least 170 CTAs are admitted as occupancy-safe.

| tile / warps | CTAs | min | median | verdict |
| --- | ---: | ---: | ---: | --- |
| 128x128 / 8 | 32 | 115.616 us | 115.744 us | reject: severe underfill |
| 64x64 / 2 | 128 | 73.472 us | 73.504 us | reject: below SM count |
| 64x64 / 4 | 128 | 61.504 us | 61.632 us | reject: below SM count |
| 32x64 / 2 | 256 | 74.080 us | 74.560 us | pass |
| 64x32 / 2 | 256 | 65.024 us | 65.280 us | pass |
| **64x32 / 4** | **256** | **59.328 us** | **59.584 us** | **winner** |
| 32x32 / 1 | 512 | 71.744 us | 72.864 us | pass |

The fresh hot confirmation gives 58.944 us minimum and 59.232 us median for
the winner.  The first samples are 59.968 us cold and 59.744 us hot, so this is
not a minimum-only clock artifact.

Winner SASS/resource facts:

- 8 static `IMMA.16832.S8.S8` sites;
- 2 static barriers;
- 96 registers/thread;
- 8,704 bytes shared memory/CTA;
- zero stack, local memory, LDL, or STL;
- grid `(32,8,1)`, block `(32,2,2)`.

An actual Q4_K V tensor, `blk.4.attn_v.weight`, independently passes the same
full-output, sentinel, finite, readonly, identity, and SASS gates at 59.264 us
minimum / 59.456 us median.  Q4 K and Q4 V therefore share one qualified
compiler geometry; this is not an extrapolation from the gate/up shape.

## Dependency-ordered population proxy

The population proxy loads all 36 real K buffers and captures exactly 36
compiler-owned calls.  Replay is exact, sampled real weights produce distinct
outputs, all samples are finite, and weight/activation inputs remain readonly.

An independent-output graph is allowed to overlap calls and is not an honest
layer-order model.  The authority below therefore replays the exact compiler
binary 36 times in dependency order on one queue:

| population | minimum | median | per call at minimum |
| --- | ---: | ---: | ---: |
| compiler Q4_K K, 36 calls | 2.580838 ms | 2.592229 ms | 71.690 us |
| retained FP16 K population | 6.3845 ms | profiled authority | 177.3 us |
| isolated role recovery | **3.803662 ms** | estimate boundary | — |

The compact Q8 producer is not included in this proxy.  Charging approximately
0.115 ms from the retained producer rate leaves about 3.69 ms of estimated
whole-prefill recovery.  Applied to the 83.793 ms authority only as arithmetic,
not as a booking, that is approximately:

```text
83.793 ms -> 80.104 ms
6110 prompt tok/s -> 6392 prompt tok/s
```

That estimate is superseded by the measured whole-model bracket below.

## Default-off whole-model K integration

The K route is independently leased by
`NV_COMPILER_Q4_IMMA_K_PP512=1` and requires the existing compiler gate/up arm.
It admits only exact Qwen3-8B Q4_K `attn_k` projections at
`(512,1024,4096)`.  Q and V retain their control paths.

Fresh R9, same process protocol and current gate/up compiler baseline:

| arm | minimum | median | prompt throughput |
| --- | ---: | ---: | ---: |
| gate/up compiler control | 74.395289 ms | 74.623558 ms | 6,882.16 tok/s |
| gate/up + compiler K | 70.390585 ms | 70.594409 ms | 7,273.70 tok/s |
| measured recovery | **4.004704 ms** | **4.029149 ms** | **+391.54 tok/s (+5.69%)** |

The candidate graph contains exactly 36 K compact-Q8 producers and 36 K
compiler mains, in addition to the 72 gate/up pairs.  All 36 K weight
arguments are distinct canonical model buffers.  There are zero K FP16
overlays, weight copies, fixups, or partial workspaces, and all K launches use
the qualified 256-CTA geometry.  A/B/A replay is bit-exact and activation
sensitive.  Candidate versus control full logits are finite and pass
`allclose(rtol=0.02, atol=0.5)` with maximum absolute difference 0.0865281,
mean absolute difference 0.0195507, and the same greedy token (198).

The first integration spelling routed an opaque preallocated main output
through the nested attention `FUNCTION`.  It satisfied the structural census
but failed exact A/B/A replay (maximum difference 0.100314) and regressed to
75.51196 ms.  That was an ownership defect, not a primitive limit.  Keeping
the compact-Q8 sidecar but expressing the packed main as its ordinary
compiler-owned matmul lets the scheduler own the output lifetime.  This is the
spelling that passes replay and books the 4.00 ms recovery.

## Q6_K V: exact missing compiler contract

The checkpoint contains 36 K tensors, all Q4_K.  V contains 18 Q4_K and 18
Q6_K tensors.  `PackedWeightTransform` understands scalar Q6_K dequantization,
but there is no `Q6KInt8FragmentProvider` and no Q6/Q8 group-accumulator ABI.
The Q4 accumulator must not be reused: Q4 metadata owns K32 groups, whereas
Q6_K has an independently signed scale every K16.

The smallest admissible Q6 substrate is:

```text
Q6KInt8FragmentProvider
  abi: q6_k.logical_nk_to_s8_k16.v1
  storage: canonical uint16[105] per K256 block (210 bytes)
  input coordinates: logical row, K16-aligned k_base
  output: signed char.vec(16), values reconstructed from low4 | high2, then -32
  metadata: FP16 block D, signed-int8 K16 scale, logical subgroup id

Q6KQ8SubgroupAccumulatorContract
  abi: q6_k_q8_1.k16_fp32_accumulator.v1
  owns: one K16 integer dot before outer-K reduction
  contribution: rounded(D * q6_scale) * q8_scale * int_dot
  boundary: the two K16 halves of one IMMA K32 must remain separately scalable
```

The last boundary is the real implementation wall.  NVIDIA's available signed
int8 tensor-core instruction consumes K32, but Q6 requires two differently
scaled K16 subtotals.  A correct implementation must either emit paired
masked/partitioned IMMA operations or add a specialized tensor-core result
contract that exposes both K16 subtotals before correction.  Summing the K32
integer dot first and applying one scale is mathematically invalid.

Q6 promotion gates must include all 16 subgroups of a K256 block, K16/K32 and
block-boundary fixtures, one actual Q6 V tensor, the full 524,288-output oracle,
finite/sentinel/readonly checks, SASS/resources, and the same >=170-CTA rule.
Until that typed contract exists, the Q6 half of V books zero recovery.

## Evidence and reproduction

- Harness: `extra/llm_research/prefill/nv_compiler_q4k_k_role_gate.py`
- Population proxy: `extra/llm_research/prefill/nv_compiler_q4k_k_population_proxy.py`
- Cold R9: `docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/strict-cold-r9.json`
- Hot R9: `docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/strict-hot-r9.json`
- Real Q4 V: `docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/q4-v-strict-r9.json`
- Ordered population: `docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/population-strict-r9.json`
- Whole-model K candidate: `docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/model-k-candidate-r9-v3.json`
- Matched gate/up control: `docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/model-k-control-r9-v3.json`
- Full-logit comparison: `docs/task_workflow/evidence/nv-prefill-compiler-q4k-k-role-20260828/model-k-compare-v3.json`
- Generated CUDA, cubins, and SASS are retained beside those JSON files.

One broader precontract test is currently red in the shared dirty tree:
`test_precontract_int8_lds_contract.py` expects 16 vector LDS stores but the
concurrent `kernel_lds.py` work emits 256 scalar stores.  The focused NVIDIA
typed-fragment tests pass 7/7 and every executed K/V binary above passes, but
the shared generic change must reconcile that unit test before promotion.
