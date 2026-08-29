# NVIDIA pp512 prefill compiler-native lifecycle ledger

## Decision

**SUBSTRATE PASS; Q4 GATE/UP + K + Q/O LIFECYCLE PASS.** The physical packed-Q4/Q8/IMMA primitive,
compiler-generated packed fragment/correction path, computed-input ownership,
canonical packed-weight ownership, and captured model lifecycle now pass
together. The exact default-off stack covers Q4 gate/up, K, Q, and O with 180
compact-Q8 producers and 180 compiler-owned packed-weight PROGRAMs. It beats
the resident-FP16 control and passes a 20-cycle deep replay audit. It is not
yet a production-generic route: admission is exact Qwen3-8B pp512, V/down are
still outside the winning path, and Q/O requires a graph-specific safe queue
cut. Q6 V/down arithmetic passes, but its model lifecycle loses wall time.

This audit is strictly Qwen3-8B Q4_K_M, pp512/ubatch512, RTX 5090 sm_120. It
does not use decode-token numbers and does not compare AMD and NVIDIA absolute
throughput.

## Wall authority

| arm | synchronized unprofiled wall | prompt rate | status |
|---|---:|---:|---|
| tinygrad corrected FP16, original R7 | 84.025 ms | 6,093 tok/s reported | correctness authority |
| tinygrad current FP16 comparator | 83.793 ms | 6,110 tok/s reciprocal | current planning baseline |
| llama.cpp CUDA MMQ, retained R5 | 36.608 ms | 14,074 tok/s reported | same-session comparator |
| tinygrad packed-v4 research route | 93.200 ms | 5,494 tok/s reciprocal | correct, default-off NO_GO |
| tinygrad compiler-packed gate/up | 74.695 ms | 6,855 tok/s | default-off substrate PASS |
| tinygrad compiler-packed gate/up + K | 70.173 ms | 7,296 tok/s | matched combined control PASS |
| tinygrad compiler-packed gate/up + K + Q/O | 69.378 ms | 7,380 tok/s | default-off combined PASS |
| tinygrad compiler-packed gate/up + K + Q6 V/down | 72.174--72.284 ms | 7,094--7,083 tok/s | correctness PASS, performance FAIL |

The original valid tinygrad-to-llama gap is 47.417 ms. Against the current
83.793-ms comparator it is 47.185 ms. Packed-v4 is 9.407 ms slower than the
current FP16 route and 56.592 ms slower than llama.

The final admitted Q4 stack is 14.125 ms faster than the fresh 83.503-ms FP16
control and 32.770 ms above llama's retained wall. It improves reciprocal
prompt throughput by about 20.4% over that FP16 control. K accounts for about
4.0 ms beyond gate/up in its matched model bracket. Q/O adds a smaller but
real 0.795 ms in the composed graph; its isolated 2.682-ms recovery is not
added because lifecycle interactions absorb most of it.

Profiler totals are accounting evidence only:

| region | tinygrad `PROFILE=1` | llama Nsight GPU busy |
|---|---:|---:|
| dense projections | 85.981 ms, 252 calls | 27.263 ms, 249 MMQs including Q8/fixup |
| Flash Attention | 3.286 ms | 1.714 ms |
| vocabulary head | 2.880 ms | 0.311 ms |
| norms, RoPE, residuals, KV writes, activation | 3.662 ms | 3.808 ms |
| total device entries/busy | 95.810 ms | 33.096 ms |

Llama's 44.840-ms profiled wall contains a profiler-induced host gap and is
not substituted for its 36.608-ms authority. Tinygrad profiling similarly
raises its wall from about 84 to about 96 ms.

## Dense role ledger

The byte columns are algorithmic weight requests per full M=512 projection,
including four 128-token M tiles. They are not resident model size and not
hardware-counter measurements.

| role | calls | `(M,N,K)` | model format | tiny FP16 total | tiny FP16 bytes/call | llama packed bytes/call | packed-v4 coverage |
|---|---:|---|---|---:|---:|---:|---|
| Q | 36 | 512,4096,4096 | Q4_K | about 6.605 ms | 134.2 MB | 37.75 MB | no |
| K | 36 | 512,1024,4096 | Q4_K | about 6.385 ms | 33.55 MB | 9.44 MB | no |
| V | 36 | 512,1024,4096 | Q4_K/Q6_K | about 6.385 ms | 33.55 MB | 9.44/13.76 MB | no |
| O | 36 | 512,4096,4096 | Q4_K | about 6.605 ms | 134.2 MB | 37.75 MB | no |
| gate | 36 | 512,12288,4096 | Q4_K | about 20.510 ms | 402.7 MB | 113.2 MB | yes |
| up | 36 | 512,12288,4096 | Q4_K | about 20.510 ms | 402.7 MB | 113.2 MB | yes |
| down | 36 | 512,4096,12288 | Q4_K/Q6_K | 18.982 ms | 402.7 MB | 113.2/165.2 MB | no |

Tinygrad's FP16 dense total is 85.981 ms for 3.556 TMAC. Llama's exact
population is 214 Q4 MMQs, 35 Q6 MMQs, 249 Q8 conversions, and corresponding
fixups. Their measured totals are:

| llama dense component | calls | GPU busy |
|---|---:|---:|
| Q4 main | 214 | 20.057 ms |
| Q6 main | 35 | 4.253 ms |
| Q4 fixup | 214 | 1.782 ms |
| Q6 fixup | 35 | 0.312 ms |
| Q8_1 producer | 249 | 0.859 ms |
| complete dense lifecycle | | 27.263 ms |

The resident tinygrad FP16 overlay is about 13.89 GB, versus about 4.41 GB for
the mixed packed Q4/Q6 projection population. Llama moves 3.56x fewer Q4
weight bytes and 2.44x fewer Q6 bytes. Gate/up and down service nearly the same
number of bytes per second in the two implementations; llama wins primarily
because its useful weights stay compressed.

The vocabulary head is separate from the 252 full-batch dense calls. Both
implementations request only the final row: `(M,N,K)=(1,151936,4096)`.
Llama's Q6 MMVQ takes 0.311 ms and reads about 510.505 MB of packed weight;
tinygrad's profiled head takes 2.880 ms.

## End-to-end lifecycle

### tinygrad corrected FP16

```text
model load: Q4/Q6 -> persistent FP16 overlay
each projection: FP16 activation + FP16 weight -> HMMA -> FP16 output
all 36 layers: full-batch Q/K/V/O and gate/up/down
after block 35: retain final row -> vocab -> token
```

The persistent expansion is a one-time load cost, but the larger FP16 payload
is streamed again by every projection. K/V additionally underfills the GPU:
32 CTAs on a 170-SM device, versus llama's 170-owner Stream-K topology.

### llama.cpp CUDA MMQ

```text
FP32/FP16 activation -> Q8_1 producer
packed Q4/Q6 global load -> cooperative decode into int8 shared fragments
128N x 128M x 256K, eight warps -> signed-int8 IMMA
direct complete tiles or bounded Stream-K partials -> fixup -> FP32 result
after final attention: gather requested row -> final FFN at M=1
Q6 MMVQ vocabulary -> token
```

The captured large Q4 kernel launches 170 CTAs, one per SM, with 57,856 B
dynamic shared memory plus a 1 KiB runtime reservation. Its shared layout is a
76-dword Q4 row and a 36-dword Q8 row. Llama does not require useful
cross-kernel overlap, `cp.async`, or TMA to reach its number.

### tinygrad packed-v4 research route

```text
FP16 normed activation -> exact compact Q8 producer
raw/native direct-packed Q4 IMMA main
340 bounded FP32 partial slots -> deterministic fixup
native PROGRAM output -> graph consumer
```

Coverage is gate/up only: 72 Q8 producers, 72 mains, and 72 fixups. The v4
profile is:

| component | total | per call |
|---|---:|---:|
| Q8 producer | 0.229536 ms | 3.188 us |
| packed main | 39.842176 ms | 553.364 us |
| fixup | 1.863392 ms | 25.880 us |
| native projection lifecycle | 41.935104 ms | 582.432 us |

The corresponding FP16 gate/up bodies total about 41.075 ms in the matched
profile. Thus the current packed lifecycle is already slightly slower before
charging the remaining graph handoff. Llama's large Q4 projection lifecycle is
roughly 0.18 ms/call, so the problem is not Q8 production; it is main-kernel
service and lifecycle ownership.

### tinygrad compiler-packed Q4 route

```text
FP16 normalized activation -> exact compact Q8 producer
canonical model-owned Q4_K words -> compiler-generated K64 IMMA PROGRAM
direct FP32 output -> ordinary graph consumer
```

The compiler path corrects each K32 signed-int8 dot with tile-local Q4
scale/min and Q8 scale/raw-sum metadata before the outer FP32 K reduction. K64
amortizes barrier/metadata staging without crossing the K128 register-spill
cliff. The retained primitive is 465.025 us versus 463.968 us for v4
main+fixup, a 1.0023x ratio.

At gate/up model scale it has exactly 72 Q8 producers and 72
compiler-generated mains, with zero old fixups, packed transport copies,
expanded gate/up FP16 weights, or partial workspace. All 72 main weight
arguments are the 72 canonical model parameter bases. The fresh R9 wall is
74.695 ms versus 83.503 ms FP16; full
logits are finite, choose token 198 in both arms, and pass the declared
`rtol=0.02, atol=0.5` gate (`max_abs=0.114424`, `mean_abs=0.008292`).

The occupancy-safe K geometry is `(64,32,64)`, 128 threads, 256 CTAs. Its
real-role primitive is about 59 us and the 36-role model population reduces
the matched wall from about 74.4 to 70.4 ms. The composed gate/up+K control
reproduces at 70.173 ms with exact 20-cycle replay.

Q/O uses the same typed Q4 contract but an exact frozen compiler PROGRAM at
the nested model boundary. A fresh graph-derived Flash dependency cut is
required: the admitted cut runs at 69.378 ms and is exact for all 20 replay
cycles across 72 gate/up, 36 K, 72 Q/O record/output pairs, all 36 KV slices,
logits, and token. Default ready placement is slightly faster at 68.965 ms but
fails one of 20 cycles and is rejected. Primary-only and one-queue controls
are exact but regress to about 74.07 ms.

The Q6 typed K16-scale substrate also passes full V and down oracles with
paired masked IMMA. Its model route is not admitted: V-only is a marginal
~0.12-ms signal, down-only regresses by ~1.61 ms, and combined V/down regresses
by 1.84--1.95 ms against the corrected 70.333-ms gate/up+K control.

## What is proven and what is still a wall

The direct packed emitter is **not missing**. The v4 research kernel:

- reads canonical packed Q4_K words directly;
- performs nibble decode, scale/min correction, LDSM, and signed-int8 IMMA;
- contains 256 static IMMA and 32 LDSM sites, uses 255 registers, and has no
  local-memory spill instructions;
- passes all 6,291,456 outputs with max absolute error
  `1.6689300537109375e-6`;
- preserves finite values, complete sentinel coverage, and read-only inputs;
- runs the isolated full chain in 470.112 us minimum / 470.976 us median,
  with a 455.040-us main.

The compiler campaign closes the first five historical walls for the admitted
Q4 gate/up, K, Q, and O roles:

1. `PackedWeightTransform` now owns typed Q4 and compact-Q8 fragments before
   tensor-core range permutation.
2. K32 scale/min correction stays tile-local and reduces in FP32; there is no
   global group-partial tensor.
3. Single-output and nested computed PROGRAM values preserve exact producer
   and canonical model-parameter identities through callification and replay.
4. Packed transport copies fall from 70 to zero.
5. Direct-output compiler kernels remove partial/fixup workspace for all 180
   admitted Q4 projections.

The remaining boundaries are:

1. **Role coverage.** Q4 gate/up, K, Q, and O are covered. V and down remain
   FP16 in the winning graph. Q6 typed arithmetic is solved, but its current
   down lifecycle is slower than FP16.
2. **Kernel service.** The compiler K64 projection is about 465 us while
   llama's corresponding packed Q4 lifecycle is about 0.18 ms. Compiler
   ownership is solved; the remaining service lead requires executed-path and
   issue-schedule evidence rather than static SASS-count guesses.
3. **Binding storage.** Gate/up and K immutable compiler assets are separated
   from per-model/per-capture state, and the admitted combined wrapper gives
   K/Q/O graph-owned lazy buffers. The physical gate/up capture footprint fell
   by about 293 MiB. The standalone Q/O research binding still caches mutable
   record/output pools and a cursor per device; it is harness-only and is not
   concurrency- or multi-model-safe.
4. **Admission.** The model seam is intentionally exact Qwen3-8B pp512 and
   process-start environment selection is authoritative; dynamic same-process
   environment switching is not supported because `getenv` is cached.
   Packed warmstart keys also still scan packed-dtype parameters across the
   whole AST rather than proving operand-scoped A/B ownership. This fails
   closed on collisions but is not production-generic matching.

## AMD-to-NVIDIA transfer map

| AMD packed-WMMA asset | NVIDIA disposition |
|---|---|
| exact route admission and shape guards | reuse the contract; add NVIDIA-qualified rows |
| `PackedWeightTransform` model-buffer identity | reuse |
| movement-only packed carrier | reuse concept and logical ownership |
| postrange packed decode at fragment production | implemented for exact Q4_K/Q8 K64 |
| scheduler-generated `a @ b.T` | reuse the compiler entry point |
| one `Ops.PROGRAM`, ABI output/A/B | reuse ownership principle; NVIDIA may require explicit Q8 metadata and fixup values |
| compiler-known inputs, output, and graph dependencies | implemented through finalized compiler PROGRAM identity |
| canary, binary identity, and fail-closed qualification | reuse |
| AMD wave32 lane map and WMMA instructions | do not reuse; replace with warp32 LDSM/IMMA mapping |
| AMD LDS geometry and frozen 14B shapes | do not copy; qualify NVIDIA shapes independently |

AMD's promoted route decodes packed weights inside the compiler-generated WMMA
projection. The NVIDIA target is the same ownership model, not the same
instruction sequence. Because IMMA consumes int8 activations, its Q8 producer
may remain a separate compiler-owned graph value; it need not be fused if
fusion duplicates quantization across output tiles.

## Recovery ledger

Measured conversion uses the fresh 83.503-ms FP16 control and
`rate = 512 / wall_seconds`. Profile exposures below are not additive wall
bookings.

| lever | recovery interpretation | resulting wall | reciprocal prompt rate |
|---|---|---:|---:|
| compiler-packed gate/up, measured | recover 8.808 ms | 74.695 ms | 6,855 tok/s |
| add compiler-packed K, measured matched bracket | recover about 4.0 ms | 70.4 ms class | about 7,274 tok/s |
| add compiler-packed Q/O, composed measured | recover 0.795 ms | 69.378 ms | 7,380 tok/s |
| Q6 V/down current spelling | regress 1.84--1.95 ms | 72.174--72.284 ms | rejected |
| V remaining packed-service/topology exposure | profile exposure only | not a wall booking | Q4/Q6 lifecycle gate required |
| down remaining packed-service exposure | about 12.1 profiled ms | not a wall booking | Q4/Q6 role gates required |
| llama-bound remaining lifecycle endpoint | recover 32.770 ms | 36.608 ms | 13,986 tok/s reciprocal |
| final-layer requested-row prune | recover 1--2 ms | 68.378--67.378 ms | 7,488--7,599 tok/s |
| attention parity bound | recover at most 1.572 profiled ms | not a wall booking | profile-only bound |
| vocabulary parity bound | recover at most 2.569 profiled ms | not a wall booking | profile-only bound |

The 58.718-ms difference between the two profiled dense regions is exposure,
not a directly bookable 58.718-ms wall recovery. The all-role dense estimate,
final-row pruning, and vocabulary work overlap at their boundaries and must not
be blindly summed.

## Exact next discriminators

Gate A (compiler-generated packed main) and Gate B (captured computed-input
model lifecycle) are complete. The next discriminators are:

1. **Q4 V:** reuse the qualified 256-CTA K geometry on real Q4 V, then prove
   it inside the composed model rather than assuming K's wall transfers.
2. **Q4 down:** qualify `(512,4096,12288)` independently. Q6 down arithmetic
   is exact but its current lifecycle regression shows that primitive success
   is insufficient.
3. **Q6 service scheduling:** explain why the 638.752-us down primitive loses
   1.61 ms at model scale before revisiting it. Do not book V's marginal
   ~0.12-ms signal without a stronger re-bracket.
4. **Queue ownership:** generalize the graph-derived Flash dependency cut so
   admission is tied to exact dependency identity, not a stale graph digest.
5. **Promotion:** only after all admitted rows pass fresh-process full-logit,
   call-census, canonical-buffer, memory, and R9 wall gates. The route remains
   default-off now.

## Evidence and open observations

Primary authorities:

- `nv-llama-prefill-lifecycle-audit.md`;
- `nv-q4-imma-combined-chain-integration-20260828.md`;
- `nv-llama-mmq-warp-ownership-audit.md`;
- `nv-prefill-q4-imma-gate-result-20260828.md`;
- `nv-compiler-packed-fragment-gate-result.md`;
- `nv-compiler-packed-prefill-model-integration.md`;
- `nv-prefill-compiler-q4k-kv-role-result-20260828.md`;
- `nv-compiler-q4k-qo-qualification-result.md`;
- `nv-compiler-q4k-gkqo-combined-result.md`;
- `nv-compiler-q6k-imma-substrate-result.md`;
- `nv-compiler-q6k-model-lifecycle-result.md`;
- retained JSON and cubins under
  `docs/task_workflow/evidence/nv-q4-imma-combined-chain-20260828/`.

Production HCQ kernels are not visible to the existing CUPTI counter route.
Unknown physical counters remain unknown; analytic bytes and retained SASS are
not relabeled as counter measurements. No production route or default was
changed by this audit.
