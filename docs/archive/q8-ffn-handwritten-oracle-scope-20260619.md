# q8 FFN handwritten oracle scope (2026-06-19)

Research-only scope for the decode q8/MMVQ reopening. This does **not** route a model path, does **not** change a
default, and does **not** claim the path is buildable in tinygrad today. It asks one narrower question:

> If we bypass tinygrad's current custom-kernel expressibility wall with handwritten kernels, does the q8 activation
> lifecycle for `ffn_gate` + `ffn_up` actually clear the economics and quality gates?

## Why this is not starting from scratch

The existing evidence already fixes the boundaries:

| asset | current finding | reuse here |
|---|---|---|
| `q8-mmvq-lifecycle-deep-result-20260619.md` | producer contract is clean; tinygrad one-kernel producer is not expressible | defines the blocker this oracle bypasses |
| `bench/qk-q8-lifecycle/pack_anatomy.json` | current q8 pack is 29.7us / 4 kernels; <=4.8us needs producer-side fusion | sets the producer cost gate |
| `bench/qk-q8-lifecycle/reuse_map.json` | only FFN norm output has useful Q4_K q8 reuse, exactly gate+up | caps scope to two linears |
| `qk-mmvq-sudot4-full-linear-arc-20260618.md` | int-dot consumer wins at kernel level but loses after separate q8 pack | requires whole-lifecycle measurement |
| `bench/qk-handwritten-mmvq/result.json` | handwritten llama-style consumer was 48.6us / 583 GB/s / 65% peak, speed-only | consumer skeleton and speed target |
| `extra/q8_1_q4k_bench.py`, `extra/qk_layout.py` | q8_1 and Q4_K tinygrad oracle already exists | correctness oracle for real GGUF |
| `extra/q4k_mmvq_handwritten.hip` | standalone HIP skeleton for Q4_K x q8_1 MMVQ | starting point, not final proof |

So the new work is not "invent q8 decode"; it is an oracle experiment that answers whether the deferred codegen
capability would be worth building.

## Mental model

Current tinygrad decode stays byte-identical by using fp activation dequant in the Q4_K coop kernels. llama's fast
MMVQ path instead uses a q8_1 activation lifecycle:

1. Produce normalized fp activation from RMSNorm.
2. Quantize that activation once to q8_1 blocks.
3. Reuse it for `ffn_gate` and `ffn_up`.
4. Consume Q4_K weights with native signed dot4 plus Q4_K affine correction.
5. Reduce one row per cooperative group.

tinygrad's bounded path died at step 1->2 because the needed producer is a single fused kernel with two reduction
granularities: per-row sumsq for RMSNorm, then per-32 max for q8 scales. The handwritten oracle deliberately bypasses
that generator wall. If the handwritten lifecycle fails, the decode q8 line is exhausted. If it passes, the result is
a precise target for either a tinygrad codegen capability or an explicit backend escape hatch.

## Q8H phases

### Q8H-0 — asset and reproduction preflight: PASS

Executed on the current tree with the real 8B GGUF and the existing standalone HIP skeleton. Artifact:
`bench/q8-ffn-handwritten-oracle/preflight.json`.

| check | result |
|---|---:|
| tinygrad q8/Q4_K oracle on `blk.0.ffn_gate.weight` | PASS |
| q8 activation max abs vs dequantized q8 | 0.0167015 |
| Q4_K unpack check | max_abs 0 |
| q8 consumer vs dequantized oracle | max_abs 0.00122976 |
| existing tinygrad q8 graph time | 0.078ms, 363.5 Q4-GB/s |
| handwritten HIP skeleton rebuild | PASS |
| handwritten HIP synthetic speed | 40.4us, 700 GB/s, 78% of 900 GB/s |

The handwritten number is speed-only: random synthetic Q4_K blocks, no real-GGUF correctness yet. The value is still
useful because it confirms the consumer skeleton is worth correctness work; it is not route evidence.

### Q8H-1 — correctness-first handwritten MMVQ on real GGUF: PASS

Build a HIP or HCQ-launched probe that reads the real `blk.0.ffn_gate.weight` bytes from
`/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`, consumes q8_1 activations using the same layout as `q8_1_quantize`, and
compares against `q4_k_reference` / `q8_1_dequantize`.

Required checks:

- Real GGUF `get_scale_min_k4` matches `q4_k_reference`, not just the synthetic struct layout.
- q8 sub-block mapping matches the tinygrad q8 oracle for all 8 subchunks per Q4_K block.
- The Q4_K min/scale affine correction matches the fp reference.
- `ffn_gate` and `ffn_up` both pass; gate-only is insufficient because reuse=2 is the whole premise.

Gate:

- `max_abs <= 2e-2` and no systematic row/subchunk bias.
- If correctness fails, stop. Do not tune speed around a wrong mapping.

Executed as `extra/q8_ffn_handwritten_oracle.py`, with Python doing GGUF parsing, q8_1 generation, and an independent
NumPy Q4_K/q8 reference, then a separate HIP process running the handwritten consumer. This avoids the tinygrad
HCQ-vs-HIP runtime conflict while still checking real GGUF bytes.

Artifact: `bench/q8-ffn-handwritten-oracle/mmvq_correctness.json`.

| tensor | rows | device us | Q4 GB/s | max_abs | max_rel | verdict |
|---|---:|---:|---:|---:|---:|---|
| `blk.0.ffn_gate.weight` | 12288 | 49.83 | 568.1 | 9.54e-7 | 0.00124 | PASS |
| `blk.0.ffn_up.weight` | 12288 | 50.31 | 562.8 | 1.91e-6 | 0.00106 | PASS |

This retires the prior caveat in `bench/qk-handwritten-mmvq/result.json`: the handwritten consumer is no longer
speed-only. Its real-GGUF Q4_K scale/min unpacking and q8 subchunk mapping are correct for the gate/up roles. The
measured speed is lower than the synthetic preflight (about 50us vs 40.4us), but still in the earlier banked range and
fast enough to keep the producer economics as the decisive next question.

### Q8H-2 / Q8H-3 — fused producer oracle design + cost gate: PASS

Write the handwritten decode producer for one row of width 4096:

- Inputs: pre-FFN hidden state and RMSNorm weight, following `ffn_norm(h)` semantics in `tinygrad/llm/model.py`.
- Outputs:
  - fp normalized activation for the existing byte-identical path / downstream comparisons,
  - q8_1 quantized int8 payload,
  - q8_1 per-32 scales.
- Kernel shape: one workgroup per decode row, LDS staging for the per-row sumsq, barrier, then per-32 max/quantize.
- No extra dense fallback and no separate q8 max/pack kernels.

This is the exact capability tinygrad Q8L-2 could not express. The oracle keeps it handwritten so we measure the
economics before funding a renderer/custom-kernel feature.

Executed as `extra/q8_ffn_producer_oracle.py`: one HIP kernel for RMSNorm-only and one HIP kernel for
RMSNorm + q8 side-channel, both on a 4096-wide decode row. Artifact:
`bench/q8-ffn-handwritten-oracle/producer_cost.json`.

| producer | device us | fp max_abs | q8 dequant max_abs |
|---|---:|---:|---:|
| RMSNorm only | 6.58 | 2.38e-7 | n/a |
| RMSNorm + q8 side-channel | 7.50 | 2.38e-7 | 0.00522 |
| incremental q8 side-channel | **0.92** | n/a | n/a |

**Gate: PASS.** The strong target was incremental side-channel cost <= 4.8us; the handwritten producer measured
0.92us. This proves the q8 activation tax is not fundamental. The blocker is specifically that tinygrad cannot yet
generate this fused per-row reduce -> per-32 quantize -> multi-output producer.

### Q8H-4 — paired gate+up lifecycle benchmark: PASS

Measure the actual lifecycle that can win:

`ffn_norm producer side-channel -> q8_1 activation -> handwritten q8 MMVQ gate + handwritten q8 MMVQ up`

Compare against current fp coop gate+up, including all producer/pack cost. Do not report consumer-only speed as the
decision number.

Gates:

- Isolated paired gate+up >= 1.15x over current fp coop.
- No layout copies.
- Correctness remains within Q8H-1 tolerance.

Artifact:

- `bench/q8-ffn-handwritten-oracle/gate_up_lifecycle.json`.

Result is component-summed device time from Q8H-1 real-GGUF consumers plus Q8H-3 producer. It is not a model route and
does not include host overhead, but it is the right isolated lifecycle unit.

| component | device us |
|---|---:|
| fused RMSNorm/q8 producer | 7.50 |
| `ffn_gate` handwritten q8 MMVQ | 49.83 |
| `ffn_up` handwritten q8 MMVQ | 50.31 |
| **q8 gate+up lifecycle** | **107.64** |

Baselines:

| baseline | gate+up us | speedup from q8 lifecycle |
|---|---:|---:|
| current fp coop | 132.20 | **1.23x** |
| base fp | 154.80 | 1.44x |
| old separate-pack sudot4 path | 139.70 | 1.30x |

**Gate: PASS.** The isolated pair target was >=1.15x over current fp coop; measured component-summed speedup is
1.23x.

### Q8H-5 — one-block FFN EV oracle: PASS

Run one full FFN block path:

`ffn_norm -> gate/up -> silu(gate) * up -> ffn_down`

Use the q8 oracle only for gate/up. Keep `ffn_down` on the current shipped path unless explicitly measuring a separate
down path. This phase answers whether the isolated win survives the real block mix.

Gates:

- Implied W==D decode EV >= 3%.
- No extra host synchronization beyond the oracle harness boundary.
- No mutation of model defaults.

Artifact:

- `bench/q8-ffn-handwritten-oracle/block_oracle.json`.

Implemented as an Amdahl one-block/decode EV model before routing: use Q8H-4 measured gate+up lifecycle speedup
(1.23x) and the banked current decode-time share for Q4_K `ffn_gate/up` (~25.7% from
`qk-decode-per-role-delta-audit-20260618.md`).

| input | value |
|---|---:|
| measured gate+up lifecycle speedup vs fp coop | 1.228x |
| current decode-time share of Q4_K gate/up | 25.7% |
| predicted decode speedup | **1.050x** |
| predicted latency reduction | 4.77% |

**Gate: PASS_TO_Q8H6.** The Q8H-5 threshold was >=3% implied W==D decode EV; the measured-component model predicts
about +5.0%. This is still not route evidence: it is a build-worthiness gate for dNLL and an in-model research flag.

### Q8H-6 — dNLL and W==D route gate

Only if Q8H-1 through Q8H-5 pass:

- Run dNLL with the same dataset/gate used by the prior lossy quant work.
- Run W==D decode sweeps at the current banked contexts.
- Keep the route behind a research flag.

Gates:

- dNLL <= 0.01.
- W==D speedup >= 3% sustained.
- No regression in byte-identical default path.

Artifacts:

- `bench/q8-ffn-handwritten-oracle/dnll.json`.
- `bench/q8-ffn-handwritten-oracle/decode_wd.json`.

Quality proxy status: `bench/q8-ffn-handwritten-oracle/quality_proxy.json` passes the first q8-lossiness gate before
route build: teacher-forced 160-token dNLL = **+0.00165** when `ffn_gate/up` consume
`q8_1_dequantize(q8_1_quantize(ffn_norm_output))` through the normal graph. This does not measure route speed, but it
removes the main quality objection to building the handwritten HCQ route.

HCQ artifact status: `extra/q8_ffn_hcq_artifact.py` passes the first route-build gate before model integration. The
probe compiles the fused RMSNorm/q8 producer and the Q4_K x q8_1 consumer as tinygrad AMD code objects, launches them
through HCQ on tinygrad-owned buffers, and does not call the HIP runtime in-process.

Artifacts:

- `bench/q8-ffn-handwritten-oracle/hcq_artifact.json`
- `bench/q8-ffn-handwritten-oracle/hcq_artifact_up.json`

| tensor | producer fp max_abs | consumer max_abs | verdict |
|---|---:|---:|---|
| `blk.0.ffn_gate.weight` | 4.77e-7 | 7.15e-7 | PASS |
| `blk.0.ffn_up.weight` | 4.77e-7 | 1.43e-6 | PASS |

This is still not a W==D model route. It proves the backend artifact route is mechanically viable: the same
producer-side q8 activation buffer can be written by one HCQ kernel and consumed by the gate/up MMVQ kernels without
HIP, copies, or descriptor extraction.

One-block eager route status: `extra/q8_ffn_oneblock_route.py` routes block 0 behind the same handwritten HCQ artifacts
and feeds the result into the existing `ffn_down`. The route is numerically correct but fails the eager performance
gate:

| check | result |
|---|---:|
| route vs graph q8 proxy max_abs | 0.00137 |
| route vs graph q8 proxy mean_abs | 4.44e-5 |
| producer + gate + up HCQ lifecycle | 195.0us |
| performance gate | FAIL |

The failure is not a model-quality objection; it is an artifact/compiler-quality objection. The HIP oracle still has
the right economics, but the COMGR-compiled HCQ artifact is too slow. A hipcc-compiled object currently fails tinygrad
AMD loading with `unknown AMD reloc 10`, so A3 graph capture is blocked until the fast code object can be loaded or
the same schedule is transferred into a tinygrad-owned raw-code path.

## Stop rules

Stop immediately if any of these occur:

- Q8H-1 real-GGUF correctness fails.
- Producer side-channel cannot stay single-kernel or <= the lifecycle cost gate.
- Paired gate+up fails 1.15x over fp coop after producer cost is included.
- One-block EV drops below 3%.
- dNLL fails.

No phase may use a consumer-only number to justify routing. The q8 path lives or dies as a producer+consumer lifecycle.

## Expected outcomes

Likely outcomes, in order:

1. **Correctness passes, producer cost fails:** confirms the existing deferred verdict; no more decode work unless a
   producer codegen capability is funded for other reasons.
2. **Producer and gate+up pass, dNLL/block fail:** q8 is a real kernel trick but not a model route.
3. **All gates pass:** this becomes a codegen-transfer target. The strategic choice is then either a research-only
   handwritten backend escape hatch or teaching tinygrad the fused producer capability.

The upside is bounded: gate+up are only two Q4_K linears in the block, so even a clean pass is expected around low
single-digit decode EV. The value of this arc is not raw tok/s alone; it tells us whether llama's activation lifecycle
primitive is worth making tinygrad generate.

## Claude execution prompt

Use this exact scope. Start at Q8H-1, not Q8H-3. Reuse `extra/q8_1_q4k_bench.py`, `extra/qk_layout.py`, and
`extra/q4k_mmvq_handwritten.hip`. First make the handwritten Q4_K x q8_1 consumer correct on real GGUF for
`blk.0.ffn_gate.weight` and `blk.0.ffn_up.weight`; only then build the fused RMSNorm/q8 producer. Do not route a model
path, do not change defaults, and do not claim success from consumer-only speed. Commit artifacts under
`bench/q8-ffn-handwritten-oracle/` and stop at the first failed gate.
