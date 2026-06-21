# Decode Fusion Build Scope: Attention Reduce/Stat + FFN Activation

Date: 2026-06-20

Executor: Claude

## Objective

Build the two remaining decode lifecycle fusions identified by timed attribution:

1. **Attention reduce/stat fusion** in the flash-decode path.
2. **FFN activation fusion** to remove the standalone `silu(gate) * up` elementwise launch.

This is no longer an exploration scope. The cost splits and cheap candidate A/Bs already ran:

- `docs/decode-attention-cost-split-result-20260620.md`
- `docs/decode-elementwise-cost-split-result-20260620.md`
- `docs/decode-attention-candidate-ab-result-20260620.md`
- `docs/decode-ffn-activation-fusion-result-20260620.md`
- `docs/decode-attention-elementwise-result-20260620.md`

The cheap levers are closed. Proceed only with bounded custom-kernel/lifecycle fusion work.

## Current Evidence

Current decode W/D:

| route | ctx512 | ctx1024 | ctx2048 | ctx4096 | host-sync |
|---|---:|---:|---:|---:|---:|
| baseline | `68.0` | `66.5` | `63.5` | `60.8` tok/s | `0.0%` |
| q8 opt-in | `72.8` | `~71.0` | — | `64.5` tok/s | `0.0%` |
| llama.cpp | `98.6` | `97.6` | `95.4` | `92.2` tok/s | — |

Attention split:

| ctx | attention ms | reduce_fixup | softmax_stats | partial_compute |
|---:|---:|---:|---:|---:|
| 512 | `3.22` | `1.66` | `0.79` | `0.77` |
| 1024 | `3.51` | `1.79` | `0.94` | `0.78` |
| 2048 | `4.17` | `2.08` | `1.26` | `0.83` |
| 4096 | `5.18` | `2.43` | `1.85` | `0.90` |

Elementwise split:

| family | ms/token | note |
|---|---:|---|
| `E_49152_32_3` | `~1.24` | FFN `silu(gate)*up`, launch-bound, flat across ctx |
| residual adds | `~0.32` | smaller |
| rope | `~0.28` | smaller |
| casts/copies | `~0.34` | smaller |

Closed cheap candidates:

| candidate | result |
|---|---|
| `FLASH_L=256/512` | regress; partial parallelism collapses |
| `FLASH_DECODE=0` SDPA | catastrophic |
| remove FFN `.contiguous()` | no target movement; `E_49152` remains |

## Global Measurement Policy

1. Use pinned-clock local attribution/A/B where short-kernel timing is needed:
   - `extra/qk_clock_pin.py`
   - always restore `auto`.
2. Final promotion still needs full W==D decode timing at ctx `512,1024,4096`.
3. Report both:
   - diagnostic pinned-clock local movement;
   - user-realistic final W==D route movement.
4. No decode default changes unless all gates pass and owner explicitly approves.
5. Every candidate must write JSON artifacts under:

```text
bench/qk-decode-fusion-build/
```

6. Every result doc must state:
   - default behavior changed: yes/no;
   - correctness/quality status;
   - exact commands;
   - artifact paths;
   - whether q8 route remains compatible.

## Phase A: Attention Fusion Build

### Goal

Reduce the number and cost of flash-decode reduction / online-softmax-stat kernels in the current `gqa_coop_vec`
path.

Current target cost:

```text
reduce_fixup + softmax_stats = ~2.73 ms/token @ctx1024
reduce_fixup + softmax_stats = ~4.28 ms/token @ctx4096
```

The QK/V partial compute is not the target; it is small and flat (`~0.8-0.9ms`).

### Read First

- `extra/qk_flash_decode.py`
- current flash-decode selection logic in `tinygrad/llm/model.py`
- `docs/decode-attention-cost-split-result-20260620.md`
- `docs/decode-attention-candidate-ab-result-20260620.md`
- any tests around `qk_flash_decode`, especially exactness vs SDPA.

### Target Kernel Families

The cost split names these dominant families:

- reduce/fixup:
  - `r_2_8_128_16_4_2_32n1`
  - `r_1024_16_4_2_32`
  - start_pos-dependent `r_2_...start_pos..._8_4_4_16`
- online-softmax stats:
  - `flash_max`
  - `flash_den`
  - `flash_prob`
  - `flash_gmax`
  - `flash_combine`

### Candidate A1: Fuse Online-Softmax Stats Per Chunk

Build a candidate path that combines the stat chain into fewer kernels.

Current conceptual lifecycle:

```text
partial_compute
-> flash_max
-> flash_den
-> flash_prob
-> flash_gmax
-> flash_combine
-> reduce/fixup rows
```

Desired candidate lifecycle:

```text
partial_compute
-> fused_stat_or_partial_stat
-> fused_combine_or_reduce
```

Target:

- collapse `flash_max/den/prob/gmax/combine` into one or two kernels where possible;
- preserve online-softmax numerics within existing decode policy;
- keep `FLASH_L=128` unless a new fused path proves a different `L` wins.

Suggested implementation files:

- modify or extend `extra/qk_flash_decode.py`;
- add env flag:

```text
FLASH_DECODE_FUSED_STATS=1
```

or equivalent candidate flag, default off.

Suggested artifacts:

- `extra/qk_decode_attention_fused_stats_ab.py`
- `bench/qk-decode-fusion-build/attention_fused_stats_ab.json`
- `docs/decode-attention-fused-stats-result-20260620.md`

Local gate:

- exactness vs current flash-decode or SDPA within existing tolerance;
- reduces `softmax_stats` by `>=0.4ms/token @ctx1024` or `>=0.8ms/token @ctx4096`;
- no increase in `partial_compute` larger than the recovered stat cost;
- no GPU hang/OOM.

Full W==D gate:

- ctx1024 speedup `>=2%` for this subcandidate, or ctx4096 speedup `>=4%`;
- no ctx512 regression `>1%`;
- host-sync remains non-target.

Stop condition:

- If fusing stats increases register/LDS pressure enough that partial or combine cost cancels the win, stop and
  document. Do not continue widening the fused body blindly.

### Candidate A2: Fuse Cross-Chunk Reduce/Fixup

If A1 passes or if A1 shows stats are not independently fusible, target the reduce/fixup rows.

Goal:

- reduce the `r_*` fixup chain;
- fold combine into reduce where possible;
- reduce per-KV-chunk finalization kernels.

Suggested flag:

```text
FLASH_DECODE_FUSED_REDUCE=1
```

Suggested artifacts:

- `extra/qk_decode_attention_fused_reduce_ab.py`
- `bench/qk-decode-fusion-build/attention_fused_reduce_ab.json`
- `docs/decode-attention-fused-reduce-result-20260620.md`

Local gate:

- reduces `reduce_fixup` by `>=0.5ms/token @ctx1024` or `>=1.0ms/token @ctx4096`;
- does not increase `softmax_stats + partial_compute` enough to erase the gain;
- correctness/quality passes.

Full W==D gate:

- ctx4096 speedup `>=5%`;
- ctx1024 speedup `>=3%`;
- no ctx512 regression `>1%`.

### Candidate A3: Combined Attention Fusion

Only after A1/A2 show positive local movement:

```text
FLASH_DECODE_FUSED_STATS=1 FLASH_DECODE_FUSED_REDUCE=1
```

Artifact:

- `bench/qk-decode-fusion-build/attention_fused_combined.json`
- `docs/decode-attention-fused-combined-result-20260620.md`

Gate:

- recovers `>=1.0ms/token @ctx1024` or `>=1.8ms/token @ctx4096`;
- W==D ctx1024 `>=72 tok/s` baseline or `>=76 tok/s` q8-compatible route;
- W==D ctx4096 improves `>=8%`;
- correctness/quality passes.

## Phase B: FFN Activation Fusion Build

### Goal

Eliminate the standalone FFN activation elementwise:

```text
E_49152_32_3 = silu(gate) * up
```

Current target cost:

```text
~1.24 ms/token
~36 launches/token
~33 us/call
```

This is launch-bound, not memory-bandwidth-bound. A faster standalone elementwise kernel is not enough; the launch
must disappear or be absorbed into another existing launch.

### Read First

- `tinygrad/llm/model.py` around the FFN forward:
  - `self.ffn_gate(x).silu().contiguous() * self.ffn_up(x)`
  - `self.ffn_down(...)`
- `extra/q4_k_gemv_primitive.py`
- `extra/q6_k_gemv_primitive.py`
- q8 FFN route files if using q8-compatible fusion.
- `docs/decode-elementwise-cost-split-result-20260620.md`
- `docs/decode-ffn-activation-fusion-result-20260620.md`

### Candidate B1: Fused FFN Activation Producer

Extend the gate/up producer path so it emits the already-activated vector:

```text
act = silu(gate) * up
```

instead of materializing `gate`, `up`, then launching `E_49152`.

Potential sites:

- gate/up epilogue in Q4 custom GEMV path;
- q8 gate/up route if the implementation is q8-only;
- a new narrow custom producer that takes gate/up outputs and writes activation while amortizing launch with existing
  gate/up lifecycle.

Suggested flag:

```text
FFN_ACT_FUSED_PRODUCER=1
```

Artifacts:

- `extra/qk_decode_ffn_activation_producer_fusion_ab.py`
- `bench/qk-decode-fusion-build/ffn_activation_producer_fusion_ab.json`
- `docs/decode-ffn-activation-producer-fusion-result-20260620.md`

Local gate:

- `E_49152_32_3` disappears or shrinks by `>=50%`;
- elementwise recovers `>=0.5ms/token @ctx1024`;
- full logits/greedy or dNLL policy passes;
- q8 compatibility stated.

Full W==D gate:

- ctx1024 speedup `>=3%`;
- no ctx4096 regression;
- no quality regression.

Stop condition:

- If producer fusion requires duplicating gate/up GEMV work or materially worsens GEMV time, stop.

### Candidate B2: Fused `ffn_down` Prologue

Extend `ffn_down` so it consumes gate and up inputs and applies:

```text
silu(gate[i]) * up[i]
```

inside the down GEMV input path, avoiding the realized activation buffer.

Potential site:

- `extra/q6_k_gemv_primitive.py`

Tradeoff:

- may add math in the `ffn_down` inner loop or input load path;
- must avoid recomputing activation per weight if that explodes work.

This candidate is only viable if the implementation computes the activation once per input element or otherwise
amortizes it safely. Do **not** naively recompute `silu(gate[i])*up[i]` for every output weight.

Suggested flag:

```text
FFN_ACT_FUSED_DOWN_PROLOGUE=1
```

Artifacts:

- `extra/qk_decode_ffn_activation_down_prologue_ab.py`
- `bench/qk-decode-fusion-build/ffn_activation_down_prologue_ab.json`
- `docs/decode-ffn-activation-down-prologue-result-20260620.md`

Gate:

- same as B1;
- plus prove no per-weight activation recompute blowup.

### Candidate B3: q8-Compatible Activation Fusion

If B1/B2 naturally fit only the q8 route, measure it separately and label it q8-only.

Gate:

- must beat existing q8 route, not baseline;
- recover `>=0.5ms/token @ctx1024` over q8;
- quality policy must remain within the q8 opt-in policy.

Artifact:

- `bench/qk-decode-fusion-build/ffn_activation_q8_fusion_ab.json`
- `docs/decode-ffn-activation-q8-fusion-result-20260620.md`

## Phase C: Stacked Route

Only after at least one attention candidate and one FFN activation candidate pass local gates.

Measure:

| route |
|---|
| baseline |
| q8 only |
| attention fusion only |
| FFN activation fusion only |
| attention + FFN activation |
| q8 + attention + FFN activation, if compatible |

Create:

- `extra/qk_decode_fusion_stacked_timing.py`
- `bench/qk-decode-fusion-build/stacked_timing.json`
- `docs/decode-fusion-stacked-result-20260620.md`

Required ctx:

- `512,1024,2048,4096`.

Promotion gate:

- ctx1024 reaches `>=80 tok/s`, or recovers `>=2.5ms/token`;
- ctx4096 improves `>=8%`;
- no ctx512 regression `>1%`;
- correctness/quality passes;
- no default change unless owner-approved.

Projected target:

| route | realistic recovery | projected ctx1024 |
|---|---:|---:|
| attention fusion | `~1.3-1.9ms` | `~73-77 tok/s` |
| FFN activation fusion | `~1.0-1.2ms` | `~72-73 tok/s` |
| stacked | `~2.3-3.0ms` | `~83-88 tok/s` |
| stacked + q8 | must measure | possible best route |

## Phase D: Lifecycle Search Encoding

After a candidate passes, encode it as a lifecycle-search template. This is required by the project north star.

Read:

- `docs/project-north-star-llama-and-lifecycle-search-20260620.md`
- `docs/primitive-lifecycle-search-scope-20260619.md`

Create or update:

- candidate schema rows for attention fusion and FFN activation fusion;
- refutation rows for `FLASH_L`, SDPA bypass, and `.contiguous()` removal;
- runner bindings to local A/B and W==D promotion scripts.

Artifacts:

- `bench/qk-lifecycle-search/generated_candidates.json`
- `bench/qk-lifecycle-search/refutations.json`
- or successor v2 lifecycle-search artifacts if already cut over.

Minimum encoding:

```json
{
  "id": "decode_attention_reduce_stat_fusion",
  "phase": "decode",
  "role": "attention",
  "template": "gqa_flash_reduce_stat_fused",
  "target_kernels": ["flash_max", "flash_den", "flash_prob", "flash_gmax", "flash_combine", "r_*fixup"],
  "quality_gate": "exact greedy / dNLL policy",
  "timing_gate": "W==D ctx512/1024/4096",
  "known_refutations": ["FLASH_L_256_512", "FLASH_DECODE_0_SDPA"]
}
```

```json
{
  "id": "decode_ffn_activation_fusion",
  "phase": "decode",
  "role": "ffn",
  "template": "silu_mul_fused_into_producer_or_down",
  "target_kernels": ["E_49152_32_3"],
  "quality_gate": "exact greedy / dNLL policy",
  "timing_gate": "W==D ctx512/1024/4096",
  "known_refutations": ["remove_contiguous_no_target_movement"]
}
```

## Do Not Do

- Do not reopen Q6/MMVQ/GEMV work.
- Do not reopen q8 lifecycle unless the fusion candidate is explicitly q8-only and beats q8 baseline.
- Do not use `FLASH_L` tuning or SDPA bypass again except as refutation rows.
- Do not count pinned-clock local timing as a product benchmark.
- Do not change defaults before W==D + quality + owner policy.
- Do not start `tinygrad-v2` cleanup in the middle of a fragile fusion build unless the owner explicitly switches
  the task to v2 migration.

## Final Report

Create:

- `docs/decode-fusion-build-result-20260620.md`

It must include:

1. Attention fusion candidate results.
2. FFN activation fusion candidate results.
3. W==D full-route timing for any passing candidate.
4. Correctness/quality status.
5. Stacked route timing or reason not stacked.
6. Lifecycle-search encoding status.
7. Exact commands.
8. Artifact paths.
9. Default behavior changed: yes/no.
10. Recommendation: ship, keep opt-in, continue build, or stop.

## Success Definitions

Minimum success:

- one custom fusion candidate built and locally gated, even if it fails, with clear evidence.

Strong success:

- attention or FFN activation fusion recovers `>=0.5ms/token @ctx1024` and gives `>=3%` W==D speedup.

Best success:

- stacked route reaches `>=80 tok/s @ctx1024` with correctness intact and is encoded as a lifecycle-search template.

