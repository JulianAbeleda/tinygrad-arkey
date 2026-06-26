# Decode Attention A3.1 v_dot2 Score Lowering Scope

## Goal

Close the first concrete speed gap between the A2 lifecycle-clean generated attention route and the owned flatline
attention route.

A2 is clean but slow:

| ctx | owned tok/s | A2 whole-cache tok/s | A2 / owned |
|---:|---:|---:|---:|
| 512 | 105.1 | 78.2 | 74.4% |
| 1024 | 103.4 | 75.6 | 73.1% |
| 2048 | 101.0 | 69.7 | 69.0% |
| 4096 | 96.1 | 60.6 | 63.1% |

The owned route already proves the target shape can flatline decode better:

- `owned_flash_tile_gqa_whole`
- `owned_flash_combine`
- whole-cache KV input
- no `E_49152`
- token-correct
- W==D flatline around `105 -> 96 tok/s` from ctx512 to ctx4096

A3.1 asks whether the first missing primitive, `v_dot2`, can be exposed in the generated whole-cache score path
without losing A2 lifecycle cleanliness.

## Current generated score path

A2 program:

- `flash_score_whole_cache_32_128`

Role:

- Computes QK scores from `q:[Hq,Hd]` and whole `cache_kv:[2,1,Hkv,MAXC,Hd]`.
- Emits a scalar generated reduction over `Hd=128`.
- Feeds generated `flash_max`, `flash_prob`, partial, denominator, and combine programs.

Known problem:

- It is generated and lifecycle-clean, but it does not use the owned tile's dot-product primitive shape.

Expected current ISA:

- no `v_dot2`
- no owned tile
- no LDS/cross-lane attention tile primitive

## Why `v_dot2` first

The owned flatline route wins because it is not just "a generated attention shape with better constants." It has
lower-level primitive structure:

- dot-product primitive for QK work
- cross-lane cooperation
- LDS-staged tile layout
- split-KV tile plus combine lifecycle

`v_dot2` is the smallest first slice because it targets the score kernel without requiring a full rewrite of partial,
combine, or TILE+COMBINE lifecycle. If `v_dot2` cannot be emitted in the generated score path, search cannot discover
the flatline shape yet.

## Non-goals

- Do not hand-write a new full attention kernel.
- Do not call inline whole-kernel assembly pure search.
- Do not change the default route.
- Do not promote on isolated speed.
- Do not accept any candidate that reintroduces `E_49152`.
- Do not skip the route-clean/token gate just because ISA improves.

## Files likely involved

Read-first:

- `extra/qk_flash_decode.py`
- `tinygrad/renderer/cstyle.py`
- `tinygrad/renderer/llvmir.py`
- `tinygrad/uop/ops.py`
- `tinygrad/codegen/*`
- `extra/qk_isa_primitive_audit.py`
- `extra/qk_decode_attention_a3_baseline.py`

Likely new files:

- `extra/qk_decode_attention_a3_1_vdot2_probe.py`
- `bench/qk-decode-attention-a3-1-vdot2/latest.json`
- optional result doc: `docs/decode-attention-a3-1-vdot2-score-lowering-result.md`

## Execution plan

### Step 0: Baseline proof

Capture A2 score-kernel ISA/resource attribution.

Required output:

- program name: `flash_score_whole_cache_32_128`
- code object or disassembly path if available
- instruction counts:
  - `v_dot2`
  - `v_fma`
  - fp16/global loads
  - LDS ops
  - cross-lane ops
  - scratch/spill
- W==D rows from A3 baseline

Expected verdict:

- `A3_1_BASELINE_SCORE_NO_VDOT2`

If the tool cannot isolate/disassemble the generated score program, stop and build that capture before changing code.

### Step 1: Minimal renderer/codegen probe

Answer:

```text
Can this fork emit v_dot2 from generated code at all?
```

Minimal acceptable probe:

- small generated UOp kernel
- fp16 pair dot over a known layout
- stable program name, for example `vdot2_probe`
- ISA confirms `v_dot2`
- correctness check against scalar dot

Expected verdicts:

- `A3_1_RENDERER_VDOT2_PROBE_PASS`
- `A3_1_BLOCKED_BY_RENDERER`

If this fails, do not keep modifying attention. The blocker is below attention search.

### Step 2: Generated whole-cache score v_dot2 candidate

Add a default-off candidate:

```text
DECODE_ATTN_SCORE_VDOT2=1
```

Target generated program name:

```text
flash_score_whole_cache_vdot2_32_128
```

Requirements:

- Reads whole `assigned_kv`, not sliced K/V.
- Keeps the same output score buffer contract as `flash_score_whole_cache_32_128`.
- Does not fire owned attention.
- Does not create `E_49152`.
- Tokens match A2/owned baseline.
- ISA confirms `v_dot2`.

Expected verdicts:

- `A3_1_VDOT2_SCORE_ROUTE_CLEAN`
- `A3_1_FAIL__E_49152_REINTRODUCED`
- `A3_1_FAIL__TOKEN_MISMATCH`
- `A3_1_FAIL__OWNED_FLASH_FIRED`
- `A3_1_FAIL__NO_VDOT2_IN_ISA`

### Step 3: W==D transfer test

Compare three arms:

| Arm | Meaning |
|---|---|
| owned | flatline oracle/current shipped route |
| A2 | lifecycle-clean generated baseline |
| A3.1 | A2 + score `v_dot2` |

Context ladder:

- `512`
- `1024`
- `2048`
- `4096`

Required table:

| ctx | owned tok/s | A2 tok/s | A3.1 tok/s | A3.1 vs A2 | A3.1 vs owned |
|---:|---:|---:|---:|---:|---:|

Transfer verdict:

- `A3_1_VDOT2_SCORE_TRANSFERS` if W==D improves beyond spread at at least two ctx points and no lifecycle gate regresses.
- `A3_1_VDOT2_SCORE_NO_TRANSFER` if ISA improves but W==D is flat/noisy.
- `A3_1_VDOT2_SCORE_REGRESSES` if W==D regresses.

## How this maps to the flatline target

The owned route already proves a flatline-ish decode attention path is achievable on this GPU and model shape.

A3.1 does not need to match owned alone. It needs to answer:

```text
Is the dot-product primitive one of the missing generated-code reasons A2 loses?
```

Interpretation:

| Result | Meaning |
|---|---|
| v_dot2 cannot be emitted | The search wall is renderer/codegen exposure. |
| v_dot2 emits but no W==D transfer | The immediate gap is probably cross-lane/LDS/lifecycle, not dot op alone. |
| v_dot2 transfers partially | Continue with cross-lane reduction next. |
| v_dot2 nearly closes gap | Optimize/bundle before LDS. |

## Kill conditions

Stop A3.1 and record the blocker if:

- generated `v_dot2` requires hand-owned whole-kernel assembly
- the candidate needs sliced K/V and reintroduces `E_49152`
- tokens diverge
- route capture cannot distinguish A2 from A3.1
- ISA cannot be attributed to the specific score program
- W==D improves only in isolated timing but not in real decode

## Promotion rule

A3.1 alone is not automatically promotable.

It can only become a promotion candidate if:

- route-clean gate passes
- `v_dot2` is present in generated score ISA
- W==D approaches owned-route threshold across all ctx points
- search manifest records it as generated/search-owned

Otherwise A3.1 is an evidence step for A3.2.

## Expected end states

One of:

- `A3_1_BLOCKED_BY_RENDERER`
- `A3_1_BASELINE_SCORE_NO_VDOT2`
- `A3_1_VDOT2_SCORE_ROUTE_CLEAN`
- `A3_1_VDOT2_SCORE_TRANSFERS`
- `A3_1_VDOT2_SCORE_NO_TRANSFER`
- `A3_1_VDOT2_SCORE_REGRESSES`

## Recommended first implementation

Build the probe artifact first:

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_a3_1_vdot2_probe.py
```

The probe should fail fast with `A3_1_BLOCKED_BY_RENDERER` if generated `v_dot2` is not currently exposable.
