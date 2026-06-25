# Decode Attention Pure Search Scope

## Goal

Replace the owned AMDGCN decode-attention route with a generated/search-owned route selected by BubbleBeam, without losing the current lifecycle guarantees.

Current truth:

```text
DECODE_ATTN_AMDGCN_TILE=1
  -> owned_flash_tile_gqa_whole
  -> owned_flash_combine
  -> no E_49152 with KV identity
  -> token-correct and W==D-promoted
  -> not pure machine search
```

Target truth:

```text
BubbleBeam / search-space candidate
  -> generated split-KV attention tile and generated combine lifecycle
  -> no owned_flash_* programs fire
  -> no E_49152 materialization
  -> tokens match and W==D meets the owned-route threshold
  -> DECODE_ATTENTION_PURE_SEARCH_GENERATED
```

## Current Blocker

The current search-space manifest classifies decode attention as `SEARCH_BLOCKED_BY_CODEGEN` because the owned winner uses primitives that are not yet generated/search-owned:

- `v_dot2` / equivalent fp16 dot lowering.
- Cross-lane reduction for softmax/PV lifecycle.
- LDS-staged tile layout and split-KV scheduling.
- TILE + COMBINE lifecycle represented as one candidate, even if it remains two programs.

GEMV is no longer the blocker. Tracked Q4_K decode GEMV is pure/generated under BubbleBeam; decode attention is the next pure-machine-search target.

## Execution Sequence

### A0: Baseline purity capture

Tool: `extra/qk_decode_attention_purity_capture.py`.

Gate:

- `owned_flash_tile_gqa_whole` fires.
- `owned_flash_combine` fires.
- `E_49152` is absent.
- Buffer identity is preserved.
- Verdict is `DECODE_ATTENTION_NOT_PURE__OWNED_TILE_COMBINE`.

### A1: Generated skeleton candidate

Goal: add a default-off generated candidate that is route-attributed separately from the owned tile and combine.

Gate:

- Candidate route can be captured without owned `owned_flash_*` names.
- If it is correct but slow, classify as `SEARCH_GENERATED_WD_FAIL`.
- If it falls back to owned tile/combine, classify as not pure.

### A2: Primitive lowering increments

Add missing generated primitives one at a time:

- `v_dot2` lowering or an equivalent generated dot primitive.
- Cross-lane reduction lowering.
- LDS-staged tile layout/search knobs.
- Generated combine or explicit TILE+COMBINE lifecycle candidate.

Each increment must have a structural artifact before W==D speed claims.

### A3: W==D promotion gate

Run ctx 512/1024/2048/4096.

Required pass conditions:

- Tokens match.
- Generated route fires instead of owned AMDGCN tile/combine.
- No `E_49152` materialization.
- Unknown-bucket attribution remains closed.
- Throughput meets the selected owned-route threshold.

### A4: BubbleBeam binding

Only after A3 passes:

- Bind BubbleBeam to the generated attention candidate.
- Keep owned AMDGCN tile/combine as explicit fallback/reference.
- Update the search-space manifest, handoff, and final pure-search status docs.

## Non-Goals

- Do not change attention defaults before A3 passes.
- Do not chase cheaper combine alone unless the full lifecycle W==D gate says it transfers.
- Do not claim pure search from a hand-owned HIP code object.
- Do not regress KV identity or reintroduce `E_49152`.
