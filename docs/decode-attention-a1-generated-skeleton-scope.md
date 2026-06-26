# Decode Attention A1 Generated Skeleton Scope

## Goal

Build the first generated/search-owned decode-attention skeleton candidate after GEMV purity completion.

The A1 target is not speed. The target is route ownership and attribution:

- BubbleBeam can select a generated decode-attention candidate.
- The generated candidate appears in graph/program capture with a stable name.
- The owned AMDGCN tile route does not fire for the candidate.
- The route preserves KV buffer identity.
- The route does not reintroduce `E_49152`.
- Token correctness is checked on a small decode sample.

If A1 passes, later phases can add the missing performance primitives: `v_dot2`, cross-lane reduction, LDS tile
layout, and TILE+COMBINE lifecycle controls.

## Current baseline

A0 capture tool:

- `extra/qk_decode_attention_purity_capture.py`

A0 artifact:

- `bench/qk-decode-attention-purity/latest.json`

A0 verdict:

- `DECODE_ATTENTION_NOT_PURE__OWNED_TILE_COMBINE`

Observed default route:

- `owned_flash_tile_gqa_whole: 1`
- `owned_flash_combine: 1`
- `generated_attention_programs: 0`
- `E_49152_present: false`
- `buffer_identity_inputs: true`

This means decode attention is functionally good, avoids the old materialization penalty, but is still not pure
machine search because the selected fast path is the owned AMDGCN tile plus combine.

## What `E_49152` means and why it must stay absent

`E_49152` is the graph/materialization signature for the unwanted KV slice/copy buffer in the old decode-attention
route. In practical terms, it means the model created a temporary materialized cache slice instead of letting the
attention route read the existing KV cache buffer directly.

Why it matters:

- It is a lifecycle regression, not just a naming detail.
- It adds extra memory traffic and dispatch work.
- It hides whether the attention candidate is testing the real decode path or a copy-heavy substitute.
- It previously explained a large decode gap, so reintroducing it invalidates speed conclusions.

A generated attention candidate is not promotable if it brings `E_49152` back, even if the local kernel looks fast.

## A1 candidate shape

Create a generated skeleton route with the same external contract as the owned attention route:

- Input: query, key cache, value cache, decode position/context metadata.
- Output: attention result with the same shape/dtype contract as the existing route.
- KV access: whole-buffer identity read, no slice materialization.
- Dispatch lifecycle: one candidate identity that can later own TILE+COMBINE, even if A1 internally starts as simple
  generated programs.
- Naming: stable program/candidate name, for example `decode_attention_generated_skeleton`.

A1 can be slow. It can use simple scalar/generated operations at first. The non-negotiable requirement is that the
candidate is generated and attributable.

## Implementation steps

1. Add a default-off route flag.

   Suggested flag:

   - `DECODE_ATTN_GENERATED_SKELETON=1`

   Default:

   - `0`

   Rule:

   - It must not alter the default promoted owned-attention route.

2. Add a candidate binding.

   Requirements:

   - The route should be visible to BubbleBeam/FutureSight or the decode-attention search manifest.
   - The candidate must carry provenance fields:
     - `search_space_id`
     - `search_generation_status: generated_skeleton`
     - `blocked_primitives`
     - `promotion_status: attribution_only`

3. Implement the generated skeleton.

   Minimal acceptable A1 implementation:

   - Use tinygrad-generated operations only.
   - Preserve output shape and token correctness.
   - Keep the KV cache as identity input.
   - Avoid owned HIP/AMDGCN tile code.
   - Avoid `owned_flash_tile_gqa_whole`.
   - Avoid `owned_flash_combine` if possible; if combine reuse is unavoidable in the first patch, classify the verdict
     as partial and do not call it generated attention purity.

4. Extend the purity capture.

   Add A1 mode to:

   - `extra/qk_decode_attention_purity_capture.py`

   Required capture fields:

   - selected candidate name
   - generated skeleton program count
   - owned tile count
   - owned combine count
   - `E_49152_present`
   - buffer identity status
   - token sample
   - verdict

5. Add a gate artifact.

   Suggested output:

   - `bench/qk-decode-attention-generated-skeleton/latest.json`

   Passing A1 verdict:

   - `DECODE_ATTENTION_A1_GENERATED_SKELETON_ROUTE_CLEAN`

   Partial verdicts:

   - `DECODE_ATTENTION_A1_PARTIAL__OWNED_COMBINE_REMAINS`
   - `DECODE_ATTENTION_A1_FAIL__E_49152_REINTRODUCED`
   - `DECODE_ATTENTION_A1_FAIL__TOKEN_MISMATCH`
   - `DECODE_ATTENTION_A1_FAIL__OWNED_TILE_STILL_FIRES`

## Gate

A1 passes only if all are true:

- Generated candidate fires.
- Owned tile does not fire.
- `E_49152_present == false`.
- `buffer_identity_inputs == true`.
- Token sample matches baseline.
- The artifact records missing performance primitives instead of hiding them.

Speed is recorded but not used as the promotion criterion in A1.

## Kill conditions

Stop and classify instead of iterating blindly if:

- Avoiding `E_49152` requires the owned runtime path.
- Token correctness requires owned combine semantics that are not representable yet.
- The generated route cannot be named/captured as a distinct candidate.
- BubbleBeam cannot bind the candidate without hard-coding model-specific control flow.

Expected blocker classification:

- `SEARCH_BLOCKED_BY_CODEGEN` for missing `v_dot2`, cross-lane reduction, or LDS tile layout.
- `SEARCH_BLOCKED_BY_RUNTIME` if TILE+COMBINE cannot be represented as a candidate lifecycle.

## Exit state

At completion, one of these must be true:

- A1 passes and creates a route-clean generated attention skeleton artifact.
- A1 fails with a precise blocker and an artifact that proves why.

Either result is useful. A pass unlocks primitive performance work. A precise fail tells us which wall is still below
the current search/codegen layer.

## Completed result

A1 completed with:

- `DECODE_ATTENTION_A1_FAIL__E_49152_REINTRODUCED`
- Artifact: `bench/qk-decode-attention-generated-skeleton/latest.json`
- Result doc: `docs/decode-attention-a1-generated-skeleton-result.md`

The generated route fired and tokens matched, but the route consumed sliced KV inputs and reintroduced
`E_49152_32_3`. The next step is a generated whole-cache KV skeleton that accepts `assigned_kv` directly and indexes
K/V internally.
