# True Generation: Rediscover G3 From a Topology Grammar (TG0–TG7)

Date: 2026-06-30

Status: execution scope for the bridge from **machine selects a generated route** to **machine authors the route**.

## The milestone (not "beat G3")

The key milestone is NOT beating G3. It is: **can the system rediscover G3 from a topology grammar without hardcoding the
route?** That is the bridge from "machine selects generated route" to "machine authors route." Success = a grammar over
the lane-map topology, given only quant facts + shape + GPU features, regenerates the G2 LaneMap that G3 uses
(lossless per PMS-R5) — with `block_groups=4` / `words_per_group=8` / the axis-role assignment NOT pre-baked.

## Phase plan

1. **TG0** — audit G3 provenance: generated vs templated vs still-human. **(DONE: `TG0_PASS_G3_PROVENANCE_PINNED`)**
2. **TG1** — define a `LaneMapTemplate` IR that can losslessly express G3 (exposes the still-human topology DOF as free fields).
3. **TG2** — build a topology candidate author that can rediscover G3 from grammar/profile (the crux).
4. **TG3** — make quant semantics data-driven across Q4_K, Q5_K, Q6_K, Q8_0, fp16.
5. **TG4** — add a new-profile opener for model/quant/GPU variants.
6. **TG5** — separate target features for AMD/NVIDIA/Metal.
7. **TG6** — evaluate template-authored candidates with the existing authority gates (PMS-R2 evaluator).
8. **TG7** — run the first true new-profile search.

## TG0 result (provenance, pinned)

Source: `bench/qk-g3-provenance-audit/latest.json` (`extra/qk_g3_provenance_audit.py`), built on PMS-R5
(`bench/qk-lanemap-template-audit/latest.json`, which proved the EMISSION is a lossless template).

| bucket | count | what it is |
|---|---|---|
| generated | 1 | UOp → AMDGCN lowering (tinygrad codegen, automatic) |
| templated | 5 | kernel emission from the LaneMap + rows/k/dequant-body/store (R5-proven lossless) |
| quant_data | 3 | Q4_K block layout constants (qk_k=256, words_per_block=36, quant_word_base=4) — TG3 makes data-driven |
| gpu_data | 1 | lane_extent = wave32 — TG5 makes a target feature |
| **still_human** | **5** | **the lane-map topology — the bridge target** |

**The 5 still-human components (the human design surface):** `block_groups`, `words_per_group`, `axis_role_assignment`
(GLOBAL/LOCAL/REDUCE), `cross_lane_reduction`, `packed_word_lane_index`.

**Topology DOF a grammar (TG1/TG2) must span:**
- `k_to_lane_decomposition` — factor k_blocks into (block_groups × words_per_group × local_block × group_pair), with
  `block_groups * words_per_group == lane_extent`;
- `axis_roles` — assign each factor to GLOBAL | LOCAL | REDUCE;
- `lane_ownership_index` — the coalesced packed-word index per lane (derivable from the decomposition + quant packing);
- `reduction_pattern` — cross-lane wave reduce vs partials+separate-reduce.

Verdict headline: **G3 is generated emission of a templated spec, but the lane-map topology is still human.** The bridge
only needs to make that topology machine-discoverable; everything else is already mechanical or data.

## Non-goals / discipline

- Do not chase speed; the target is rediscovery, not a new fast kernel.
- Do not change defaults or reopen refuted routes.
- Reuse PMS-R0–R2 (manifest + table-driven evaluator) and PMS-R5 (lossless template) — build on them, do not rebuild.
- A rediscovered topology is only "authored" if the grammar (not a person) produced the winning decomposition, and the
  PMS-R2 evaluator proves route/token/speed equivalence.
