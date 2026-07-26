# LUNA-061: G5 Resource and Lifetime Design

Verdict: `INCONCLUSIVE`; no production candidate is admitted.

## Immutable invariants

| Invariant | Required state |
|---|---|
| Geometry | `B=1,Hq=40,Hkv=8,Hd=128,G=5,S=48`, wave32, five full waves/workgroup |
| Tile ownership | One workgroup per `(kv_head, split)`; warp owns its GQA query head |
| Traffic | Cooperative K/V staging remains single-read per staged tile; no second pass or duplicated K/V work |
| Synchronization | Preserve the post-stage barrier and the WAR barrier before LDS overwrite |
| Semantics | Causal mask, live `Tc` bound, KV layout/positions, split partial layout, and fused combine output remain unchanged |
| Route | Keep `decode_flash_live_split_g5_kvboth`; no threshold or fallback policy change |

## Candidate matrix

| Candidate | Static basis | Allowed next gate | Stop condition |
|---|---|---|---|
| G5-specific address recomputation | G5 has a longer-lived address/staging expression surface across six deep blocks; source cannot show allocation | Compile-only only after G5 code-object evidence identifies address/live-range pressure | No G5-only live-range/resource delta, or recomputation increases scratch/VGPR occupancy limit |
| G5-specific expression lifetime/order primitive | Tile recurrence, barriers, and five-wave ownership are fixed while long-loop cost diverges | Compile-only only after disassembly/resource comparison identifies a specific lifetime region | Changes route, output ownership, barrier placement, or useful work; or lacks predicted resource change |
| G5 route/kernel selection change | None | Not admitted | No positively observed equivalent route/resource advantage; do not alter flash threshold or candidate binding |
| Four-loader-wave staging | Source gives only a theoretical staging slack and changes cooperative ownership | Not admitted beyond an isolated compile-only discriminator explicitly preserving all ownership/barriers | Any changed traffic, grid, barrier, or output ownership; no material resource reduction |
| `K_ONLY`, split size, G reduction, fifth-head serialization, second pass | Explicitly refuted or unsafe in scope | Prohibited | New contradictory evidence must be reviewed before reopening |

## Design conclusion

The safe search space is empty today. The two conditional primitive families
above are contracts, not proposed edits: the source does not expose emitted
VGPR allocation, register live intervals, spills, or occupancy. A source-only
change would be schedule speculation and is rejected.

## Required evidence before a candidate exists

- Final-commit G4/G5 512/4096 tile and combine disassembly/code objects.
- Same-schema VGPR, SGPR, LDS, private/scratch/spill, occupancy, instruction,
  barrier, and wait rows.
- A named differentiator: e.g. G5-only spill, occupancy limit, or a specific
  address/live-range expansion whose removal predicts a resource change.
- A route positive control proving the expected G5 tile/combine pair ran.
- A bounded numerical/route control before timing any changed program.
