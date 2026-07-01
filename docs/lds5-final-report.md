# LDS5 — Final Report: LDS Staging Primitive Track

Date: 2026-07-01.

## Track outcome: LDS0_EMITTER_BLOCKED

The LDS staging primitive track ran LDS0 only. All subsequent phases (LDS1-LDS4) are skipped
per track spec: the first hard blocker is reported and the track stops.

## What LDS0 found

### Correction to EB5

EB5 classified `decode_bypass_kv_slice_lds` as PRIMITIVE_MISSING (LDS-alloc UOp unavailable).
**This was wrong.** `Ops.DEFINE_LOCAL` (AddrSpace.LOCAL) and `Ops.BARRIER` are already used in
production code in `extra/qk_flash_decode.py` (lines 214, 253) for K staging in the LDS+crosslane
score kernel. AMD renderer lowers them to `__local` alloc and `__builtin_amdgcn_s_barrier`.

### Actual blocker

Full-split V staging (the only angle with a real benefit) requires a "prologue range" before the
main j REDUCE loop — a structural UOp kernel pattern that does not currently exist. The three
angles checked:

| angle | verdict |
|-------|---------|
| Cross-kernel LDS staging | GRAPH_LIFETIME_BLOCKED (HIP LDS is workgroup-scoped) |
| Per-token V staging inside j loop | NOT_BENEFICIAL (no data reuse) |
| Full-split V staging (prologue + barrier + reduce) | **EMITTER_BLOCKED** — needs prologue range UOp |

### Corrected BoltBeam classification

`decode_bypass_kv_slice_lds`: PRIMITIVE_MISSING → **EMITTER_BLOCKED**

Rollback from EB5's wrong classification: the primitive (AddrSpace.LOCAL) exists; the missing
piece is a structural kernel pattern (prologue range), not a UOp opcode.

## Reopen condition

A "prologue range" UOp primitive allows a pre-stage indexed loop that (1) runs before the main
REDUCE, (2) writes to a DEFINE_LOCAL buffer, (3) implies a barrier before the REDUCE starts.
Once this exists:
- `flash_partial_coop_vec_whole_cache_kernel` gains a V prologue stage
- E_49152_32_3 (6.69% GPU at ctx512) is eliminated
- On-chip LDS provides equivalent or better warming than current L2 copy
- Upper Amdahl: 6.69% at ctx512 → projected +3.0-5.0 tok/s

## Next step

No tinygrad code changed in this track (LDS0 is docs + BoltBeam update only).
The prologue range UOp is a new kernel DSL capability, to be designed when the
"emitter primitive" track is picked up as a primary goal.
