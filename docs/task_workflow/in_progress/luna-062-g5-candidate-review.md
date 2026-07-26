# LUNA-062 through LUNA-064: G5 Candidate Review

## LUNA-062 verdict

`NOT_RUN` with an `INCONCLUSIVE` design result. No bounded candidate is safe
to compile or time because LUNA-060 has no owner/Amdahl proof and LUNA-061 has
no named resource mechanism.

### Exact ROCm gate for LUNA-062

Before BoltBeam receives one candidate record, require all of the following on
the same final Track-A commit/model/device session identity:

1. 512 and 4096 tinygrad traces with a collector positive control and observed
   G5 tile/combine code-object hashes.
2. G4 and G5 resource/disassembly ledgers at both depths, including VGPR,
   SGPR, LDS, scratch/private/spill, wave/workgroup geometry, occupancy limit,
   VMEM/LDS/VALU/matrix/barrier/wait counts normalized by loop trip or tile.
3. Per-family device-wall attribution and an Amdahl upper bound demonstrating
   the named family can meet the full-model target.
4. Focused counters only if the installed gfx1100 tool exposes counters that
   distinguish the named mechanism; record tool version, definition, passes,
   and normalization. Missing tools are a stated limitation, not permission to
   invent a counter conclusion.
5. Compile-only candidate evidence preserving every LUNA-061 invariant,
   followed by deterministic finite/token and KV-bounds checks with exact
   route identity. Only then may serialized shallow/deep timing rank survivors.

Stop rather than expanding axes if no candidate meets the Amdahl bound, a
resource delta is absent, a code object is not positively matched, or any
route/traffic/semantic control changes.

## LUNA-063 verdict

`NOT_RUN`. No qualifying LUNA-062 candidate exists, so no implementation,
regression, or production routing change was made.

## LUNA-064 verdict

`NOT_RUN`. Without a passing LUNA-063 candidate, there is no basis for a
full-depth authority claim. Required later authority remains three
same-session interleaved baseline/candidate pairs at 512/1024/2048/4096,
same-session llama at 512/4096, token parity, route identity, 8B
non-regression, power/fault state, and ordinary-timing bandwidth calculation.
