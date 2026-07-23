# Selective LDS PV-accumulator demotion cost model

Date: 2026-07-24

## Scope

This is a static admission model for the current single-wave `gfx1100` Hd128
attention shared by the 8B overlay and 14B bounded routes. It makes no GPU
performance claim. No kernel was compiled, launched, or replayed for this
model.

## State and byte accounting

A wave32 lane owns eight fp32 values for each of eight Hd16 PV accumulator
blocks: 64 VGPR values per lane. Therefore:

```text
one block = 8 fp32/lane * 32 lanes * 4 B = 1024 B
all blocks = 8 * 1024 B = 8192 B/workgroup
existing probability LDS = 512 B/workgroup
full rotating backing = 8192 + 512 = 8704 B/workgroup
```

The old 256-byte calculation was `8 blocks * 8 values * 4 B` and omitted all
32 lanes. Its address formula also placed `lane` in an eight-entry dimension;
the correct flat index is `(block*32 + lane)*8 + element`.

| Policy | Register PV window | Demoted blocks | Acc LDS | Total LDS | Steady LDS accumulator traffic/KV tile | Arithmetic-only VGPR estimate |
|---|---:|---:|---:|---:|---:|---:|
| full register baseline | 8 blocks / 64 | 0 | 0 | 512 B | 0 | measured 254 |
| full rotating backing, 1 block | 1 / 8 | 8 | 8192 B | 8704 B | 8192 B load + 8192 B store = 16384 B | `254-56=198` |
| pinned 1-block window | 1 / 8 | 7 | 7168 B | 7680 B | 7168 B load + 7168 B store = 14336 B | `254-56=198` |
| pinned 2-block window | 2 / 16 | 6 | 6144 B | 6656 B | 6144 B load + 6144 B store = 12288 B | `254-48=206` |

The estimates are optimistic lower bounds before LDS addresses, counters,
wait state, and allocator fragmentation. They prove that accumulator demotion
alone does not imply the 192-VGPR gate: one block needs at least six additional
non-PV values removed, and two blocks needs at least fourteen.

With vectorized 16-byte LDS operations, one block/lane requires two loads and
two stores. Full backing therefore adds 16 `ds_load_b128` and 16
`ds_store_b128` wave instructions per KV tile. A one-block loop has eight
load/update/store phases; two blocks has four phases but does not reduce total
bytes under full backing. A pinned window reduces both traffic and LDS size as
shown. First-tile zero initialization and the final drain can be scheduled so
their combined bytes replace, rather than add to, one steady read/write cycle;
the long-KV steady cost remains the admission authority.

## Static residency bounds

The captured device has 64 KiB LDS/CU, wave32 workgroups, two SIMDs/CU, a
32-wave/CU limit, and a 1024-work-item/CU limit. LDS alone gives exact bounds:

| Policy | Total LDS | `floor(65536/LDS)` one-wave groups/CU |
|---|---:|---:|
| full rotating backing | 8704 B | 7 |
| pinned one block | 7680 B | 8 |
| pinned two blocks | 6656 B | 9 |
| rejected G2 | 9216 B | 7 |

The installed device record does not expose physical VGPR-file capacity or
allocation granularity. Do not claim an exact residency count from 192 VGPR.
The combined static bound is:

```text
waves/CU <= min(32, floor(65536/LDS),
                2 * floor(VGPR_capacity_per_SIMD / allocated_VGPR_per_wave_lane))
```

At roughly 192 VGPR, LDS becomes limiting only if the VGPR file would
otherwise admit at least 7/8/9 waves per CU for the respective policy. A
profiler or device occupancy API must establish the actual count before a
residency claim is promoted.

## Cost comparison

### Rejected G2 K/V sharing

G2 used 9216 B LDS, 250 VGPR, two workgroup barriers per KV tile, and 36 B
private memory. It reduced scalar K/V load sites but was 0.20% slower at KV512
and 1.65% slower at KV4096. Full accumulator backing uses 512 B less LDS but
adds 16 KiB explicit accumulator traffic per tile. Its only plausible
advantage is a large VGPR/residency gain with wave-local waits and zero private
memory. If compiled VGPR does not cross the admission bucket, G2 already
predicts rejection.

### Split score-state/PV recomputation

The accepted compile-only split has Stage A at 154 VGPR and all four two-block
Stage B slices at 192 VGPR, each with 512 B LDS and zero spill/scratch. Per KV
tile it executes:

```text
monolithic: 8 QK WMMA + 8 PV WMMA = 16 WMMA
split: Stage A 8 QK + four Stage B * (8 QK + 2 PV)
     = 40 QK + 8 PV = 48 WMMA
delta: +32 QK WMMA, 3x total WMMA, five kernels, explicit stats traffic
```

LDS demotion retains one QK pass and 16 total WMMAs, but replaces split's QK
recomputation with 12-16 KiB accumulator LDS traffic/tile. This trade is worth
testing only after resource metadata proves residency, not from operation
counts alone.

## Window decision

Use a **one-block window for the first diagnostic**.

- It is the only option close enough to 192 VGPR to falsify the hypothesis;
  even its optimistic estimate is 198, requiring six further live-value cuts.
- Two blocks starts at an optimistic 206 and cannot prove a better residency
  class unless at least fourteen other values disappear.
- Two blocks has lower traffic and half as many window transitions, so it is a
  follow-up only if one block crosses the resource gate and profiling shows the
  same occupancy bucket can tolerate eight more VGPR.

If one block compiles above 192, reject this monolithic demotion route before
GPU replay. A two-block candidate cannot improve its VGPR admission result.

## Fail-closed gates

### Numeric

- fp32 LDS storage only; no fp16 accumulator demotion or additional rounding.
- Bit-identical output versus the direct monolithic fp32-accumulator path is
  expected because update order is unchanged; any nonzero direct-path delta
  requires an explained ordering change.
- Independently satisfy the existing causal fp32 reference tolerance:
  `rtol=0.02`, `atol=0.004`, for first and prefix geometry on both 8B and 14B.

### Structural proof

- One CALL, one QK/online-softmax pass, exactly eight attributed QK and eight
  attributed PV WMMAs per KV tile; no split recomputation.
- Exactly eight disjoint LDS accumulator blocks, each `32*8*4=1024` B, with
  lane/block address proof and complete output ownership `0..7`.
- No materialized score/probability tensor, no output overlap/gap, and no
  implicit compiler spill masquerading as demotion.
- Wave32/local-size32 only, wave-private waits only, zero workgroup barriers;
  each LDS load must be dominated by its publication and wait.

### Resource

- One-block compiled HIP metadata `vgpr <= 192`, `lds_bytes <= 8704`,
  `scratch_bytes=0`, both spill counts zero, private bytes zero, SGPR no worse
  than the 26-SGPR baseline.
- ISA census must show only one live eight-value PV C fragment and explicit
  fp32 LDS load/store sites. Removing fixed register names without lowering
  compiled VGPR fails.
- Static occupancy calculation and a later measured occupancy counter must
  both improve over the 254-VGPR/512-B baseline; allocation alone is not proof.

### Performance, only after the other gates pass

- Alternating synchronized replay with preallocated buffers, fixed inputs,
  completed JIT capture, and the same sample protocol as G2.
- At KV512 and KV4096, median latency must beat the direct monolithic baseline
  by at least 5%; KV64 may regress by at most 1%. This margin is deliberately
  larger than G2's sub-2% effects.
- It must also beat the complete five-launch split path at identical geometry,
  including Stage A, all four Stage B launches, and stats traffic.
- Full 8B and 14B prefill authority runs must improve end-to-end tokens/s with
  no decode or non-attention regression before any production promotion.

## Verdict

The primitive is admissible only as a one-block compile-only microgate. The
correct footprint is 8-9 KiB, not 256 B, and its 14-16 KiB/tile traffic is
large enough that an actual residency step is mandatory. Compile above 192
VGPR, any private/spill allocation, unchanged occupancy, or less than the
performance margin rejects the route and retains the split implementation as
the resource proof point.
