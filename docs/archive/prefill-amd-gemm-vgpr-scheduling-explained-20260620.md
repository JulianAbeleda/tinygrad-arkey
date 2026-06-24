# Prefill AMD GEMM — Explaining the "VGPR-Bound" PLR Wall (from Tensile source)

Date: 2026-06-20

## The question

I claimed the last ~60→66 (intra-substep PLR) is "VGPR-walled: 2× fragments + accumulators overflow 256." Is
that a hardware limit, or something else? Reading the cloned Tensile source answers it precisely — and
**softens the claim**: it is an **allocator-capability gap**, not a physics wall.

Source: `/home/ubuntu/rocm-libraries-tensile-sparse/shared/tensile/Tensile/` (`KernelWriterAssembly.py`,
`AsmRegisterPool.py`).

## What PLR costs (confirmed)

`KernelWriterAssembly.py:1340-1373`: PLR multiplies the **fragment** buffers (`ValuA`/`ValuB`) by `(1+PLR)`:

```
PLR = PrefetchLocalRead
valuBlocksA = (1+PLR) * InnerUnroll
numVgprValuA = numVgprValuAPerBlock * valuBlocksA      # PLR1 -> 2x fragments
```

So PLR1 does need a second fragment buffer (~64 VGPR for our 4×4 tile). That part of my claim was right. The
accumulators (`ValuC`) are independent of PLR. The question is whether `ValuC(128) + 2×frag` fits 256.

## How Tensile fits it — a lifetime register POOL with overlap

It fits because Tensile **does not allocate every buffer a disjoint permanent range.** It uses a register
pool (`AsmRegisterPool.py`: `checkOut`/`checkIn`, `Status.Available/Unavailable`) that **reuses physical
VGPRs once a buffer's live range ends**, and it deliberately **overlaps** buffers whose lifetimes don't
conflict:

- **Global-load temps overlap the fragment regs** — `KernelWriterAssembly.py:1571-1574` /`1591-1594`:
  ```
  # g2l can overlap valu
  self.startVgprG2LA = self.startVgprValuA
  vgprIdx = startVgprValuA + max(numVgprValuA + numVgprValuPackA, numVgprG2LA)
  ```
  (`max`, not sum — they share the same physical registers.)
- **C-accumulators overlap the A/B tile "up until writeback"** — `1524-1531` (verbatim):
  ```
  # MI kernels can overlap C-tile w/ AB-tile up until writeback. Illustrated below:
  # |<-------------- valuC -------------->|
  # |------------|-----------|xx|---------|
  #   lastValuAB ^   startVgprReuse ^   lastValuC ^
  ```
  The accumulators aren't all live until the end, so the fragment/prefetch registers live *inside* the C
  range during the loop, freed (`startVgprReuse`, `1772`) for reuse.

So Tensile's "256 VGPR" is a **packed** 256 — the same physical registers serve accumulators, fragments,
global-load temps, and the PLR prefetch buffer at *different points in the loop*, managed by lifetimes.

## Why ours overflows — static disjoint allocation

`build_gemm_lds2` (`extra/gemm/rdna3_wmma_matmul.py:412`) assigns every buffer a **fixed, permanent,
non-overlapping** range:

```
FA=10; FB=FA+WM*8; ACCb=FB+WN*8; CTA=ACCb+WM*WN*8; CTB=CTA+loadsA*4; SCR=CTB+loadsB*4
```

| buffer | VGPRs | live when |
|---|---|---|
| FA (A frags) | v10–41 (32) | during WMMA |
| FB (B frags) | v42–73 (32) | during WMMA |
| ACC (accumulators) | v74–201 (128) | whole loop, critical at writeback |
| CTA/CTB (coop-load temps) | v202–233 (~32) | **only during global→LDS store; DEAD during WMMA** |
| total | **234** | (assert ≤ 256) |

There is **no `checkIn`** — `CTA/CTB` stay reserved for the whole kernel even though they're dead during the
compute phase. Adding a PLR prefetch buffer (+64) → 298 > 256 → the static allocator's `assert` fails. The
data would fit a *pool*; it doesn't fit our *fixed map*.

## The corrected conclusion

**The "VGPR wall" is not a hardware limit — it's that our hand-asm uses static disjoint allocation while
Tensile uses a lifetime register pool.** The ~32 VGPRs our kernel wastes on permanently-reserved coop-load
temps (dead during WMMA) are exactly where Tensile would place the PLR prefetch buffer.

So a dependency-free PLR1 is **more achievable than I said** — it doesn't need "Tensile's whole register
scheduler," just **register-lifetime reuse for one overlap**: manually assign the PLR prefetch fragments onto
the `CTA/CTB` range (which is dead once the LDS store is issued), ordering the code so their lifetimes don't
collide. That is intricate hand-asm bookkeeping, but it is a *bounded, known* technique — not a physics wall
and not a multi-month codegen capability.

## Honest status update

- My earlier "intra-substep PLR overflows 256 = separate VGPR-bound project / register-allocation wall" is
  **refined**: the mechanism is a missing **register-lifetime overlap** (Tensile's pool), and there is a
  concrete dependency-free path — overlap the PLR buffer onto the dead coop-load temps (~32 VGPR of headroom).
- It is still real work (manual register-lifetime management on the `assemble_linear` path, correctness-risky)
  and the payoff is the ~60→66 (~9%) that PLR latency-hiding buys on top of the bank-conflict fix.
- The dependency-free arc currently **rests at Tensile-class ~60.7** (BK32+PAD16, wg2). This explanation
  reopens a *bounded* path to attempt the rest, distinct from the (declined) vendored `.co`.

## If pursued next (one bounded experiment, no BEAM)

Hand-write a PLR variant of `build_gemm_lds2` that, after the cooperative LDS store + barrier, **reuses the
CTA/CTB register range** as the prefetch buffer for the next K-substep's `ds_load`, issued before the current
WMMAs, with a targeted `lgkmcnt`. Gate on: correctness (rel RMSE < 0.02), VGPR ≤ 256 (no spill), and TFLOPS
under the interleaved harness vs the ~60.7 frontier. Outcome bounds the dependency-free ceiling definitively.
