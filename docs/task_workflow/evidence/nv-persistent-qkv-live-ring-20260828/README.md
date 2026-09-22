# Persistent Q/K/V live-ring physics gate

## Verdict

Closed for this construction.  Live producer-owned publication is correct and
deadlock-safe in a single resident grid, but it is slower than the standalone
Q4_K projection in both cache regimes.  Do not invest in production graph or
allocation support from this evidence.

## Construction

- The arithmetic is mechanically copied from tinygrad's generated phased
  one-row Q4_K body; only row/lane ownership changes.
- Inputs use finite Q4_K headers (`d` and `dmin` are finite half values),
  deterministic scales/quants, and 16 rotated 14.16 MB weight groups.
- CTA 0 is the producer.  It waits until every consumer CTA reports residency,
  then publishes epoch 1 with a device-scope release store.
- Consumers acquire the epoch, process deterministic row stripes, and publish
  completion.  A device-clock watchdog aborts stalled admission/publication.
- The selected geometry is 339 consumer CTAs x 256 threads (2,712 warps) on
  170 SMs.  One producer CTA is explicitly reserved.  One-SM and three-SM-wide
  occupancy variants were also tested and were slower.

## R9 authority

Source: `gate-r9.json`.

| arm | L2-hot median | rotated-cold median |
|---|---:|---:|
| standalone one-row grid | 9.567 us | 12.574 us |
| persistent live service | 15.366 us | 15.476 us |
| persistent debt | +5.799 us (+60.6%) | +2.902 us (+23.1%) |

All 6,144 outputs are bitwise identical and finite.  The persistent kernel uses
37 registers, one barrier, and zero spills; the standalone uses 43 registers,
zero barriers, and zero spills.  This is not register spilling.

## Walls discovered

1. A host-mapped epoch is not viable here.  Even after removing per-row
   system atomics, host/device visibility and completion polling cost roughly
   milliseconds, orders of magnitude above the projection.
2. Separate producer and consumer grids did not obtain live cross-stream
   progress before the watchdog, including high-priority producer and half-SM
   reservation.  A same-grid producer removes that admission ambiguity.
3. The corrected same-grid service still loses.  Persistent fixed-stripe
   ownership and its residency/publication/completion protocol cost more than
   they recover from launch/scheduling.  The cold result is the relevant wall:
   DRAM service is 23% worse despite identical bytes and arithmetic.

Because timing failed, DRAM/service counter collection was intentionally not
promoted to the expensive counter gate.  The next Q/K/V lever needs a changed
work/byte topology, not merely persistent ownership of the existing row body.
