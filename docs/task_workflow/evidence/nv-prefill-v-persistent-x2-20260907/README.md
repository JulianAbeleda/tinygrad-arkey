# Q4V/Q6V persistent-x2 topology closure

Both selected generated V bodies use direct grid `(32,8,1)`, block `(32,2,2)`, 256 CTAs and no fixup. A common generated-owned transform reduced grid x to16 and made every CTA execute logical x tiles `blockIdx.x` and `blockIdx.x+16`. The loop encloses all per-tile state, accumulator initialization, barriers and output ownership.

Representative canonical Q4V and Q6V layers are bit exact and read-only. Alternating R15 rejects performance: Q4V 156.845→176.231 us and Q6V169.198→209.524 us. Across18 roles each this projects a1.075ms regression. Direct256-CTA topology remains authoritative; no production route is retained.
