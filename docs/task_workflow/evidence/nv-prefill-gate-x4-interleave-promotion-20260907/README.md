# Generated gate/up Q4-A x4 plus interleaved WMMA promotion

The compatible composition of the generated Q4-A/x4 body, logical coalesced output, and WMMA-update interleave clears the frozen 0.5 ms population gate. The interleave transform now derives its exact 64 scale declarations from data ownership instead of renderer-local cast numbering; it still requires 32 WMMAs, 64 scales, 64 accumulator updates, and consumes each declaration once.

Deep smoke passes token 198, exact replay and every recorded stage buffer, 252 canonical generated projection mains, 198 Q8 producers, 90 active fixups, and zero overlays or weight copies. Composed SASS retains 256 IMMA and x4/SHFL output mapping while reducing registers to 250, stack/local to zero, and LDL/STL to zero.

Stable warmup-9 R9 control/candidate/control medians are 47.951482 / 47.121582 / 47.933117 ms; minima are 47.685803 / 46.949819 / 47.670343 ms. Against the control midpoint, candidate wins 0.820718 ms median and 0.728254 ms minimum. Control endpoint drift is 0.018365 ms.

The exact generated route is selected automatically inside the qualified compiler pp512 gate/up Stream-K route. Rollback: `NV_COMPILER_Q4_GATE_X4_INTERLEAVE=0`.
