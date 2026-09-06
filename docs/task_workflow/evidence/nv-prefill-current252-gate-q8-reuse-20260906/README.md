# Current252 gate/up Q8 pair-reuse promotion

The existing graph-owned pair-reuse substrate is now requalified on the complete current252 route after generated Flash K staging. It shares one compact Q8 activation record across each ordered gate/up projection pair, removing 36 redundant producers without changing either packed-weight main or fixup.

A control/candidate/control bracket measured 49.598182 / 49.368896 / 49.690918 ms medians. The candidate improves 0.275654 ms against the mean controls; its 49.217232 ms minimum is 0.177252 ms below the better control minimum. An immediately preceding candidate replication measured 49.265036 ms median.

Every retained arm is finite, selects token 198, and passes five exact recurrent replay cycles. The candidate retains 252 generated mains, 252 canonical packed-weight bases, 90 active fixups, zero overlays and zero weight copies. Q8 producers fall from 252 to 216. The stale harness branch still expected the earlier current234 composition; it now checks the complete current252 population, including Q6 V and both down families.

Pair reuse is the ordinary compiler-Stream-K behavior. `NV_COMPILER_Q4_GATE_Q8_REUSE=0` is the explicit rollback.
