# Tile-major gate fragment plus interleave qualification

The two retained scheduling changes were tested together after repairing the
fragment transform's tile-major ABI boundary. The current252 route passed five
exact replay cycles and measured 47.519338 ms median (47.246300 ms minimum),
versus 47.642309 and 47.870729 ms baseline replications.

The 0.237181 ms reduction against the mean baseline does not clear the frozen
0.5 ms population threshold. Both schedules remain default-off; this closes
their tile-major interaction without a wider toggle sweep.
