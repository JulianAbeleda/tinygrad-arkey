# Tile-major gate interleave qualification

The retained Stream-K WMMA-update interleave was requalified after the gate/up
path moved to its tile-major Q8 carrier. Two exact current252 model runs measured
47.599797 and 47.558841 ms medians, versus the promoted tile-major baseline's
47.642309 and 47.870729 ms replications. Both candidates passed five exact
replay cycles and the generated/canonical route census.

The 0.177200 ms mean reduction does not clear the frozen 0.5 ms population
investment threshold, so the schedule remains default-off. Set
`NV_COMPILER_Q4_STREAMK_INTERLEAVE_WMMA_UPDATES=1` to reproduce it.
