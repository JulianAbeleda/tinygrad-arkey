# Corrected exact segmented Flash-to-O complete-span gate

## Decision

`EXACT_AND_OVERLAP_REAL__COMPLETE_SPAN_NEGATIVE__STOP_AT_16_16`

No production route was changed and no recovery is booked.

The corrected 16/16 gate removes every internal diagnostic timing event from
the timed candidate.  The early score, combine, and O-partial kernels execute
in order on the high-priority stream.  The remaining score and combine execute
on the low-priority stream and publish one `cudaEventDisableTiming` readiness
event.  The high-priority stream waits for that event only before O-finish.
Timeline events exist only in a separate trace invocation.

## R9 authority

| arm | hot median | 16-copy rotated-cold median |
| --- | ---: | ---: |
| unsplit score -> combine -> O | 11.3875 us | 25.267999 us |
| corrected segmented 16/16 | 17.5885 us | 27.467999 us |

The candidate is bitwise exact at score, combine, and final O.  Score metadata
contains the legal negative-infinity sentinel for empty split lanes but no
NaNs; combine and final O are finite.  All candidate kernels compile without
spills.

The separate trace shows a 3.903999-us median overlap between remaining Flash
and early O-partial.  That overlap is real, but it is not useful recovery: the
candidate loses 6.201 us hot and 2.200 us rotated-cold after charging the split
score/combine launches, 512 KiB scratch store/read traffic, the dependency
edge, and the second O body.

The requested 8/24 and 4/28 sweeps were not run because the advance condition
was not met.  The clean 16/16 construction showed overlap but no net recovery;
widening the imbalance cannot remove any charged component and would require
a different partial-O lane owner, making it a new construction rather than a
parameter sweep of this exact two-row/warp gate.

Primary authority: `complete-span-corrected-16-16-r9.json`.

