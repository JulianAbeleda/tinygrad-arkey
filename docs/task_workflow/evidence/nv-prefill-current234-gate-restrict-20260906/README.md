# Current234 gate Stream-K restrict-pointer experiment

The generated gate/up Stream-K transform was compiled with restrict-qualified
kernel pointers while retaining U8, ownership, producer, fixup, and the full
current234 route. Correctness and structural census pass, including token 198
and 20/20 exact recurrent replay cycles.

The candidate median is 55.702979 ms. Matched current234 controls are 54.935668
and 55.076140 ms, so restrict qualification regresses 0.697075 ms or 1.267%
against their mean. The source option was reverted and is not retained in the
runtime. This closes alias qualification as a gate-main optimization.
