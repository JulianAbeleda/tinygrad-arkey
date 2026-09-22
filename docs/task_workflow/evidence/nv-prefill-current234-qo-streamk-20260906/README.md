# Current234 Q/O Stream-K experiment

The existing N4096 Stream-K scheduler had never been exercised for the complete
36 Q plus 36 O population; its dormant capture limit was incorrectly 36. The
research arm fixes that limit, gives the N4096 program a distinct symbol, and
requires 72 mains plus 72 additional deterministic fixups. It passes the
current234 structural census, token 198, 20/20 exact recurrent replay, and the
full-logit tolerance against wide Q/O (max absolute 0.2363677, mean 0.0150453).

The candidate median is 55.995463 ms against current234 wide-Q/O controls of
54.935668 and 55.076140 ms. It regresses 0.989559 ms or 1.799% against their
mean, so Q/O Stream-K remains research-only and wide Q/O remains selected.
The first attempt is retained as the pre-timing population-limit failure.
