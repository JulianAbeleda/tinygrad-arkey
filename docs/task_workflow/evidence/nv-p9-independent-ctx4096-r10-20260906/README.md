# NV P9 independent ctx4096 R10

Commit `c13da6ade`, two ordinary independent requests in one fresh process,
max-context 4608 and expected decode horizon four. Both prefills pass and return
the same first token, 34208. Request one takes 204.669 s including cold
compile/prewarm; request two takes 28.533 s. Post-request GlobalCounters are
19,711,028,320 and 19,711,046,756 bytes, an 18,436-byte difference. GPU state is
P0, 2572 MHz SM, 14001 MHz memory, 50--51 C, with no reported throttle reason.
This is request-lifecycle correctness and capacity evidence, not latency
qualification.
