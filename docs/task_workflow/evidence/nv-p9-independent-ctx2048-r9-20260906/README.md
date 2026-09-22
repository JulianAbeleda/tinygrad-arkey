# NV P9 independent ctx2048 R9

Commit `ce714b3df`, two ordinary independent requests in one fresh process,
max-context 2560 and expected decode horizon four. Both prefills pass and return
the same first token, 13876. Request one takes 140.220 s including cold
compile/prewarm; request two takes 14.386 s. Post-request GlobalCounters are
19,398,663,264 and 19,398,673,508 bytes, a 10,244-byte difference. GPU state is
P0, 14001 MHz memory, 49--51 C, with no reported throttle reason. This is
request-lifecycle correctness and capacity evidence, not latency qualification.
