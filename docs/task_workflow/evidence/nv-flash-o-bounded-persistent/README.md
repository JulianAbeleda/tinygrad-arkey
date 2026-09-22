# Bounded persistent O evidence

The canonical result is `final-w1024-r9.json`. It compares the installed O
kernel, the first-class bounded UOp emitter after readiness, and the same
emitter launched persistently ahead of the full-grid Flash producer.

`bracket-w512-r5.json` and `bracket-w1024-r5.json` select the useful end of the
worker-population sweep. `sweep-w*.json` and `smoke-w128-v2.json` are topology
discriminators for smaller worker populations.

All reported candidates are bit-exact and finite. The final emitter uses 39
registers, one block barrier, and zero spills. No production route was edited.

Verdict: the O body reaches parity at 1,024 workers, but launching it ahead of
Flash regresses the dependent span. No recovery is booked.
