# S0 privileged clock probe

Decision: **STOP (wall drift only)**.

Non-interactive sudo was available. The exact PROFILE=0 safe-cut candidate was
run under `/tmp/gpu-bench.lock` with two compute queues, three warmups, nine
R9 rounds, and deep-20 replay while `nvidia-smi -lgc 3090` was active.

Correctness passed: finite logits, token `198`, exact deep-20 replay, and the
198-role canonical census. The R9 samples were
`72.103998, 72.070646, 72.121643, 72.079867, 72.109481, 72.107778,
72.133305, 72.139357, 72.136782 ms`; median `72.109481 ms`.

This is 7.25% slower than the `67.235719 ms` authority and fails the required
1% band. Telemetry recorded graphics clocks from 2797 to 2947 MHz. The trap
restored the prior state: persistence mode disabled and graphics clock lock
inactive. Complete evidence is in
`docs/task_workflow/evidence/nv-prefill-post-substrate-privileged-clock-20260829/`.
