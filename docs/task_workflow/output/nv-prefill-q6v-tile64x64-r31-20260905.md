# Q6-V 64x64 tile R31

The V-only schedule constructor was run with 18 canonical type-14 V weights and alternating dynamic TinyJit candidate/llama calls. Correctness passed on both activations (`max_abs=0.009955287`), but candidate median was `1.276592 ms` versus llama `0.928517 ms`, slower than the default candidate `1.235886 ms`. The schedule is rejected for performance and remains research-only.
