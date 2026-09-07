# Gate weight/record restrict cadence closure

The canonical Q4 weight and DS4 Q8 record are distinct read-only allocations. Adding `__restrict__` to both generated CUDA inputs preserved composed 8-call x 10-cycle bit-exact replay. Static instruction counts were unchanged (LDG152, PRMT176, LDS448, IMMA256), but NVRTC changed scheduling/register allocation from REG237 to REG242 without spills. Alternating R15 regressed from 335.210 us control to 338.356 us candidate (3.146 us/call; minima 323.649/327.816). The qualifier is rejected and no production path is retained.
