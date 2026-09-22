# Q6-V schedule sweep

The existing compiler Q6 research constructor was tested against its admitted 128-thread warp variants. Baseline `(warp_m,warp_n)=(2,2)` compiles with an exact V candidate identity. `(4,1)` and `(1,4)` both fail closed while matching the V PROGRAM, reporting `expected one compiler Q6 attn_v PROGRAM, found 0`. No identity or geometry gate was weakened and no nonbaseline runtime result is claimed.
