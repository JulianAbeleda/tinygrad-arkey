# B0.1 gate/up executed-path locator

Status: **PASS**.

The exact compiler path is frozen for the real `blk.0.ffn_gate.weight` fixture:
72 roles (36 gate and 36 up), `M=512, N=12288, K=4096`, K64, grid `(96,4,1)`,
block `(32,2,4)`, and entry point
`r_4_96_32_2_4_2_2_4_4_64_2_2_2_8cefca4b229d8048634693df0dd2614ca8c3df3f9d8527d95c0d4432b3a5a49e`.

The retained unroll-4 cubin is `03b44392...3e0ef389`; resources are 255
registers, 8 bytes stack, 0 local, and 21,504 bytes shared. The real fixture
passes finite-output, zero-sentinel, nonzero-output, and read-only input checks;
its retained R9 is 283.008 us minimum and 285.120 us median.

The three authorized B0 mutations are frozen as fragment-distance reordering,
metadata-only distance movement, and one register-safe alternating fragment
buffer. Tile, ownership, IMMA arithmetic/count, queue placement, cp.async/TMA,
and fusion are immutable.

The direct one-real-gate harness was rerun under the GPU flock and passed. Root
`/usr/local/bin/ncu` version 2026.2.1.0 captured the complete baseline: occupancy
limits registers/shared/warps `1/1/6`, active warps `16.58%`, eligible warps
`0.62`, long-scoreboard `16.43%`, issue rate `41.22%`, tensor duty `0.01
inst/cycle`, and `1,049,901.18` executed instructions. Static compiler metadata
is 255 registers, 8-byte stack, 0 local, and 21,504-byte shared allocation. No
compiler source was edited; B0.2 is the sole next packet.
