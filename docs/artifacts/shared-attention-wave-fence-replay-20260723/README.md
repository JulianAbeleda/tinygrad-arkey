# M10e4b wave-fence replay delta

This record compares the single-wave LDS fence against the published
workgroup-barrier replay matrix using the identical benchmark closure:
prebuilt mask, captured `TinyJit`, three warmups, ten synchronized replay
samples, and no allocation, compilation, or host copy inside the samples.

| Route | KV | Barrier median | Wave-fence median | Latency delta |
|---|---:|---:|---:|---:|
| 8B | 512 | 0.5444895 ms | 0.5444015 ms | -0.016% |
| 14B | 512 | 0.5776465 ms | 0.5787870 ms | +0.197% |
| 8B | 4096 | 3.4831830 ms | 3.4764935 ms | -0.192% |
| 14B | 4096 | 3.5523230 ms | 3.5480730 ms | -0.120% |

Negative latency delta is faster. All changes are below 0.2%, so the measured
performance result is neutral rather than a claimed speedup. Full numeric
prechecks retained the published errors: `6.103515625e-05` at KV512 and
`1.9073486328125e-06` at KV4096.

The change is retained for compiler and resource correctness. The admitted
grid proves one wave per workgroup, generated HIP and ISA contain one ordered
`s_waitcnt(64519)` and no workgroup barrier, and VGPR/SGPR/LDS/spill resources
are unchanged. The conservative workgroup barrier remains the fallback for
unknown or multi-wave geometry.
