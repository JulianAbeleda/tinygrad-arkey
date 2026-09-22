# Q6 owner SASS instruction ledger

This is a binary-first comparison of the current generated owner kernel and
the pinned llama Q6 MMQ oracle. Counts below come from `nvdisasm -c`, not from
the Python/UOp source. The generated capture is `/tmp/current_q6_owner.cubin`
with entry `nv_generated_q6k_streamk_owner_partials`; the oracle is
`evidence/nv-packed-q4k-q8-llama-extracted-20260830/q6k-mmq-dense.sm_120a.cubin`.

## Resource and instruction census

| metric | generated owner | llama oracle | ratio / delta |
|---|---:|---:|---:|
| registers | 255 | 255 | equal ceiling |
| stack frame | 512 B | 72 B | +440 B |
| shared reservation | 1024 B | 1024 B | equal in cubin metadata |
| IMMA.16816.S8.S8 | 256 | 512 | 0.50x |
| LDSM | 32 | 64 | 0.50x |
| LDG (all) | 277 | 210 | +67 |
| STG (all) | 66 | 128 | -62 |
| LDS (all) | 208 | 488 | -280 |
| STS (all) | 24 | 142 | -118 |
| LDL | 155 | 31 | +124 |
| STL | 183 | 29 | +154 |
| BAR | 4 | 9 | -5 |
| BRA | 7 | 5 | +2 |
| BSSY/BSYNC | 3 / 3 | 64 / 64 | fewer structured regions |

The `IMMA` and `LDSM` counts are whole-entry counts. The oracle has two
IMMA/LDSM instructions per corresponding generated instruction because its
entry contains two output-row paths. Therefore they are not evidence that the
generated arithmetic is missing half the mathematical work without first
normalizing the row-path topology.

## K=256 alignment

The representative Q6 shape has 48 K=256 work units per output tile. A raw
whole-entry division gives:

| per K=256 unit (raw entry average) | generated | llama |
|---|---:|---:|
| IMMA | 5.33 | 10.67 |
| LDSM | 0.67 | 1.33 |
| LDL | 3.23 | 0.65 |
| STL | 3.81 | 0.60 |

These averages are only for scale; the two binaries have different CTA
topologies and must not be treated as a performance model by themselves.

## Conclusion

The first measured convergence target is spill/stack traffic, not fragment
load spelling. Both entries allocate 255 registers, but the generated entry
uses a 512-byte stack frame and emits approximately 5x as many local loads and
6x as many local stores. This is the strongest direct binary explanation for
the generated kernel's slowdown and should be the next optimization gate.

The next experiment must reduce `LDL/STL` and stack bytes while preserving
exact output and the owner/fixup ABI. A candidate only qualifies if the
post-compile SASS census improves those fields; source-level load counts alone
do not qualify it.

## Reproduction

```sh
NV=.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm
$NV -c /tmp/current_q6_owner.cubin > /tmp/current_q6_owner.nvdisasm
$NV -c docs/task_workflow/evidence/nv-packed-q4k-q8-llama-extracted-20260830/q6k-mmq-dense.sm_120a.cubin > /tmp/llama_q6.nvdisasm
/usr/local/cuda-13.2/bin/cuobjdump --dump-resource-usage /tmp/current_q6_owner.cubin
/usr/local/cuda-13.2/bin/cuobjdump --dump-resource-usage docs/task_workflow/evidence/nv-packed-q4k-q8-llama-extracted-20260830/q6k-mmq-dense.sm_120a.cubin
```
