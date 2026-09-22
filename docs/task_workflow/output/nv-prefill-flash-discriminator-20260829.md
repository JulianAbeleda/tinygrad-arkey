# NVIDIA pp512 prefill Flash discriminator

## Source/trace extraction

The exact cross-runtime accounting identifies Flash as 36 tinygrad attention
main/reduction pairs versus 36 llama score and 36 llama reduction launches.
Tinygrad launches 1,449 total kernels and llama 1,186. Flash active time is
3.328096 ms tinygrad versus 1.657447 ms llama, an active debt of 1.670649 ms;
device idle is only 0.307232 ms versus 0.201442 ms, so overlap is not the
primary discriminator.

The llama source map records grouped Q/KV attention, causal masking, and a
layer-35 requested-row gather. The exact launch/accounting artifacts are under
`docs/task_workflow/evidence/nv-prefill-exact-cross-runtime-trace/llama/`;
`cross-runtime-accounting.json` is the machine-readable authority.

## Gate status

The direct oracle is retained at `evidence/nv-prefill-flash-20260829/oracle.npz`.
The current NV model route names the finalized program
`nv_sm120_q16_grid_hd128_loop_attention`; its source is installed through the
model binding rather than the AMD `FlashPrefillAttentionSpec`. The AMD
`prefill_flash_e2e_parity.py` harness is not valid NVIDIA evidence and is not
used. A direct invocation of the AMD semantic wrapper on NV was rejected by
the target guard before launch, so that traceback is not a Flash wall result.
The NV finalized PROGRAM still needs a dedicated ProgramInfo adapter for
separate score/reduction timing and direct fixture binding.

## Decision: STOP pending exact pp512 gate

The retained trace supports topology testing (CTA ownership, head/tile
partition, causal reduction placement, and KV order) but does not identify a
winning mechanism. No fusion, overlap, or Stream-K investment is authorized.
