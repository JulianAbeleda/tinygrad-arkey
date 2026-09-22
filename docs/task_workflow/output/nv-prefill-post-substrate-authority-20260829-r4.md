# NVIDIA pp512 post-substrate S0 authority freeze, r4

Status: **PASS**

The F1 exact route is restored byte-for-byte: `QK_PRIMITIVE=1`, compiler
Q4/K/QO pp512 flags, `NV_COMPILER_Q4_IMMA_UNROLL=4`, and the required HCQ
safe-cut settings. `NV_Q4_IMMA_PP512` and `NV_UNROLL` were explicitly unset.

Two fresh candidate R9 brackets passed bitwise deep-20, finite logits, token
`198`, and the exact 198-role census. Bracket medians were `67.251540 ms`
and `67.254339 ms` (delta `0.002799 ms`); settled authority is `67.2529395 ms`
median and `67.099104 ms` minimum. This reproduces the prior authority within
noise; no route/config regression is present. Fresh llama R9 is retained in
the r4 evidence directory.

Evidence: `docs/task_workflow/evidence/nv-prefill-post-substrate-authority-20260829-r4/`

No downstream packet is authorized by this S0 artifact (`next_packet=null`).
