# NVIDIA pp512 post-substrate S0 authority freeze (2026-08-29)

Status: **STOP**

The fresh tinygrad composed `unroll-4 + Q4-V` arm completed the required 9
warmup-excluded samples, but the deep-20 recurrent gate failed at cycle 17:
`max_abs=0.07578897476196289`, `mean_abs=0.02062172256410122`. The greedy
token remained `198` and outputs were finite, but replay was not exact.

The same arm measured `69.243034 ms` minimum and `69.283518 ms` median,
outside the frozen `67.235719 ms +/- 1%` authority band. Therefore llama.cpp
was not run and no cross-runtime wall claim is admitted.

Observed census before the failure: `198` canonical weights, `198` compiler
mains, `198` Q8 producers, `72` gate/up, `36` K, `72` Q/O, `18` V, zero
weight-copy kernels, zero partial workspace, and `54` remaining FP16
overlays. Hardware was RTX 5090 `sm_120`, driver `595.84`; PROFILE was off.

Evidence: `docs/task_workflow/evidence/nv-prefill-post-substrate-authority-20260829/`

Runner: `extra/llm_research/prefill/nv_prefill_post_substrate_authority.py`

Next packet: `null`.
