# TG4 New-Profile Opener -- verdict: **TG4_PASS_NEW_PROFILE_OPENER_READY**

The opener runs the 9-step new-profile flow (model/shape, quant-mix, GPU features, route census, ceiling, wall-share, declared rows, inherited do_not_search, first candidate) from a profile descriptor.

## Acceptance

- **Regenerates the existing Qwen3-8B/gfx1100 profile**: True (promoted/shipped routes: ['decode_q4k_g3_generated', 'decode_q6k_coop_shipped', 'decode_attention_owned_two_kernel', 'prefill_pipe_role_selective_default']; first action: NO_ACTION_ALL_GEMV_ROLES_PROMOTED)
- **Drafts a new gfx1100 target without route flags**: True (`qwen3_8b_q5_k_m_gfx1100`; quants resolved ['Q5_K', 'Q6_K', 'fp16']; open rows ['attn_qo__Q5_K', 'ffn_down__Q5_K', 'ffn_gate_up__Q5_K', 'lm_head__Q6_K'])
- **Refuses incomplete profiles**: True (model=True, quant=True, gpu=True)

## Drafted profile per-role weight-read ceiling (HBM speed-of-light)

| role | quant | N | K | weight bytes | roofline ms @ measured bw |
|---|---|---:|---:|---:|---:|
| attn_qo | Q5_K | 4096 | 4096 | 11534336 | 0.0141 |
| ffn_gate_up | Q5_K | 12288 | 4096 | 34603008 | 0.0422 |
| ffn_down | Q5_K | 4096 | 12288 | 34603008 | 0.0422 |
