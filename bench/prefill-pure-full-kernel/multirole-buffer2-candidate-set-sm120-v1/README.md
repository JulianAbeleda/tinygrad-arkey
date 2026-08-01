# Qwen3-8B multi-role buffer2 candidate set for nvidia_sm120

`candidate-set.json` is the deterministic BoltBeam expansion of the admitted
gate/up buffer2 schedule into four independently hashed exact workloads for the
`nvidia_sm120` target: `ffn_gate_up`, `ffn_down`, `attn_qo`, and `attn_kv`.

Minted through the typed path (2026-08-01, T6 of
`target-schedule-derivation-scope-20260801.md`): the schedule and
`static_constraints` come from `derive_target_schedule` on the NV sm120
capability row (`extra/llm_research/mint_typed_candidate_template.py`), not a
clone of AMD's. The schedule carries NV vocabulary (`cuda_mma_*` fragment,
`wmma_f32_8x16x16_f16`, `max_lds_bytes: 49152`, null waitcnt -- no `rdna3_*`,
no AMD `lgkm` value), and BoltBeam's decoupled builder
(`build_qwen3_8b_buffer2_candidate_set` with `target_id="nvidia_sm120"` and
`--schedule-template`) stamps the four exact workloads onto it. The profile
identity is `qwen3_8b_q4k_m_sm120`, the target part of each identity is
`CUDA:sm120:wave32`, and no `gfx1100` string appears anywhere in the artifact.

This is a minted qualification artifact, not a completed promotion decision.
The mint now ADMITS through the canonical precontract lane (the old
`capability_tc` skip is gone) and fails in the child compile only at the two
known lowering boundaries (accumulator carrier vec(8) vs vec(4);
operand lane-layout derivation), recorded in
`docs/bringing-up-a-new-target-20260731.md`.
