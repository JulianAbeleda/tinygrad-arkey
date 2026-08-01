# Qwen3-8B multi-role buffer2 candidate set for apple_m4_10c

`candidate-set.json` is the deterministic BoltBeam expansion of the admitted
gate/up buffer2 schedule into four independently hashed exact workloads for the
`apple_m4_10c` target: `ffn_gate_up`, `ffn_down`, `attn_qo`, and `attn_kv`.

Minted by BoltBeam's decoupled builder (`build_qwen3_8b_buffer2_candidate_set`
with `target_id="apple_m4_10c"`, exp branch, 2026-08-01). The profile identity
is `qwen3_8b_q4k_m_m4_10c`, the target part of each identity is
`Metal:m4_10c:wave32`, and no `gfx1100` string appears anywhere in the artifact.

This is a minted qualification artifact, not a completed promotion decision.
GPU execution, admission, and timing remain the authority and are pending via
the M1e Metal precontract lane (`extra/llm_research/prefill/metal_precontract_lane.py`).
