# Qwen3-8B multi-role buffer2 candidate set for nvidia_sm120

`candidate-set.json` is the deterministic BoltBeam expansion of the admitted
gate/up buffer2 schedule into four independently hashed exact workloads for the
`nvidia_sm120` target: `ffn_gate_up`, `ffn_down`, `attn_qo`, and `attn_kv`.

Minted by BoltBeam's decoupled builder (`build_qwen3_8b_buffer2_candidate_set`
with `target_id="nvidia_sm120"`, main branch, 2026-08-01). The profile identity
is `qwen3_8b_q4k_m_sm120`, the target part of each identity is
`CUDA:sm120:wave32`, and no `gfx1100` string appears anywhere in the artifact.

This is a minted qualification artifact, not a completed promotion decision.
GPU execution, admission, and timing remain the authority and are pending on
the RTX 5090 that is present in the bring-up environment.
