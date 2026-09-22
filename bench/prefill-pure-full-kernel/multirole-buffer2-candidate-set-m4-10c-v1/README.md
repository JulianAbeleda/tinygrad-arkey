# Qwen3-8B multi-role buffer2 candidate set for apple_m4_10c

`candidate-set.json` is the deterministic BoltBeam expansion of the admitted
gate/up buffer2 schedule into four independently hashed exact workloads for the
`apple_m4_10c` target: `ffn_gate_up`, `ffn_down`, `attn_qo`, and `attn_kv`.

Minted through the typed path (2026-08-01, T6 of
`target-schedule-derivation-scope-20260801.md`): the schedule and
`static_constraints` come from `derive_target_schedule` on the Metal m4_10c
capability row (`extra/llm_research/mint_typed_candidate_template.py`), not a
clone of AMD's. The schedule carries Metal vocabulary
(`metal_simdgroup_matrix_*` fragment, `wmma_f32_8x8x8_f16`,
`max_lds_bytes: 32768`, null waitcnt), and BoltBeam's decoupled builder
(`build_qwen3_8b_buffer2_candidate_set` with `target_id="apple_m4_10c"` and
`--schedule-template`) stamps the four exact workloads onto it. The profile
identity is `qwen3_8b_q4k_m_m4_10c`, the target part of each identity is
`Metal:m4_10c:wave32`, and no `gfx1100` string appears anywhere in the artifact.

This is a minted qualification artifact, not a completed promotion decision.
Admission-level outcomes are recorded in
`docs/bringing-up-a-new-target-20260731.md`: the buffer2 geometry is rejected
by `capability_lds` (40960 > 32768) and the single-buffer shape admits; the
full GPU run happens on a Mac via the M1e precontract probe lane
(`extra/llm_research/prefill/precontract_probe_lane.py`).
