# Qwen3-8B multi-role buffer2 candidate set

`candidate-set.json` is the deterministic BoltBeam expansion of the admitted
gate/up buffer2 schedule into four independently hashed exact workloads:
`ffn_gate_up`, `ffn_down`, `attn_qo`, and `attn_kv`.

Tinygrad admission and GPU execution remain the authority. The three new role
execution reports are recorded after full-output nonconstant comparisons. The
existing gate/up authority remains in
`../anchor-ffn-gate-up/two-buffer-candidate-v1/` and is not duplicated here.

This directory does not by itself establish a whole-model performance result.
Kernel timing, combined route census, parity, and pinned context sweeps are
separate acceptance gates.
