# QK Flywheel Triage Adapter V0 Attempt

This records the first Phase 3.2 adapter-training attempt after the Phase 3.0
diagnostic and Phase 3.1 SFT export.

Input:

- `bench/amd-decode-flywheel-proof-20260614/triage-sft-v0/adapter-input.jsonl`

Attempted candidates:

- `last1_ffn` suffix-cache LoRA, rank `4`, alpha `8`, generated training mode.
- `last1_ffn` suffix-cache LoRA, rank `4`, alpha `8`, baseline training mode.

Both attempts were terminated after repeated 30 second polls with no stdout and
no adapter artifact. The generated-mode attempt is not the prior working
internal-adapter training path. The baseline-mode attempt matches the prior
working path more closely, but this kernel-triage dataset has much longer
prompts and did not produce a practical first-candidate loop in this run.

Conclusion:

- status: `blocked`
- result: `training_path_latency_blocked`

This does not prove or disprove the flywheel. It means Phase 3.2 needs a
smaller/progress-reporting adapter candidate, a prompt-compression pass, or a
more practical adapter training loop before a held-out adapter rollout can be
scored.
