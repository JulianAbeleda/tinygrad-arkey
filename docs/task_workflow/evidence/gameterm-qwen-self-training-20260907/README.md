# GameTerm Qwen self-training MVP — 2026-09-07

This evidence records a controlled Qwen3-8B Q4_K_M output-LoRA training experiment on the RTX 5090 and evaluates the frozen base and qualified adapter through the real `gameterm-harness` binary.

## Result

- GameTerm exact completion score: **0/4 base -> 1/4 adapted**.
- Offline held-out first-token loss: **13.283194 -> 1.631391**.
- Offline held-out first-token accuracy: **0/4 -> 1/4**.
- The base GGUF SHA-256 stayed `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
- The saved adapter reloaded byte-exactly according to digest `caf4c689cbc9a5b33102a0cd3a20f8d0408e3a8b5ce1997e577915b91e723d27`.

The four evaluation prompts use wording absent from the eight training prompts. Every score row has raw normalized GameTerm events under `base/events` or `adapted/events`. Infrastructure failures are not counted as model misses; both final runs completed all turns without a harness failure.

## Scope

This proves the tinygrad train/save/load/serve/GameTerm-evaluate loop can change held-out harness behavior. The labels are synthetic and supplied by the host verifier, so the model is not grading itself. This adapter trains only the first completion token and the output projection. `WITHHOLD` and `OBSERVE` each require three tokenizer tokens, which bounds this MVP and motivates the autonomous multi-token continuation.

The 96-step comparison is retained because it lowered loss further but did not improve accuracy. It was rejected rather than silently selected.

## Reproduction settings

Training used `DEV=NV`, rank 8, alpha 8, learning rate 0.0005, 24 steps, max context 64, and eval split cadence 3. Serving used `--default-max-tokens 1 --prefill-chunk-size 64`. Evaluation used `/home/ubuntu/gameterm_beta/target/debug/gameterm-harness` with the system prompt stored in the training summary.
