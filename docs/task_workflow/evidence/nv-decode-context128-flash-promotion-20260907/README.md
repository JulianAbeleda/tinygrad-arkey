# Ordinary generated Flash selection at context 128

A fresh SDPA/Flash/SDPA context128 bracket gives SDPA midpoint5.3700731ms median and5.2334655ms minimum versus generated S48 Flash4.3663465/4.2483540ms. Flash wins1.0037266ms median and0.9851115ms minimum, clearing both gates. Ordinary auto selection now starts at context128. Set `FLASH_DECODE_THRESHOLD=512` to restore the previous threshold; explicit route overrides are unchanged.

S24 was tested once because halving partial/combine traffic could plausibly recover the remaining llama gap. It regressed to4.3926811ms median versus S48's4.3663465 and is rejected. The final clean-process ordinary run selects `route=flash`, generated `rollout_jit_flash` with418 programs, and records token/GPU/census evidence. Context128 still has a smaller kernel/endpoint gap to matched llama, so this policy promotion closes only the larger SDPA selection deficit.
