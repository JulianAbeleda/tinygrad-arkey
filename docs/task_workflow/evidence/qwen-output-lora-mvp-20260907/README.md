# Qwen3-8B tinygrad LoRA MVP

This is the first real-model training gate: Qwen3-8B Q4_K_M remains frozen while tinygrad trains an output LoRA on NVIDIA.

- status: `pass`
- held-out loss: `12.734298` -> `11.476225`
- adapter changed/reloaded exactly: `True` / `True`
- model file unchanged: `True`
- steps: `12` in `31.023s`

The bounded scope is deliberate: one completion token and the output projection qualify the real GGUF/model/gradient/optimizer/artifact path. Multi-token completion masking and transformer-block QLoRA remain the next stage.
