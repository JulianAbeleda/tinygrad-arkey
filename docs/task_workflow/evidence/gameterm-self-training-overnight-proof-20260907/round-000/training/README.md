# Qwen3-8B tinygrad LoRA MVP

This is the first real-model training gate: Qwen3-8B Q4_K_M remains frozen while tinygrad trains an output LoRA on NVIDIA.

- status: `pass`
- held-out loss: `16.242996` -> `15.052478`
- adapter changed/reloaded exactly: `True` / `True`
- model file unchanged: `True`
- steps: `1` in `4.027s`

Completion scope is `all` and only the output projection is adapted. Transformer-block QLoRA remains a later stage.
