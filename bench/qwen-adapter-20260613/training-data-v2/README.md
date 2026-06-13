# Adapter Signal Dataset V2

This dataset gives the output-LoRA path a real supervised signal: ordinary
question prompts must answer with the one-token sentinel `OK`. The base
model should fail the held-out exact-match rollout; a trained adapter should
learn the override. This is a behavior-change plumbing gate, not a capability
benchmark.

- rows: `60`
- train rows: `48`
- eval rows: `12`
- target: `OK`
