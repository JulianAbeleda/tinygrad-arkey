# Rotating-window RL with captured activations: idea and prior art (2026-09-25)

## Idea (Julian)
Train only a window of blocks [k, k+w) per RL update and rotate the window across updates. The sampler captures the
exact fp32 input to block k, so the trainer runs only the window (bit-identical below it; memory and compute scale
with the window). Every on-policy update re-samples with the current full model, so captured inputs are always fresh.
Enabled by the one stack (capture at any block; no weight sync). Phase 1 (last block only) is predeclared and runs
first; this is the follow-up experiment.

## Prior art (web check; sources' claims vs our inference kept separate)
- "Is One Layer Enough? Training a Single Transformer Layer Can Match Full-Parameter RL Training", arXiv 2607.01232:
  RL gains concentrate in middle layers; one middle layer or ~10/36 matches or beats full-parameter RL
  (e.g. 69.1% vs 66.4%). Fixed layer per run, full forward/backward each step, no activation caching, no rotation,
  no rollout/train bit-identity.
- BAdam, arXiv 2404.02827: block-coordinate Adam, rotates blocks, SFT only; notes the waste of backpropagating
  through inactive deeper blocks.
- LISA, arXiv 2403.17919: importance-sampled layer unfreezing per iteration, SFT only.
- Egeria, arXiv 2201.06227: freezes converged lower layers and caches their activations; supervised, monotonic freeze.
- Thinking Machines, "LoRA Without Regret" (2025): for RL, rank-1 LoRA matches full fine-tuning (~1 bit/episode);
  LoRA on all layers (esp. MLP) recommended.
- No source found combining RL + sampler-captured hidden states as trainer input + a rotating window.

## Design implications
1. Prioritize middle blocks in the rotation (2607.01232).
2. Ablate narrow windows (1-2 blocks); low capacity per update should suffice (LoRA Without Regret).
3. State explicitly that re-sampling every update removes BAdam-style stale frozen context.
4. Windows containing an attention block need fp32 K/V (see train-inference-mismatch-localization.md).
5. Risk for phase 1: training only the last block may under-deliver relative to middle layers.
