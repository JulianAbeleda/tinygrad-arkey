# Nemotron-H one-stack RLOO: goal board

Goal: one tinygrad stack (`exp`) that samples AND trains Nemotron 3 Nano 4B BF16 for DayCare's RLOO on Countdown
(`~/DayCare/research/rloo-llama-countdown.md`), and a pass/fail answer on its R5 gate. Parity numbers: `README.md`.

## Rules
- Main loop manages; agents build. One owner per file; new workstreams go in new files.
- GPU: `~/scratchpad/bin/gpu-run time <cmd>` (exclusive, the only source of reported numbers) or `gpu-run check <cmd>`
  (shared, correctness). Both cap host RAM at 24 GB. Never run on the GPU outside it.
- Results go into the repo (commit + push to `exp`), never only into /tmp.
- Method: reverse engineer -> test -> solve. Every number is measured.

## Workstreams (2026-09-25 afternoon)

| ID | Item | Owner | Files | Status | Gate |
|---|---|---|---|---|---|
| T0 | Profile one RLOO update; which tinygrad the trainer uses | profiler agent | none (scratchpad only) | running | stage table + ranked levers |
| W8 | Continuous batching / slot refill in the sampler | sampler agent | `tinygrad/llm/nemotron_h_sampler.py`, `test/unit/test_nemotron_h_sampler*.py` | starting | >=90% active slots on a skewed length mix; parity holds |
| I1 | One-stack RLOO loop: tinygrad sampler -> recompute -> LoRA update in place -> sample | integration agent | `~/DayCare/daycare/nursery/rloo_tinygrad.py` + its test | starting | 2 updates on the 4B: loss moves, parity <=0.1 nats, no llama.cpp |
| R3 | Smoke on the llama.cpp loop | after T0 | `~/DayCare` | pending | R3 in rloo-llama-countdown.md |
| F16 | Why fp16 fails the consistency gate; mixed fp16/hi-lo scheme | fp16 agent | `~/scratchpad/fp16/` (no product code) | running | same-numerics sampler vs recompute mean <=1e-3, max <=1e-2 |
| RS | How vLLM/verl/NeMo-RL correct rollout/trainer mismatch (TIS etc.) | research agent | none | running | ranked fixes + hypotheses |
| W5/W10 | Fusion, per-B graphs | - | - | parked | - |
| P | 10k prefill 1.65 s vs vLLM 0.37 s | - | - | parked (about 10% of a group) | - |

## Done today
- 257d19bd2 merge of self-training + nvidia-bringup into `exp`; 4aafc5e7c prefill per-block-kind graphs (10k 1.80 -> 1.65 s).
