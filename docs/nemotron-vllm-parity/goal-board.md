# Nemotron-H one-stack RLOO: goal board

Goal: one tinygrad stack (`exp`) that samples AND trains Nemotron 3 Nano 4B BF16 for DayCare's RLOO on Countdown
(`~/DayCare/research/rloo-llama-countdown.md`), and a pass/fail answer on its R5 gate. Parity numbers: `README.md`.

## Rules
- Main loop manages; agents build. One owner per file; new workstreams go in new files.
- GPU: `~/scratchpad/bin/gpu-run time <cmd>` (exclusive, the only source of reported numbers) or `gpu-run check <cmd>`
  (shared, correctness). Both cap host RAM at 24 GB. Never run on the GPU outside it.
- Results go into the repo (commit + push to `exp`), never only into /tmp.
- Method: reverse engineer -> test -> solve. Every number is measured.

## Priority (2026-09-25 ~14:45)
1. I1 real-model smoke owns the GPU. 2. W8 lands if passing, then parks. 3. fp16 diagnosis resumes after the smoke. R5 gate stays exactly as predeclared.

## Workstreams (2026-09-25 afternoon)

| ID | Item | Owner | Files | Status | Gate |
|---|---|---|---|---|---|
| T0 | Profile one RLOO update | profiler agent | none | DONE: vendored tinygrad 0.13; 1 prompt x 8 on llama.cpp path: prefix 186 s, sample 228 s, features 443 s, parity recompute 115 s, **backward ~4000 s (~84%, CPU-bound in `_apply_map_to_tensors`, 16-token slices)**; est. ~5 h/update at 4 prompts |
| W8 | Slot refill: default slots 2*batch//rows+2 (~99% active while queued, CPU sim) | sampler agent | `nemotron_h_sampler.py`, `test_nemotron_h_sampler_refill.py` | parked uncommitted in ~/wt/w8; needs one GPU bench (B=32/64) |
| I1 | One-stack RLOO loop + TIS (C=2) | integration agent | `~/DayCare/daycare/nursery/rloo_tinygrad.py` + its test | **GPU PRIORITY**; DayCare 8d6e7ff loop, e691a13 TIS, c7f6955 predeclared; real-model smoke (R3) queued; watch backward cost | 2 updates on the 4B: loss moves, parity <=0.1 nats, no llama.cpp |
| R3 | Smoke on the llama.cpp loop | after T0 | `~/DayCare` | pending | R3 in rloo-llama-countdown.md |
| F16 | Why fp16 fails the consistency gate; mixed fp16/hi-lo scheme | fp16 agent | `~/scratchpad/fp16/` (no product code) | running | same-numerics sampler vs recompute mean <=1e-3, max <=1e-2 |
| RS | How vLLM/verl/NeMo-RL correct rollout/trainer mismatch (TIS etc.) | research agent | none | running | ranked fixes + hypotheses |
| W5/W10 | Fusion, per-B graphs | - | - | parked | - |
| P | 10k prefill 1.65 s vs vLLM 0.37 s | - | - | parked (about 10% of a group) | - |

## Done today
- 257d19bd2 merge of self-training + nvidia-bringup into `exp`; 4aafc5e7c prefill per-block-kind graphs (10k 1.80 -> 1.65 s).
