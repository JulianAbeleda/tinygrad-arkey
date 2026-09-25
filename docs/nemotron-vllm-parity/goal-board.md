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
| W8+CAP | Sampler tail capture + NaN guards + 2x prompt slots | sampler agent | `nemotron_h_sampler.py` + tests | **landed ad59f6338**: captured tail input reproduces sampled logprobs bit-exact (3016/3016 rows, 4B); NaN now raises; lane rings zeroed on handout; W8 GPU tok/s not measured |
| I1 | One-stack RLOO loop + TIS (C=2) | integration agent | `~/DayCare/daycare/nursery/rloo_tinygrad.py` | **GPU PRIORITY**: DayCare a9a821e uses the capture; 0c207c3 logs R3 attempt 1 (parity stop) honestly; R3 attempt 2 (smoke-003) queued |
| R3 | Smoke on the llama.cpp loop | after T0 | `~/DayCare` | pending | R3 in rloo-llama-countdown.md |
| CORE | tinygrad core bugs | core-bugs agent | tinygrad core | ad956809e backward: one grad per aliased base; 31050972c liveness: read-modify-write kernel reads its output slot (Adam NaN candidate); NaN-with-LoRA repro under test |
| WIN | Rotating-window RL: capture at any block + replay window bit-exact (attention at 12,17,24,32) | sampler agent | sampler | d8924c8d9 landed (tiny: 55/55 bit-exact, all k); real-4B check queued; prior art 5a7c2df9c |
| TIM | Localize train/inference reduction-order mismatch (Mamba scan vs recurrent step, split-KV attention, row-count kernel choice, RMSNorm); invariance plan | fp16 agent | `~/scratchpad/fp16/` -> docs/.../train-inference-mismatch-localization.md | leave-one-out: source = row-count-dependent projection kernels, amplifier = bf16 K/V/q/P at the 4 attention layers; mm+kv32 cuts gap ~250x (64 steps); 2048-step prod: mean 4.2e-4, max 2.0e-2 | site table with gap reduction when unified |
| RS | How vLLM/verl/NeMo-RL correct rollout/trainer mismatch (TIS etc.) | research agent | none | running | ranked fixes + hypotheses |
| W5/W10 | Fusion, per-B graphs | - | - | parked | - |
| P | 10k prefill 1.65 s vs vLLM 0.37 s | - | - | parked (about 10% of a group) | - |

## Done today
- 257d19bd2 merge of self-training + nvidia-bringup into `exp`; 4aafc5e7c prefill per-block-kind graphs (10k 1.80 -> 1.65 s).
