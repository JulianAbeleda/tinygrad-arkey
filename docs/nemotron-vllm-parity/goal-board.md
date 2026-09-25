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
| I1 | One-stack RLOO loop + TIS (C=2) | integration agent | `~/DayCare/daycare/nursery/rloo_tinygrad.py` | **R3 PASSED** (DayCare 5356e60): bit-exact parity 4096/4096, loss moves, in-place adapter, 84 s/update warm (512-token smoke; rewards 0 at that cap). 7b5f367 compiled tail slices (~80x/token). **R4 running** (rloo-tinygrad-r4-001; update 1: reward 0.281, parity 0, 225 s) |
| R3 | Smoke on the llama.cpp loop | after T0 | `~/DayCare` | pending | R3 in rloo-llama-countdown.md |
| CORE | tinygrad core bugs | core-bugs agent | tinygrad core | DONE: ad956809e, 31050972c, **b5ef62d16 NaN race = relaxed QMD membar across 2 compute queues** (diag2 before 3/3 dirty, after 6/6 clean; 32-lane gate 5/5 @512, 2/2 @4096). Throughput cost of restored membars not measured |
| WIN | Rotating-window RL capture/replay (attention at 12,17,24,32) | sampler agent | sampler | d8924c8d9; real 4B: k=41/36 bit-exact 1480/1480; k=30/12 first replay mismatches (same runtime bug as CORE) |
| TIM | Localize train/inference reduction-order mismatch (Mamba scan vs recurrent step, split-KV attention, row-count kernel choice, RMSNorm); invariance plan | fp16 agent | `~/scratchpad/fp16/` -> docs/.../train-inference-mismatch-localization.md | leave-one-out: source = row-count-dependent projection kernels, amplifier = bf16 K/V/q/P at the 4 attention layers; mm+kv32 cuts gap ~250x (64 steps); 2048-step prod: mean 4.2e-4, max 2.0e-2 | site table with gap reduction when unified |
| RS | How vLLM/verl/NeMo-RL correct rollout/trainer mismatch (TIS etc.) | research agent | none | running | ranked fixes + hypotheses |
| W5/W10 | Fusion, per-B graphs | - | - | parked | - |
| P | 10k prefill 1.65 s vs vLLM 0.37 s | - | - | parked (about 10% of a group) | - |

## Done today
- 257d19bd2 merge of self-training + nvidia-bringup into `exp`; 4aafc5e7c prefill per-block-kind graphs (10k 1.80 -> 1.65 s).
