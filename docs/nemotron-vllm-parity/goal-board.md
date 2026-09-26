# Nemotron-H one-stack RLOO: goal board

Goal: one tinygrad stack (`exp`) that samples AND trains Nemotron 3 Nano 4B BF16 for DayCare's RLOO on Countdown
(`~/DayCare/research/rloo-llama-countdown.md`), and a pass/fail answer on its R5 gate. Parity numbers: `README.md`.

## Definition of done (Julian, 2026-09-25 evening; ordered by dependency)
1. **Finish phase 1**: RLOO R5-R7, then the LR sweep and the v2 run (does RL help at a real step size?).
2. **Kernel audit (measure first)**: ncu of vLLM/cuBLAS vs BoltBeam-emitted kernels on every Nemotron shape (decode
   M=8-128 and prefill M=512-8k): tile, stages, warp layout, stalls, % of roofline -> a replication spec per shape.
   Only head-to-head so far: ssm_in M=4096 BoltBeam 141 TF (56% of ~250 TF) vs cuBLAS 209 TF (84%). Scripts: bench/spec/.
3. **Training parity with vLLM**: RL sampling throughput at B=32-128 (now ~70%). Decode shapes first (thin GEMMs at
   ~36-38% of roofline are the biggest lever), plus fusion, attention length bucketing and one-graph priming
   (built: d6491e857/d97387365, held until 1 finishes). Gate: every decode kernel matches or beats vLLM's shape.
4. **Prefill parity + 10k batched prefill**: promote Nemotron's large-M shapes onto the tuned substrate (shape by
   shape; it was a shape problem), fix the batched 10k memory failure, test plain-bf16 prompt prefill (capture may make
   hi/lo unnecessary there). Now 1.65 s vs 0.37 s.
5. **Rotating-window RL** (capture at any block; bit-exact on the real 4B) as the next experiment.
6. **External write-up**: bit-exact RL on a hybrid Mamba model + mismatch localization (vLLM RFC #55524, NeMo-RL).

## Priority (2026-09-26 ~10:30, Julian)
PAUSED: kernel search, sampler prefix sharing. FIRST: update BoltBeam with the new techniques (strategy space derived from GPU facts: smem, cp.async/ldmatrix, SM count -> split-K, swizzle, occupancy/wave fit; record NV routes + evidence in BoltBeam; NCU collection/import/audit owned by BoltBeam, tinygrad keeps only a cubin+launch-spec exporter). Then (Julian): run a NEW SCAN with the updated BoltBeam (derived search space, BoltBeam-owned NCU) over every Nemotron shape (decode M 8-128, prefill M 512-8k), compare to vLLM and roofline, and use that scan to decide the next moves for goals 3-4 (instead of resuming the paused plans as-is).

## Rules
- Main loop manages; agents build. One owner per file; new workstreams go in new files.
- GPU: `~/scratchpad/bin/gpu-run time <cmd>` (exclusive, the only source of reported numbers) or `gpu-run check <cmd>`
  (shared, correctness). Both cap host RAM at 24 GB. Never run on the GPU outside it.
- Results go into the repo (commit + push to `exp`), never only into /tmp.
- Method: reverse engineer -> test -> solve. Every number is measured.
- Before any long run: a short real-model run proves the outputs (save -> reload -> exact check); trained weights get a raw backup before any formatted export. Never edit code a running job imports.

## Priority (2026-09-25 ~14:45)
1. I1 real-model smoke owns the GPU. 2. W8 lands if passing, then parks. 3. fp16 diagnosis resumes after the smoke. R5 gate stays exactly as predeclared.

## Workstreams (2026-09-25 afternoon)

| ID | Item | Owner | Files | Status | Gate |
|---|---|---|---|---|---|
| T0 | Profile one RLOO update | profiler agent | none | DONE: vendored tinygrad 0.13; 1 prompt x 8 on llama.cpp path: prefix 186 s, sample 228 s, features 443 s, parity recompute 115 s, **backward ~4000 s (~84%, CPU-bound in `_apply_map_to_tensors`, 16-token slices)**; est. ~5 h/update at 4 prompts |
| W8+CAP | Sampler tail capture + NaN guards + 2x prompt slots | sampler agent | `nemotron_h_sampler.py` + tests | **landed ad59f6338**: captured tail input reproduces sampled logprobs bit-exact (3016/3016 rows, 4B); NaN now raises; lane rings zeroed on handout; W8 GPU tok/s not measured |
| AUD | Goal 2: kernel audit | audit agent | kernel-audit-nemotron.md | **DONE c9798805d**: BoltBeam slower on every shape; decode GEMMs B=32 6.8 vs vLLM 4.8 ms, gap 5.4/10.7 ms at B=64/128; 10k prefill GEMMs 1110 vs 286 ms. Cause: register-staged 2-buffer, scalar LDS (no ldmatrix/cp.async), 48 KB LDS cap, ~50x bank conflicts; hi/lo 2x at prefill |
| PRIME | Sampler memory + throughput | sampler agent | sampler/prefill, schedule/memory.py (opt-in) | 7cc433d27, 3067908d6, 96009907e: shared arenas, one prompt Mamba state (-2.8 GB), flush in pool -> **B=128 cap 4096 fits** (32.2/32 GB, no headroom); W8 B=128 2448 tok/s (lanes 99% while queued, 70% whole run); step 36.2 ms whole ring. bf16 SSM state deferred (not needed; needs KL gate). Next: real RL shape (10k shared envelope prefix) -> prefix sharing |
| KS | Goal 3 decode GEMM search | kernel-search agent | codegen, dense candidates | **6348f8776 promoted 23/24 decode routes** (cp.async 2-6 stages, ldmatrix, swizzle, split-K<=14; gate 24/24): rollout step B=32 12.02->11.26, B=64 18.82->17.40, B=128 33.01->30.89 ms; per kernel B=32 within 3-32% of vLLM, output beats it; B>=64 blocked by hi/lo doubling rows (decision pending Julian); ncu on ssm_in next |
| I1 | One-stack RLOO v1 + v2 (predeclared) | integration agent | DayCare | **PHASE 1 CONCLUDED**: engineering gates all passed (bit-exact parity ~3.1M tokens, deterministic, lossless export). v1 (lr 2e-5) R5: 134 vs 126 /300, -2.7 pts CI[-9.0,+3.7] FAIL. Sweep picked 2e-4 (rule). v2 (lr 2e-4, KL 150x v1, train reward 0.481 vs 0.428) R5: 134 vs 132 /300, -0.7 pts CI[-7.3,+6.0] FAIL (DayCare 94149d3). Step size not the bottleneck at this data scale; next RL experiment (data scale / placement / reward) awaits Julian |
| R3 | Smoke on the llama.cpp loop | after T0 | `~/DayCare` | pending | R3 in rloo-llama-countdown.md |
| CORE | tinygrad core bugs | core-bugs agent | tinygrad core | DONE: ad956809e, 31050972c, **b5ef62d16 NaN race = relaxed QMD membar across 2 compute queues** (diag2 before 3/3 dirty, after 6/6 clean; 32-lane gate 5/5 @512, 2/2 @4096). Throughput cost of restored membars not measured |
| W8b | Refill bench + re-timing after b5ef62d16 | sampler agent | sampler | ff4f033ec: refill 1.70x (B=32) / 1.58x (B=64) vs fixed; 99% active lanes while queued (gate met); whole-run 0.71/0.58 of B/step (drain + slow priming 0.67 s/256 tok); b5ef62d16 costs +5-7%/step (B=8 7.58, B=64 18.75, B=128 32.03 ms); next: rollout attention length bucketing, priming cost |
| WIN | Rotating-window RL capture/replay (attention at 12,17,24,32) | sampler agent | sampler | d8924c8d9; **real 4B after b5ef62d16: k=41/36/30/12 all 1480/1480 bit-exact on first replay** |
| TIM | Localize train/inference reduction-order mismatch (Mamba scan vs recurrent step, split-KV attention, row-count kernel choice, RMSNorm); invariance plan | fp16 agent | `~/scratchpad/fp16/` -> docs/.../train-inference-mismatch-localization.md | leave-one-out: source = row-count-dependent projection kernels, amplifier = bf16 K/V/q/P at the 4 attention layers; mm+kv32 cuts gap ~250x (64 steps); 2048-step prod: mean 4.2e-4, max 2.0e-2 | site table with gap reduction when unified |
| RS | How vLLM/verl/NeMo-RL correct rollout/trainer mismatch (TIS etc.) | research agent | none | running | ranked fixes + hypotheses |
| W5/W10 | Fusion, per-B graphs | - | - | parked | - |
| P | 10k prefill 1.65 s vs vLLM 0.37 s | - | - | parked (about 10% of a group) | - |

## Done today
- 257d19bd2 merge of self-training + nvidia-bringup into `exp`; 4aafc5e7c prefill per-block-kind graphs (10k 1.80 -> 1.65 s).
