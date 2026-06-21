# Prefill DEFAULT-POLICY evaluation — should PREFILL_V2 / PREFILL_CONCRETE_KV be default-on?

Date: 2026-06-20. Repo `/home/ubuntu/tinygrad-arkey`. GPU gfx1100 **RX 7900 XTX 24 GB**. Model Qwen3-8B-Q4_K_M.
Harness: `extra/qk_prefill_default_policy_eval.py`. Data: `bench/qk-prefill-default-policy-eval/result.json`.
This is a **policy/integration** evaluation (e2e TTFT, load, VRAM, amortization) — kernel throughput is settled.

## Method & honesty caveats

Fresh subprocess per (mode, prompt length) so COLD (first generation, incl. compile) is cleanly separated from
WARM (a 2nd divergent-prompt generation, jits reused). VRAM = tinygrad-tracked `mem_used_per_device` (per-process).
- **Contamination:** the run shared the GPU with another session's intermittent small decode benchmarks (watcher
  flagged them). VRAM / load_s / call-schedule are **not** contention-sensitive → robust. Prefill seconds may
  carry ±~20% contention noise, but match prior clean standalone spot-checks (mode1/512 cold 5.6 vs 6.2; mode2/4096
  cold 12.3 vs 12.4) — and every conclusion below is an order-of-magnitude / VRAM call, not a close one.
- Clocks pinned `high` during measurement, restored to `auto`. tok0 byte-identical across all modes (greedy-exact).

## Full matrix (clean re-run)

| mode | prompt | load_s | peak_vram_gb | cold_prefill_s | warm_prefill_s | TTFT_cold_s | decode_tok/s | tok0_match | call_schedule |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---|
| 0 `V2=0` | 512 | 6.7 | 5.26 | 27.08 | 7.49 | 33.8 | 77.6 | ✓ | 32-tok ×16 |
| 0 | 1024 | 6.7 | 5.42 | 34.8 | 15.71 | 41.5 | 72.4 | ✓ | 32-tok ×32 |
| 0 | 2048 | 6.5 | 5.73 | 51.87 | 33.35 | 58.4 | 68.9 | ✓ | 32-tok ×64 |
| 0 | 4096 | 6.6 | 6.35 | 91.7 | 73.53 | 98.3 | 61.4 | ✓ | 32-tok ×128 |
| 1 `V2=1` | 512 | 12.7 | 19.15 | 5.64 | 2.03 | 18.3 | 77.6 | ✓ | int512 |
| 1 | 1024 | 13.0 | 19.30 | 9.44 | 5.05 | 22.5 | 74.6 | ✓ | int512 + sym512 |
| 1 | 2048 | 13.1 | 20.00 | 13.33 | 3.55 | 26.4 | 68.9 | ✓ | int512 + sym512×3 |
| 1 | 4096 | 12.8 | 20.88 | 17.35 | 7.51 | 30.2 | 61.3 | ✓ | int512 + sym512×7 |
| 2 `V2=1 CKV=1` | 512 | 18.9 | 19.21 | 1.91 | 0.17 | 20.8 | 77.5 | ✓ | int512 |
| 2 | 1024 | 22.2 | 19.47 | 3.44 | 0.33 | 25.7 | 74.5 | ✓ | int512 ×2 |
| 2 | 2048 | 26.5 | 20.07 | 6.34 | 0.69 | 32.9 | 69.4 | ✓ | int512 ×4 |
| 2 | 4096 | 35.0 | 21.68 | 12.28 | 1.60 | 47.3 | 61.8 | ✓ | int512 ×8 (all concrete) |

No OOMs on the clean run (the earlier matrix's OOMs were the concurrent-process contamination). 4096 fits (21.7 GB
< 24 GB).

## Findings

**1. The true default (`PREFILL_V2=0`) prefill is broken-slow — order of magnitude, not marginal.** It routes the
whole prompt through 32-token symbolic decode-style chunks (16/32/64/128 calls for 512/1024/2048/4096): warm
prefill **7.5 / 15.7 / 33 / 73.5 s**. At 4096 that is **46× slower** than mode 2 (1.6 s). This is the real
out-of-box experience and it is unusable for interactive prefill beyond a few hundred tokens.

**2. `PREFILL_V2` costs ~+14 GB VRAM — the gating constraint.** 5.3 GB (Q4 only) → **19–21 GB** (the realized fp16
weights coexist with the Q4 storage). Fits comfortably on this 24 GB card for 8B; **would OOM a 16 GB card**
(e.g. 7900 GRE) and leaves little headroom for very-long-context KV or larger models. VRAM, not speed, decides
PREFILL_V2 default-on.

**3. `PREFILL_CONCRETE_KV` trades load-time precompile for the best prefill.** Precompile-at-load scales with
context (1/2/4/8 jits → load **18.9 / 22.2 / 26.5 / 35.0 s**). In return, every chunk is concrete: warm prefill
**0.17 / 0.33 / 0.69 / 1.6 s** — best in class, ~4.7× over mode 1 at 4096 (concrete chunks 2+). But the precompile
is **wasted on a one-shot**: cold one-shot TTFT at 4096 is **mode 1 = 30.2 s < mode 2 = 47.3 s** (mode 2's +18 s
load isn't repaid by one generation).

**4. The right METRIC is e2e TTFT, and it is dominated by different things in different regimes:**
- **Cold one-shot:** TTFT ≈ **load + first prefill**, and **load dominates** (13–35 s). Prefill kernel speed is
  secondary. Best cold TTFT = **mode 1** (PREFILL_V2 without the precompile tax): 18–30 s vs mode 2's 21–47 s and
  mode 0's 34–98 s.
- **Warm / server (load paid once at startup):** every request is warm prefill + decode. **mode 2** wins TTFT
  decisively (0.17–1.6 s). 
**5. Decode dominates total latency — prefill policy only moves TTFT.** Decode is **~61–78 tok/s** (the banked
~67% llama, unchanged by any prefill mode). A 128-token generation costs ~1.7–2.1 s of decode; warm prefill
(mode 2) is 0.17–1.6 s. **For total request latency, decode dominates; no prefill policy fixes decode throughput.**

## Policy verdicts

| question | verdict | why |
|---|---|---|
| **PREFILL_V2 default-on?** | **server-default / VRAM-gated** (not global default-on) | 5–15× faster prefill + the only path to llama-parity throughput, BUT +14 GB VRAM OOMs ≤16 GB cards. Safe global default only with VRAM-aware auto-enable. |
| **PREFILL_CONCRETE_KV default-on?** | **server-default / long-prompt-default** (opt-in otherwise) | Best warm prefill, but precompile-at-load hurts cold one-shot TTFT. Amortizes only across repeated/long generation. |
| advertised **default** row | mode 0 out-of-box (works everywhere) — but flag it slow; **recommend mode 1 on ≥24 GB** | mode 0 is the only thing that fits every card; it should warn that real prefill wants PREFILL_V2. |
| advertised **opt-in / server** row | **mode 2** (`PREFILL_V2=1 PREFILL_CONCRETE_KV=1`) | best warm prefill (0.17–1.6 s), the server/long-prompt profile. |

## End-state recommendation

1. **PREFILL_V2:** keep **default-off** today (safe for 16 GB cards), but treat it as the **recommended setting on
   ≥24 GB** and **server-default**. The ideal is a **VRAM-aware auto-enable** (enable iff free VRAM ≥ fp16-covered-
   weight size + KV headroom) — a small engineering feature, the only thing standing between "broken-slow default"
   and "fast default that doesn't OOM small cards." Until then: opt-in, prominently documented.
2. **PREFILL_CONCRETE_KV:** **server-default and long-prompt-default; opt-in for one-shot.** Recommend it whenever
   the process serves >1 generation or the prompt is long (≥~1024); do NOT pair it with a cold one-shot short
   prompt (the precompile load tax loses). A future **prompt-length / first-request-async** precompile trigger
   would remove even that caveat.
3. **Advertise as default:** the out-of-box `PREFILL_V2=0` row (universal), with an explicit "for prompts beyond a
   few hundred tokens, set `PREFILL_V2=1` (needs ~19 GB for 8B)" note. The **headline pp512 = 3394 tok/s (112%
   llama)** number is the mode-1/2 concrete chunk, not the default.
4. **Advertise as opt-in/server:** `PREFILL_V2=1 PREFILL_CONCRETE_KV=1` — the parity profile (warm prefill
   0.17–1.6 s, 73–111% llama across contexts), for 24 GB+ servers / long / repeated prompts.
5. **Remaining prefill work:** **none on kernels** (settled; flash v2 explicitly out of scope). The open items are
   pure integration: (a) VRAM-aware PREFILL_V2 auto-enable; (b) a per-layer fp16 realize to shrink the +14 GB
   (so PREFILL_V2 fits 16 GB cards); (c) async/length-gated CONCRETE_KV precompile to kill the one-shot tax; (d)
   route the prefix-cache "short remainder" through prefill-v2 instead of the 32-token path. **Decode (~67% llama)
   is the real frontier — prefill policy will not move it.**
