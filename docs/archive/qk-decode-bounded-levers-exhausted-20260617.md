# 8B decode — bounded levers EXHAUSTED (2026-06-17)

> **SUPERSEDED 2026-06-17 (same day): NOT exhausted.** The one lever this doc flagged as un-refuted — the full
> MMVQ work-decomposition — was built and **SHIPPED**: cooperative-K Q6_K lm_head (pos→LOCAL coalesced loads),
> **+19% in-model decode, byte-identical**, decode ~48%→~57% of llama (`qk-mmvq-q6k-lm-head-arc-20260617.md`).
> The lever is general (applies to every Q6_K/Q4_K role) — more decode wins are open. The "bounded levers
> exhausted" framing below was wrong: the bounded *knobs* (dp4a/schedule/fusion) were exhausted, but the
> *work-decomposition rewrite* was not, and it won big. Treat the rest of this doc as the pre-cooperative-K state.

Campaign milestone. After the gqa_coop_vec ship + the Q6_K dp4a refutation, every **bounded/local** decode
lever is shipped or refuted. Remaining work is high-risk (decode-block/runtime) or a different phase
(prefill/14B). RX 7900 XTX, Qwen3-8B-Q4_K_M.

## 1. Current accepted decode (default path)

`FLASH_DECODE=auto`, threshold 512, **`FLASH_VARIANT=gqa_coop_vec`**, `FLASH_L=128`, Q4_K/Q6_K primitives on,
ffn_down demotion dNLL-gated (default-off). In-model W==D, byte-identical greedy:

| ctx | tinygrad | llama | % of llama |
|---|---|---|---|
| 512 | 47.7 | 98.6 | 48% |
| 1024 | 46.9 | 97.6 | 48% |
| 2048 | 45.7 | 95.4 | 48% |
| 4096 | 43.9 | 92.2 | 48% |

**~48% of llama, FLAT** (slope −8% ≈ llama −7%). Up from the campaign's decaying 45/42/38/32. Prefill (v2) is
81% of llama.

## 2. Attention levers

| lever | status |
|---|---|
| flash-decode auto + threshold 512 | SHIPPED |
| hoisted exp (once/key) | SHIPPED |
| gqa_coop (cooperative GQA V-reuse) | SHIPPED (+3.9…+19.8% over hoisted) |
| **gqa_coop_vec (coalesced LOCAL-d loads)** | **SHIPPED (+6.5…+48.8% over gqa_coop; closed the slope gap)** |
| Stream-K / adaptive split | REFUTED by audit (slope flat, GPU filled at long ctx, attention now 18% — `qk-streamk-decode-attention-result-*`) |
| decode_attention_v3 (LDS/WMMA at decode-M) | REFUTED (regime mismatch — `qk-decode-attention-v3-result-*`) |

The decode-attention **slope gap is closed.**

## 3. GEMV / base-decode levers

| lever | status |
|---|---|
| Q4_K / Q6_K primitives, shared storage | SHIPPED (banked) |
| Q6→Q4 demotion (dNLL-gated) | SHIPPED (default-off) |
| Q4K_FUSE horizontal fusion | REFUTED (−18%) |
| dp4a / Q4K_VDOT (Q4_K int8 dot) | REFUTED e2e (+1%; decode GEMV is in-pipeline bandwidth-bound; standalone 1.77× was a warm-cache artifact) |
| **Q6_K split-K dp4a (ffn_down/lm_head)** | **REFUTED at Phase-0 gate (this arc) — share 31.5% but realized ~1.04× → +1.2% e2e; not built (`qk-q6-splitk-dp4a-result-*`)** |
| parts/tile schedule search; weight layout | REFUTED/optimal (exhausted) |

## 4. Q6_K split-K dp4a verdict (this arc)

Q6_K roles (lm_head + half the ffn_downs) are parts==1 Q6_K, 31.5% of decode @ctx512. Optimistic Amdahl
(1.77×) = +15.9%, but the **realized in-pipeline dp4a speedup is ~1.04×** (Q4_K precedent: same memory-bound
GEMV class, same q8_1+dp4a mechanism; lm_head reads ~500MB/token = pure bandwidth) → **+1.2% e2e**, below the
5% gate. Refuted without building (build-only-if-earned + no-broad-GEMV-rewrite).

## 5. Remaining gap to llama

~**52% gap, flat** (48% of llama at all ctx). It is the **base decode**: GEMV ~58% of decode @ctx4096 + the
~780 progs/token program-granularity.

**MMVQ structural diagnosis (2026-06-17, `qk-mmvq-primitive-roadmap-20260617.md`, `qk_q6_splitk_dp4a_probe.py`):**
the GEMV achieves only ~**10% of HBM peak** (lm_head Q6_K 91.8 GB/s, ffn_down 129.7 GB/s) — it is **NOT
raw-bandwidth-saturated and NOT dot-bound** (dp4a +1% both quants). READRAW shows the memory schedule reaches
~730 GB/s *without* dequant, and the **dequant/unpack ALU per weight** is the limiter (Q4_K fp 365 → Q6_K 91
GB/s). dp4a removes the *dot*, not the *unpack* → +1%. The unpack is **format-mandated** (must extract 4/6-bit
per weight). The only path to llama's ~2× is **Phase F: a full llama-shaped MMVQ kernel** (unpack→int8 once +
dp4a + block-amortized affine + q8_1) — high-risk, uncertain (+1% piecemeal precedent), a substantial build.
Bounded knobs (dp4a, schedule search, q8_1 reuse) are refuted/low-EV.

## 6. Explicit status

**Bounded decode levers are EXHAUSTED.** Everything local/bounded is shipped (attention: 4 wins; primitives +
demotion) or refuted (Stream-K, v3, Q4K_FUSE, dp4a Q4_K + Q6_K, schedule search). The remaining decode gap is
distributed/structural.

Remaining work, all out of "bounded decode" scope:
- **Prefill WMMA attention** (Primitive 7) — different phase, the revived WMMA's right regime; prefill 81%→ more.
- **Decode-block graph fusion** (Primitive 5) — very high risk, compiler-arch (the ~780→260 progs gap).
- **Low-sync speculative runtime** (Primitive 6) — very high risk, needs a TinyJit/runtime change (the
  jit-alternation wall); +30-60% if solved.
- **14B/32B matrix** — different target (more GPU-bound).

Recommendation: stop bounded-decode work; pick **prefill WMMA** (highest-confidence remaining) or accept the
current state (decode ~48% of llama flat, prefill 81%).
