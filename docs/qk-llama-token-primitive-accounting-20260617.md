# llama vs tinygrad decode — token-primitive accounting (2026-06-17)

Why llama is ~92–98 tok/s and tinygrad ~44–48 tok/s on the SAME RX 7900 XTX / Qwen3-8B-Q4_K_M. Accounting only,
no kernels built, no defaults changed. Provenance: `bench/qk-llama-token-primitive-accounting/provenance.json`
(tinygrad `bd8d5104b`, llama `b9592`/`ac4cddeb0`, ROCm 7.2.4). measured / inferred / hypothetical tagged.

## The one-line answer

Decode is **weight-bandwidth-bound** (both read ~4.68 GB of quantized weights per token). **llama's MMVQ reads
those weights at ~626 GB/s (~70% of HBM peak); tinygrad's quantized matvec at ~349 GB/s aggregate (~39%), and
its big Q6_K roles at 91–129 GB/s (10–14%).** That ~1.8× on the dominant matvec (plus tinygrad attention still
~2× larger share) is the ~2× decode gap. It is **dequant/work-decomposition efficiency of the quantized matvec**,
not the dot instruction (dp4a alone = +1% e2e, refuted).

## 1. llama primitive ledger [MEASURED: rocprofv3 kernel-trace, decode-only shares]

| primitive | llama impl | kernels | cost (decode share) | citation |
|---|---|---|---|---|
| Q4_K/Q6_K matvec (MMVQ) | int8 dp4a (`__builtin_amdgcn_sdot4`), q8_1 activations, fp affine on block sums, 1 kernel/linear | ~253/token | **73.4%**, ~626 GB/s (~70% peak) | rocprofv3; `vecdotq.cuh`/`mmvq.cu` |
| decode attention | `flash_attn_tile` + `stream_k_fixup` + `combine` | ~3/layer | **7.5%** | rocprofv3 (AMD_LOG_LEVEL confirmed names) |
| q8_1 activation quant | quantize-once-per-activation, reused across linears | — | **3.8%** (cheap) | rocprofv3 `cpy/quant` |
| RMSNorm | separate kernels | ~2/layer | 5.0% | rocprofv3 |
| RoPE | separate | ~1/layer | 2.5% | rocprofv3 |
| elementwise (residual/SwiGLU) | separate/fused | — | 1.0% | rocprofv3 |
| graph/launch | HIP graph (`GGML_HIP_GRAPHS=ON`); ~260 kernels/token | — | amortized | CMakeCache + count |
| lm_head, sampling | same MMVQ (Q6_K, vocab 151936) | 1 | within MMVQ | source |

llama decode ≈ **10.25 ms/token** @ctx1024 (97.6 tok/s); ~73% is the efficient MMVQ weight read.

## 2. tinygrad primitive ledger [MEASURED: in-model W==D + isolated role BW; eager-proxy shares marked]

| primitive | tinygrad impl | programs | cost | artifact |
|---|---|---|---|---|
| Q4_K matvec | QK primitive, **fp dequant + fp dot**, parts/opts schedule | ~7/layer | ~31% (proxy); Q4_K roles ~40% peak | `qk-base-decode-gemv-structural-plan` |
| Q6_K matvec (lm_head, ½ ffn_down, k/v) | Q6 primitive, fp dequant | — | ~31% (proxy); **lm_head 91.8 GB/s = 10% peak**, ffn_down 129.7 = 14% | `qk_q6_splitk_dp4a_probe` |
| activation | **fp16** (no q8_1 by default) | — | — | — |
| reduction | parts split-K → partials + `.sum` (extra kernels for parts>1 roles) | +1/role | small | — |
| decode attention | **gqa_coop_vec** (cooperative GQA V-reuse + LOCAL-d coalesced loads) | ~6/layer | ~13–18% (slope closed, −8%) | `qk-gqa-coop-vector-load-result` |
| RMSNorm/residual/RoPE/cast | separate small kernels (scheduler-fused elementwise) | ~15/layer | ~12–19% combined | `qk-decode-block-map` |
| graph/launch | TinyJit HCQ graph; **~1000 programs/token**; W==D (GPU-bound, host ~0) | — | not host-bound | `qk-decode-runtime-overhead` |

tinygrad decode ≈ **21.3 ms/token** @ctx1024 (46.9 tok/s); matvec aggregate ~349 GB/s (~39% peak).

## 3. Gap table (the user's table, filled — MEASURED where shown)

| primitive | llama | tinygrad | measurable gap | reason | status | searchable | expected e2e upside |
|---|---|---|---|---|---|---|---|
| **Q4_K matvec** | MMVQ dp4a, ~70% peak | QK fp-dequant, ~40% peak | **~1.7×** on Q4_K roles | dequant ALU + work-decomp (not the dot: dp4a +1%) | dp4a refuted; full MMVQ open | yes (work-decomp) | +5–15% if full-MMVQ |
| **Q6_K matvec** | MMVQ dp4a | Q6 fp-dequant, **10–14% peak** | **~2–3×** (lm_head worst) | heavier 6-bit unpack ALU + low effective BW | dp4a refuted (+1%) | yes | +5–10% if full-MMVQ |
| **activation quant** | q8_1, **3.8%**, amortized | fp16 (none default) | small | q8 cheap; not the bottleneck | — | low priority | ~0 |
| **reduction** | fused in MMVQ | parts split-K + `.sum` | small | extra reduce kernels for parts>1 | part of MMVQ search | yes (split strategy) | small alone |
| **memory layout** | GGUF K-quant, no repack | GGUF shared storage, no repack | ~none | same layout | parity | no (repack high-risk) | ~0 |
| **decode attention** | tile + stream-k, **7.5%** | gqa_coop_vec, **~13–18%** | ~2× share | tinygrad attn still less efficient, but slope CLOSED | gqa_coop_vec shipped; stream-K refuted | mostly closed | small (≤+3%) |
| **graph/kernel boundaries** | ~260 kernels/token | ~1000 programs/token | 4× count | **NOT host-bound (W==D)**; hurts via per-kernel matvec inefficiency, not launches | GPU-bound | decode-block fusion (very-high-risk) | unclear |

**Crucial:** program count (1000 vs 260) does NOT hurt via launch overhead — W==D proves decode is GPU-bound,
host ~0. It hurts because tinygrad's matvec is **many individually-inefficient per-role GEMV kernels** (each at
10–40% peak) vs llama's MMVQ (one efficient kernel/linear at ~70% peak). The granularity is a *symptom* of the
matvec work-decomposition gap, not a launch-cost gap.

## Why llama is ~95 tok/s / why tinygrad is ~46 tok/s / the remaining gap

- **llama ~95:** MMVQ reads the 4.68 GB weights at ~70% HBM peak (efficient unpack→int8 + dp4a + block-amortized
  affine + q8_1-once), attention is cheap (7.5%), ~260 fused kernels under a HIP graph.
- **tinygrad ~46:** the quantized matvec reads the same 4.68 GB at ~39% peak aggregate (10–14% on the big Q6_K
  roles) — the **fp-dequant/unpack ALU + per-role work-decomposition** cap effective BW; attention is ~2× larger
  share (~13–18%); ~1000 programs (GPU-bound, not launch-bound).
- **Remaining gap = the MMVQ matvec efficiency** (626 vs 349 GB/s). It is **dequant + work-decomposition**, and
  the *full* llama-shaped MMVQ is required (the dot alone, dp4a, was refuted at +1%).

See `qk-machine-search-primitive-rows-20260617.md` for the search rows.
