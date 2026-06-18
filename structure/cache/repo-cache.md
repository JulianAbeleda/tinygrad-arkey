# Repo Cache

Last updated: 2026-06-16
Analyzed source: `tinygrad/llm/`, `extra/qk_*`/`llm_*`, `test/external/`, `docs/`, `structure/`.
Scope: stable implementation context for token-saving orientation. Full doc map: `../../docs/README.md`.

## Project Shape

- `tinygrad-arkey`: an **AMD-only hard-fork of tinygrad** for **quantized LLM decode** (Q4_K/Q6_K GGUF) on
  RDNA3 (RX 7900 XTX, gfx1100). Non-AMD backends (NV/CUDA/METAL/QCOM/CL/WEBGPU/DSP) were pruned.
- User-facing surface: `tinygrad/llm/cli.py` (chat / `--benchmark` / OpenAI-compat `--serve`).
- The fork's value is **decode primitives + a bounded machine-search layer** in `extra/qk_*`.
- What it is NOT: a general multi-backend tinygrad; a training framework (inference/decode focus).

## Core Facts

- Python 3.12, deps via `uv` (`uv sync --extra testing_minimal --extra costmodel`); interpreter `.venv/bin/python`.
- Run decode: `DEV=AMD .venv/bin/python -m tinygrad.llm.cli -m <gguf> --warmup --benchmark N`.
- Models: `/home/ubuntu/models/Qwen3-{1.7B,4B,8B,14B,32B}*.gguf`.
- Key files: `tinygrad/llm/model.py` (`Transformer.from_gguf`, `QKConfig`, Q4K/Q6K primitive install +
  demotion), `tinygrad/llm/cli.py`, `extra/q4_k_gemv_primitive.py` / `q6_k_gemv_primitive.py` (kernels),
  `extra/qk_quantize.py` (Q4_K quantizer), `extra/qk_flash_decode.py`, and the machine-search system
  (`extra/qk_search_spec.py` + `qk_nll_eval.py` + `qk_demote_search.py`).
- Env flags (names only): `DEV`, `JIT`, `Q4K_PRIMITIVE`/`Q6K_PRIMITIVE` (auto-on for AMD GGUF), `Q6K_COVER_MORE`,
  `Q6K_DEMOTE_FFNDOWN`/`QK_DEMOTE_TENSORS`, `FLASH_DECODE`/`FLASH_L`, `QK_GENERATED_POLICY`,
  `QK_PRIMITIVE_STORAGE`. Generated artifacts under `bench/**` are gitignored.

## Main Flow

1. `cli.py` loads a GGUF → `Transformer.from_gguf` (`model.py`) installs Q4_K/Q6_K decode primitives
   (path-aware, default-on for AMD), applies any demotion/policy.
2. `model.generate` runs the JIT'd decode loop (symbolic `start_pos`); primitives swap the GEMV kernels.
3. Machine search (when used): `qk_search_spec` rows → isolated runner (`cli --benchmark` + `qk_nll_eval`) →
   quality gate → `AcceptedPolicy` artifact.

## Key Boundaries

- **AMD-only**; non-AMD paths gated/removed. Env-ordering invariant: set `DEV`/`JIT`/QK flags **before**
  `from tinygrad import ...` (loaders import tinygrad lazily to preserve this).
- **Do NOT run BEAM / risky schedule search on gfx1100** (hangs). Generated policies are **never a runtime default**.
  Lossy quant demotions are **dNLL-gated, default-off**. Exact wins (Q6_K coverage, flash) are byte-identical.
- Dangerous-power surface (custom kernels, raw queue/ring) stays contained + gated (see `coding-principles`).

## Verification

- `.venv/bin/python -m pytest test/external/ -q` → **237 pass / 56 skip** (56 skip = artifact-absent reproduce
  tests, by design; `bench/**` is gitignored).
- `.venv/bin/python -m py_compile <file>`; `git diff --check`.
- Decode smoke: the `cli --benchmark` command above (steady median ~55 tok/s default-on).

## Current Direction (refreshed 2026-06-17)

**Matched llama.cpp baseline measured on this RX 7900 XTX** (`../../docs/qk-llama-baseline-xtx-20260617.md`;
the host is an XTX — rocm-smi misreports "GRE", 24GB VRAM + rocminfo confirm XTX): llama decode 99.5 (d0) /
98.6 / 97.6 / 95.4 / 92.2 tok/s @ctx 0/512/1024/2048/4096; prefill pp512 = 3069.

**Decode (shipped):** flash-decode default-ON — `FLASH_DECODE=auto` threshold **512**, **`FLASH_VARIANT=gqa_coop_vec`**
(gqa_coop cooperative GQA V-reuse + output-dim `d` mapped to LOCAL threads → coalesced fp16 loads; gqa_coop ran
as pathological 1-thread workgroups), **`FLASH_L=128`** (`../../docs/qk-gqa-coop-vector-load-result-20260617.md`).
Measured: **47.7 / 46.9 / 45.7 / 43.9 tok/s @ctx 512/1024/2048/4096 = ~48% of llama FLAT** (slope **−8%** ≈
llama −7%; the decode-attention SLOPE GAP IS CLOSED). Was gqa_coop 44.8/41.3/36.3/29.6 (45/42/38/32, −34%).
`FLASH_VARIANT={v1,hoisted,gqa_coop,gqa_coop_vec}` override. The "~64 tok/s" figure is short-ctx ~ctx8 +
demotion. Plus Q6_K coverage, ffn_down demotion (dNLL-gated, default-off).
**MMVQ_COOP family SHIPPED (cooperative-K coalesced Q6_K GEMV, default on):** pos→LOCAL lane → coalesced
packed-weight loads (default is one-row-per-thread at ~10-14% HBM peak). `Q6K_LM_HEAD_COOP=1` (lm_head 91→457
GB/s, 10%→51% peak) + `Q6K_FFN_DOWN_COOP=1` (ffn_down 125→347 GB/s, 14%→39% peak), kernel
`q6k_coop_partial_kernel`; **`Q4K_ATTN_QO_COOP=1`** (attn_q/o 4096×4096 169→258 GB/s, 19%→29% peak, kernel
`q4k_coop_partial_kernel`, `Q4K_COOP_RT=16`). **Decode 68.3 / 66.3 / — / 60.9 tok/s @ctx 512/1024/4096 = ~68%
of llama (+44/+43/+40% over the pre-coop 47.3/46.5/43.6), byte-identical greedy, W==D.**
`../../docs/qk-mmvq-q6k-lm-head-arc-20260617.md`, `../../docs/qk-mmvq-coop-q4k-attn-result-20260617.md`. The
work-decomposition rewrite (NOT the refuted dp4a knob) was the base-decode answer; "bounded decode exhausted" is
SUPERSEDED. **MMVQ_COOP RULE: apply coop only where baseline coalescing is bad (<~30% peak); Q4_K ffn_gate/up
(41% peak) + ffn_down (Q4_K) REFUTED — already coalesced. Low-risk role-by-role coop is DONE. **Q4_K ffn_gate/up
full-MMVQ (Family A: q8_1+dp4a) REFUTED 2026-06-18 (`../../docs/qk-mmvq-q4k-ffn-full-result-20260618.md`): dp4a
kernel 39% peak (no faster than fp), whole-linear 0.82×; the dot was never the bottleneck — the 40→70 gap is the
format-mandated UNPACK ALU (nibble+scale decode), which dp4a doesn't touch (READRAW 70% unreachable while unpack
ALU present). ffn_gate/up unpack-ALU-bound at ~48% ceiling, no bounded kernel breaks past. Remaining decode gap
(~32%) is unpack-ALU/codegen-ILP-structural; next levers OUT of decode-kernel scope: prefill WMMA / 14B / deep
codegen-ILP (very high risk).**

**Key diagnosis (`../../docs/llama-rocm-decode-attention-audit-20260617.md`):** llama decode is ~context-FLAT
(−7%), tinygrad decays −43% → the long-ctx gap is **attention**. llama uses `flash_attn_tile` + stream-K split +
combine (GQA-batched tile, fp16 LDS staging, vectorized loads); tinygrad's hoisted flash_partial re-reads V 4×
at ~33 GB/s. **CEILING:** perfect attention only removes the slope → still ~44% of llama; the **base-decode 2.3×
gap (GEMV + ~780 progs/token)** is the bigger structural limiter.

**Refuted/closed (do NOT reopen):** B1/Q4K_FUSE GEMV horizontal fusion (−18%), norm/small-op fusion, sub-4-bit
(dNLL), register-blocking flash_partial, **decode_attention_v3** (LDS/WMMA at decode-M; naive LDS 0.5-0.77× vs
IC-served baseline). **WMMA custom-kernel idiom REVIVED** (`spec_tensor` rule) — a lasting asset, but for the
**PREFILL** regime (large-M), not decode. **Prefill (best gap): pp512 81% of llama** (PREFILL_V2 ~2486, opt-in).
Next decode-attention target (design, not built): GQA-batched cooperative tile + vectorized LDS load
(`../../docs/tinygrad-decode-attention-next-primitive-spec-20260617.md`). The bounded machine-search system is
the reusable asset.

## Known Risks

- `bench/**` gitignored → reproduce/golden tests skip-if-absent on a fresh tree (not regressions).
- The two gated builds (overlap 2nd compute ring in `ops_amd.py`; sub-4-bit quantizer+kernel) are
  dangerous-power surface — scope deliberately, don't bolt on.

## Update Rule

Update when command routing, architecture boundaries, core data contracts, storage layout, verification
commands, or important file ownership change. No transient logs, secrets, or session content.
