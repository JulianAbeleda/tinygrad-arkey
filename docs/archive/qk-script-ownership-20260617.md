# QK script ownership / classification (2026-06-17, after the gqa_coop ship)

Which `extra/qk_*` (and related) scripts are current authorities vs frozen historical probes. Updated after
`FLASH_VARIANT=gqa_coop` became the decode-attention default. No deletions — frozen scripts keep their value
(GEMV_ROLE source, banked methods, regression reproducers); this is the index. Most frozen scripts also carry
an in-file `# STATUS:` header.

## Durable runtime (the shipped path — edit with care)

| file | owns |
|---|---|
| `tinygrad/llm/model.py` | the model; `should_use_flash_decode` policy + flash-decode call site; defaults `FLASH_DECODE=auto`, threshold 512, **`FLASH_VARIANT=gqa_coop_vec`**, `FLASH_L=128` |
| `extra/qk_flash_decode.py` | the decode-attention primitive: `flash_decode_attention` + UOp kernels (max/prob/partial_v2/partial_coop/**partial_coop_vec**/gmax/den/combine); `FLASH_DECODE_VARIANTS` SSOT (default **gqa_coop_vec**); `__main__` exactness self-test |
| `extra/q6_k_gemv_primitive.py` | Q6_K decode GEMV: `q6k_gemv_partial_kernel` (default) + **`q6k_coop_partial_kernel`** (MMVQ_COOP cooperative-K: pos→LOCAL lane coalesced; shipped lm_head + ffn_down) |
| `extra/q4_k_gemv_primitive.py` | Q4_K decode GEMV: `q4k_gemv_partial_kernel`/`q4k_gemv_packed_load_partial_kernel` (default) + **`q4k_coop_partial_kernel`** (MMVQ_COOP lane4→LOCAL coalesced; shipped attn_q/o; ffn_gate/up refuted) |
| `test/external/test_q6k_coop.py` | MMVQ_COOP correctness + routing locks (lm_head/ffn_down/attn_q/o coop vs base; attn_q/o routed, ffn_gate/up not) |

## Durable search / gates (reusable machine-search layer)

| file | owns |
|---|---|
| `extra/qk_search_spec.py` | schema authority (SearchSpace incl. `flash_variant` {v1,hoisted,gqa_coop}, AcceptedPolicy, Constraints) |
| `extra/qk_flash_variant_search.py` | Track-3 variant search (grid now includes gqa_coop; W==D method) |
| `extra/qk_flash_sweep.py` | continuous tok/s-by-ctx sweep worker |
| `extra/qk_demote_search.py`, `extra/qk_nll_eval.py` | demotion search + dNLL quality gate |
| `extra/qk_prefill_v2_*`, `extra/qk_spec_decode_acceptance_gate.py` | prefill-v2 gates; spec-decode acceptance gate |

## Current measurement authorities

| file | method |
|---|---|
| `extra/qk_decode_block_map.py` | the current per-region decode census (post-hoisted; flash path) — supersedes layer/primitive census |
| `extra/qk_decode_runtime_overhead.py` | the **W==D** warm device-feed method (the trustworthy in-model gate; isolated DEBUG2 misled both ways — see gqa_coop result doc) |
| `extra/qk_gqa_coop_decode_attention.py` | Target-A harness (hoisted baseline + cooperative-partial probe) |

## Frozen / refuted (kept; do not treat as current)

| file | status |
|---|---|
| `extra/qk_decode_attention_v3.py`, `extra/qk_decode_attention_v3_tile.py` | **REFUTED** — v3 LDS/WMMA at decode-M (regime mismatch); v3 harness kept for the baseline method |
| `extra/qk_wmma_custom_smoke.py`, `qk_wmma_qk_tile.py` | WMMA-revival probes — capability proven; the WMMA win belongs to **prefill** (large-M), not decode |
| `extra/qk_flash_search.py` | FROZEN — threshold 512 shipped (funcs renamed `*_threshold_*`) |
| `extra/qk_decode_layer_census.py` | FROZEN runner — **but its `GEMV_ROLE` map is the SSOT** imported by `qk_decode_block_map.py` |
| `extra/qk_decode_primitive_census.py`, `qk_decode_smallop_audit.py`, `qk_decode_copy_diagnostic.py`, `qk_attention_kernel_map.py` | FROZEN one-off audits (verdicts recorded) |
| `extra/qk_spec_decode_generate.py` | REFUTED integration (jit-alternation runtime-bound) |
| `extra/qk_gemv_*`, `extra/qk_flash_prefill_*` | REFUTED (Q4K_FUSE −18%; reuse-free flash-prefill) |

## Key docs (current authority)

- Decode attention current: `qk-gqa-coop-decode-attention-result-20260617.md` (gqa_coop, the default).
- Bank: `qk-8b-decode-banked-20260617.md`. Matched baseline: `qk-llama-baseline-xtx-20260617.md`.
- llama audit + next levers: `llama-rocm-decode-attention-audit-20260617.md`,
  `qk-gqa-coop-next-attention-levers-20260617.md`.
- **Superseded-for-default (historical, still valid):** `qk-8b-flash-variant-result-20260617.md` (hoisted —
  superseded as default by gqa_coop), `amd-decode-flash-threshold-20260616.md`, `qk-flash-decode-auto-20260617.md`.
