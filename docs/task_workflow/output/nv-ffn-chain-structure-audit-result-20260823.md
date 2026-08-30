# NV FFN chain-structure audit (2026-08-23)

Deterministic re-test of five claims about the installed tinygrad FFN fold.
Inputs are retained at the same commit as HEAD (`6570abc02`); each input SHA is
re-checked against its manifest and each claim is re-derived from primary
artifacts rather than re-cited from an older report.

## Verdicts

| claim | verdict |
| --- | --- |
| C1 gate/up -> down is one buffer edge, no intermediate kernel | PASS |
| C2 fp16 cast is a candidate-only regression | PASS |
| C3 intermediate is 24 KB against ~100 MB+ weights | PASS |
| C4 Q8 provider feeds attention Q/K/V, never down | PASS |
| C5 DRAM gap is wall-time, not byte-count | PASS |

## C1 - the chain does not exist

* [MEASURED] 36 gate/up nodes, 36 down nodes.
* [MEASURED] Every gate/up has exactly one RAW consumer and every consumer is a
  down kernel (`q4k/q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd`).
* [MEASURED] The gate/up -> down edge span is uniformly `24576` bytes across all
  36 layers. There is no separable cast/activation/provider kernel between them.

## C2 - the cast is the regression, not an opportunity

* [MEASURED] The installed control contains `0` `E_128_32_3` nodes and no
  cast/silu/quant/pack-like kernel name in the FFN path.
* [MEASURED] The candidate-only `E_128_32_3` cast costs `39.52 us/token`.
* [MEASURED] Four-warp body is `1352.384 us` vs control `1359.424 us`
  (`-7.04 us`); net gate/up swap is `+32.48 us`.
* [MEASURED] Authoritative unprofiled wall bracket regresses `+28.36 us/token`
  with token SHA `f25083e5...d7905` across all arms.

## C3 - byte counts

* [MEASURED] Intermediate activation: `24576` bytes (`12288 x fp16`).
* [MEASURED] Weight streams per layer: gate/up `2 x 28311552` bytes, down q4
  `28311552`, down q6 `41287680`.
* [MEASURED] Total FFN weight traffic: `3291217920` bytes/token (~3.29 GB).
* [MEASURED] Packing the intermediate fp16->Q8 saves `12288` bytes, which is
  `0.00037%` of the FFN weight traffic. Representation is not the lever.

## C4 - provider is attention-only

* [MEASURED] 17 `rmsnorm_q8_1_llama_provider_4096` nodes, Q8 output uniformly
  `4608` bytes.
* [MEASURED] Every provider consumer is an attention projection: 8 providers
  feed Q+K+V (`q4k_warp_coop_q8_dp4a_partial_4096_4096`,
  `q4k_warp_coop_q8_dp4a_partial_1024_4096`, `q6k_q8_warp_direct_1024_4096`),
  9 feed Q+K+K (V routed through the KV cooperative kernel).
* [MEASURED] No provider feeds the down projection.

## C5 - DRAM gap is wall-time

* [MEASURED] tinygrad DRAM bytes are NCU-measured; llama shares the same GGUF
  weight bytes. The throughput gap is therefore wall-time only.

```text
gate/up   tinygrad 1.501 TB/s   llama 1.609 TB/s   (0.933)
down q6   tinygrad 1.353 TB/s   llama 1.449 TB/s   (0.934)
down q4   tinygrad 1.357 TB/s   llama 1.501 TB/s   (0.904)
```

* [INFERRED] The llama side is `shared_weight_bytes / measured_body`, an
  effective rate; only the tinygrad byte count is independently NCU-measured.
  The same-weight premise is model-defined, not a per-run counter.

## Evidence

* Machine-readable result:
  `docs/task_workflow/evidence/nv-ffn-chain-structure-audit-20260823/result.json`
* Harness: `extra/llm_research/decode/nv_ffn_chain_structure_audit.py`
* Input hashes: capture `c34b5752...`, decomposition `cdec4339...`,
  closure `950f6985...` (all verified against their retained manifests).

No production model, renderer, scheduler, runtime, or route code was changed.
