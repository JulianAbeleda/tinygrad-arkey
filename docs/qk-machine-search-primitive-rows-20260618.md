# Machine-search primitive rows — refreshed (2026-06-18)

Successor to `qk-machine-search-primitive-rows-20260617.md` (now SUPERSEDED). Reflects shipped/refuted/deferred
state after the MMVQ-coop ships, the int-dot/q8 closeout, the spec-verify close, and the prefill PWR-0/1 audit.
Source of truth: `what-makes-a-performance-primitive-efficient-20260618.md`; per-role evidence:
`qk-decode-per-role-delta-audit-20260618.md`. Only the rows below are live; everything else is closed (table at end).

Schema = full primitive boundary (per `performance-primitive-research-principles.md`): each live row carries
primitive name, phase, current impl, reference impl, required dataflow, legal knobs, correctness/quality gate,
isolated gate, in-model gate, expected Amdahl, known refutations, fallback. New schema concepts needed in
`extra/qk_search_spec.py` (scoped update): `Q8_SIDECHANNEL`, `WMMA_DENSE_ISSUE`, `EXTERNAL_BLAS_BOUNDARY`.

## Live rows

```json
[
  {
    "primitive": "decode_q4k_ffn_q8_sidechannel", "phase": "decode", "state": "deferred behind codegen capability (Q8L-2 KILL: q8-mmvq-lifecycle-deep-result-20260619.md)",
    "current_impl": "Q4_K ffn_gate/up byte-identical fp dequant+fp dot, ~41% HBM peak [M]; coop refuted (already coalesced)",
    "reference_impl": "llama MMVQ: q8_1 activation (produced once) + native sudot4 + block-amortized affine, ~70% peak",
    "required_dataflow": "fused custom RMSNorm/apply PRODUCER emits (fp output + q8_1-packed activation + per-32 scales) in one launch -> gate/up int-dot consumes q8 -> sudot4 -> qsum/affine correction",
    "legal_knobs": ["producer_fusion(rmsnorm_apply_epilogue)","q8_pack_granularity(per32)","multi_output_store_layout","sudot4_path","reuse_count(gate+up=2)","scale_decode_location"],
    "correctness_quality_gate": "dNLL <= 0.01 multi-window (lossy q8, rel ~0.006); value-tested sudot4 signedness",
    "isolated_gate": "producer effective q8 overhead <= 4.8us (break-even vs fp coop at reuse=2)",
    "in_model_gate": "W==D >= +5% decode @ctx512 byte/quality-exact, no long-ctx regression",
    "expected_amdahl": "+3-4% e2e (gate/up = 2 of 7 linears; reuse ceiling 2 because k/v are Q6_K) [I]",
    "known_refutations": "separate q8 pack (29.7us/4 kernels) -> 0.96x; graph-reuse 0.94-0.96x; sudot4 whole-linear loses; per-32 max cannot piggyback RMSNorm sum-of-squares",
    "fallback": "byte-identical fp coop (shipped default) on unsupported shapes / non-AMD",
    "blocked_by": "Q8L-2 PROVEN: fused per-row-reduce->broadcast->per-32-reduce->multi-output store NOT expressible via UOp store-group idiom (GROUP-of-ENDs fails verify; two granularities = serial separate kernels). Needs LDS-reduction flash-style kernel = deep codegen capability. Q8L-0 contract clean, Q8L-1 cost <=4.8us plausible if single-kernel."
  },
  {
    "primitive": "decode_q4k_ffn_coop_subgate", "phase": "decode", "state": "sub-gate candidate",
    "current_impl": "default ffn_gate/up = base q4k_gemv_partial (LOCAL:0:64, 41% peak); coop kernel exists but routed only to attn_q/o",
    "reference_impl": "the shipped q4k_coop_partial_kernel (coalesced lane4->LOCAL), already exact for attn_q/o",
    "required_dataflow": "route ffn_gate/up (12288x4096) through q4k_coop_partial_kernel behind Q4K_FFN_COOP flag",
    "legal_knobs": ["Q4K_FFN_COOP(on/off)","Q4K_COOP_RT","unsupported_shape_fallback"],
    "correctness_quality_gate": "byte-identical greedy (fp-reassoc-tol, same kernel class as shipped attn_q/o coop)",
    "isolated_gate": "1.16x (PASS a >=10% bar) [M]",
    "in_model_gate": ">= +5% W==D decode; MEASURED +1.0/1.5/1.8/2.3% @ctx128/512/1024/4096 [M] -> FAILS",
    "expected_amdahl": "+1.0-2.3% e2e, grows with ctx [M]",
    "known_refutations": "isolated 1.16x / Amdahl ~+6% did NOT translate (in-model only +1-2.3%); below 5% route gate",
    "fallback": "default base kernel (no flag)",
    "blocked_by": "sub-gate; stackable-only. Bank as candidate, do not route standalone"
  },
  {
    "primitive": "decode_attention_residual_audit", "phase": "decode", "state": "audit-only",
    "current_impl": "gqa_coop_vec flash-decode (default), threshold 512, hoisted exp, L=128; slope gap closed (~-8% ~ llama)",
    "reference_impl": "llama flash_attn_tile + stream_k_fixup + combine, ~7.5% decode share",
    "required_dataflow": "current HEAD block-map attention ms/token + slope @ctx 512/1024/4096 vs llama trace",
    "legal_knobs": ["FLASH_VARIANT","FLASH_L","FLASH_DECODE_THRESHOLD"],
    "correctness_quality_gate": "byte-identical greedy (flash exact vs SDPA)",
    "isolated_gate": "n/a (audit)",
    "in_model_gate": "close if residual <= 3% e2e; else name a specific attention primitive",
    "expected_amdahl": "<= +3% e2e [I]; attention ~13-18% share but slope already closed",
    "known_refutations": "stream-K refuted (GPU filled at long ctx); decode_attention_v3 LDS/WMMA refuted (decode-M regime mismatch)",
    "fallback": "shipped gqa_coop_vec",
    "blocked_by": "no bounded target; likely closes <=3%"
  },
  {
    "primitive": "prefill_fp16_wmma_lds_tiling", "phase": "prefill", "state": "REFUTED as LDS-lever (PWLT-A2, prefill-wmma-lds-tiling-result-20260619.md): hand-LDS WMMA = 1.02x default, both ~34% peak; LDS-tiling IC-served on gfx1100. Real lever = dense WMMA issue / Tensile-class scheduling -> see prefill_wmma_dense_issue and external_blas rows",
    "current_impl": "PREFILL_V2 inc-1: fp16 realized weights + WMMA + warmstart-TC; ~74% of forward is WMMA matmul but LDS=0 (re-reads operands) [M]",
    "reference_impl": "rocBLAS/Tensile: 128x128 macro-tile staged in 25.6KB LDS -> ~80% peak",
    "required_dataflow": "stage fp16 operand tiles into LDS/shared, reuse across WMMA macro-tiles before HBM re-read (Boehm step 2)",
    "legal_knobs": ["LDS_tile_M","LDS_tile_N","LDS_tile_K","wmma_macro_tile","workgroup","double_buffer","GROUP/LOCAL_into_LDS_opt"],
    "correctness_quality_gate": "fp16 prefill dNLL <= 0.01 (already passed for PREFILL_V2 inc-1); no decode regression",
    "isolated_gate": "isolated tiled matmul >= 1.5x current in-model prefill-shaped linear (warm, not cold TFLOPS)",
    "in_model_gate": ">= 1.5x full warm pp512 candidate (>=3x strong)",
    "expected_amdahl": "~74% matmul share; 2x matmul -> ~1.6x full pp [I]",
    "known_refutations": "PREFILL_FP16/REALIZE/Q4K_UNFUSE/Q4K_BATCHED no in-model win (controls); reuse-free custom kernels slow",
    "fallback": "PREFILL_V2 inc-1 (shipped, opt-in)",
    "blocked_by": "not blocked; refuted. Explicit LDS staging did not move the shape. Do not reopen as a locality-only row"
  },
  {
    "primitive": "prefill_wmma_dense_issue", "phase": "prefill", "state": "REFUTED as bounded config sweep (prefill-own-wmma-kernel-result-20260619.md): best 42.0 TFLOPS, below 62 TFLOPS gate",
    "current_impl": "PREFILL_V2 fp16 realized weights + WMMA plateau around 40.8 TFLOPS; hand-LDS WMMA 41.5 TFLOPS; fp16 ALU path ~40 TFLOPS [M]",
    "reference_impl": "measured external ceiling/control: hipBLASLt 69.8 TFLOPS ffn_gate/up, rocBLAS 70.9/76.7 TFLOPS ffn_down/attn_q/o (prefill-external-blas-result-20260619.md)",
    "required_dataflow": "fp16 realized weights -> global/register load -> dense independent WMMA issue -> fp32 accumulate -> output; LDS optional/off because PWLT-A2 showed IC-served operands",
    "legal_knobs": ["threads_per_block(128|256|512)","macro_tile_MN","accumulator_depth","independent_wmma_ops","K_unroll","BLOCK_K","LDS(on|off)","load_wmma_overlap"],
    "correctness_quality_gate": "fp16 oracle mse tolerance per kernel; in-model fp16 dNLL <= 0.01; no decode regression",
    "isolated_gate": "FAIL: best 42.0 TFLOPS, same as baseline; gate was >=62 TFLOPS",
    "in_model_gate": "not reached",
    "expected_amdahl": "~74% matmul share; if the bucket moves 40.8->~70 TFLOPS, full-pp upper bound is roughly 1.4-1.45x before overhead [I]",
    "known_refutations": "LDS staging alone 1.02x; POWN-1: more waves, bigger tiles, BK32, W1x1, and noLDS all regress; quant-weight reuse closed for 8B",
    "fallback": "PREFILL_V2 inc-1 (shipped, opt-in)",
    "blocked_by": "not a bounded knob issue; remaining gap to external BLAS likely needs deeper codegen/software-pipelining/assembly/Tensile-class control"
  },
  {
    "primitive": "prefill_attention_lds_flash", "phase": "prefill", "state": "deferred D",
    "current_impl": "SDPA (~24% of PREFILL_V2 forward [M]); custom score-free kernel expressible+correct but ~170-760x slower (reuse-free)",
    "reference_impl": "llama mature tiled/flash attention: K/V tiles in LDS, online softmax state in registers, compact writes",
    "required_dataflow": "K/V tile -> LDS (64KB/wg cap, L<=128@Hd=128) + register-resident online max/sum/acc + 1 barrier/tile + compact store",
    "legal_knobs": ["kv_tile_L","lds_layout(packed|fp)","online_softmax_state","warp_reduce(WR1-3)","causal_mask"],
    "correctness_quality_gate": "exact vs SDPA (rel ~1e-3); dNLL if dtype changes",
    "isolated_gate": "LDS attention tile > 1x global-reread baseline at prefill T (the Phase-5 fail point to beat)",
    "in_model_gate": "long-prefill +10% warm pp, quality accepted",
    "expected_amdahl": "~24% share now, grows at long prompts (~51% @sp3072) [M] -> matters mainly for long ctx",
    "known_refutations": "reuse-free score-free kernel 170-760x slower (LDS reuse missing); naive single-query LDS tile 0.5-0.74x (cache-served, low-occupancy)",
    "fallback": "SDPA",
    "blocked_by": "needs high-occupancy warp/WMMA flash (256-thread query blocks); SHAPED_WMMA custom-kernel convention stale (WR4 wall); WR1-3 + LDS-tiling assets exist"
  },
  {
    "primitive": "external_blas_rawhip_boundary", "phase": "prefill", "state": "Lane A HIP-runtime bridge KILL (EBT-1); Lane B Tensile HSACO extraction/codegen-transfer scoped",
    "current_impl": "pure tinygrad codegen for all prefill matmuls (WMMA, LDS=0)",
    "reference_impl": "rocBLAS / hipBLASLt / Tensile measured by extra/qk_prefill_blas_ceiling.cpp",
    "required_dataflow": "PREFILL_V2 fp16 tiles -> selected Tensile HSACO kernel launched through tinygrad HCQ with exact solution/symbol/kernarg/launch/workspace contract; fallback to PREFILL_V2",
    "legal_knobs": ["backend(rocblas|hipblaslt|rawhip|tinygrad)","fallback_policy","artifact_portability","authority_boundary"],
    "correctness_quality_gate": "bit/dNLL parity with tinygrad path; clean fallback when lib absent",
    "isolated_gate": "PASS: hipBLASLt 69.8 TFLOPS on ffn_gate/up = 1.71x tinygrad; rocBLAS 70.9/76.7 TFLOPS on ffn_down/attn_q/o",
    "in_model_gate": ">= 1.5x full warm pp with fallback intact",
    "expected_amdahl": "moderate-high for prefill: measured ~1.7x large matmuls gives roughly 1.4-1.45x full-pp upper before bridge/layout overhead [I]",
    "known_refutations": "the tinygrad-internal LDS alternative (prefill_fp16_wmma_lds_tiling) is REFUTED -- LDS-tiling doesn't help (PWLT-A2). Lane A in-process HIP runtime bridge is KILLED by EBT-1: HIP runtime and tinygrad HCQ/KFD are mutually exclusive.",
    "fallback": "pure tinygrad PREFILL_V2 (~70-83% llama)",
    "blocked_by": "external-artifact authority decision plus contract recovery: selected solution, code object, kernel symbol, named .kd descriptor, kernarg layout, launch geometry, workspace, and shape matrix. Full scope: prefill-tensile-primitive-extraction-and-codegen-scope-20260619.md; start TPE-1 only if Tensile artifacts are accepted."
  }
]
```

## Closed / superseded rows (do NOT reopen as build tasks without new evidence)

| row | disposition | reason |
|---|---|---|
| `prefill_quant_weight_reuse_8b` | **REFUTED / CLOSED** | PWR-1: 8B PREFILL_V2 already realizes fp16 weights + WMMA → no in-forward dequant to amortize → zero Amdahl room. VRAM-frugal 14B/32B niche only (excluded by no-pivot). |
| old broad `mmvq_q6k` | **SUPERSEDED** | SHIPPED for lm_head (51%) + ffn_down (39%) via coop coalescing; Q6_K dp4a/int-dot refuted (+1% e2e). Residual to 70% folded into `decode_q4k_ffn_q8_sidechannel` (same q8 wall). |
| old broad `mmvq_q4k` | **SUPERSEDED** | SHIPPED attn_q/o (29%) coop; ffn_gate/up = `decode_q4k_ffn_q8_sidechannel` (deep q8 lifecycle only); ffn_down subordinate; dp4a-only / fp-codegen / sudot4-whole-linear all refuted. |
| old `prefill_wmma_attention` (vague) | **SPLIT / SUPERSEDED** | now split into `prefill_wmma_dense_issue` (pure tinygrad matmul), `external_blas_rawhip_boundary` (measured ceiling/control), and `prefill_attention_lds_flash` (long-prompt attention). |
| `decode_block_fusion` | **REFUTED / low-EV** | per-role delta audit found norms/RoPE/elementwise ~12–19% spread over ~380 tiny kernels, GPU-bound, no ≥5% fused target; only a norm→MMVQ epilogue could help and that IS `decode_q4k_ffn_q8_sidechannel`. |
| dp4a/Q4K_VDOT, Q6_K split-K dp4a, Q4K_FUSE, stream-K, decode_attention_v3, schedule-knob-only, q8 amortization-alone, weight repack, sub-4-bit, naive spec, ring2 decode | **REFUTED** | carried from the 06-17 rows doc; see its closed table + `qk-mmvq-int-dot-closeout-20260618.md`. |
| `spec_verify_q4k_batched_k` | **CLOSED** | spec verify is distributed T-scaling; no single kernel (`qk-spec-verify-component-breakdown-20260618.md`). |

## Live-row priority (Amdahl-ranked, all gated, none routable cheaply)

1. `external_blas_rawhip_boundary` — isolated ceiling passes, but routing is an authority/runtime boundary
   (HCQ-vs-HIP runtime, fallback, external dependency policy), not a kernel tweak. EBT-1 killed the HIP-runtime
   bridge; current full scope is `prefill-tensile-primitive-extraction-and-codegen-scope-20260619.md`.
2. `decode_q4k_ffn_q8_sidechannel` — the only decode lever left (~+3–4%), deep + lossy + multi-output-precedent-less.
3. `prefill_attention_lds_flash` — matters at long prompts; deep, SHAPED_WMMA-walled.
4. `decode_q4k_ffn_coop_subgate` — +1–2.3% stackable, not routable alone.
5. `decode_attention_residual_audit` — likely closes <=3%.

No live row is a bounded cheap edit; all are deep (codegen/lifecycle/BLAS-boundary) or sub-gate. This matches the
source-of-truth conclusion: the decode primitive space is exhausted; remaining progress needs a deep
activation-lifecycle (decode) or an external/raw-HIP boundary. The bounded pure-tinygrad dense-WMMA sweep is closed.
