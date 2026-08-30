# NV FFN GEMV DRAM-streaming implementation scope (2026-08-22)

Status: implementation scope; authorizes codegen/renderer changes only after
the isolated gate passes. Requires production edit approval.

1. Row and edge: gate/up `q4k_g3_lanemap_gemv_w1w3fused16_12288_4096`
   (36 x 37.728 us) and down `q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd`
   (18 x 30.560) + q4 down (18 x 20.896). Edge: ffn_norm -> gate/up ->
   activation -> down -> residual.
2. Dominant term: DRAM-cold streaming efficiency. tinygrad 83.9% vs llama
   89.9% of peak on gate/up; 75.6-75.8% vs 81.0-83.8% on down. Body is
   L2-hot faster; D is clear.
3. Code paths: `tinygrad/renderer/cuda.py`, GEMV codegen topologies
   (four-warp gate/up, memory coalescing), no scheduler/runtime change.
4. Legality: target-derived facts only (Q4/Q6 quant, 12288x4096 and
   4096x12288 shapes, fp16 epilogue residual). Not model identity.
5. Fallback: first arm is four-warp gate/up; down q6/q4 epilogue and residual
   contract preserved exactly.
6. Contract: identical weights, quant, and output; token SHA identical.
7. Arms: isolated = NCU cold replay (retained `phase5/ffn-decomposition.json`);
   installed = fresh reverse wall bracket.
8. Census gate: 36 + 36 node counts unchanged; no copy/materialization added.
9. Reverse wall bracket (control/candidate/control), +50 us promotion bar.
10. Rollback: revert renderer/codegen; non-regression on non-NV targets.
11. Projected ceiling 163.5 us, labelled unmeasured.
12. Prohibited: model-name or block-list dispatch.
