# NV Flash live qualification — 2026-08-29

Status: **PASS** for the research-only observer seam on a real Qwen3-8B pp512 forward.

- Fresh model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`, seed `20260829`, 512 tokens, `start_pos=0`.
- The HCQGraph construction observer captured 26 graph calls and 1,018 admission events. Exactly one selected call was `nv_sm120_q16_grid_hd128_loop_attention` (graph index 6).
- Bound ABI is slots `out=0`, `Q=1`, `K=2`, `V=3`; byte sizes are `4194304, 4194304, 1048576, 1048576`; launch geometry is global `(1024,1,1)`, local `(32,1,1)`.
- All captured arrays are finite. The candidate covers all `2,097,152` output elements. Independent fp32 score/reduction oracle comparison gives max absolute error `0.00930290`, mean absolute error `0.000178066`, and passes `rtol=2e-2, atol=2e-3`.
- Replay freshness passed: output and input hashes were unchanged on a subsequent graph replay; input slots 1–3 remained readonly.
- Matched post-capture wall: baseline `84.234 ms`, observer `84.137 ms` (delta `-0.097 ms`).

Focused graph tests: `22 passed, 1 skipped` (`test_graph_admission`, `test_hcq_graph_profile_export`, `test_hcq_nv_ready_placement`, `test_nv_prefill_k_program_discovery`).

Raw evidence: [`live-qualification.json`](live-qualification.json), [`oracle.json`](oracle.json).
