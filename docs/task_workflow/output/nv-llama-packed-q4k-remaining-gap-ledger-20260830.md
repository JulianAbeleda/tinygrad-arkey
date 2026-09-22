# Packed Q4K llama-versus-tinygrad remaining-gap ledger

The retained packed-Q4K candidate is correctness `PASS`: finite output, token
`198`, zero mismatches in the attention-arm comparison, and the complete
packed candidate census. Its nine-sample wall has minimum `41.112815 ms` and
median `41.2644 ms`. Against llama authority minimum `34.680367 ms` and
median `35.019399 ms`, the remaining gap is **6.432448 ms minimum** and
**6.245001 ms median**.

## Ranked families

1. **Attention core:** Q `4.480512 us`, K `2.205536 us`, V `3.199936 us`,
   Flash `3.341056 us`, and O `4.492096 us` of current HCQ exposure. QKV/O
   arms pass correctness. Run a matched native attention-core R9 with markers,
   complete logits, token, and role census before claiming debt reduction.
2. **Norms/elementwise:** norm/conversion `1.424128 us`, activation/multiply
   `0.745696 us`, and residual/RoPE/KV support `4.224608 us`. Run a
   low-perturbation census and exact-shape fused-kernel probes; accept only a
   whole-model minimum improvement beyond control noise.
3. **Vocabulary/final row:** vocabulary exposure `2.911360 us`. The earlier
   final-row candidate regressed the graph, so the next test must produce full
   logits, recurrent token correctness, and a complete launch census. Top-1
   proxies are not admissible.
4. **Launch/lifecycle:** the HCQ ledger has zero unknown intervals and a
   `0.569723 ms` host/graph boundary residual estimate. Run an HCQ-native
   timestamp/submit/allocation/materialization census with observer overhead
   below 2%; do not book PROFILE walls.

The HCQ figures are exposure-ranking evidence from the exact captured graph,
not additive subtraction from the 6.432448 ms wall gap. No family is promoted
by the existing evidence. GPU tests, if authorized, must use a fresh matched
R9 and serialize under `flock -w 1200 /tmp/gpu-bench.lock`; `tinygrad/llm/model.py`
must remain untouched.

Machine-readable ledger: `docs/task_workflow/evidence/nv-llama-packed-q4k-model-20260830/remaining-gap-ledger.json`.
