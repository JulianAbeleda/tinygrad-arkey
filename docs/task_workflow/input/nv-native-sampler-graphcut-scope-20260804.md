# Native NV sampler graph-cut scope

Date: 2026-08-04. Diagnostic scope only. No production route, lowering, or
default is authorized by this record.

## Established boundary

On the authority d512 prompt (`Qwen3-8B-Q4_K_M.gguf`, 151,936-token
vocabulary), native NV materializes correct final logits and a correct backend
argmax:

| expression | result |
|---|---:|
| `model.logits(... )[:, -1, :].argmax(-1).item()` | 13876 |
| host argmax of the same materialized row | 13876 |
| `model(tokens, 0, Tensor([0.0]), use_flash=False).item()` | 151936 |

`151936` is one-past-vocabulary and therefore invalid. The construction uses
the fused-prefill-off convention *before model construction*; see
`scratchpad/shared_prompt_final_logits_probe.py`.

## Refuted local changes

The following were evaluated only in a disposable working tree and reverted.
All still returned 151936 in the full production forward graph:

1. Equivalent finite Gumbel form `argmax(logits - max(t,1e-12)*gumbel)`.
2. Plain `logits.argmax(..., keepdim=True)` in `Transformer.forward`.
3. `contiguous()` and `clone()` between logits and that argmax.
4. `where(temperature == 0, greedy, legacy_sampled)`.

Small standalone NV and TinyJit wide-vocabulary argmax controls are correct.
Thus this is not established as an RNG, temperature arithmetic, keepdim, or
ordinary TinyJit argmax defect. The remaining minimal owner is the final wide
argmax/scalar result when it is compiled in the full production model graph.

## Next candidate: pre-trace graph cut

Implement only in an explicitly authorized experiment:

1. Add a dedicated greedy `forward_greedy(tokens, start_pos)` whose final
   operation is `logits.argmax(-1, keepdim=True)` and a separately captured
   `TinyJit` selected by Python control before tracing.
2. Select it only from `generate(..., temperature=0.0)`, where the Python
   float is authoritative. Do not inspect a device Tensor with `.item()` in
   the hot path and do not change the generic `model(..., temperature)` ABI.
3. Keep the current sampled `forward` byte-for-byte for every nonzero
   temperature and for direct generic callers.
4. Validate native NV shared d512: in-range output exactly 13876; validate a
   fixed-depth generated token stream; then run the CUDA control to establish
   that the greedy branch itself is backend-neutral.
5. Measure d512/d2048 graph node count, steady wall time, and any copy node.
   Reject the candidate if it adds a host synchronization or an unbounded copy
   tax. Promotion requires a separately approved parity/correctness scope.

If a separate pre-trace greedy graph still returns 151936, stop: the owner is
below model sampling and needs a captured-program/UOp-level trace of the final
reduction and output buffer rather than another model-level rewrite.
