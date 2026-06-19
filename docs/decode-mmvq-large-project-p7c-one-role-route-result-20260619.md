# Decode MMVQ large project P7c one-role route result - 2026-06-19

Purpose: execute `decode-mmvq-large-project-p7c-one-role-route-scope-20260619.md`.

Artifacts:

- `tinygrad/llm/model.py`
- `extra/qk_decode_mmvq_p7c_one_role_smoke.py`
- `bench/qk-decode-mmvq-large-project/p7c_one_role_smoke.json`

## Result

Verdict: **PASS_ONE_ROLE_ROUTE_SMOKE**.

P7c routes one real model role through the imported Q4_K MMVQ primitive behind a default-off research flag:

```text
DECODE_MMVQ_IMPORT_Q4=1
blk.0.attn_output
real GGUF Q4_K weights
real block activation
model.py side-buffer lifecycle
imported q8 producer + imported llama Q4 consumer
```

The route is intentionally narrow:

- AMD only;
- decode shape only: `B=1`, `T=1`, `dim=4096`;
- `TransformerBlock.attn_output` only;
- requires `q4k_storage`;
- persistent per-block q8/out side buffers;
- flag unset keeps the original `self.attn_output(out_in)` path.

## Smoke

`extra/qk_decode_mmvq_p7c_one_role_smoke.py` loads Qwen3-8B Q4_K_M, preinstalls the imported `rows=4096` programs, runs
`block._attention(block.attn_norm(x), 0)`, and checks that block 0 allocated the imported-route side buffers.

Recorded result:

| field | value |
|---|---|
| role | `blk.0.attn_output` |
| output shape | `[1, 1, 4096]` |
| routed blocks | `[0]` |
| q8 launch | global `[1,1,1]`, local `[128,1,1]` |
| MMVQ launch | global `[4096,1,1]`, local `[32,1,1]` |

## Integration Notes

The imported program install must happen outside the precompiled model function. A lazy first install from inside
`TransformerBlock._attention` hits tinygrad's device-usage guard during precompile. P7c therefore treats install as
setup and the model path as launch-only.

The smoke uses one block instead of a full `model(...)` call. A full forward is not the P7c gate and exceeded available
allocation headroom while building the whole JIT graph on this host. The block-level route still exercises the real
`model.py` branch, real GGUF weights, real activation, and real persistent side-buffer lifecycle.

## Consequence

The imported Q4 path has crossed from probe-only to model-integrated, still behind a research flag.

Next phase:

1. P7d timing: compare `blk.0.attn_output` baseline vs imported route with clock-controlled, interleaved measurement.
2. P7e quality: dNLL or logits diff on the q8 activation path.
3. P7f coverage: extend from `attn_output` to `ffn_gate/up` if P7d clears a local speed gate.
4. P7g W==D verdict: only after timing and quality gates pass.
