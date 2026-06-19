# q8 FFN handwritten A4 decode result (2026-06-19)

This is the final gate for the q8 FFN handwritten research route.

Route:

- flag: `Q8_FFN_HANDWRITTEN=1`;
- default: off;
- scope: dense decode FFN gate/up for Qwen3-8B Q4_K_M shape `4096 -> 12288`;
- primitive: hipcc/LLD fused RMSNorm+q8 producer plus fused Q4_K x q8 gate/up consumer;
- graph path: Tensor-visible placeholder `PROGRAM` nodes with runtime-cache swap, verified by contract audit and
  TinyJit replay.

## Verdict

**PASS_RESEARCH.**

The route clears both final gates:

- W==D decode speedup >=3%;
- dNLL <=0.01.

It remains research-only because it depends on external hipcc/LLD artifacts and a runtime-cache injected artifact
runner. No default behavior changes.

## Speed

Artifacts:

- `bench/q8-ffn-handwritten-oracle/decode_wd_baseline.json`
- `bench/q8-ffn-handwritten-oracle/decode_wd_q8_route.json`

Harness:

- `extra/qk_decode_runtime_overhead.py`
- W path: real decode with `.item()` per token;
- D path: dispatch-only replay with one final sync;
- same model, seed, contexts.

| ctx | baseline tok/s | q8 route tok/s | speedup |
|---:|---:|---:|---:|
| 128 | 79.5 | 84.5 | 1.063x |
| 512 | 73.0 | 77.4 | 1.060x |
| 1024 | 71.3 | 75.4 | 1.058x |
| 4096 | 65.1 | 68.4 | 1.051x |

Minimum speedup: **1.051x**.

Average over these rows: **1.058x**.

## Quality

Artifacts:

- `bench/q8-ffn-handwritten-oracle/nll_baseline.json`
- `bench/q8-ffn-handwritten-oracle/nll_q8_route.json`

Harness:

- `extra/qk_nll_eval.py`;
- teacher-forced decode-path NLL;
- 160 tokens;
- same seed/window as earlier q8 quality proxy.

| route | NLL |
|---|---:|
| baseline | 2.855476 |
| q8 route | 2.858363 |
| dNLL | **+0.002887** |

Quality gate: `dNLL <= 0.01` -> **PASS**.

## Important correction

The first actual-route dNLL attempt failed catastrophically (`dNLL +12.25`) because the route was placed inside
`_feed_forward`, whose input is already `ffn_norm(h)`. The artifact producer also performs RMSNorm, so that path
double-normalized the FFN input.

The fixed route is one level higher, in `FFNBlock.__call__`:

`h -> q8 producer does ffn_norm(h) + q8 -> fused gate/up -> ffn_down -> h + routed`

That matches the one-block probe and restores quality.

## Current status

| gate | status |
|---|---|
| q8 activation quality proxy | PASS |
| isolated lifecycle | PASS (`114.12 us`) |
| eager one-block route | PASS (`121.38 us`) |
| graph injection contract | PASS |
| TinyJit replay | PASS |
| W==D decode | PASS (`+5.1-6.3%`) |
| actual-route dNLL | PASS (`+0.002887`) |

## Next decision

This proves the research route is valid. Remaining choices:

1. keep it as a research flag only;
2. transfer the primitive into tinygrad-owned codegen/ASM to remove the external artifact dependency;
3. evaluate whether the maintenance cost is worth a roughly 5-6% decode gain on this 8B path.
