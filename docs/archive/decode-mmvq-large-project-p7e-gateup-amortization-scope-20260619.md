# Decode MMVQ large project P7e gate/up amortization scope - 2026-06-19

Purpose: answer the fresh question left open by P7d: does the imported Q4_K lifecycle become worthwhile for the
`ffn_gate/up` pair, where one q8 activation can feed two larger `12288 x 4096` consumers?

## Target

Measure only the FFN input-sharing pair:

- `blk.0.ffn_gate`;
- `blk.0.ffn_up`;
- real Qwen3-8B Q4_K_M weights;
- real FFN input: `block.ffn_norm(x + block._attention(block.attn_norm(x), 0))`;
- baseline: current tinygrad `ffn_gate(inp)` and `ffn_up(inp)`;
- candidate: one q8 producer plus imported Q4_K consumer for gate and imported Q4_K consumer for up;
- both paths timed as TinyJit graph calls, interleaved in one process.

## Why This Is Fresh

P7d refuted `attn_output`, but that role has only one 4096-row consumer. `ffn_gate/up` has three favorable differences:

- three times the rows (`12288`);
- two consumers sharing the same activation;
- one q8 producer amortized across both consumers.

## Gates

P7e passes only if:

- baseline and imported pair both run;
- imported pair replay is stable;
- candidate median wall time is at least `1.10x` faster than baseline;
- q8-path numerical drift is recorded for both outputs.

If P7e fails, the imported Q4 decode route is not a local timing win for the two Q4 buckets we can currently import
(`attn_output` and `ffn_gate/up`), and the large MMVQ path should stop before model-wide routing.
