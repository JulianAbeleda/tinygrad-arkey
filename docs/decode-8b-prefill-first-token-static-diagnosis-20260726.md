# 8B decode `prefill_first_token` static diagnosis (2026-07-26)

## Scope

This is a CPU/static comparison only.  No GPU command was run, and this does
not claim a runtime fix.

## Finding

`decode_runtime_overhead.py` reaches `prefill_first_token` through
`_warm_depth -> _prefill -> Transformer.generate -> next(gen)`.  For an exact
512-token prompt on the selected 8B prefill-v2 configuration, `generate`
does **not** use its `chunk_size` argument: it takes its `prefill_ubatch`
(512) branch and invokes one concrete `[1,512]` prefill call at start position
zero.

That call selects `prefill_v2_jits.setdefault(0, TinyJit(self.forward))`.
The selected concrete-KV JIT is explicitly lazy, so a new model instance pays
the first compile in `next(gen)`.  The source documents this as approximately
five seconds per previously unseen concrete chunk offset.  Thus a long silent
period at the new `prefill_first_token` lifecycle stage is consistent with a
cold prefill compilation; it is not evidence that the decode-token JIT or the
measurement loop has started.

The first warm request should cache the start-position-zero concrete JIT on
the same model instance.  The decoder then proceeds to its flash/SDPA decode
warmup and only afterwards enters measurement.  Every independently-created
decode authority process repeats the cold first-prefill cost.

## Comparison with the working prefill authority

`prefill_whole_synced.py` constructs the same model and uses the same
`max_context=4608`, but it bypasses `generate`.  Its timing body calls
`model(chunk, concrete_start_pos, temp, use_flash=True)` with a concrete
`[1,512]` tensor, realizes it during explicit warmups, and synchronizes around
the timed burst.  Its smoke profile limits that to start position zero,
`K=1`, `warmups=1`, and `rounds=1`.

Consequently, the smallest discriminating invocation is the existing
prefill-only smoke authority, not a decode measurement:

```sh
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk/prefill/prefill_whole_synced.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --model-profile qwen3_8b_q4k_m_gfx1100 --mode smoke \
  -K 1 --warmups 1 --rounds 1 --start-positions 0 --whole-lengths 512 \
  --max-context 4608 --no-artifact
```

This command exercises the one cold `[1,512]`, start-position-zero prefill
JIT that decode needs before first token, while excluding generation feedback,
sampling/readback, decode JIT capture, repetitions, and other prompt depths.
It is the direct prefill-only command expected to reach completion when the
stalled stage is merely the normal cold compile.

## Non-discriminator

Changing decode's `--chunk-size` from 32 to 512 is **not** a meaningful test
for this 512-token case.  With `prefill_v2` enabled and at least 512 prompt
tokens remaining, `Transformer.generate` selects `config.prefill_ubatch`
(512) before it reaches the fallback that consults `chunk_size`.  Both values
therefore compile/run the same start-position-zero concrete prefill JIT.

## Boundaries and follow-up

If the smoke authority also remains stuck before its first output, the static
evidence localizes the issue to the shared load/first-prefill compile route,
not decode.  If it completes but decode remains stuck at `prefill_first_token`,
the next discriminating instrumentation is to report immediately before and
after `_prefill`/`next(gen)` plus the model's `prefill_v2_jits` keys; this is
observability only, not a runtime correction.

Relevant source anchors: `extra/qk/decode/decode_runtime_overhead.py`
(`_prefill`, `_warm_depth`), `tinygrad/llm/model.py` (`generate`, `__call__`,
`prefill_v2_jits`), and `extra/qk/prefill/prefill_whole_synced.py`
(`prefill_authority`/`burst`).
