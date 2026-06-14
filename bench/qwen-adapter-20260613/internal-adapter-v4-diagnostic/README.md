# Internal Adapter V4 Diagnostic

This artifact records the first attempt to move beyond output-head LoRA for the
strict JSON-answer gate.

## What Was Added

- `lastN_ffn` target groups expand to exact dense FFN module paths:
  `blk.*.ffn_gate`, `blk.*.ffn_up`, and `blk.*.ffn_down`.
- Unknown target groups and zero-install expansions fail loudly.
- Internal adapters preserve activation gradients; output-only adapters keep the
  original exact-preserving detached behavior.
- The adapter trainer has a plain-block training path so internal adapter params
  are not hidden behind the block-level precompiled wrapper.

## What Failed

- Training internal adapters on the generated QK inference path fails during
  backprop through frozen quant bit-unpack ops (`Ops.OR`).
- Training with fully realized fp16 weights gets past that but OOMs on 8B:
  `23.78 GB` used and a `1.64 MB` allocation fails.
- Baseline/no-REALIZE training can run a one-step smoke, but full `last4_ffn`
  and reduced `last1_ffn` runs were too slow to be usable as the next gate and
  were manually stopped before writing artifacts.

## Smoke Result

A one-step baseline/no-REALIZE `last4_ffn` smoke over 10 rows passed:

- status: `pass`
- adapter kind: `lora`
- installed target group: `last4_ffn`
- adapter L2 delta: `0.876816`
- train loss: `0.0828 -> 0.0772`
- eval loss: `0.3223 -> 0.3159`

This proves the internal-adapter graph can train in principle. It does not prove
held-out generation improvement.

## Verdict

Do not promote V4 and do not run more target sweeps yet. The next engineering
step is a dedicated internal-adapter training path that is both differentiable
and practical on 8B: lower memory than `REALIZE=1`, faster than plain-block
no-REALIZE, and still compatible with loading the resulting adapter into the
generated QK inference path.
