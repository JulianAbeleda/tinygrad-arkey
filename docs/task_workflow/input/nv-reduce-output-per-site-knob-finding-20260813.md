# NV reduce-output per-site knob: ffn-norm site not CPU-admissible (2026-08-13)

Date: 2026-08-13
Branch: `nvidia-bringup-20260731`
Status: **measurement record. CPU-only, no GPU arm, no production change.**
Unblocks the per-site census the reduce-output blocker asked for, and reports
the empirical result it exposes.

## What changed

- Added `_decode_reduce_output_ffn_rmsnorm_promoted` in
  `tinygrad/llm/model.py`. The FFN-norm call site is now independently
  gateable while the fp32 q/k site stays on the global route (production
  unchanged: absent the knob it follows the global flag).
- Added `test/unit/test_decode_reduce_output_site_flags.py` pinning the
  q/k vs FFN split (5 tests).
- Extended `scratchpad/nv_reduce_output_rmsnorm_census.py` with
  `--arm ffn-before` and `--arm ffn`: q/k promoted + FFN closed vs q/k
  promoted + FFN promoted, both under the callify Context.

## Census result (DEV=CPU, depth 256, typed-semantic-producer on)

| arm | programs | `reduce_output_rmsnorm_1_4096` | weight stores |
| --- | ---: | ---: | ---: |
| ffn-before (q/k on, ffn off) | 734 | 36 | 0 |
| promoted (q/k on, ffn on) | 734 | 36 | 0 |

Net program delta is **0**. The selector trace is the tell: `promoted` creates
3 more `1_4096` marker entries than `ffn-before` (13 vs 10), and every one of
the 3 extra is `marker_not_eligible` (10 vs 7 rejected). So the FFN-norm
markers are created when the site is promoted but rejected at lowering; the 36
admitted `1_4096` bodies are the attention-side site gated by the global flag,
not the FFN site.

## Finding

The per-site knob works as a flag, but it does not by itself expose a body-free
M1 removal: the M1 ffn-norm `1_4096` markers (residual input `h = x + attn_out`)
do not carry the precompiled-output identity proof the current fp16-consumer
spelling requires, so they fail `marker_not_eligible` on CPU. The next CPU step
is the residual-identity proof for the ffn-norm input (declared-typed-output /
producer-STORE, step 1 of
`nv-m1-norm-epilogue-generic-primitive-scope-20260812.md`), not another flag.

No NV arm until that CPU gate passes.
