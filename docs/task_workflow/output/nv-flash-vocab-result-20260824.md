# NV flash-combine and vocab-tail result

Date: 2026-08-24  
Repo: `/home/ubuntu/tinygrad-arkey`  
GPU: RTX 5090 (`NV sm_120`), model: Qwen3-8B-Q4_K_M

## Outcome

The two clean ledger targets produced one promotion:

1. Wider flash combine is exact but does not improve production wall time.
   Production remains at 32 lanes.
2. A one-CTA native fp32/int32 vocab argmax is exact, removes the three serial
   scheduler reductions, and recovers `56.386 us/token` in a reps=9 reverse
   bracket. It is promoted only for `NV sm_120`.

The new comparable plain-wall endpoint is `4.578813 ms/token`, or
`218.397 tok/s`. The remaining gap to 240 is `412.146 us/token`.

## 1. Flash combine: rejected

All 32-, 64-, and 128-lane combine kernels produce identical fp16 output.
Only the combine width changes; the score/PV tile remains the installed
32-lane kernel, and all arms retain 36 combine calls per token.

| width | combine device us/token | result |
| ---: | ---: | --- |
| 32 | 103.568 | control |
| 64 | 125.392 | `+21.824 us`, reject before wall |
| 128 | 96.080 | `-7.488 us`, advance to wall |

The 128-lane device saving did not survive the plain reverse bracket:

| control A ms | candidate ms | control C ms | candidate vs midpoint |
| ---: | ---: | ---: | ---: |
| 4.606416 | 4.604076 | 4.599669 | `+1.034 us/token` slower |

The token streams match, but the candidate loses to Control C and to the
control midpoint. This is a clean `NO_GO_WALL`; no target policy promotes a
wider combine.

## 2. Vocab tail: promoted

The accepted construction does not retry the rejected packed-u64 top-1 path.
It materializes the existing fp32 sampled-score row, then performs a paired
value/index reduction in one 1024-thread CTA:

- every thread scans a strided slice in registers;
- each warp reduces `(fp32 value, int32 first index)` with shuffles;
- warp zero reduces the 32 shared warp winners;
- exact value ties choose the lowest index, including `-0.0 == +0.0`;
- a held one-element clone preserves the JIT feedback-output lifetime.

Random finite logits, first-index ties, signed zero, all-equal rows, and finite
extrema all match ordinary `Tensor.argmax` for 256, 512, and 1024 threads.
The first full-model attempt exposed a stale held-output view; it was rejected,
root-caused, and fixed before qualification. The final warmup sequence and all
27 timed token windows match both controls.

The three old reduction rows contribute `53.120 us/token`; the native body is
`8.864 us/token`. The decisive wall result is:

| control A ms | candidate ms | control C ms | recovery | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 4.605720 | 4.549334 | 4.605719 | 56.386 us/token | 1.239% |

The route is selected by
`tinygrad/llm/generated/decode-native-argmax-route-policy.json` only for
`NV sm_120`. `TINYGRAD_NATIVE_ARGMAX_DISABLE=1` restores the scheduler tail at
model load.

## Updated installed ledger

The fresh no-override production profile closes as:

```text
scheduled nodes  516
node sum          4352.288 us/token
device union      4349.500 us/token
overlap              2.788 us/token
```

Versus the preceding ledger, this is one fewer node, `41.488 us` less node
sum, and `41.250 us` less union. The graph contains
`native_finite_fp32_argmax_151936_t1024` and none of
`r_32_4_1187`, `r_128_16_8_1187`, or `r_16_8`.

The comparable reps=15 plain production endpoint moves from
`4.635946` to `4.578813 ms/token`, a `57.133 us/token` recovery:

```text
current          218.397 tok/s
target           240.000 tok/s
remaining        412.146 us/token
latency cut left   9.8915%
```

Holding the independent llama node-sum authority fixed, the remaining
tinygrad-minus-llama device gap falls from `515.506` to `474.018 us/token`.
The vocab role residual is inferred to fall from `65.440` to about
`23.952 us/token`; this inference subtracts the measured installed node-sum
change and does not pretend to be a fresh llama capture.

## Next sequence

Flash combine width and the serial vocab tail are now adjudicated. The next
work should be production-conditioned rather than another isolated body
rewrite:

1. remove a real launch/output boundary in the K/V-to-cache or
   score-to-combine chain while preserving the installed arithmetic;
2. test whether a projection construction can raise achieved streaming rate
   or remove output transport for down/gate-up/Q/O without claiming DRAM-byte
   savings;
3. re-run the same wall and installed ledgers after each accepted change.

Structured evidence is in
`docs/task_workflow/evidence/nv-post-flash-vocab-ledger-20260824/`, with raw
flash evidence under `nv-flash-combine-width-20260824/` and vocab evidence
under `nv-native-argmax-20260824/`.

Verification: `104 passed` focused tests. Verdict:
`ONE_PROMOTION_FLASH_NO_GO_218_397_TOK_S_240_NOT_REACHED`.
