# NV parity resume playbook (2026-08-19)

Status: parked. Everything below is measured or explicitly flagged as a
ceiling, not a claim. Goal remains: close the decode wall gap to llama.cpp on
Qwen3-8B-Q4_K_M / RTX 5090 sm_120 / `DEV=NV`.

## Measured anchors (same-session, flocked)

| side | tok/s | us/token |
| --- | ---: | ---: |
| tinygrad landed (2 GPFIFO + reuse lanes) | 210.97 | 4740.1 |
| tinygrad serial | 205.99 | 4854.6 |
| llama same-session | 246.37 | 4058.9 |
| llama-bench fresh | 254.4 | 3931 |

Fair gap is vs the same-session llama anchor: `-685.6 us/token` (`-35.4 tok/s`).

## Wall equation (first principles)

```text
device wall = critical_path + unoverlapped_nonpath
tinygrad     = 4187            + 332             = 4519 us (overlap 0)
llama        = 3902 span       (overlap ~946)   = ~3902 us
device gap   = 617 us

host gap: tinygrad ~221 vs llama ~157 = +64 us
total gap  = 681 us
```

The 617 us device gap splits into:

```text
critical-path gap  285 us = support +626 - gemv -329
unoverlapped work  332 us
```

tinygrad's gemv chain is already 329 us faster than llama's mmq chain
(3259 vs 3588). The loss is non-gemv support: tinygrad carries 928 us of it
on the dependency spine, llama exposes ~302 us.

Critical-path classes (from the stale `PROFILE=1` DAG, see risk note below):
gemv 3259, flash 314, reduce_output 235, rmsnorm 155, residual 135,
scatter_vocab 56, q8_provider 33 = 4187 us.

## Exhaustive lever ledger (tested status)

| # | lever | ceiling | status |
| --- | --- | ---: | --- |
| 1 | flash score body -> llama fattn-vec shape | ~163-248 (was claimed) | **FALSIFIED** (see below) |
| 2 | q/k norm launch geometry -> llama 1.30 us | ~130-147 | landed P1 +55-67; remainder bitwise-gated |
| 3 | 1x4096 norm tree/warp reduce | ~97 (body), 2.45x | real; NOT bitwise-gated for fp16 (see below) |
| 4 | residual/cast glue into gemv epilogue | ~135 | construction NO-GO (opaque custom gemv) |
| 5 | vocab scatter -> host top-1 | ~56 | measured ~5.6 us real (closed) |
| 6 | gemv speedup | already -329 | bonus only |
| 7 | more queues / overlap | +26 max | closed (2 queues landed +140) |
| 8 | host gap / replay merge | ~64 | measured flat/negative (closed) |

## Fresh tests this session

### Flash body claim is falsified

Device-side CUPTI (nsys, 400 back-to-back launches):

| kernel | us/launch |
| --- | ---: |
| tinygrad production tile (S=48) | 4.16 |
| tinygrad llama-shape transcription (S=4) | 24.7 |
| llama `flash_attn_ext_vec` (matched isolated, 08-16) | 4.10 |

tinygrad's production flash score is already at llama body parity. The
llama-shape transcription is 6x slower, not faster. The earlier "flash body
gap" was an in-situ-vs-isolated comparison artifact. There is no flash
codegen win. The remaining flash gap (~143 us) is overlap, measured at 33 us
ceiling (below the +50 us bar).

### Norm warp-reduce body speed + SHA flip

New harness `extra/llm_research/decode/nv_norm_body_device_timing.py`,
nsys CUPTI, isolated realized buffers:

| shape | serial (bitwise) | warp-reduce | speedup |
| --- | ---: | ---: | ---: |
| 1x4096 (ffn/output) | 5.09 us | 2.08 us | 2.45x |
| q 32x128 | 1.28 us | 1.18 us | 1.08x |
| k 8x128 | 1.25 us | 1.15 us | 1.08x |

SHA flip splits by output dtype:

| norm | output | warp-reduce flips SHA? |
| --- | --- | --- |
| ffn 1x4096 | fp16 | no (bitwise identical) |
| attn 1x4096 | fp16 | no |
| q/k 32x128, 8x128 | fp32 | yes |
| output 1x4096 | fp32 | yes |

Evidence: `docs/task_workflow/evidence/nv-norm-warp-vs-serial-device-20260819.json`.

## Key reframe: bitwise is not the blocker

The big norm win (2.45x on 1x4096) is available WITHOUT touching the
bitwise contract, because the fp16 output swallows the association
difference. The fp32 norms where bitwise actually matters only gain ~8%.

The real blocker is view-passing materialization, located at
`tinygrad/schedule/rangeify.py:249`:

```python
x_arg = x.reshape(numel).contiguous()   # materializes 144 copies/token
```

The fast warp-reduce norm (`rmsnorm_native_*`) is byte-identical when forced
open, but the opaque boundary copies every lazy input (+144 kernels/token,
~+142 us from the Path 3 / M3 records). The manual `kernel.call` alternative
crashes `symbolic` with `bad reshape: () -> (4096,)` when the lazy producer
collapses. This is the same scheduler substrate gap as the overlap question,
and it is precisely scoped, not vague.

## Full-token wall A/B state (partially run)

New harness `extra/llm_research/decode/nv_norm_native_wall_ab.py`.

Smoke (count 8, reps 1), ffn site:

| arm | ms/token | token SHA |
| --- | ---: | --- |
| control | 4.753 | 9e6664fd... |
| candidate ffn (warp-reduce) | 4.738 | 9e6664fd... (identical) |

ffn native norm is token-identical and wall-neutral-to-positive even with
the copies in the smoke. This needs the proper bracket to trust.

Proper bracket (count 24, reps 4), ffn site:

| arm | result |
| --- | --- |
| control A | 4.7267 ms/token = 211.57 tok/s, token SHA 1e73e557... (done) |
| candidate ffn | ABORTED mid-run (needs re-run) |
| control B | not run |

## Next steps (ordered)

1. Re-run the aborted ffn bracket, then a full 1x4096 bracket
   (`--sites attn,ffn,output`), control/candidate/control, under flock:

   ```bash
   cd /home/ubuntu/tinygrad-arkey
   flock -w 600 /tmp/gpu-bench.lock env PYTHONPATH=. DEV=NV \
     python3 -m extra.llm_research.decode.nv_norm_native_wall_ab \
     --arm candidate --sites ffn --count 24 --reps 4 --out /tmp/nn_cand.json
   ```

   Decision bar: token SHA must equal control; wall must clear +50 us/token.
   The smoke already proves SHA identity for the fp16 ffn site.

2. If wall is neutral/positive even with the 144 copies, the view-passing fix
   is the highest-value build: removing ~144 copies is ~-216 us
   (~220 tok/s) before the 2.45x body win is even counted. Scope it from
   `rangeify.py:249` plus the `symbolic` collapse, then re-test.

3. After view-passing lands, re-run the native norm bracket and book it if it
   clears the bar.

4. Re-address overlap only after the above: the co-schedule scan measured a
   33 us ceiling on this topology, so it is likely closed; do not reopen it
   without a new mechanism.

## Open questions and risks

- The 4187 us critical path comes from a stale `PROFILE=1` DAG whose durations
  are inflated at JIT replay boundaries. Production device busy is 4519 us
  (node_sum, overlap 0). A clean `PROFILE=0` critical-path re-derivation is
  still owed before trusting exact end-state arithmetic.
- The native norm materialization penalty at current HEAD is not yet measured
  with a proper bracket (only smoke). The old Path 3 `-1%` was at ~172 tok/s,
  before the host-gap / S4 PDL / reuse-lane landings.
- Relaxing bitwise for the fp32 output norm is a real decision but low value
  (~8% body gain); the fp16 norm win does not require it.

## Artifacts created this session

- `extra/llm_research/decode/nv_norm_body_device_timing.py`
- `extra/llm_research/decode/nv_norm_native_wall_ab.py`
- `docs/task_workflow/evidence/nv-norm-warp-vs-serial-device-20260819.json`
- nsys traces under `/tmp/nv_flash_body_fresh*` and `/tmp/nv_norm_body_fresh*`
