# NV pp512 K/V split-K occupancy discriminator

## Verdict

The locally expressible split-K composition is a decisive no-go.  Splitting
the exact FP16 `(512,1024,4096)` projection into 2/4/8 independent GEMMs and
charging the FP32 fixup makes the complete lifecycle 1.80x/2.04x/2.18x slower
than the installed 32-CTA candidate in the hot R9 bracket.  All arms are finite
and pass the declared tolerance (`atol=0.125`, `rtol=0.002`; observed maximum
absolute error at most `4.883e-4`).

This does **not** disprove a native single-launch stream-K kernel.  Tinygrad has
no local FP16 stream-K body for this production shape.  The measured split
arms are serial partial launches (32 concurrent CTAs per launch) followed by a
fixup, not llama's 170-CTA main kernel.  Their purpose is to reject graph-level
split-K composition and establish that a new native substrate is required.
No production route was changed.

## R9 hot and fresh-process R7

| arm | aggregate CTAs | hot R9 min | cold R7 min | hot speedup | correctness |
| --- | ---: | ---: | ---: | ---: | --- |
| installed FP16 candidate | 32 | 278.413 us | 278.283 us | 1.000x | retained bit-exact authority |
| split-K 2 + reduction | 64 serial | 501.173 us | 511.903 us | 0.556x | PASS, max abs 4.883e-4 |
| split-K 4 + reduction | 128 serial | 568.597 us | 576.552 us | 0.490x | PASS, max abs 3.662e-4 |
| split-K 8 + reduction | 256 serial | 606.797 us | 629.570 us | 0.459x | PASS, max abs 3.662e-4 |

These are synchronized TinyJit call walls and therefore include graph launch
cost.  The lower-overhead profile authority for the installed body remains
177.3 us/call.  The close hot/cold control agreement confirms the discriminator
itself is stable, while its absolute wall must not replace the profiler's
kernel-body number.

At 72 K/V projections per pp512 pass, substituting the measured compositions
would add approximately 16.0 ms (split-2), 20.9 ms (split-4), or 23.6 ms
(split-8) to the synchronized substrate wall.  Recoverable pp512 wall is
therefore zero for these arms.

## Roofline and required native gate

The installed K/V population consumes 12.769 ms in the profiled pass.  Its
algorithmic FP16 tiled-weight payload is 33.55 MB/call, so even an optimistic
1.8 TB/s DRAM roof is about 18.6 us/call.  The exact GEMM contains 2.147 GMAC;
at the measured 127.7 TMAC/s FP16 issue ceiling its compute roof is about
16.8 us/call.  The roofline lower bound is therefore roughly 19 us before
launch and epilogue costs, versus 177.3 us observed: topology leaves real
headroom, but the serial composition cannot access it.

A faithful next gate must be one main kernel with about 160--192 resident work
units, each owning a K interval, plus one coalesced fixup.  It must measure the
main+fixup pair together and beat 177.3 us.  Useful thresholds are:

- 100 us/call: 5.57 ms pp512 recovery;
- 60 us/call: 8.45 ms recovery;
- 35 us/call: 10.25 ms recovery;
- 25 us/call: 10.97 ms recovery and near the analytic FP16 roof.

The retained llama compressed lifecycle (Q8 conversion + Q4/Q6 MMQ + fixup)
runs around 20 us for Q4 K/V and 31 us for Q6 V.  That arm is authoritative
evidence for the target, but it is not locally callable through this isolated
FP16 harness and was not relabeled as a tinygrad result.

## Reproduction and evidence

Harness:
`extra/llm_research/prefill/nv_prefill_kv_splitk_discriminator.py`

Run with:

```sh
DEV=NV PYTHONPATH=. python3 extra/llm_research/prefill/nv_prefill_kv_splitk_discriminator.py \
  --rounds 9 --warmups 3 \
  --out docs/task_workflow/evidence/nv-prefill-kv-splitk-20260828/hot-r9.json
```

Evidence:

- `docs/task_workflow/evidence/nv-prefill-kv-splitk-20260828/hot-r9.json`
- `docs/task_workflow/evidence/nv-prefill-kv-splitk-20260828/cold-r7.json`
