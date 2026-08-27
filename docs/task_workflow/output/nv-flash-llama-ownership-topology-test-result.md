# NV Flash llama ownership-topology test result

## Outcome

The llama-style ownership theory was constructed and tested in four causal
stages. None qualifies for production investment on tinygrad's current
eight-column/group Flash topology. All flags remain closed-default.

| construction | registers | repeated median | conditioned cold | result |
|---|---:|---:|---:|---|
| control | 96 | about 4.085 us | 5.920-6.080 us | baseline |
| shared score/probability ownership, FP32 PV | 70 | 4.253 us | 7.104 us | ownership real; service negative |
| shared ownership + complete early V tile, FP32 PV | 167 | 4.540 us | 7.104 us | negative |
| above + packed FP16 PV | 148 | 4.558 us | 6.880 us | negative |
| packed topology + warp-local probability barriers | 148 | 4.393 us | 6.976 us | best faithful arm; negative |

The synthetic partial-output harness remained bit-exact for its deterministic
input, and every arm had zero spills. The packed arm must still be treated as
numerically non-equivalent in general because FP16 loop-carried PV state can
round at every update; performance fails before that broader numerical gate is
needed.

## What the test establishes

Shared ownership does exactly what the theory predicts locally: it reduces the
score/probability register population from 96 to 70 registers. Packed PV also
reduces the complete early-V construction from 167 to 148 registers. Replacing
the two probability-exchange CTA barriers with warp barriers improves the
repeated result by about 0.17 us.

Those changes do not yield llama's cold service rate. The faithful warp arm is
still about 1.06 us slower conditioned-cold than its matched control. DRAM
bytes and sectors remain effectively fixed, so the loss is internal service:
extra probability shared traffic, synchronization, conversion/update work,
and the 148-register early-V live set outweigh the latency overlap.

This means the previously stated ownership chain is an accurate description
of llama but not a transferable isolated lever. Llama's result depends on its
entire token/warp mapping and update grammar, not merely these features:

```text
single-score lane ownership + warp-local probability exchange + dense V issue cadence
```

Tinygrad's current kernel assigns each 8-lane group eight columns and retains a
different score/reduction grammar. Adding llama's ownership components to that
grammar creates extra work rather than reproducing llama's kernel.

### NVIDIA-path correction

The packed-PV arm was a research construction, not a faithful description of
the retained NVIDIA llama binary. In the pinned llama source,
`V_DOT2_F32_F16_AVAILABLE` selects the packed `half2` PV path, and that macro is
only enabled for the relevant HIP/RDNA path. The NVIDIA build therefore uses
`float2` PV accumulation. Packed FP16 PV is closed both numerically and as an
explanation of llama's NVIDIA result.

## Disposition

No token bracket is warranted and no tok/s recovery is booked. A genuine retry
would need a clean-room port of llama's complete warp/token mapping as a new
kernel, not more flags layered onto the current emitter. That is a kernel
replacement project with its own numerical contract, not a V-schedule tweak.

## Evidence

- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/vsharedprob-counter.json`
- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/vsharedprob-vdim-counter.json`
- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/vllama-topology-counter.json`
- `docs/task_workflow/evidence/nv-flash-v-schedule-20260827/vllama-topology-warp-counter.json`
- `extra/llm_research/decode/nv_flash_v_schedule_counter_probe.py`
