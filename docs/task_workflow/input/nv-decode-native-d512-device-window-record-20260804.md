# Native NV d512 decode device-window record

Date: 2026-08-04. Route: `DEV=NV`; model: Qwen3-8B-Q4_K_M; depth 512;
RTX 5090 / driver 595.84. Status: **outer token-gap equation reconciled within
4.75 us (0.29%); semantic device partition remains open.**

## Direct result

`scratchpad/nv_decode_group_window_ledger.py` inserts native queue timestamp
markers immediately before the first and after the fifth production HCQ graph
submission. It does not change graph programs, buffers, or dependencies. Five
alternating marker-off/marker-on runs all emitted token `330` and reproduced the
same five group hashes.

| native boundary | median us/token |
| --- | ---: |
| marker-free token wall | 5613.208 |
| marked token wall | 5642.484 |
| native graph-device window | 5291.424 |
| marked outside-window remainder | 347.915 |
| marker wall perturbation | +29.276 (0.522%) |

The five graph groups contain `32 + 64 + 128 + 256 + 468 = 948` programs. This
is the native topology, not the 1021-program CUDA diagnostic topology. In
particular, native NV already contains the promoted tinygrad w1w3 fusion.

## Reconciliation against llama

The independent llama unprofiled boundary is 3889.808 us of graph-device span
and 81.979 us outside the graph, or 3971.787 us total. Use the marker-free
native wall for the outside term and the stable marked native timestamp for the
device term:

```text
native device delta = 5291.424 - 3889.808 = 1401.616 us
native outside      = 5613.207933 - 5291.424 = 321.783933 us
outside delta       = 321.783933 - 81.979 = 239.804933 us
diagnostic sum      = 1401.616 + 239.804933 = 1641.420933 us
authority gap       = 1646.170000 us
reconciliation      = 4.749067 us (0.288%)
```

Thus the measured authority gap is boundary-accounted within the campaign
tolerance `max(50 us, 2%)`:

- **85.14%** is extra native graph-device time;
- **14.57%** is extra work outside the graph-device window;
- **0.29%** is cross-session/median reconciliation error.

This is not yet the requested semantic explanation. The 1401.616 us device
term must still be partitioned by disjoint real-token family A/Bs or native
class attribution. The 239.805 us outside term must be split into pre-dispatch,
graph update/submission, synchronization/copyout, and Python sampling work.

The raw five-repetition payload is `/tmp/nv_decode_group_window_ledger_20260804.json`.
The compact durable payload is
`docs/task_workflow/output/nv-decode-native-d512-device-window-20260804.json`.
