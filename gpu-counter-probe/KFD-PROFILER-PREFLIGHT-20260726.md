# KFD profiler preflight, 2026-07-26

The committed JSON artifact was captured without `sudo`, dispatch, PMC register programming, or a GPU reset.
Run it again with:

```bash
python3 extra/qk/decode/kfd_profiler_preflight.py --output gpu-counter-probe/kfd-profiler-preflight.json
```

## Result and exact blocker

`/dev/kfd` and `/dev/dri/renderD128` are `root:render` mode `0660`, and the
current user is in `render`; ordinary KFD access is therefore not the blocker.
The profiler setup request is `AMDKFD_IOC_PROFILER` (`0xc0284b86`), operation
`KFD_IOC_PROFILER_PMC` (`op=0`), with `{gpu_id: <KFD topology GPU ID>, lock: 1,
perfcount_enable: 1}`. The existing decode PMC attempt returned `EPERM` for
that request.

This process has neither `CAP_SYS_ADMIN` nor `CAP_PERFMON` in `CapEff` and has
`perf_event_paranoid=4`. These facts explain why render-node membership does
not authorize privileged performance monitoring. The installed headers do not
contain the matching amdkfd implementation, so this report deliberately does
not claim a specific in-kernel capability check beyond the observed `EPERM`.

The default preflight invokes only `AMDKFD_IOC_GET_VERSION` and the profiler
`VERSION` operation. It does **not** invoke the PMC lock operation because a
successful lock changes perfmon state. To make that state-changing probe, an
operator must explicitly run both `--probe-pmc-lock --allow-state-change`; do
not use it in this campaign without approval and an unlock plan.

## Required operator change

Run the approved ROCm profiler interface in the intended privilege domain (or
grant only the capability/driver policy it documents), then rerun the preflight
and retain the JSON. Do not treat `sudo`, a broad device-node permission change,
or a GPU reset as a fix. A successful preflight is still only permission/tool
readiness; it does not authorize decode PMC or graph PMC work.
