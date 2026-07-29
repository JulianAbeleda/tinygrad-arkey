# TinyGPU v11 early tunnel-observation scope

Date: 2026-07-29

Status: host-side recorder implemented; hardware execution not yet authorized.
This scope defines the next single-transition R6.2 observation. It does not
authorize an automatic reboot, reset, replug, cable/port change, DEXT install,
AMD initialization, socket server, BAR access, or workload.

Repository/worktree: `/Users/julianabeleda/env/tinygrad-arkey-exp`, branch
`exp`, starting HEAD `3900b2c96446c9bfee46f9e1a8aac56a4a1e8e72`.

Audit:
`docs/task_workflow/output/egpu-usb4-acio-tunnel-boundary-audit-20260729T030650Z.md`.

Runtime evidence:
`docs/task_workflow/output/egpu-usb4-v11-post-reboot-R6.1-20260729T025229Z.md`.

## Objective

Capture the evidence the prior delayed R6.1 missed: v11 handshake, keepalive,
power confirmation, and the PCI/tunnel registry state before any ACIO link
loss. Preserve one-Hz paired samples incrementally so the artifact survives the
first provider-query failure.

This observation distinguishes two branches:

1. `IOPCITunnelL1Enable=false` is already present before another identical
   failure: the Apple reference implementation's tunnel veto is not sufficient,
   and a BAR-descriptor v12 is rejected.
2. the property is absent or true while v11 power and canary state are healthy:
   a separately scoped, source-only BAR-descriptor candidate may be justified,
   but no install follows automatically.

Neither branch alone proves a cable defect. The repeated both-lane ACIO failure
still warrants a later one-variable cable/port isolation if software-visible
tunnel policy is not causal.

## Recorder

`extra/usbgpu/tests/capture_tunnel_idle.py`:

- requires the inherited `/tmp/gpu-bench.lock` and the feature worktree as the
  lock runner's current directory;
- refuses to start if the TinyGPU socket server is running or reachable;
- captures registration and boot time before the first provider query;
- queries handshake, keepalive, then power immediately;
- records `tinygpu` and all `IOPCIDevice` registry entries after the first
  healthy paired sample;
- samples validated keepalive and power payloads every second for 300 seconds;
- atomically rewrites the artifact after each phase/sample;
- stops at the first malformed/unhealthy/unavailable provider result; and
- records the relevant ten-minute kernel log and final USB4/PCI topology.

The recorder has no configuration write, BAR, DMA, reset, lease, server, or
workload path.

## Future R6.2 execution

Execution requires a separate, explicit operator-coordinated clean reboot while
leaving the enclosure, cable, port, installed v11, and power settings unchanged.
After login, run exactly:

```sh
.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py --wait-s 10 -- \
  .venv/bin/python extra/usbgpu/tests/capture_tunnel_idle.py \
  --duration-s 300 --interval-s 1
```

Do not run the v11 installer again. Do not recover the endpoint a second time
if the recorder fails or the tree disappears. Preserve its JSON path and stop.

## Admission and stop conditions

R6.2 passes only if:

- exactly installed v11 remains active and the native handshake identifies
  protocol v1 with capabilities 11;
- every keepalive payload is `active_healthy` with zero failures/resources;
- every power payload is confirmed/publishable, generation-matched, On, and
  free of request, release, downgrade, or teardown errors;
- the same provider generation persists for five minutes; and
- keepalive successes and the power canary advance.

Stop on the first failed handshake/status/power query, generation change,
endpoint loss, unhealthy payload, active TinyGPU server, or resource count.
Do not proceed to A1, reset/replug, install a v12, change ASPM/link speed, or
run a workload from this scope.
