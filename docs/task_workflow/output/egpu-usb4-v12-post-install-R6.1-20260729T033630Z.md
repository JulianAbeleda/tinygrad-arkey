# TinyGPU v12 post-install R6.1 audit

Collected: 2026-07-29T03:34:37Z through 2026-07-29T03:36:48Z

Status: the audited v12 development build installed, activated, and verified successfully, but
R6.1 stopped at the first live-provider gate. The UT4G USB4 bridge is connected at 40 Gb/s while
the downstream AMD PCI functions are absent. v12 therefore has not received a matching
`1002:744c` provider and its BAR5-residency change has not yet been exercised.

Authority:
`docs/task_workflow/input/egpu-usb4-v12-provider-bar-residency-scope-20260729.md`.

Installed source commit:
`258d4f0e0114500b7c96b9117bcd4f6499331302`.

Install provenance:
`docs/task_workflow/output/tinygpu-development-install-provenance.txt`.

## Installation result

The audited installer ran under `/tmp/gpu-bench.lock` with the literal
`APPROVE_TINYGPU_DEVELOPMENT_INSTALL` token on invocation and at its immediate interactive
confirmation. It installed `/Applications/TinyGPU.app` and retained the prior application at:

```text
/Applications/.TinyGPU.previous.20260729T033437Z-3313.app
```

The provenance record contains the exact source commit and completed `before`, `activated`, and
installed/built signature sections. Strict verification of the installed app and DEXT succeeded.
The installed DEXT has `CFBundleVersion=12` and the live registration set is clean:

```text
* * - org.tinygrad.arkey.tinygpu.driver2 (1.0.0/12) org.tinygrad.arkey.tinygpu.driver2 [activated enabled]
  * 9YG3G8543N org.tinygrad.tinygpu.driver2 (1.0.0/3) org.tinygrad.tinygpu.driver2 [activated disabled]
```

The registration-only native check passed:

```text
Extension registration is clean and active. Verify keepalive and power status before using the GPU.
status_exit=0
```

## First live-provider gate

The three live-provider diagnostics were unavailable with the expected provider-absent exit code:

```text
keepalive handshake unavailable
handshake_exit=3
keepalive status unavailable
keepalive_exit=3
power status unavailable
power_exit=3
```

The DEXT process census was empty. Logs show macOS selected and activated v12 at
`2026-07-28 23:34:51.266-0400`, but there is no subsequent DriverKit server launch,
`tinygpu::start(display)` event, or child publication. Activation registers the DEXT personality;
it does not instantiate the driver without a matching PCI provider.

## Topology and link evidence

The post-install topology contains the ADTLINK UT4G on USB4 bus 0 with route 1, link status `0x2`,
and 40 Gb/s speed. `SPPCIDataType` contains no AMD device; specifically `1002:744c` and its
`1002:ab30`, `1002:7446`, and `1002:7444` companion functions are absent.

Starting at `2026-07-28 23:35:17.089-0400`, ACIO repeatedly reported Gen2/3 link errors on both
lanes with codes `83` and `87`. The first unavailable handshake was already observed before that
burst. This is consistent with the previously recorded upstream tunnel/link-loss state and does
not establish a v12 start or BAR5-mapping failure.

No reset, replug, reboot, power cycle, AMD initialization, socket server, DMA operation, workload,
or model load ran after installation.

## Disposition and bounded recovery gate

The software installation passes. Runtime admission is unresolved, not failed: v12 cannot retain
BAR5 until macOS enumerates `1002:744c` and calls its provider start.

The next hardware experiment is one separately operator-authorized enclosure recovery event. Under
the GPU lock, capture the pre-event registration and topology, perform exactly one reset or power
transition, and stop unless the full four-function AMD tree returns with `1002:744c` linked at x16
and 16.0 GT/s. If it returns, immediately require all of the following before AMD initialization:

1. a v12 DriverKit server and published `tinygpu` child bound to `1002:744c`;
2. selector-4 handshake and selector-5 keepalive with advancing `0x744c1002` canaries;
3. `tinygpu.power-residency.v3` with policy `driverkit_bar5_mapping_v1`, active nonzero BAR5 bytes,
   and zero BAR-residency error;
4. zero workload leases, workload BAR mappings, and DMA allocations; and
5. no new both-lane ACIO burst, provider-generation change, or endpoint disappearance.

Only after those immediate gates pass should the existing 120-second and five-minute awake-idle
A1 capture run. Do not use a model load as the recovery event. If BAR5-only residency later loses
the endpoint with the same ACIO signature, preserve that evidence and move to the already scoped
BAR0 escalation rather than retrying in the same run.
