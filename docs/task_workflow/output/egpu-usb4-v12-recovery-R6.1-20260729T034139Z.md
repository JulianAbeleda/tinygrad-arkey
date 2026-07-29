# TinyGPU v12 bounded recovery R6.1 audit

Collected: 2026-07-29T03:41:39Z through 2026-07-29T03:42:44Z

Status: R6.1 failed closed on repeated both-lane ACIO errors. The v12 provider nevertheless
started correctly, retained BAR5, disabled tunneled L1 on the AMD display function, and completed
128 consecutive identity canaries without a provider loss during the captured interval. No
agent-initiated reset or power cycle was performed because the endpoint returned before the
authorized recovery action began.

Authority:
`docs/task_workflow/input/egpu-usb4-v12-provider-bar-residency-scope-20260729.md` and the explicit
operator token `APPROVE_ONE_EGPU_RECOVERY`.

Installed source commit:
`258d4f0e0114500b7c96b9117bcd4f6499331302`.

## Recovery boundary

The locked pre-event capture began at `2026-07-29T03:41:39Z`. v12 remained the only active arkey
registration. The UT4G was visible at 40 Gb/s and `SPPCIDataType` still returned no AMD rows, but
the native-v12 selector-4 handshake unexpectedly succeeded and the v12 DEXT process was already
running as PID 4836.

Logs establish that an uncommanded endpoint-return transition had occurred immediately before the
capture. v12 launched and published against the AMD display function at:

```text
2026-07-28 23:40:24.195-0400 DK: TinyGPUDriver-0x1000017f0 server launched, validating
2026-07-28 23:40:24.195-0400 DK: tinygpu-0x1000017f0::start(display-0x1000017ce) ok
2026-07-28 23:40:24.195-0400 (IOPCIFamily) child tinygpu(0x1000017f0) published
```

This happened before any agent-issued power command. Spending the authorized destructive event on
an already returned provider would have destroyed the only clean v12 observation window, so no
Shelly action, reset RPC, replug, reboot, or other power transition ran. The causal mechanism for
the endpoint return is not asserted by this evidence.

## v12 residency evidence

At `03:42:04Z`, the handshake identified `tinygrad-arkey-native-v12`. Keepalive status reported
provider generation 1, attempts/successes `92/92`, identity `0x744c1002`, no failures, and zero
workload leases, BAR mappings, or DMA allocations. Two seconds later the counters advanced to
`94/94` in the same generation. The final locked read at `03:42:44Z` reported `128/128`.

The v3 residency payload was stable across all reads:

```text
bar_residency_policy_id=driverkit_bar5_mapping_v1
bar_residency_requested=true
bar_residency_active=true
bar_residency_bar=5
bar_residency_bytes=1048576
bar_residency_error=0
```

IORegistry bound the child to `display@0`, identity `1002:744c`, at PCI location `5:0:0`. The
display provider and the upstream tunneled bridge chain reported `IOPCITunnelL1Enable=No`. All
four AMD functions were present in IORegistry, and the display function reported link status
`4356` (`0x1104`, x16 at 16.0 GT/s). The child CDHash matched installed v12:
`c2671c57a8877a8e676700f8da16ab154fdf09a2`.

These observations prove that the v12 BAR5 implementation works as written and establishes the
intended IOPCIFamily L1 veto. They do not prove awake-idle acceptance.

## Independent power predicate

The provider remained `active_degraded` and `publishable=false` because the inherited full-power
predicate never confirmed a transition after its most recent request:

```text
full_power_requested=true
power_request_accepted=true
power_request_confirmed=false
power_request_attempts=3
desired_power_flags=2
last_observed_power_flags=2
power_request_error=0
transition_count=1
unexpected_downgrade_count=0
```

IORegistry simultaneously showed the TinyGPU child at power state 3 with capability flags 2 and
the PCI provider at current power state 2. The status is therefore a strict ordering failure: the
only observed On transition preceded the third request, and requesting an already-On service did
not produce a later callback. It is not a BAR5 acquisition failure.

## ACIO hard stop

The first post-start both-lane ACIO burst occurred at `23:40:33.105-0400`, 8.910 seconds after v12
published. Both lanes reported error `83`, immediately followed by error `87`. The same paired
signature repeated at `23:40:35`, `23:41:10`, `23:41:40`, and `23:41:42` while BAR5 residency and
identity canaries remained active.

The scope requires stopping on the first such burst. No A0/A1 runner, controlled 120-second idle
gate, five-minute capture, AMD initialization, socket server, DMA, workload, or model load ran.
The later read-only status samples only preserved the provider state after the hard stop.

## Disposition

BAR5 residency is real and kept the display function's tunneled L1 policy disabled, but it did not
prevent the historical ACIO lane-error signature. The endpoint survived for at least the captured
140 seconds after provider start, so the evidence does not establish that BAR5 has no benefit; it
does establish that BAR5 alone is insufficient for the existing zero-ACIO admission contract.

The result raises the physical USB4 path (cable, connector/port, UT4G/ASM2464 bridge, or signal
margin) relative to a missing-L1-veto explanation: errors continued after IORegistry showed the
veto active. A controlled known-good cable/port A/B is now more discriminating than merely adding
another BAR mapping. If source escalation continues first, BAR0 is the already scoped next
candidate, but it should be treated as a separate descriptor-residency experiment rather than an
expectation that a second mapping will change the already-disabled L1 state. Minimal AMD
initialization/allocation and a loaded model remain later controls, not recovery actions for this
failed run.
