# TinyGPU v10 post-Start power-residency source candidate

Collected: 2026-07-29T02:05:47Z

Status: source implementation, clean-commit host-side verification, and the
single audited installation are complete. The v10 upgrade is pending its first
reboot. No reboot, NVRAM change, reset/replug, AMD initialization, A0/A1 gate,
or workload occurred.

Scope:
`docs/task_workflow/input/egpu-usb4-v10-post-start-power-residency-scope-20260729.md`.

Audit:
`docs/task_workflow/output/egpu-usb4-v9-power-management-api-audit-20260729T014837Z.md`.

Starting HEAD: `7f4974d3b7d1386828e63ccab44db3ab6c921c14` on `exp`.

Implementation commit: `016b3a106` (`[runtime] defer DriverKit power request
until post-start`).

Installed source/provenance commit:
`0d9f3a1bf6ff0111fdb405b3959000b741603205` (`[docs] record v10 clean-commit
verification`). Any later documentation-only handoff commit is not part of the
installed payload.

## Implementation outcome

The source-only v9 identity was advanced to v10 and must never be installed.
The v10 driver:

- removes `SetPowerOverride(true)` and every override-dependent readiness
  predicate;
- records a pre-join `SetPowerOverride(false)` lifecycle probe during Start,
  but performs no power-state request there;
- requests `kIOServicePowerCapabilityOn` only from a non-null timer callback,
  after Start can return and DriverKit can join the service to its PM tree;
- retries an accepted but unconfirmed request at most three times;
- confirms the request only through `SetPowerState(On)` and requires a later
  successful PCI identity canary before entering `active_healthy`;
- withdraws the service's desire with `kIOServicePowerCapabilityOff`;
- publishes `tinygpu.power-residency.v2`, including request attempts and time,
  pre/post-join probes, observed transitions, post-request canary evidence, and
  Stop resource counts;
- keeps diagnostics structurally decodable while runtime admission and A0/A1
  qualification remain fail-closed on complete healthy evidence;
- guarantees that Stop drains and releases what it can, closes the PCI session,
  and reaches `Stop(provider, SUPERDISPATCH)` even when resources are still
  recorded or timer drain reports an error; and
- corrects the user-client selector-10 branch layout without changing its ABI.

The inference-based `SetASPMState` and `EnablePCIPowerManagement` experiments
were deliberately excluded. Activating them in the same build would make a
successful A1 result causally ambiguous.

## Wire and identity changes

- `tinygpu.keepalive.v1` and selector/command IDs remain frozen.
- The separate power payload advances from
  `tinygpu.power-residency.v1` to `tinygpu.power-residency.v2`.
- The canonical fixture advances to
  `extra/usbgpu/protocol/fixtures/power-residency-status-v2.json`.
- The installer and DEXT bundle version advance together to `10`.
- Native build IDs are `tinygrad-arkey-native-v10` and
  `tinygrad-arkey-v10`.

## Focused tests

Command:

```sh
.venv/bin/python -m pytest -q \
  test/unit/test_tinygpu_native_source.py \
  test/unit/test_tinygpu_server_source.py \
  test/unit/test_tinygpu_wire_spec.py \
  test/unit/test_tinygpu_remote_protocol.py \
  test/unit/test_egpu_qualify.py \
  test/unit/test_tinygpu_install_script.py
```

Result: `76 passed` in 0.25 seconds. Pytest emitted the same three environment
warnings for the unavailable owner of the `timeout`, `timeout_func_only`, and
`timeout_method` configuration keys; no test failed or skipped.

`git diff --check` also passed.

## DriverKit analyzer and build

The full analyzer command was:

```sh
xcodebuild -project TinyGPUDriverExtension.xcodeproj \
  -scheme TinyGPU -configuration Debug \
  -derivedDataPath build/DerivedData clean analyze \
  CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO
```

Result: `ANALYZE SUCCEEDED`. The five analyzer plists produced by that pass
(driver implementation, driver iig, user-client implementation, user-client
iig, and server) each contained zero diagnostics.

The final signed build command was:

```sh
bash extra/usbgpu/tbgpu/installer/install_nosip.sh --build
```

Result: `BUILD SUCCEEDED`. The script ad-hoc signed both bundles and its strict
bundle verification succeeded. The four DriverKit analyzer plists retained by
the final normal build each contain zero diagnostics. The only Xcode warning
was the expected AppIntents metadata skip because the app has no AppIntents
dependency.

## Final build identity

| Component | Value |
|---|---|
| DEXT version | `1.0.0/10` |
| App launcher SHA-256 | `0e6fc687159328ae49aa11a2fd642cb8c452bbc504a645a77cbbb51ad4bbadfa` |
| App debug dylib SHA-256 | `0df32a2846f77b2a4d17c2671293f48079dead0aeddf0d379ca609a674a29f3b` |
| App CDHash | `0c511e3f993d283ecb272ae31ae3af9c0d5b06d2` |
| DEXT executable SHA-256 | `dbc59982e1c9022332a203348705d88d7140b37f10de55b252147ea4f5beca33` |
| DEXT CDHash | `e27775e8c48445543358d25fa77de8d4a438a002` |
| Signing | ad-hoc, no TeamIdentifier |

## Installation and pre-reboot state

The audited installer ran under `/tmp/gpu-bench.lock` with the exact interactive
approval token. It replaced `/Applications/TinyGPU.app`, retained the previous
app at `/Applications/.TinyGPU.previous.20260729T021204Z-5026.app`, and exited
with its documented code `10`: installation succeeded, but the DEXT upgrade
requires a reboot.

The local provenance transcript is
`docs/task_workflow/output/tinygpu-development-install-provenance.txt`. It is
intentionally ignored by Git through `*.txt`, records run
`20260729T021204Z-5026`, and binds the installed payload to full source commit
`0d9f3a1bf6ff0111fdb405b3959000b741603205`.

The final locked, read-only pre-reboot observation showed:

- legacy version 3 is `activated disabled`;
- arkey version 8 is `terminating for upgrade via delegate`;
- arkey version 10 is `activated enabled`;
- the app reports `Extension upgrade is pending. Restart macOS before using
  TinyGPU.`; and
- handshake, keepalive status, and power status are unavailable, as expected
  until the registration transition is resolved by reboot.

No reset, replug, endpoint power-cycle, AMD initialization, or workload was
used to obtain that state. Do not run the installer again before reboot.

## Exact post-reboot handoff

All eGPU observations and gates below must run from this feature worktree
through:

```sh
.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py -- <command>
```

Do not reset or replug the enclosure, initialize AMD hardware, start the
TinyGPU socket server, or run a workload during the first diagnostic boot.

### R6.1 — registration and diagnostic admission

Acquire the lock once and collect, in order:

1. `sysctl -n kern.boottime` and `uptime`, to prove this is the post-install
   boot.
2. `systemextensionsctl list`. Require exactly one arkey registration at
   version 10 with `activated enabled`; version 8 must be gone and legacy
   version 3 may remain only as disabled. Any transitional arkey row is a stop.
3. `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU status`. Require the clean
   active message, not pending upgrade or approval.
4. `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU keepalive handshake`.
   Require `tinygpu.handshake.v1`, capabilities containing `11`, and
   `server_build_id` equal to `tinygrad-arkey-native-v10`.
5. `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU keepalive status`. Require
   `state=active_healthy`, `enabled=true`, identity `1002:744c`, at least one
   successful tick, no failures, no timer error, and zero active workload/BAR/
   DMA resources.
6. `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU power status`. Require the
   full `tinygpu.power-residency.v2` prediction table below.
7. `ioreg -r -n tinygpu -l`, plus read-only DEXT process/log and PCI/USB4
   endpoint observations, to bind publication and endpoint identity to this
   boot.

| Field | Required result |
|---|---|
| `override_probe_prejoin_error` | `-536870212` |
| `override_probe_postjoin_error` | `0` |
| `power_request_attempts` | ideally `1`; any value outside `1..3` is a stop |
| `power_request_error` | `0` |
| `transition_count` | at least `1` |
| `desired_power_flags`, `last_observed_power_flags` | `2`, `2` |
| `power_request_accepted`, `power_request_confirmed`, `publishable` | all `true` |
| `last_canary_identity_dword` | `0x744c1002` |
| request/callback/canary times | callback and canary later than the request |
| `unexpected_downgrade_count` | `0` |
| `stop_busy_leases`, `stop_busy_bars`, `stop_busy_dma` | all `0` |

If any R6.1 predicate fails, capture the raw output and relevant boot logs, then
stop. Do not use reset/replug as recovery and do not proceed to A1.

### Provenance checkpoint before A1

The current pending-reboot transcript has no `=== activated ===` phase because
the installer correctly returned at the reboot boundary. Also, this
documentation-only commit moves branch HEAD past the installed source commit.
As written, `extra/usbgpu/tests/qualify.py` requires both an activated phase and
the provenance source commit to equal current HEAD, so invoking A1 immediately
would fail provenance admission before the 120-second observation. That is a
host-tooling failure, not evidence about tunnel continuity.

Before A1, add and verify an audited post-reboot provenance-finalization path
that only reads the active registration and installed bundles, retains
`0d9f3a1bf6ff0111fdb405b3959000b741603205` as the installed source commit, and
teaches qualification to validate the recorded installed commit rather than an
unrelated later documentation HEAD. It must not rebuild, reinstall, activate,
reset, replug, or initialize the GPU. Commit that host-only change separately.

### A1 — idle continuity

Only after R6.1 is entirely green and provenance admission has been finalized,
run the existing A1 gate under the GPU lock. It performs two healthy status and
power samples separated by 120 seconds with no client and requires the same
provider generation, advancing keepalive/canary evidence, no failures or
unexpected downgrade, a continuously visible endpoint, no server restart, and
no reachable TinyGPU socket.

A1 is the first test that answers the remaining hardware question: whether the
DEXT child's confirmed full-power desire prevents removal of the upstream
USB4/ACIO tunnel. Still do not reset/replug, initialize AMD, or run a workload
during A1.

## Remaining uncertainty and stop boundary

The source fixes the proven lifecycle error and makes API acceptance,
power-state observation, and endpoint evidence distinct. It cannot establish
from a host-only build that a DriverKit child's On desire prevents macOS from
removing the upstream USB4/ACIO tunnel. Only the future A1 idle-continuity gate
can answer that.

The focused tests, analyzer, signed build, and audited install are complete.
The only authorized next system transition is the separately approved reboot.
After reboot, follow R6.1 and the provenance checkpoint above in order. The
first boot remains diagnostic only: no reset/replug, AMD initialization, or
workload is permitted before the lifecycle evidence and A1 gate are green.
