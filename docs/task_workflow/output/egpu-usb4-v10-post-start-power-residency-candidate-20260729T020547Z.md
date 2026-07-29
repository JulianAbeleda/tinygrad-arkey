# TinyGPU v10 post-Start power-residency source candidate

Collected: 2026-07-29T02:05:47Z

Status: source implementation and clean-commit host-side verification complete.
The implementation is committed as `016b3a106`; it is uninstalled. No system-extension activation,
reboot, NVRAM change, endpoint observation, reset/replug, AMD initialization,
A0/A1 gate, or workload occurred.

Scope:
`docs/task_workflow/input/egpu-usb4-v10-post-start-power-residency-scope-20260729.md`.

Audit:
`docs/task_workflow/output/egpu-usb4-v9-power-management-api-audit-20260729T014837Z.md`.

Starting HEAD: `7f4974d3b7d1386828e63ccab44db3ab6c921c14` on `exp`.

Implementation commit: `016b3a106` (`[runtime] defer DriverKit power request
until post-start`).

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

## Remaining uncertainty and stop boundary

The source fixes the proven lifecycle error and makes API acceptance,
power-state observation, and endpoint evidence distinct. It cannot establish
from a host-only build that a DriverKit child's On desire prevents macOS from
removing the upstream USB4/ACIO tunnel. Only the future A1 idle-continuity gate
can answer that.

The focused tests, analyzer, and signed build above were rerun from the clean
implementation commit. Before activation, retain a clean `exp` worktree so the
installer's provenance gate can bind the binary to the exact branch HEAD, and
obtain separate explicit installation/reboot approval. The first boot is
diagnostic only: validate service publication, both lifecycle probes, request
attempts, On callback, post-request canary, and `active_healthy` before A1. Do
not reset/replug, initialize AMD, or run a workload in that gate.
