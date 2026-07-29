# TinyGPU v12 provider BAR-residency candidate

Collected: 2026-07-29T03:31:13Z

Status: source candidate complete and locally admitted. It is built but not installed. No hardware,
system-extension registration, reset, replug, AMD initialization, DMA, workload, or model load ran.

Starting HEAD: `10b9e00010965b9d400eeb2570f3d9bb929e7822` on clean `exp`.

Authority:
`docs/task_workflow/input/egpu-usb4-v12-provider-bar-residency-scope-20260729.md`.

## Implemented boundary

- The provider requests BAR5 immediately after PCI open and `1002:744c` identity validation.
- It retains the `IOMemoryDescriptor` and `IOMemoryMap` before the one-Hz timer is created.
- Provider health and workload admission require active nonempty BAR5 residency with zero error.
- The mapping is independent of `active_bar_mappings`; workload lease cleanup cannot release it.
- Stop drains the timer, releases workload MMIO caches, releases BAR5, releases the power request,
  closes PCI, and then stops the service.
- Explicit function reset releases and reacquires BAR5 around reset. No automatic reset exists.
- `tinygpu.power-residency.v3` reports policy `driverkit_bar5_mapping_v1`, requested/active,
  BAR number/type/bytes, and the native error.
- App, DEXT, installer, capture, and native/server identities advance atomically to v12.

No MMIO access, BAR0/VRAM retention, BAR2 retention, DMA, shared memory, AMD initialization,
allocation, GPU work, ASPM change, or model residency was added.

## Source verification

The focused command covered all eight eGPU/TinyGPU test modules:

```text
.venv/bin/python -m pytest -q \
  test/unit/test_egpu_capture_tunnel_idle.py \
  test/unit/test_egpu_minimal_compute.py \
  test/unit/test_egpu_qualify.py \
  test/unit/test_tinygpu_install_script.py \
  test/unit/test_tinygpu_native_source.py \
  test/unit/test_tinygpu_remote_protocol.py \
  test/unit/test_tinygpu_server_source.py \
  test/unit/test_tinygpu_wire_spec.py
```

Result: `85 passed` in 0.25 seconds. The three warnings are unchanged pytest configuration
warnings for unavailable timeout options.

Additional checks passed:

- Python compilation of the changed runtime, qualification, capture, and focused tests.
- `bash -n` for the audited installer and registration classifier.
- JSON parsing for power-residency v3 and system-extension fixtures.
- `git diff --check`.

## DriverKit analyzer and signed build

The clean analyzer command completed with `ANALYZE SUCCEEDED`:

```text
xcodebuild -project TinyGPUDriverExtension.xcodeproj \
  -scheme TinyGPU -configuration Debug \
  -derivedDataPath build/DerivedData clean analyze \
  CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO
```

All five produced static-analyzer plists contain zero diagnostics: the provider source,
provider IIG, user-client source, user-client IIG, and native C server.

The audited build-only path then completed with `BUILD SUCCEEDED`:

```text
bash extra/usbgpu/tbgpu/installer/install_nosip.sh --build
```

The result is a strictly verified ad-hoc app and universal DEXT at:

```text
extra/usbgpu/tbgpu/installer/build/DerivedData/Build/Products/Debug/TinyGPU.app
```

The DEXT `CFBundleVersion` is exactly `12`. Built native strings contain
`tinygrad-arkey-native-v12`, `tinygrad-arkey-v12`, and `tinygpu.power-residency.v3`; no v11
build identity was found.

## Remaining admission boundary

Installation is intentionally not implicit. The installer requires a clean feature commit, the
inherited `/tmp/gpu-bench.lock`, the literal approval token on invocation, and the same token
again from an interactive terminal immediately before replacing `/Applications/TinyGPU.app`.

After approved installation and any required reboot, run R6.1 then A0/A1 without a socket server
or workload. Admission requires active BAR5 residency with nonzero bytes and zero error, advancing
one-Hz canaries in one provider generation, zero workload resource counts, and no ACIO link-loss
signature. The escalation order on failure is BAR0, then minimal AMD initialization/allocation,
then a loaded model as the known-good control.
