# TinyGPU v9 diagnostic-start source candidate

Collected: 2026-07-29T01:18:47Z through 2026-07-29T01:25:59Z

Status: source/build candidate complete for review. It is uncommitted and was
not installed or activated. The installed v8 registration and eGPU were not
changed.

## Causal boundary

The locked post-reboot v8 audit is recorded in
`egpu-usb4-v8-post-reboot-R6.1-20260729T011847Z.md`. Registration, binary
identity, USB4, and PCI enumeration passed, but the kernel recorded
`TinyGPUDriver::start(display) fail` before the `tinygpu` diagnostic service was
published. v8 returned from `Start_Impl` on the first native power-request
error, making the error fields in `tinygpu.power-residency.v1` unreachable.

## Source disposition

- `RequestPowerResidency` attempts both `SetPowerOverride(true)` and
  `ChangePowerState(kIOServicePowerCapabilityOn)` and preserves both native
  results, even when the first request fails.
- `Start_Impl` logs both errors and does not discard an otherwise valid
  read-only identity/first-canary result solely because power residency is not
  confirmed.
- The service can publish in a degraded diagnostic state. Workload leases,
  reset, BAR, DMA, configuration, and MMIO remain fail-closed behind the
  existing `PowerResidencyReady` contract.
- The DEXT bundle and native handshake build identities advance from v8 to v9.
  The installer and complete-registration classifier fixtures advance together,
  so a same-version v8 reinstall cannot occur.
- The invalid user-client cast path now makes its single owned reference
  transfer explicit. This is runtime-equivalent to the prior scoped release and
  removes conflicting full/shallow analyzer ownership reports.

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

Result: `74 passed` in 0.23 seconds. Pytest emitted three configuration warnings
because the environment does not have the plugin that owns the repository's
`timeout`, `timeout_func_only`, and `timeout_method` settings; no test failed or
skipped.

## DriverKit analyzer and build

- A clean `xcodebuild ... clean analyze` completed with `ANALYZE SUCCEEDED`.
- All four DriverKit shallow-analyzer plists from the final normal build contain
  zero diagnostics.
- `bash extra/usbgpu/tbgpu/installer/install_nosip.sh --build` completed with
  `BUILD SUCCEEDED`, ad-hoc signed both bundles, and passed deep/strict bundle
  verification. No install mode was invoked.
- The only Xcode warning was the expected app-intents metadata skip because the
  app has no AppIntents framework dependency.

Final build identity:

| Component | Value |
|---|---|
| DEXT version | `1.0.0/9` |
| App SHA-256 | `a5d4ac76ba67bce50fc29d5057fd5499d0ba151438b0b0780adb3d2ba9a63742` |
| App CDHash | `4da668e66e1a531af8df1ca4582d0bc20e2555dd` |
| DEXT SHA-256 | `ab269b17d9895e47aa91ab4a46f736449d5bcc8a0ef457c8d55e129d4628fee5` |
| DEXT CDHash | `c7d37c19a7dcd37d52b09d14e2a9a99b63e6da81` |
| Signing | ad-hoc, no TeamIdentifier |
| Native build IDs | `tinygrad-arkey-native-v9`, `tinygrad-arkey-v9` |

## Stop boundary

The source worktree is intentionally dirty for review, so it cannot pass the
installer's clean-source provenance gate. Before any hardware action, review
and commit the complete candidate, rerun the focused tests and signed build from
that clean commit, present the final source/build hashes and exact bounded
install/activation sequence, and obtain separate operator approval. Do not
install v9, reset/replug the endpoint, or run A0/A1 from this uncommitted state.
