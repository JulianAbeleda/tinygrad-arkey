# TinyGPU v11 synchronous power-confirmation candidate and handoff

Collected: 2026-07-29T02:26:01Z through 2026-07-29T02:40:38Z

Status: v10 runtime diagnosis, v11 implementation, clean-commit focused tests,
clean analyzer, and signed build-only verification are complete. The v11
candidate is not installed. No reset, replug, AMD initialization, TinyGPU
socket server, A1, or workload ran.

Scope:
`docs/task_workflow/input/egpu-usb4-v11-sync-power-confirmation-scope-20260729.md`.

Starting HEAD: `eaeff7ee89092e906d109e6d2bdeb2442fd83e7e` on `exp`.

Implementation commit: `544bde177a9bb0ab0dc542a8e4773b0dc321cd38`
(`[runtime] reconcile synchronous DriverKit power callback`).

## Runtime finding

The first v10 boot proved that the DEXT now starts and publishes and that the
PCI identity canary remains reachable. R6.1 stopped because keepalive state was
`active_degraded`. The separately authorized selector-10 query returned the
following decisive fields:

| Field | v10 result |
|---|---|
| `full_power_requested`, `power_request_accepted` | `true`, `true` |
| `power_request_confirmed`, `publishable` | `false`, `false` |
| `power_request_attempts` | `3` |
| desired / observed flags | `2` / `2` |
| request / probe errors | all zero except expected pre-join probe |
| transition count | `1` |
| last transition / retained last request | `684269405208` / `687485345833` |
| last canary | `717402579833`, identity `0x744c1002` |
| unexpected downgrades / resources | all zero |

The API accepted the request and DriverKit delivered On. The defect was the v10
confirmation latch. A synchronous callback ran before `ChangePowerState`
returned and therefore before its successful return could set
`powerRequestAccepted=true`. v10 did not reconcile that already-recorded
transition after the return. It retried until the attempt limit, clearing
confirmation and retaining a later request time without another callback
because the service was already On.

## v11 implementation

`FullPowerTransitionConfirmsRequest` now centralizes the evidence invariant. It
requires an accepted zero-error request, no release, an observed On transition,
and `lastPowerTransition > lastPowerRequestTick`.

Before `ChangePowerState(On)`, the request path marks acceptance and
confirmation false and records the current request time. After the call
returns, it records the error and acceptance and immediately recomputes
confirmation. A synchronous callback is therefore reconciled; a stale On
transition from before the request is not. `SetPowerState_Impl` uses the same
predicate for later asynchronous delivery. `PowerResidencyReady` requires the
confirmation latch, the live shared predicate, and a later successful canary.

No PM API, policy, retry limit, schema, selector, capability, hardware gate,
timer, or teardown behavior changed.

The installer, both DEXT project configurations, classifier fixtures, native
direct-status build ID, and socket-server build ID advance atomically to v11.
The v2 power payload and `driverkit_full_power_v1` policy remain frozen.

## Clean-commit focused tests

From clean implementation commit `544bde177`:

```sh
.venv/bin/python -m pytest -q \
  test/unit/test_tinygpu_native_source.py \
  test/unit/test_tinygpu_server_source.py \
  test/unit/test_tinygpu_wire_spec.py \
  test/unit/test_tinygpu_remote_protocol.py \
  test/unit/test_egpu_qualify.py \
  test/unit/test_tinygpu_install_script.py
```

Result: `77 passed` in 0.22 seconds. Pytest emitted the existing three unknown
configuration warnings for `timeout`, `timeout_func_only`, and
`timeout_method`; there were no failures or skips. `git diff --check` passed.

The source guards now require post-return reconciliation after API acceptance,
the same ordered predicate in `SetPowerState_Impl`, strict transition/request
timestamp ordering, and matching v11 installer, project, and native identities.

## Clean analyzer and signed build

The clean analyzer command was:

```sh
xcodebuild -project TinyGPUDriverExtension.xcodeproj \
  -scheme TinyGPU -configuration Debug \
  -derivedDataPath build/DerivedData clean analyze \
  CODE_SIGN_IDENTITY= CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO
```

Result: `ANALYZE SUCCEEDED`. All five reports contained zero diagnostics:

- `TinyGPUDriver-5fa57021850eb8e64216c432553d91f0.plist`;
- `TinyGPUDriver.iig.plist`;
- `TinyGPUDriverUserClient-cfd27415bc7fce648c99ad4c242095da.plist`;
- `TinyGPUDriverUserClient.iig.plist`; and
- `server.plist`.

The final build command was:

```sh
bash extra/usbgpu/tbgpu/installer/install_nosip.sh --build
```

Result: `BUILD SUCCEEDED`. The installer ad-hoc signed both bundles, strict
verification passed, and nothing was installed. The only Xcode warning was the
expected AppIntents metadata skip because the app has no AppIntents dependency.

Toolchain: Xcode 26.5 build 17F42, macOS SDK 26.5, DriverKit SDK 25.5.

## Verified build identity

| Component | Value |
|---|---|
| DEXT version | `1.0.0/11` |
| Native diagnostic ID | `tinygrad-arkey-native-v11` |
| Socket server ID | `tinygrad-arkey-v11` |
| App launcher SHA-256 | `0c1886782b9d2f329f4409367275df7cb87b4c79e0177afd199ed197cb36d664` |
| App debug dylib SHA-256 | `5924282408a4d14572f0816420ba3de2c73f6daa9846e1beb30a54529629eed6` |
| App CDHash | `0c3527f89d62efd87e7828b9d37a72fd72508b20` |
| DEXT executable SHA-256 | `2f6bec82e6457ef5a42567106dce4ffeba17405405a5dbdf647f6885e3c34b55` |
| DEXT CDHash | `c9e068c1a6cd880262944e52dce81dafbd0476a7` |
| Signing | ad-hoc, no TeamIdentifier |

## Exact installation boundary

Do not reinstall v10. After this evidence is committed and the `exp` worktree
is clean, the only permitted replacement path is the audited installer under
the GPU lock with both its literal command-line token and immediate interactive
approval:

```sh
.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py -- \
  bash extra/usbgpu/tbgpu/installer/install_nosip.sh \
  --install APPROVE_TINYGPU_DEVELOPMENT_INSTALL \
  --provenance-out docs/task_workflow/output/tinygpu-development-install-provenance.txt
```

If the installer exits 10 with a pending upgrade, that is success at the reboot
boundary. Do not run it again. Record the installed source commit, retained
rollback app, registration rows, and provenance transcript, then reboot once.

## Exact first-boot R6.1 handoff

Every command must run from this feature worktree through
`extra/usbgpu/tools/with_gpu_lock.py`. Do not reset/replug, initialize AMD,
start the socket server, or run a workload.

Collect in order and stop on the first failure:

1. Boot time and uptime must prove a post-install boot.
2. Exactly one arkey registration must be v11 `activated enabled`; v10 must be
   gone and legacy v3 may remain only disabled.
3. Native `status` must report a clean active registration.
4. `keepalive handshake` must report capability 11 and
   `tinygrad-arkey-native-v11`.
5. `keepalive status` must be `active_healthy`, enabled, generation-stable,
   identity `1002:744c`, successful, error-free, and resource-zero.
6. `power status` must satisfy the complete v2 table below.
7. Only after those gates pass, capture `ioreg`, DEXT process/log, and read-only
   PCI/USB4 endpoint identity for the same boot.

| Field | Required result |
|---|---|
| lifecycle probes | pre-join `-536870212`, post-join `0` |
| `power_request_attempts` | ideally `1`; outside `1..3` is a stop |
| request state | requested, accepted, confirmed, and publishable all `true` |
| request/release errors | all `0`; release not attempted |
| transition evidence | count at least `1`, observed/desired flags `2` |
| timestamp order | transition and canary strictly later than retained request |
| identity | `0x744c1002` |
| downgrade/resource evidence | unexpected downgrade and Stop resource counts all `0` |

If R6.1 fails, capture raw output and relevant boot logs and stop. Do not use a
reset or replug as recovery.

Before A1, the previously identified host-only provenance-finalization path is
still required so qualification validates the installed source commit rather
than a later documentation HEAD. It must only read registration and installed
bundles; it must not rebuild, reinstall, activate, reset/replug, or initialize
the GPU. A1 remains unauthorized until both R6.1 and provenance admission are
green.
