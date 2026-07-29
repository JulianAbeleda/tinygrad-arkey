# eGPU over USB4/Thunderbolt: the idle link-drop, its cause, and the keepalive fix

Durable record of a hardware issue that cost real debugging time and whose fix was
lost in a refactor. The historical symptom and root cause were previously recorded
only in a commit message (`554800bef`, 2026-06-10) and deleted code. The current
implementation and measured installation state are recorded below.

## Symptom

Running the RX 7900 XTX (`gfx1100`, PCI device id `0x744c`) as an eGPU from a Mac Mini
over Thunderbolt 4 / USB4 (ASM2464-class UT4G adapter, tinygrad AMD-over-USB path): after
an idle period the GPU **disappears** from the host — `system_profiler SPDisplaysDataType`
no longer lists `0x744c`, and it cannot be brought back in software.

## Root cause (proven, not assumed)

**The USB4/UT4G link dies on idle→low-power-state transitions.** When traffic is idle the
chain descends into ASPM / CLx low-power link states and **fails retraining on exit**. May
2026 captures show repeated ACIO Gen2/3 link errors before macOS marks the Thunderbolt tree
dead; a 40-minute idle period after a successful session reproduced it reliably.

This is **link-side power management**, not:
- macOS system idle-sleep (it drops while the host is otherwise awake),
- the GPU "idling off" (GPU occupancy is irrelevant — the PCIe endpoint goes away), nor
- a PSP fault as primary cause. The PSP-DIRTY state (`extra/remote/amd_repro.py`
  `classify_psp_clean_gate`: SOS-alive / mailbox all-ones) is the **downstream aftermath** —
  once the link drops and the device loses power/config, the PSP has re-booted its Secure OS
  and the native `am` driver can no longer re-init it in software, which is why recovery then
  needs a full physical power cycle.

## The fix: a 1 Hz config-space-read keepalive

Suppress the low-power transitions by touching the link ~once a second whenever a device is
open and the bridge is otherwise idle. A config-space read is harmless and keeps the link out
of the deep ASPM/CLx states that fail to retrain. The current implementation owns cadence in
the native DriverKit provider under the named policy `usb4_amd_744c_v1` (1000 ms interval,
100 ms maximum leeway). `REMOTE_KEEPALIVE_S` is explicitly unsupported; Python environment
state and TinyGPU.app liveness are not keeper evidence.

Reference implementation (from the deleted `extra/remote/serve.py` @ `554800bef` — recover
the full file with `git show 554800bef:extra/remote/serve.py`):

```python
# REMOTE_KEEPALIVE_S sets the cadence; 0 disables. Default on only for OSX.
KEEPALIVE_S = float(os.getenv("REMOTE_KEEPALIVE_S", "1.0" if OSX else "0"))
_keepalive_fail = 0
def keepalive_tick():
  global _keepalive_fail
  if not KEEPALIVE_S or dirty_error: return
  for dev_id, pci_dev in opened_devices.items():
    try:
      pci_dev.read_config(0, 4)                 # harmless 1Hz config-space read
      if _keepalive_fail: log(f"KEEPALIVE recovered after {_keepalive_fail} failures")
      _keepalive_fail = 0
    except Exception as e:
      _keepalive_fail += 1
      if _keepalive_fail in (1, 10, 100): log(f"KEEPALIVE failed x{_keepalive_fail} dev={dev_id}: {e}")

# driven from the serve loop between requests, and from the accept loop while waiting:
#   if KEEPALIVE_S:
#     readable, _, _ = select.select([conn], [], [], KEEPALIVE_S)
#     if not readable: keepalive_tick(); continue
```

The keepalive must run **in the process that holds the initialized device state**, not merely
in an unrelated process capable of issuing a config-space read. The historical server also
retained its device handle, BAR mappings, and system-memory allocations across client sessions.

## Where the fix lives (architecture migration)

The historical keepalive lived in `extra/remote/serve.py`, a 243-LOC Python socket bridge.
That bridge was deleted at `4c5e67cff` during migration to a native macOS app. The current
feature branch restores bounded TinyGPU source under `extra/usbgpu/tbgpu/installer/` and
places the keeper in the DriverKit provider, not in Python or the workload server. Python
negotiates the protocol and acquires workload leases, while the provider owns the timer,
PCI read, lifecycle gate, read-only status counters, and—starting with the v12 candidate—a
provider-lifetime BAR5 descriptor and mapping. BAR5 is the first mapping requested by the
historical AMD initialization path and is the smallest native experiment that restores its
persistent device-memory state.

The wire contract is frozen in
`extra/usbgpu/protocol/tinygpu-wire-v1.md`. The provider rejects unknown identities and
never resets or power-cycles hardware after a failed tick. Workload DMA, BAR mappings, and
shared memory are released at lease disconnect. The provider BAR5 mapping is separately
owned, separately reported by `tinygpu.power-residency.v3`, excluded from workload resource
counts, and released only during provider stop/failure or explicitly around a function reset.

## v12 BAR-residency recovery candidate (2026-07-29)

Git history establishes that the RX 7900 XTX completed full Qwen inference before the native
migration. The old bridge at `554800bef` retained `opened_devices`, `mapped_bars`, and
`sysmem_allocs`; `4c5e67cff` removed that bridge; later commits `29be4c9fa` and `5bff0135c`
recorded working inference. This demotes the cable hypothesis and makes persistent initialized
device state the first recovery target.

The v12 candidate requests and maps BAR5 immediately after PCI open and identity validation,
before creating the one-Hz timer. It performs no MMIO read/write, AMD initialization, DMA,
allocation, reset, or model load. Provider health and workload admission require the BAR5
mapping to remain active with zero error. Full model residency remains a known-good control if
this minimal state is insufficient; it is not part of v12.

## Measured implementation status (2026-07-28)

- Original implementation branch: `feature/egpu-usb4-keeper`, source commit `f23c05c57468ccdd7035789fbf38320d2c2c0d68`.
  The maintained installation owner is now the clean `exp` worktree; the installer enforces that branch so
  production `master` cannot be used for development DEXT replacement.
- The audited development installer replaced `/Applications/TinyGPU.app`; the app and
  DriverKit extension are ad-hoc signed, and DEXT
  `org.tinygrad.arkey.tinygpu.driver2` reached `[activated enabled]` at version `1.0.0/5`.
- A later reinstall attempt was correctly refused by macOS because this host has
  DriverKit development mode off and retains the prior ad-hoc registration. The CLI had
  also mislabeled `OSSystemExtensionErrorDomain` code 4 as a missing-entitlement error;
  code 4 means the extension was not found, while code 2 is the entitlement error. The
  installer now requires development mode explicitly and the next DEXT version is `5` so
  macOS can distinguish an upgrade from the already-active v4 registration.
- The endpoint subsequently re-enumerated: `system_profiler` reported `1002:744c`, link up
  at 16.0 GT/s, and tinygrad's macOS PCI scan returned `1002:744c`.
- The locked minimal AMD probe still failed during `AMDev.is_smu_alive()` on the PCIIface
  path. The TinyGPU Unix RPC received EOF while servicing an MMIO write, and lease cleanup
  later saw a broken pipe. The app-level keepalive status then reported unavailable even
  though v5 remained enabled.
- No reset, power-cycle, sleep, replug, workload benchmark, or idle-duration claim was made.
- The complete machine-readable A0 artifact is
  `docs/task_workflow/output/egpu-usb4-persistent-pcie-A0-20260728T105840Z-89079.json`.
  The ignored installation transcript remains at
  `docs/task_workflow/output/tinygpu-development-install-provenance.txt` for local audit.

Qualification is now gated on the runtime initialization/RPC blocker documented in
`docs/task_workflow/input/egpu-usb4-tinygpu-runtime-initialization-scope-20260728.md`.
Do not infer compute, awake-idle, load-power, or sleep/wake behavior from activation,
enumeration, or the provider's installed/ready message.

## Current handoff (2026-07-28)

**Owner:** continue from the clean `exp` worktree at the current task commit. The installer
is development-only and must run from `exp`; do not install from `master`.

**Host state:** macOS 26.5 (25F71), Apple M4 Mac mini, SIP disabled, DriverKit development
mode on, and the ADTLink UT4G bridge visible at USB4 40 Gb/s. The v5 DEXT is enabled and
the legacy v3 DEXT is disabled. The endpoint can enumerate as `1002:744c` at 16.0 GT/s.
The first observed failure is a TinyGPU RPC/provider disconnect while issuing a BAR5
register write during AMD initialization. Earlier BAR0 initialization writes occur and
have not yet been ruled out as the causal boundary.

**Current blocker:** this is no longer an activation or enumeration task. The provider's
service termination callback, native IOKit error propagation, direct BAR/MMIO access
contract, and Python runtime-error classification are scoped in
`docs/task_workflow/input/egpu-usb4-tinygpu-runtime-initialization-scope-20260728.md`.
Do not reinstall or remove the v5 extension as a response to this runtime failure.

After the blocker scope's CPU/native checks pass, rerun the locked M0-M7 microprobes,
then fresh A0, A1, and A2. A4/A8 remain deferred until those gates pass; no local GGUF
model is currently available for the A8 Qwen3 8B smoke gate.

**Evidence boundary:** the tracked A0 artifact is
`docs/task_workflow/output/egpu-usb4-persistent-pcie-A0-20260728T105840Z-89079.json`.
The later failed A0 attempts (`...152615Z-18390.json` and `...152626Z-18408.json`) are
local diagnostic artifacts only; neither is acceptance evidence. The runtime blocker
scope is the current task authority for the post-activation failure.

**Secondary defense:** `pmset disablesleep 1` and a launchd KeepAlive daemon are separate
sleep investigations, not substitutes for the native provider keeper.

**Fallback (kept):** `extra/remote/amd_power_cycle.py` physically power-cycles via a Shelly
smart plug when the link is already dead. Prevention lowers frequency; it does not make the
power-cycle recovery obsolete.
