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

The keepalive must run **in the process that holds the device** (does the config-space read).

## Where the fix lives (architecture migration)

The historical keepalive lived in `extra/remote/serve.py`, a 243-LOC Python socket bridge.
That bridge was deleted at `4c5e67cff` during migration to a native macOS app. The current
feature branch restores bounded TinyGPU source under `extra/usbgpu/tbgpu/installer/` and
places the keeper in the DriverKit provider, not in Python or the workload server. Python
negotiates the protocol and acquires workload leases, while the provider owns the timer,
PCI read, lifecycle gate, and read-only status counters.

The wire contract is frozen in
`extra/usbgpu/protocol/tinygpu-wire-v1.md`. The provider rejects unknown identities and
never resets or power-cycles hardware after a failed tick. Workload DMA, BAR mappings, and
shared memory are released at lease disconnect and are not retained by the keeper.

## Measured implementation status (2026-07-28)

- Original implementation branch: `feature/egpu-usb4-keeper`, source commit `f23c05c57468ccdd7035789fbf38320d2c2c0d68`.
  The maintained installation owner is now the clean `exp` worktree; the installer enforces that branch so
  production `master` cannot be used for development DEXT replacement.
- The audited development installer replaced `/Applications/TinyGPU.app`; the app and
  DriverKit extension are ad-hoc signed, and DEXT
  `org.tinygrad.arkey.tinygpu.driver2` reached `[activated enabled]` at version `1.0.0/4`.
- A later reinstall attempt was correctly refused by macOS because this host has
  DriverKit development mode off and retains the prior ad-hoc registration. The CLI had
  also mislabeled `OSSystemExtensionErrorDomain` code 4 as a missing-entitlement error;
  code 4 means the extension was not found, while code 2 is the entitlement error. The
  installer now requires development mode explicitly and the next DEXT version is `5` so
  macOS can distinguish an upgrade from the already-active v4 registration.
- A0 stopped before any workload qualification. The UT4G bridge is connected at 40 Gb/s,
  but `system_profiler` reports no `1002:744c` PCI endpoint. The native diagnostic handshake
  therefore cannot open the provider. This is a signal/enumeration precondition, not evidence
  that the keeper passed or failed.
- No reset, power-cycle, sleep, replug, workload benchmark, or idle-duration claim was made.
- The complete machine-readable A0 artifact is
  `docs/task_workflow/output/egpu-usb4-persistent-pcie-A0-20260728T105840Z-89079.json`.
  The ignored installation transcript remains at
  `docs/task_workflow/output/tinygpu-development-install-provenance.txt` for local audit.

Qualification remains gated on endpoint enumeration. Once the endpoint is present, rerun A0
and then the acceptance matrix in the authoritative scope under the GPU lock. Do not infer
awake-idle, load-power, or sleep/wake behavior from this installation result.

## Current handoff (2026-07-28)

**Owner:** continue from the clean `exp` worktree at commit `530d77183` (the same change is
`b16356d95` on `dev` and `04f41ecfd` on `master`). The installer is development-only and
must run from `exp`; do not install from `master`.

**Host state:** macOS 26.5 (25F71), Apple M4 Mac mini, SIP disabled, and DriverKit
development mode now on. The ADTLink UT4G bridge is visible at USB4 40 Gb/s, but the last
check still showed no AMD `1002:744c` PCI endpoint. The installed app remains the prior
ad-hoc DEXT v4; v5 has not been activated and there is no valid v5 provenance transcript.

**Activation blocker:** `sysextd` reports two hidden registrations for
`org.tinygrad.arkey.tinygpu.driver2`: one `activated_enabled` and one
`terminating_for_upgrade_via_delegate`. This duplicate state produces
`OSSystemExtensionErrorDomain` code 4 during v5 activation. The log also prints
`package type not SYSX`; the DriverKit bundle is intentionally `.dext`/`DEXT` per Apple’s
DriverKit format, so do not change it to a generic `.system`/`SYSX` extension. Reboot the
Mac mini before retrying; if the duplicate remains, remove only this empty-team
registration with administrator authorization, then reinstall.

If a reboot does not clear it, run this locally on the Mac mini (not over SSH):

```sh
sudo systemextensionsctl uninstall - org.tinygrad.arkey.tinygpu.driver2
```

The `-` targets this development registration’s empty team ID. Do not use
`systemextensionsctl reset`, which would affect unrelated system extensions.

**Next commands after reboot:**

```sh
systemextensionsctl developer
systemextensionsctl list | rg -i -C2 'tinygpu|arkey'
cd /Users/julianabeleda/env/tinygrad-arkey-exp
/Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py -- \
  bash extra/usbgpu/tbgpu/installer/install_nosip.sh \
  --install APPROVE_TINYGPU_DEVELOPMENT_INSTALL \
  --provenance-out /Users/julianabeleda/env/tinygrad-arkey-exp/docs/task_workflow/output/tinygpu-development-install-provenance.txt
```

Approve the prompt only after the build and staged bundle checks pass. Then verify v5 is
`[activated enabled]`, run `TinyGPU keepalive handshake` and `TinyGPU keepalive status`,
confirm the PCI endpoint is present, and run qualification gates A0, A1, and A2 under the
GPU lock. A4/A8 remain deferred until those gates pass; no local GGUF model is currently
available for the A8 Qwen3 8B smoke gate. The installer rollback now preserves an already
active extension instead of trying to deactivate it on a failed replacement.

**Evidence boundary:** the tracked A0 artifact is
`docs/task_workflow/output/egpu-usb4-persistent-pcie-A0-20260728T105840Z-89079.json`.
The later failed A0 attempts (`...152615Z-18390.json` and `...152626Z-18408.json`) are
local diagnostic artifacts only; neither is acceptance evidence.

**Secondary defense:** `pmset disablesleep 1` and a launchd KeepAlive daemon are separate
sleep investigations, not substitutes for the native provider keeper.

**Fallback (kept):** `extra/remote/amd_power_cycle.py` physically power-cycles via a Shelly
smart plug when the link is already dead. Prevention lowers frequency; it does not make the
power-cycle recovery obsolete.
