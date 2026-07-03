# eGPU over USB4/Thunderbolt: the idle link-drop, its cause, and the keepalive fix

Durable record of a hardware issue that cost real debugging time and whose fix was
lost in a refactor. The symptom, root cause, and fix were previously recorded only in
a commit message (`554800bef`, 2026-06-10) and in code that has since been deleted —
banked here so it is discoverable.

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
of the deep ASPM/CLx states that fail to retrain. Cadence via `REMOTE_KEEPALIVE_S`
(`0` disables); default-on only for macOS.

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

## Where the fix now has to live (architecture migration)

The keepalive lived in `extra/remote/serve.py`, a 243-LOC Python socket bridge. On 2026-06-16
(`4c5e67cff`, "hard-fork prune Unit 1a") that bridge was deleted as part of migrating the
remote-GPU host from the Python bridge to a **native macOS app**: the client now connects to
`/Applications/TinyGPU.app` over a unix socket (`tinygrad/runtime/support/system.py`
`APLRemotePCIDevice.ensure_app()` / `__init__`, spawns `[APP_PATH, "server", sock]`).

Consequence: **the keepalive cannot be restored into this Python repo.** The device is held by
`TinyGPU.app`, so the config-space-read loop must run inside that app. As of this writing the
keepalive is **absent from the entire Python tree** (grep for `keepalive`/`REMOTE_KEEPALIVE`
→ zero hits). Whether the locally-built arkey app (`org.tinygrad.arkey`, "carries local
TinyGPU fixes" per `ensure_app`) already ports it is **unverified**.

## Action status

- **Verify (Mac-side, only the owner can):** leave the eGPU idle past the ~40-min failure
  window. If it survives (`0x744c` stays visible, PSP gate stays CLEAN), the app already carries
  the keepalive and nothing more is needed. If it still drops, port the snippet above into the
  `TinyGPU.app` server loop (gated by `REMOTE_KEEPALIVE_S`, default-on for the USB4 path).
- **Secondary defense:** `pmset disablesleep 1` + a launchd KeepAlive daemon guard against
  genuine macOS system-sleep tearing down the tree, but they are NOT the primary fix — the
  keepalive is.
- **Fallback (kept):** `extra/remote/amd_power_cycle.py` physically power-cycles via a Shelly
  smart plug when the link is already dead. Prevention lowers frequency; it does not make the
  power-cycle recovery obsolete.
