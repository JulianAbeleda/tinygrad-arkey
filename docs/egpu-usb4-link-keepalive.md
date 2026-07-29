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
owned, separately reported by `tinygpu.power-residency.v4`, excluded from workload resource
counts, and released only during provider stop/failure or explicitly around a function reset.

## v12 BAR-residency recovery candidate (2026-07-29)

Git history establishes that the RX 7900 XTX completed Qwen inference over the Mac/TinyGPU
eGPU path before the native migration. Commit `778d029c2` records the May 21 live
Qwen3-1.7B control through `REMOTE=127.0.0.1:6667 DEV=PCI+AMD`: 256 tok/s prefill and
33 tok/s decode. The exact source state used by that recorded run is its parent,
`2d317b877`. The old bridge later present at `554800bef` retained `opened_devices`,
`mapped_bars`, and `sysmem_allocs`; `4c5e67cff` removed that bridge.

Earlier versions of this document incorrectly cited `29be4c9fa` and `5bff0135c` as eGPU
proof. Those commits record later RX 7900 XTX performance work on the Linux optimization
host; they prove the optimized kernels, not the Thunderbolt transport. The May 21 run is
the relevant Mac/eGPU control. It proves this physical setup has carried real inference and
therefore demotes an intrinsically defective cable from the lead hypothesis. It does not
exclude an intermittent signal-margin fault or a failure that appears only under the larger
8B transfer load.

The v12 candidate requests and maps BAR5 immediately after PCI open and identity validation,
before creating the one-Hz timer. It performs no MMIO read/write, AMD initialization, DMA,
allocation, reset, or model load. Provider health and workload admission require the BAR5
mapping to remain active with zero error. Full model residency remains a known-good control if
this minimal state is insufficient; it is not part of v12.

## v13 historical PCI-command restoration candidate (2026-07-29)

The last old-provider revision before the USBGPU prune, `a0250a41d`, explicitly enabled PCI
command bits `0x0007` (I/O space, memory space, and bus master) before publishing `tinygpu`.
The restored native provider at `f23c05c57` omitted that operation, and versions 7-12 never
restored it. The abstract DriverKit On request added later is not a substitute for enabling
the PCI function's concrete decode and bus-master bits.

v13 restores the old read/OR/write sequence after identity validation, adds a serializing
readback, and fails provider start if the required bits do not latch. The same operation runs
after a function reset, while each keepalive tick only observes the mask and degrades on loss.
Workload admission requires the confirmed mask, BAR5 residency, an accepted DriverKit On
request, an observed On state, and an identity canary later than both the PCI command operation
and the power request. This deliberately permits the On notification to predate a request made
against an already-On service; requesting the same state is not expected to force a redundant
callback. The source-only boundary and evidence are recorded in
`docs/task_workflow/input/egpu-usb4-v13-pci-command-residency-scope-20260729.md`.

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

## Current handoff (2026-07-29)

**Owner:** continue from the clean `exp` worktree containing the v13 recovery and this
handoff. The installer is development-only and must run from `exp`; do not install from
`master`.

**Installed/runtime state:** the audited v13 provider source is `8f7afc45f`. v13 restores
the old provider's PCI command mask `0x0007` (I/O space, memory decoding, and bus master),
retains BAR5, requests full power, runs the native one-Hz keeper, and requires a later
identity canary before workload admission. Two consecutive minimal AMD computations passed
after the lifecycle-ordering repair in `d448df508`; see
`docs/task_workflow/output/egpu-usb4-v13-historical-run-recreation-20260729T041747Z.md`.

The operator reports completing a GPU reset after the latest A12 attempt. No post-reset
admission or workload result has yet been recorded in this handoff.

### Historical control and corrected interpretation

The last committed Mac/eGPU model control is the May 21 Qwen3-1.7B run recorded by
`778d029c2`, using source state `2d317b877`:

```text
Mac mini -> Thunderbolt/USB4 -> UT4G/TinyGPU -> RX 7900 XTX
REMOTE=127.0.0.1:6667 DEV=PCI+AMD
Qwen3-1.7B Q4_K_M
prefill 256 tok/s; decode 33 tok/s
```

This is the last committed Mac/eGPU model artifact, not the full boundary of what ran. The
operator reports that the Mac eGPU was serving larger models in this period, including 8B
and 14B, although their exact Mac run logs are not preserved in Git. Keep that distinction
explicit: 1.7B is repository-proven; the larger Mac runs are operator-attested.

Two independent problems then overlapped:

1. On the Mac/Thunderbolt path, an available but unserved GPU did not reliably remain at
   idle: it could power off/disappear, and recovery commonly required a reboot. The
   operator's working mitigation was to give it VRAM-backed model work promptly after it
   appeared and keep the model resident. The same GPU attached through the Linux
   motherboard's native PCIe path could remain present while idle. Moving to native Linux
   therefore made resets and repeated kernel experiments much easier; it was an operational
   choice, not a conclusion that the Mac/Thunderbolt path could not serve a model.
2. Model throughput was low. Native Linux reproduced that performance problem without the
   Thunderbolt path, and the June sequence isolated tinygrad's kernel selection: `f4876230c`
   enabled `Q4K_PRIMITIVE`, `1c247520d` recorded the 8B improvement, and `171aba1f0` recorded
   the still-slow 14B result. Later route and primitive work improved those results further.

The Linux results culminating in `29be4c9fa` and `5bff0135c` are therefore evidence about
kernel performance, not eGPU transport. The historical conclusion was that low token rate
came from tinygrad falling onto the wrong or generic kernels; eGPU disappearance/recovery
was a separate reliability problem.

The May 22 through June 10 PSP/GART arc was also not evidence of a bad cable. The enabled
`AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c` workaround contained wrong MP0/MP1/NBIO versions;
`ca3a0d816` corrected the profile and `b033d47a7` restored the NBIF alias. Commit
`9eb0b042b` separately stopped `ensure_app` from replacing the matched arkey TinyGPU app
with an incompatible upstream build. The historical Mac environment required:

```text
AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c
AM_REMOTE_SKIP_RESIZE_BAR=1
```

Do not conflate those fixes with v13's restoration of the separately omitted PCI command
mask. Both classes of software/configuration regression have existed.

### Latest 8B result

The v13 A12 attempt admitted the local 5.0 GB Qwen3-8B model and populated approximately
4.68 GB before both ACIO lanes emitted a Gen2/3 error burst. macOS then reported link status
zero, marked the AMD functions dead, force-closed TinyGPU, and powered down the tunneled
PCIe path. No token was produced and the loaded observation interval never began. Full
details are in
`docs/task_workflow/output/egpu-usb4-v13-loaded-residency-a12-20260729T044024Z.md`.

That timeline identifies the transport-collapse boundary, not its root cause. In particular,
it does not overturn the May 21 same-path inference control. The historical success used a
1.7B model, while the failed attempt was transferring the 5.0 GB 8B model. The current lead
question is therefore whether the native migration regressed the formerly working workload
or whether failure probability rises with allocation volume/DMA cadence. Cable, connector,
host port, UT4G signal margin, and power integrity remain possible intermittent contributors,
but a cable replacement is not the first discriminating experiment.

Do not collapse the historical idle-off symptom and this latest under-load failure into one
assumed cause. Historically, prompt VRAM/model residency helped prevent disappearance; A12
instead failed while filling VRAM. That difference may reflect a current software regression
or a second failure mode and is exactly why the smaller historical workload must be restored
before changing physical variables.

### Next controlled sequence

Do not swap the cable or port before collecting the software control:

1. After the operator-reported reset, run fresh R6.1/A0 and require the same v13 service,
   advancing keeper, BAR5 residency, full-power confirmation, PCI command mask 7, later
   canary, and zero leaked workload resources.
2. On current v13, recreate the committed Qwen3-1.7B control first, in one process with the
   model kept resident, using the two historical AMD environment settings above. Require a
   token, clean unload/finalization, and preserved endpoint/provider identity.
3. If 1.7B passes, increase model/weight size in one controlled intermediate step before the
   5.0 GB Qwen3-8B run. This distinguishes a general native-runtime regression from a
   transfer-volume or allocation-cadence threshold.
4. If current v13 fails at 1.7B, recreate source state `2d317b877` with its matched historical
   Python bridge/app contract in an isolated worktree. Do not install an old app/DEXT or
   replace the active provider without a separate audited install gate.
5. Only after the software controls are classified should a known-good cable or alternate
   host port be introduced as a one-variable physical A/B.

The desired end state is the historically working Mac transport/runtime behavior carrying
the now-optimized kernels. Do not roll the performance work back merely to make the old
control pass.

**Fallback (kept):** `extra/remote/amd_power_cycle.py` physically power-cycles via a Shelly
smart plug when the link is already dead. Prevention lowers frequency; it does not make the
power-cycle recovery obsolete.
