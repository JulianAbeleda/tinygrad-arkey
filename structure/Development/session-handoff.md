# Session Handoff

## Stop Point: 2026-05-22 TinyGPU No-SIP Reboot Gate

This is the current handoff for the `tinygrad-arkey` AMD/TinyGPU work.

Use this file at the start of the next session before changing code or restarting Qwen.

## Active Target

- Local repo: `/Users/julianabeleda/env/tinygrad-arkey`
- Global root: `/Users/julianabeleda/env/tinygrad`
- Global root state: symlink to `/Users/julianabeleda/env/tinygrad-arkey`
- GitHub fork: `https://github.com/JulianAbeleda/tinygrad-arkey`
- Branch: `master`
- Latest pushed commit on local `master`: `fa8780d1e [runtime] fix TinyGPU Xcode 26 build`
- Upstream PR: `https://github.com/tinygrad/tinygrad/pull/16333`

## High-Level Status

The active target is now one repo: `tinygrad-arkey`.

The remote AMD path is:

```text
tinygrad-arkey
  -> tinygrad AMD runtime
  -> RemotePCIDevice RPC
  -> extra/remote/serve.py
  -> TinyGPU DriverKit / UT4G
  -> Radeon RX 7900 XTX
```

The main blocker is now validating a patched TinyGPU app/dext on live hardware. The RX 7900 XTX is visible again in macOS as external PCIe GPU `0x744c`, but the currently active system extension is still the old signed bundle `org.tinygrad.tinygpu.driver2`. The patched local build uses `org.tinygrad.arkey.tinygpu.driver2` and cannot run while SIP/AMFI rejects ad-hoc DriverKit entitlements.

The latest research framing adds one important fact: the ASM2464PD bridge used in this path has an internal 8051-class firmware CPU with Program ROM/RAM and XDATA. That makes the UT4G path an active firmware-mediated bridge. This supports treating bad DMA/MMIO sequences as possible bridge-firmware-state triggers, but it does not prove the ASM2464PD firmware is the isolated root cause. A physical replug likely resets bridge firmware, USB4 tunnel state, TinyGPU DriverKit state, PCIe link training, and GPU endpoint state together.

## What Was Accomplished

- Renamed and standardized the active fork as `tinygrad-arkey`.
- Pointed `/Users/julianabeleda/env/tinygrad` at `tinygrad-arkey`.
- Documented AMD/ROCm/llama.cpp research in `docs/amd-rocm-llamacpp-research.md`.
- Added bridge health, dirty-state, and stats instrumentation.
- Added Qwen prefill/decode remote-pressure logging.
- Added Q4_K baseline benchmark support.
- Reproduced the GPU dropout outside model inference.
- Isolated the first clear local dropout trigger to repeated `16MB` TinyGPU `PrepareDMA` mappings.
- Documented the ASM2464PD 8051 firmware angle as relevant but unproven causality.
- Added remote-only AMD setup allocation cap:
  - default: `AMD_REMOTE_ALLOC_CAP_MB=2`
  - escape hatch: `AMD_REMOTE_ALLOC_CAP_MB=0`
- Fixed remote protocol write acknowledgements:
  - `MMIO_WRITE`
  - `SYSMEM_WRITE`
- Added explicit invalid sysmem handle errors.
- Guarded unsafe remote small-BAR discovery unless `AM_REMOTE_SMALL_BAR_DISCOVERY=1`.
- Added `AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c` to bypass the unsafe RX 7900 XTX discovery-table read.
- Found and patched a TinyGPU C server protocol bug: `CMD_MMIO_WRITE` skipped the RPC response path, causing every MMIO write to appear as an RPC timeout.
- Opened upstream PR #16333 with a narrowly scoped source fix and AI disclosure.
- Installed Xcode 26.5 and fixed the local Xcode 26 build issue by using the normal `AppIcon` asset instead of `tiny_icon.icon`.
- Built the patched TinyGPU Debug app with local `DerivedData`/module-cache paths.
- Ad-hoc signed the patched app/dext for the repo's No-SIP development path:
  - app: `org.tinygrad.arkey.tinygpu.installer`
  - dext: `org.tinygrad.arkey.tinygpu.driver2`
  - dext entitlement includes `com.apple.developer.driverkit.allow-any-userclient-access`.

## Latest Commits

```text
fa8780d1e [runtime] fix TinyGPU Xcode 26 build
08751ed0a [runtime] acknowledge TinyGPU MMIO writes
f1727ee74 [examples] generalize AMD BAR repro
a04e1d2a2 [examples] add BAR0 remote repro stages
281cef67f [test] cover remote PCI RPC framing
dfb47fd25 [runtime] centralize AMD remote detection
1bad4a9ec [docs] document ASM2464PD firmware hypothesis
```

## Latest Validation

Passed earlier:

- Capped AMD boot reached `gfx1100`.
- `has_sdma=True`.
- 16KB and 2MB host allocations completed.
- Synchronization completed.
- No `PrepareDMA size=16777216` during capped AMD boot.
- Tiny tensor sanity returned `[2, 3, 4]`.

Current state before reboot:

- `csrutil status`: SIP enabled.
- `system_profiler SPDisplaysDataType`: RX 7900 XTX visible as external PCIe GPU, vendor `0x1002`, device `0x744c`, x16.
- `systemextensionsctl list`: old signed `org.tinygrad.tinygpu.driver2` active; patched local `org.tinygrad.arkey.tinygpu.driver2` not installed yet.
- Patched app build path: `/Users/julianabeleda/env/tinygrad-arkey/extra/usbgpu/tbgpu/installer/build/DerivedData/Build/Products/Debug/TinyGPU.app`.
- Running that ad-hoc app with SIP enabled is blocked by AMFI: restricted entitlements are not validated for an ad-hoc signature.

## Current Checklist Position

See `structure/Development/amd-optimization-checklist.md`.

Current unchecked near-term items:

- Retest tensor sanity with `AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c` after the GPU re-enumerates.
- Retest Qwen 1.7B warmup after the tensor sanity passes.
- Retest Qwen 1.7B decode after allocation mitigation.
- Reduce decode-time `SYSMEM_READ`/`SYSMEM_WRITE` roundtrips.
- Add a benchmark gate for roundtrips/token regression checks.
- Prototype packed Q4_K_M fused dequant plus matvec for AMD/gfx1100.

## Next Session Plan

1. After rebooting from Recovery, confirm SIP is disabled:

```text
csrutil status
```

Apple Silicon Mac mini Recovery steps:

```text
1. Shut down the Mac mini.
2. Hold the physical power button until "Loading startup options" appears.
3. Choose Options > Continue.
4. Open Utilities > Terminal.
5. Run: csrutil disable
6. Confirm, then restart normally.
```

2. Confirm the RX 7900 XTX is visible:

```text
system_profiler SPDisplaysDataType
```

3. Install and activate the patched No-SIP TinyGPU app/dext:

```text
cd /Users/julianabeleda/env/tinygrad/extra/usbgpu/tbgpu/installer
./install_nosip.sh
```

Expected install target:

```text
/Applications/TinyGPU.app
org.tinygrad.arkey.tinygpu.driver2
```

If macOS asks for approval, use System Settings > Privacy & Security or System Settings > General > Login Items & Extensions > Driver Extensions.

4. Confirm the patched dext is active:

```text
systemextensionsctl list
```

5. Restart the bridge from the global root:

```text
cd /Users/julianabeleda/env/tinygrad
DEBUG=1 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python /Users/julianabeleda/env/tinygrad/extra/remote/serve.py 6667
```

6. Run the remote probe/health check:

```text
REMOTE_TIMEOUT=3 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python /Users/julianabeleda/env/tinygrad-arkey/extra/remote/bench.py 127.0.0.1:6667 --skip-tensor
```

7. Re-run BAR reads before writes:

```text
REMOTE_TIMEOUT=5 REMOTE_RPC_TIMEOUT=10 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python \
/Users/julianabeleda/env/tinygrad-arkey/extra/remote/amd_repro.py 127.0.0.1:6667 --stage bar-read --bars 0,2,5 --sizes 4 --offsets 0 --repeat 1
```

Known old result before patched install: 4-byte reads from BAR0, BAR2, and BAR5 pass. 4-byte writes to BAR0 and BAR2 time out while the GPU remains visible. Instrumentation showed the Python bridge blocked at `MMIO_WRITE store-start` while waiting for the nested TinyGPU app RPC. After installing the patched app/dext, this timeout should disappear if the missing TinyGPU C server response was the only failure at this node.

8. Run BAR writes:

```text
REMOTE_TIMEOUT=5 REMOTE_RPC_TIMEOUT=10 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python \
/Users/julianabeleda/env/tinygrad-arkey/extra/remote/amd_repro.py 127.0.0.1:6667 --stage bar-write --bars 0,2 --sizes 4 --offsets 0 --repeat 1
```

9. Run the smallest tensor sanity with the discovery profile:

```text
REMOTE_TIMEOUT=5 REMOTE=127.0.0.1:6667 DEV=PCI+AMD AMD_REMOTE_ALLOC_CAP_MB=2 AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c \
/Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python -c 'from tinygrad import Tensor; print((Tensor([1,2,3])+1).numpy().tolist())'
```

10. If tensor sanity passes, run Qwen 1.7B warmup with low max tokens before larger models.

11. If the GPU drops again, stop inference work and record the exact stage:

- probe/open
- BAR map
- sysmem allocation
- AMD boot
- small-BAR/discovery profile path
- prefill
- decode
- model load

## Relevant Docs

- `docs/amd-remote-dropout-investigation.md`
- `docs/amd-rocm-llamacpp-research.md`
- `structure/Development/amd-optimization-checklist.md`
- `structure/Development/tinygrad-coding-overrides.md`
- `structure/Development/coding-principles.md`
