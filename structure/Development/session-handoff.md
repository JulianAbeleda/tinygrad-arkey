# Session Handoff

## Stop Point: 2026-05-22 PSP KDB Load Stalls After Ready Mailbox

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

The patched TinyGPU app/dext has now been installed and approved on the No-SIP development path. `org.tinygrad.arkey.tinygpu.driver2` is active, the RX 7900 XTX is visible as external PCIe GPU `0x744c`, and the prior mapped-BAR write timeout is fixed. The current blocker has moved forward to AMD PSP boot: on a clean mailbox, the first PSP bootloader KDB load accepts the command, drops `C2PMSG35_BL` from ready to `0x0`, and never returns the ready bit.

The old signed extension `org.tinygrad.tinygpu.driver2` was toggled off in System Settings. After the normal reboot it is still listed by `systemextensionsctl` as `activated disabled`, but `pgrep -afil tinygpu` only shows patched `org.tinygrad.arkey.tinygpu.driver2` processes.

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
- Disabled SIP from Recovery, installed `/Applications/TinyGPU.app`, and approved the patched driver extension in System Settings.
- Validated the patched TinyGPU `CMD_MMIO_WRITE` response path on live hardware:
  - remote health bench stayed healthy
  - BAR0/BAR2/BAR5 4-byte reads passed
  - BAR0/BAR2 4-byte writes passed
  - RX 7900 XTX stayed visible after the tests

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

Post-reboot state:

- `csrutil status`: SIP disabled.
- `system_profiler SPDisplaysDataType`: RX 7900 XTX visible as external PCIe GPU, vendor `0x1002`, device `0x744c`, x16.
- `systemextensionsctl list`: patched local `org.tinygrad.arkey.tinygpu.driver2` active and enabled.
- Old signed `org.tinygrad.tinygpu.driver2`: still shown as `activated disabled`, but no old signed TinyGPU process is running.
- Installed patched app path: `/Applications/TinyGPU.app`.
- Remote health bench passed:
  - `bridge health: healthy`
  - `health: healthy`
  - BAR0 mapped
  - 8MB `MAP_SYSMEM` read/write passed
- BAR write validation passed:
  - `amd_repro.py --stage bar-read --bars 0,2,5 --sizes 4 --offsets 0 --repeat 1`
  - `amd_repro.py --stage bar-write --bars 0,2 --sizes 4 --offsets 0 --repeat 1`
- Tensor sanity with `AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c` no longer fails at the TinyGPU MMIO write response path. Before reboot it failed during PSP initialization with:

```text
TimeoutError: BL not ready. Timed out after 10000 ms, condition not met: 0 != 2147483648
```

- After reboot, the same smallest tensor sanity failed during PSP initialization with:

```text
RuntimeError: RPC failed: Connection closed
```

- Bridge log for the post-reboot tensor sanity:
  - `remote: DIRTY Connection closed`
  - `MMIO_READ_count`: `138987`
  - `MMIO_WRITE_count`: `105`
  - `RESIZE_BAR_count`: `1`
  - `CFG_WRITE_count`: `1`
- The GPU remained visible after the connection-closed failure.
- Remote health reported dirty afterward:

```text
bridge health: dirty: Connection closed
health: dirty (runtime check failed: RPC failed: bridge dirty: Connection closed)
```

- A `RESET` RPC attempt did not recover the bridge and hung before producing output.
- Restarting `extra/remote/serve.py` restored remote health:
  - `bridge health: healthy`
  - `health: healthy`
  - 8MB `MAP_SYSMEM` read/write passed
- Added PSP trace instrumentation gated by:
  - `AM_PSP_TRACE=1`
  - `AM_PSP_MSG1_READBACK=1`
  - `AM_PSP_ZERO_MSG1=1`
- Added `amd_repro.py --stage psp-status` for direct PSP mailbox reads.
- PSP trace narrowed the first boot failure:
  - initial C2PMSG35 was ready: `0x80000000`
  - first component: `PSP_FW_TYPE_PSP_KDB`
  - command: `PSP_BL__LOAD_KEY_DATABASE` / `0x80000`
  - C2PMSG36 address value: `0x80001` from `msg1_addr=0x8000100000`
  - C2PMSG35 write completed, then C2PMSG35 became `0x0`
  - C2PMSG35 never returned the ready bit and timed out with `BL not ready`
- `AM_PSP_MSG1_READBACK=1` verified the written KDB buffer bytes but did not recover the boot.
- Direct PSP mailbox status after the failed KDB command:

```text
C2PMSG33_VMBX   0x80000000
C2PMSG35_BL     0x00000000
C2PMSG36_ADDR   0x00080001
C2PMSG81_SOS    0x00000000
C2PMSG92_STATUS 0x00000000
```

- TinyGPU `RESET` RPC returned successfully and bridge health stayed healthy, but it did not restore C2PMSG35 readiness.
- Linux amdgpu PSP v13 code clears the full 1MB PSP primary firmware buffer before copying each bootloader component. tinygrad previously only overwrote the component payload plus padding. `AM_PSP_ZERO_MSG1=1` now tests full-buffer zeroing, but the current GPU state is already stuck at initial C2PMSG35 not-ready, so this needs a clean hardware reset/replug before the next tensor attempt.
- After a clean reboot/reset, `AM_PSP_ZERO_MSG1=1` still failed at the first KDB bootloader load:
  - C2PMSG35 started clean at `0x80000000`
  - KDB payload buffer was zeroed first
  - C2PMSG36 was written to `0x80001`
  - C2PMSG35 was written to `0x80000`
  - C2PMSG35 then stayed `0x0`
- Added `AM_PSP_SYSMSG1=1` to test using system memory for PSP `msg1` on remote TinyGPU. The existing direct USB path already used sysmem for `msg1`, but the `REMOTE=...` client path was using VRAM because `devfmt` is `remote:...`, not `usb:...`. The new path maps all returned sysmem pages into the boot page table instead of assuming one contiguous segment.
- After another clean reset, `AM_PSP_SYSMSG1=1` also failed at the first KDB bootloader load:
  - clean start had C2PMSG35=`0x80000000`
  - sysmem msg1 was allocated at `0x200000000000`, 256 pages, 1MB
  - C2PMSG36 was written to `0x2000000`
  - C2PMSG35 was written to `0x80000`
  - C2PMSG35 then stayed `0x0` until `BL not ready`
  - bridge health stayed healthy and the RX 7900 XTX stayed visible
- Direct PSP mailbox state after the sysmem msg1 failure:

```text
C2PMSG33_VMBX   0x80000000
C2PMSG35_BL     0x00000000
C2PMSG36_ADDR   0x02000000
C2PMSG81_SOS    0x00000000
C2PMSG92_STATUS 0x00000000
```

- Conclusion from zero-msg1 and sysmem-msg1 tests: the first KDB failure is probably not caused only by stale bytes after the firmware payload or by using VRAM vs sysmem for msg1.
- Linux source comparison found an MP0 13.0.10-specific PSP fatal-recovery quirk:
  - read `regMP0_SMN_C2PMSG_67`
  - write it back plus `0x10`
  - sleep 1000ms
  - Linux comment: "trigger PSP dram read to unhalt PSP during MP1 triggered sync flood"
- Added `AM_PSP_FATAL_QUIRK=1` to test that quirk before the first PSP bootloader component load. This is off by default and only applies when `MP0_HWIP == (13,0,10)`.
- Expanded `amd_repro.py --stage psp-status` to also report:
  - `C2PMSG64_RING`
  - `C2PMSG67_WPTR`
  - `C2PMSG90_SMU`
  - `C2PMSG115_SPI`
- After a clean reset, `AM_PSP_FATAL_QUIRK=1` also failed at the first KDB bootloader load:
  - clean start had C2PMSG35=`0x80000000`
  - clean C2PMSG67 was `0x00000000`
  - quirk wrote C2PMSG67 to `0x00000010`
  - KDB load wrote C2PMSG36=`0x80001` and C2PMSG35=`0x80000`
  - C2PMSG35 then stayed `0x0` until `BL not ready`
  - bridge health stayed healthy and the RX 7900 XTX stayed visible
- Direct PSP mailbox state after the fatal-quirk failure:

```text
C2PMSG33_VMBX   0x80000000
C2PMSG35_BL     0x00000000
C2PMSG36_ADDR   0x00080001
C2PMSG64_RING   0x00000000
C2PMSG67_WPTR   0x00000010
C2PMSG81_SOS    0x00000000
C2PMSG90_SMU    0x1e983568
C2PMSG92_STATUS 0x00000000
C2PMSG115_SPI   0x80000000
```
- Added `AM_PRE_PSP_MODE1_RESET=1` to test a more invasive pre-PSP reset/init sequence:
  - probe SMU liveness
  - clear PCI bus master
  - issue `self.smu.mode1_reset()`
  - restore PCI bus master
  - continue normal SOC/GMC/IH/PSP/SMU initialization
- This mirrors tinygrad's existing malformed-state reset sequence, but makes it available before the first KDB load on a clean remote path.
- After a clean reset, `AM_PRE_PSP_MODE1_RESET=1` did not issue mode1 because `self.smu.is_smu_alive()` returned `False` before PSP init:
  - trace: `pre-PSP mode1 reset requested smu_alive=False`
  - normal init continued
  - KDB load still wrote C2PMSG36=`0x80001` and C2PMSG35=`0x80000`
  - C2PMSG35 then stayed `0x0` until `BL not ready`
  - bridge health stayed healthy and the RX 7900 XTX stayed visible
- Direct PSP mailbox state after the pre-PSP mode1 attempt:

```text
C2PMSG33_VMBX   0x80000000
C2PMSG35_BL     0x00000000
C2PMSG36_ADDR   0x00080001
C2PMSG64_RING   0x00000000
C2PMSG67_WPTR   0x00000000
C2PMSG81_SOS    0x00000000
C2PMSG90_SMU    0x1e983568
C2PMSG92_STATUS 0x00000000
C2PMSG115_SPI   0x80000000
```

## Current Checklist Position

See `structure/Development/amd-optimization-checklist.md`.

Current unchecked near-term items:

- Investigate AMD PSP init connection close / prior bootloader timeout:
  - current stable symptom: first KDB load writes `C2PMSG36_ADDR`, writes `C2PMSG35_BL=0x80000`, then `C2PMSG35_BL` stays `0x0`
  - bridge generally remains healthy on the `BL not ready` path
  - the older post-reboot `RuntimeError: RPC failed: Connection closed` remains relevant history, but it is not the latest narrowed failure

## Current Memory Mapping Understanding

- tinygrad is already using GPU page-table-backed mappings, not an ad hoc lookup table.
- `AMMemoryManager` owns VRAM physical allocation, boot/page-table allocation, and GPU virtual address allocation at base `0x200000000000`.
- `AMPageTableEntry.set_entry()` writes real AMD PTEs into VRAM page tables.
- VRAM mappings use `AddrSpace.PHYS` and convert local physical offsets through `paddr2xgmi()`.
- System-memory mappings use `AddrSpace.SYS` and mark PTEs with the AMD system-memory bit.
- The PSP `msg1` path has now been tested both ways:
  - default remote path: 1MB VRAM allocation, PSP sees a memory-controller address such as `0x8000100000`, and `C2PMSG36_ADDR` receives `msg1_addr >> 20`
  - `AM_PSP_SYSMSG1=1`: 1MB DriverKit/sysmem allocation, all 256 returned 4KB pages mapped into the boot page table as `AddrSpace.SYS`, and `C2PMSG36_ADDR` receives the mapped GPU VA shifted by 20
- A hashmap or database would only help as debug observability for VA-to-physical mappings. It would not change what the PSP consumes. The root-cause candidate remains address-domain/aperture/setup or firmware parsing/init sequencing, not missing bookkeeping.

## Latest PSP Firmware Parse And Repro: 2026-05-23

- Added `amd_repro.py --stage psp-fw` to parse PSP SOS firmware without opening the TinyGPU bridge.
- `psp_13_0_10_sos.bin` parse result:
  - file bytes: `360352`
  - file sha256: `0bcaaad9cd8578d3841ae69155a6bd4fc3ceae8f4fb5a6ba4f576e7ace94d1d9`
  - header version: `2.0`
  - `ucode_array_offset`: `0x100`
  - component count: `9`
  - KDB component: `PSP_FW_TYPE_PSP_KDB`, descriptor offset `0x0`, blob offset `0x100`, size `0x1d40` / `7488`, sha256 `f19238a9d2401673ddbc2d7a5eba1640afd2a524b63f83d1daa91ede38226632`
- Linux `amdgpu` PSP v13 parsing also uses packed PSP SOS v2 descriptors and computes component starts from `ucode_array_offset + desc.offset_bytes`; the tinygrad KDB slicing matches that expectation.
- After the AMD GPU restart, clean PSP status before tensor sanity was:

```text
C2PMSG33_VMBX   0x80000000
C2PMSG35_BL     0x80000000
C2PMSG36_ADDR   0x00000000
C2PMSG64_RING   0x00000000
C2PMSG67_WPTR   0x00000000
C2PMSG81_SOS    0x00000000
C2PMSG90_SMU    0x1e983568
C2PMSG92_STATUS 0x00000000
C2PMSG115_SPI   0x80000000
```

- Remote health before tensor sanity was healthy.
- Smallest tensor sanity with `AM_PSP_TRACE=1` reproduced the same first-KDB failure:
  - KDB bytes: `7488`
  - wrote `C2PMSG36_ADDR=0x80001` from `msg1_addr=0x8000100000`
  - wrote `C2PMSG35_BL=0x80000`
  - `C2PMSG35_BL` became `0x0` and timed out with `BL not ready`
- PSP status after the timeout:

```text
C2PMSG33_VMBX   0x80000000
C2PMSG35_BL     0x00000000
C2PMSG36_ADDR   0x00080001
C2PMSG64_RING   0x00000000
C2PMSG67_WPTR   0x00000000
C2PMSG81_SOS    0x00000000
C2PMSG90_SMU    0x1e983568
C2PMSG92_STATUS 0x00000000
C2PMSG115_SPI   0x80000000
```

- Remote health after the timeout remained healthy. This keeps the current failure framed as PSP/GPU boot state, not bridge transport dropout.
- Linux comparison found a more relevant address-domain difference than firmware parsing:
  - Linux allocates the PSP primary firmware buffer with `amdgpu_bo_create_kernel(... PSP_1_MEG, PSP_1_MEG, GTT unless debug_use_vram_fw_buf/SRIOV ...)`.
  - Linux passes `psp->fw_pri_mc_addr >> 20` to `C2PMSG36`.
  - The older `AM_PSP_SYSMSG1=1` tinygrad experiment used sysmem but passed a GPUVM VA (`0x200000000000 >> 20`), not a direct DMA/GART-style MC address.
- Added `amd_repro.py --stage psp-sysmem-probe` to inspect TinyGPU DMA segments without invoking `AMDDevice`.
  - Default 1MB sysmem probe returned one 1MB span at `0x80104000`, but it was not 1MB-aligned.
  - `--contiguous` 1MB sysmem probe returned one 1MB span at `0x80000000`, aligned to 1MB, so the Linux-like `C2PMSG36` value would be `0x800`.
- Added runtime experiment flag `AM_PSP_SYSMSG1_DMA=1`.
  - On remote TinyGPU it allocates 1MB sysmem with `contiguous=True`.
  - It asserts the returned DMA pages form one contiguous 1MB span and are 1MB-aligned.
  - It sets `msg1_addr` to the returned DMA address directly, instead of mapping the pages into GPUVM and passing the GPU VA.
- First `AM_PSP_SYSMSG1_DMA=1` attempt on the clean restart did not reach PSP because the 1MB contiguous allocation was not 1MB-aligned (`0x80804000`). The implementation was changed to allocate 2MB contiguous sysmem and use a 1MB-aligned sub-window inside it.
- Retest with the 2MB/window implementation reached PSP:
  - raw DMA span started at `0x81948000`
  - selected aligned msg1 window offset `0xb8000`
  - `msg1_addr=0x81a00000`
  - wrote `C2PMSG36_ADDR=0x81a`
  - wrote `C2PMSG35_BL=0x80000`
  - `C2PMSG35_BL` dropped to `0x0` and timed out with `BL not ready`
- PSP status after the DMA-address failure:

```text
C2PMSG33_VMBX   0x80000000
C2PMSG35_BL     0x00000000
C2PMSG36_ADDR   0x0000081a
C2PMSG64_RING   0x00000000
C2PMSG67_WPTR   0x00000000
C2PMSG81_SOS    0x00000000
C2PMSG90_SMU    0x1e983568
C2PMSG92_STATUS 0x00000000
C2PMSG115_SPI   0x80000000
```

- Remote health after the DMA-address failure remained healthy.
- Conclusion: the first KDB failure is not fixed by switching from VRAM MC address or GPUVM sysmem VA to a direct aligned TinyGPU DMA address.
- The current mailbox is already stuck after the previous KDB attempt:

```text
C2PMSG35_BL     0x00000000
C2PMSG36_ADDR   0x0000081a
```

- Do not run `AM_PSP_SYSMSG1_DMA=1` until the next clean GPU restart/replug confirms `C2PMSG35_BL=0x80000000`.

## Next Best Branch

- Do not keep burning clean hardware resets on the same KDB test unless a new hypothesis changes the inputs.
- The PSP SOS firmware parsing branch did not show a KDB descriptor/slicing bug.
- Next hardware branch, only after a new code change: find a reset/init path stronger than DriverKit function/hot reset and not dependent on SMU being alive before PSP init.
  - narrowed trace symptom: first KDB bootloader load leaves C2PMSG35 at `0x0`
- Next code/research branch: compare Linux's pre-KDB sequencing and PSP memory/address programming more closely than component parsing:
  - `AM_PSP_SYSMSG1_DMA=1` failed, so direct DMA address alone is not enough
  - whether KDB needs another aperture/address setup before `C2PMSG36_ADDR`
  - whether an MP0/NBIO/GMC register prerequisite is missing before the first bootloader command
  - whether Linux performs an earlier PCI/device reset or PSP reset path that tinygrad does not model
- Retest Qwen 1.7B warmup after the tensor sanity passes.
- Retest Qwen 1.7B decode after allocation mitigation.
- Reduce decode-time `SYSMEM_READ`/`SYSMEM_WRITE` roundtrips.
- Add a benchmark gate for roundtrips/token regression checks.
- Prototype packed Q4_K_M fused dequant plus matvec for AMD/gfx1100.

## Next Session Plan

1. Confirm SIP is still disabled:

```text
csrutil status
```

Expected:

```text
System Integrity Protection status: disabled.
```

2. Confirm the RX 7900 XTX is visible:

```text
system_profiler SPDisplaysDataType
```

3. Confirm the patched dext is active and the old signed dext is not running:

```text
systemextensionsctl list
pgrep -afil 'org.tinygrad.*tinygpu'
```

Expected active/running target:

```text
org.tinygrad.arkey.tinygpu.driver2
```

4. Restart the bridge from the global root:

```text
cd /Users/julianabeleda/env/tinygrad
DEBUG=1 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python /Users/julianabeleda/env/tinygrad/extra/remote/serve.py 6667
```

5. Run the remote probe/health check:

```text
REMOTE_TIMEOUT=3 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python /Users/julianabeleda/env/tinygrad-arkey/extra/remote/bench.py 127.0.0.1:6667 --skip-tensor
```

6. Re-run BAR reads before writes:

```text
REMOTE_TIMEOUT=5 REMOTE_RPC_TIMEOUT=10 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python \
/Users/julianabeleda/env/tinygrad-arkey/extra/remote/amd_repro.py 127.0.0.1:6667 --stage bar-read --bars 0,2,5 --sizes 4 --offsets 0 --repeat 1
```

Known old result before patched install: 4-byte reads from BAR0, BAR2, and BAR5 pass. 4-byte writes to BAR0 and BAR2 time out while the GPU remains visible. Instrumentation showed the Python bridge blocked at `MMIO_WRITE store-start` while waiting for the nested TinyGPU app RPC. After installing the patched app/dext, this timeout should disappear if the missing TinyGPU C server response was the only failure at this node.

7. Run BAR writes:

```text
REMOTE_TIMEOUT=5 REMOTE_RPC_TIMEOUT=10 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python \
/Users/julianabeleda/env/tinygrad-arkey/extra/remote/amd_repro.py 127.0.0.1:6667 --stage bar-write --bars 0,2 --sizes 4 --offsets 0 --repeat 1
```

8. Run the smallest tensor sanity with the discovery profile. Latest post-reboot result reaches PSP initialization and closes the TinyGPU bridge connection.

```text
REMOTE_TIMEOUT=5 REMOTE=127.0.0.1:6667 DEV=PCI+AMD AMD_REMOTE_ALLOC_CAP_MB=2 AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c \
/Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python -c 'from tinygrad import Tensor; print((Tensor([1,2,3])+1).numpy().tolist())'
```

9. For the next clean PSP experiment after hardware reset/replug, first confirm PSP status:

```text
REMOTE_TIMEOUT=5 REMOTE_RPC_TIMEOUT=10 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python \
/Users/julianabeleda/env/tinygrad-arkey/extra/remote/amd_repro.py 127.0.0.1:6667 --stage psp-status
```

Expected clean starting point includes `C2PMSG35_BL 0x80000000`.

10. Do not continue Qwen work. After a clean hardware reset/replug, first confirm expanded PSP status:

```text
REMOTE_TIMEOUT=5 REMOTE_RPC_TIMEOUT=10 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python \
/Users/julianabeleda/env/tinygrad-arkey/extra/remote/amd_repro.py 127.0.0.1:6667 --stage psp-status
```

Expected clean starting point still includes `C2PMSG35_BL 0x80000000`.

11. Continue comparing early AMD init/reset and PSP bootloader prerequisites against Linux amdgpu, not msg1 placement:
  - PSP bootloader steady-state waits and status handling for MP0 13.0.10
  - whether tinygrad should do a mode1/reset sequence before first PSP KDB load on this remote path
  - whether any register/programming before PSP KDB differs from Linux for Navi31/RX 7900 XTX

Next branch: `AM_PRE_PSP_MODE1_RESET=1` was a no-op because SMU is not alive before PSP init. A useful next experiment needs a reset path that does not depend on SMU being alive, or a closer comparison with Linux's earlier PCI/device reset path before PSP KDB.
- TinyGPU reset path inspection:
  - Python `RemotePCIDevice.reset()` sends `RemoteCmd.RESET`.
  - `extra/remote/serve.py` calls `pci_dev.reset()`.
  - `APLRemotePCIDevice` inherits the remote reset RPC to the TinyGPU C server.
  - TinyGPU C server `CMD_RESET` calls DriverKit RPC selector `2`.
  - `TinyGPUDriver::ResetDevice()` calls `IOPCIDevice::Reset(kIOPCIDeviceResetTypeFunctionReset)` and falls back to `kIOPCIDeviceResetTypeHotReset`.
- Caveat: after RESET, the current Python bridge process may still hold stale opened-device and BAR mapping state. A cleaner reset experiment should run reset in its own client, then reconnect with a fresh client/process for `psp-status`.
- Added `amd_repro.py --stage reset` to send one reset and exit.

Reset recovery test sequence for current stuck PSP state:

```text
REMOTE_TIMEOUT=5 REMOTE_RPC_TIMEOUT=10 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python \
/Users/julianabeleda/env/tinygrad-arkey/extra/remote/amd_repro.py 127.0.0.1:6667 --stage reset

REMOTE_TIMEOUT=5 REMOTE_RPC_TIMEOUT=10 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python \
/Users/julianabeleda/env/tinygrad-arkey/extra/remote/amd_repro.py 127.0.0.1:6667 --stage psp-status
```

If C2PMSG35 returns to `0x80000000`, TinyGPU reset can recover PSP without reboot/replug when used with a fresh client. If it remains `0x0`, the reset is not sufficient for this PSP stuck state.
- Result: standalone TinyGPU reset did not recover PSP readiness:
  - `amd_repro.py --stage reset` returned success in ~273ms
  - fresh-client `amd_repro.py --stage psp-status` still reported C2PMSG35=`0x00000000`
  - C2PMSG36 remained `0x00080001`
  - C2PMSG90 remained `0x1e983568`
  - C2PMSG115 remained `0x80000000`
- Conclusion: DriverKit function reset / hot reset as exposed by TinyGPU is not sufficient to unwind this PSP KDB-stuck state. A full GPU/bridge replug or Mac reboot is still needed for clean PSP state.
- Added next Linux-sequencing experiment:
  - `AM_NBIO_CLEAR_STRAP2=1` clears bit 7 of `regRCC_DEV0_EPF2_STRAP2` during `AM_SOC.init_hw`, even on the NBIO 7.9.x branch where tinygrad previously skipped Linux's `nbio_v4_3_init_registers` strap clear.
  - `AM_PSP_TRACE_REGS=1` dumps selected NBIO/GMC/MMHUB/PSP registers immediately before the first PSP bootloader component load.
  - This has not been run against a clean mailbox yet. Current live status after another TinyGPU reset still shows `C2PMSG35_BL=0x00000000` and `C2PMSG36_ADDR=0x0000081a`, so the next KDB attempt must wait for full GPU/bridge restart back to `C2PMSG35_BL=0x80000000`.
- Result after clean restart on 2026-05-23:
  - Clean pre-test mailbox was ready: `C2PMSG35_BL=0x80000000`, `C2PMSG36_ADDR=0x00000000`.
  - `AM_NBIO_CLEAR_STRAP2=1` read `regRCC_DEV0_EPF2_STRAP2 old=0x0 new=0x0`; the Linux strap bit was already clear, so this is not the missing PSP prerequisite.
  - Pre-KDB trace showed `fb_base=0x8000000000`, `fb_end=0x85ff000000`, `vmhubs=1`, disabled AGP (`BOT=0x00ffffff`, `TOP=0`), system aperture over VRAM (`LOW=0x00200000`, `HIGH=0x00217fc0`), and PSP mailbox ready.
  - The first KDB load still wrote `C2PMSG36_ADDR=0x80001` from `msg1_addr=0x8000100000`, wrote `C2PMSG35_BL=0x80000`, then `C2PMSG35_BL` dropped to `0x0` and timed out.
  - Post-failure mailbox was `C2PMSG35_BL=0x00000000`, `C2PMSG36_ADDR=0x00080001`, `C2PMSG81_SOS=0x00000000`.
  - One interesting trace detail: `regBIFC_GFX_INT_MONITOR_MASK` and `regBIFC_DOORBELL_ACCESS_EN_PF` read back as `0x0` immediately after `AM_SOC.init_hw`, even though the NBIO 7.9.x branch writes them. That may be direct-vs-indirect NBIO access or harmless doorbell-only setup, but it is now a concrete discrepancy to inspect.
- Added next GART-address experiment:
  - `AM_PSP_SYSMSG1_GART=1` allocates a 2MB contiguous TinyGPU sysmem buffer, chooses a 1MB-aligned sub-window, builds a minimal 512MB Linux-style VMID0 GART table in VRAM, maps that sysmem window at `gmc.gart_start`, programs MMHUB context0 start/end/base registers, and passes `gmc.gart_start >> 20` to PSP.
  - This is the first experiment that makes PSP see a GART/MC address instead of VRAM MC, GPUVM VA, or raw DMA address.
  - First clean run was invalid as a GART test because `setup_psp_gart()` ran during PSP `init_sw`, then later `AM_GMC.init_hw()` overwrote VMID0/context0 before KDB. It still failed in the familiar way, but the trace showed context0 start/end/base did not match the intended GART window.
  - The implementation is now fixed to defer `setup_psp_gart()` until PSP `init_hw`, immediately before the pre-KDB trace and KDB load, after GMC init has completed.
  - Current live mailbox after the invalid run is stuck again: `C2PMSG35_BL=0x00000000`, `C2PMSG36_ADDR=0x07fff000`.
- Result after fixed GART run on 2026-05-23:
  - Clean pre-test mailbox was ready: `C2PMSG35_BL=0x80000000`, `C2PMSG36_ADDR=0x00000000`.
  - Pre-KDB trace confirmed the intended GART context was still programmed immediately before KDB:
    - `msg1_addr=0x7fff00000000`, `C2PMSG36_ADDR` value `0x07fff000`
    - `MMVM_CONTEXT0_CNTL=0x00000001`
    - `MMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32=0x01400001`
    - `MMVM_CONTEXT0_PAGE_TABLE_START_ADDR=(hi=0x7, lo=0xfff00000)`
    - `MMVM_CONTEXT0_PAGE_TABLE_END_ADDR=(hi=0x7, lo=0xfff1ffff)`
  - First attempt with fixed GART advanced farther than any prior run: the first KDB load did not hard-timeout at `0x0`; the trace then showed loads for `PSP_BL__LOAD_TOS_SPL_TABLE` and start of `PSP_FW_TYPE_PSP_SYS_DRV`, but the TinyGPU RPC connection closed while writing `C2PMSG36` for SYS_DRV.
  - Restarting only the Python bridge after that returned PSP status to a clean bootloader-ready mailbox, suggesting that first advanced run did not leave the PSP in the usual stuck state.
  - A second fixed-GART run from that clean mailbox regressed to the familiar first-KDB stall: `C2PMSG35_BL` became `0x0` and timed out. Post-failure mailbox was `C2PMSG35_BL=0x00000000`, `C2PMSG36_ADDR=0x07fff000`, `C2PMSG81_SOS=0x00000000`.
  - Interpretation: Linux-style VMID0 GART addressing is materially closer, but not yet stable or sufficient. The advanced first run means the old "always fails immediately at first KDB" invariant is broken, but the second run shows GART alone is not a clean fix.
- Added next GART tightening:
  - After writing the PSP GART table entries in VRAM, `setup_psp_gart()` now flushes HDP before programming context0.
  - In GART mode, `setup_psp_gart()` now expands `MMMC_VM_SYSTEM_APERTURE_LOW_ADDR/HIGH_ADDR` to cover `min(fb_base, gart_start)` through `max(fb_end, gart_end)` instead of leaving the system aperture VRAM-only.
  - Result after clean restart: pre-KDB trace confirmed `MMMC_VM_SYSTEM_APERTURE_HIGH_ADDR=0x1fffc7ff`, context0 still covered `0x7fff00000000-0x7fff1fffffff`, and msg1 readback passed. The first KDB load still dropped `C2PMSG35_BL` to `0x0` and timed out. Post-failure mailbox: `C2PMSG35_BL=0x00000000`, `C2PMSG36_ADDR=0x07fff000`, `C2PMSG81_SOS=0x00000000`.
  - Conclusion: expanding the MMHUB system aperture and flushing the GART table did not make GART mode reliable. The one prior advanced run remains an important clue but not a stable fix.
- Added next placement variant:
  - `AM_PSP_GART_LOW=1` places the test GART at MC address `0x0` instead of Linux's high placement `0x7fff00000000`.
  - This keeps the GART table/context experiment intact but changes the PSP mailbox address from high `C2PMSG36_ADDR=0x07fff000` to low `0x00000000`.
  - Rationale: every stable non-GART mailbox value so far was small (`0x800`, `0x81a`, `0x80001`), while high GART placement is the only experiment that sometimes progressed but remained unstable.
  - Result after clean restart: low-GART did not help. Pre-KDB trace confirmed `gart=0x0-0x1fffffff`, `MMMC_VM_SYSTEM_APERTURE_LOW_ADDR=0x0`, context0 start/end `0x0-0x1ffff`, and msg1 readback passed. First KDB still dropped `C2PMSG35_BL` to `0x0` and timed out. Post-failure mailbox: `C2PMSG35_BL=0x00000000`, `C2PMSG36_ADDR=0x00000000`, `C2PMSG81_SOS=0x00000000`.
- Added Linux pre-PSP HDP-remap experiment:
  - Linux `gmc_v11_0_init_golden_registers()` is effectively empty for this non-SRIOV path, so golden registers are not the missing pre-PSP step.
  - Linux `nbio_v4_3_remap_hdp_registers()` writes HDP MEM/REG remap registers before PSP: non-SRIOV `rmmio_remap.reg_offset = 0x7f000`, so MEM gets `0x7f000` and REG gets `0x7f004`.
  - Added read-only `amd_repro.py --stage nbio-status`; corrected it to use direct BAR reads for low NBIO registers and RSMU indirect reads for high NBIO registers.
  - Current stuck-state probe after low-GART failure:
    - PSP: `C2PMSG35_BL=0x00000000`, `C2PMSG36_ADDR=0x00000000`, `C2PMSG81_SOS=0x00000000`.
    - NBIO: `BIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL=0x00000000`, `BIF_BX0_REMAP_HDP_REG_FLUSH_CNTL=0x00000000`, `RCC_DEV0_EPF0_RCC_DOORBELL_APER_EN=0x0000000f`, `RCC_DEV0_EPF2_STRAP2=0x00000000`.
  - Added opt-in `AM_NBIO_REMAP_HDP=1` in `AM_SOC.init_hw`, writing MEM remap `0x7f000` and REG remap `0x7f004` before PSP init.
  - TinyGPU `amd_repro.py --stage reset` returned success but did not clear the PSP stuck state; fresh `psp-status` still showed `C2PMSG35_BL=0x00000000`. A physical/full GPU restart is still required before running the HDP-remap experiment.
  - Result after clean restart:
    - Clean pre-test mailbox was ready: `C2PMSG35_BL=0x80000000`, `C2PMSG36_ADDR=0x00000000`.
    - First run used too short an RPC timeout and timed out during BAR mapping before PSP.
    - Second run with longer timeouts programmed HDP remap (`regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL=0x0007f000`) and appeared to advance through KDB/SPL into SYS_DRV, but `C2PMSG35_BL` had changed to `0xffffffff`. That is an all-ones MMIO read, not reliable PSP readiness.
    - `_wait_for_bootloader()` now rejects `0xffffffff` as invalid even though the ready bit is set.
    - Retest with the all-ones guard failed at the same first KDB wait: after `PSP_FW_TYPE_PSP_KDB`, `C2PMSG35_BL` became `0x0` and timed out.
    - Post-failure mailbox: `C2PMSG35_BL=0x00000000`, `C2PMSG36_ADDR=0x00080001`, `C2PMSG81_SOS=0x00000000`.
  - Conclusion: HDP remap alone is not the missing PSP prerequisite. Earlier "advanced" traces that include `C2PMSG35_BL=0xffffffff` should be treated as suspect unless repeated under the all-ones guard.
- High-GART retest under the all-ones guard after clean restart:
  - Clean pre-test mailbox was ready: `C2PMSG35_BL=0x80000000`, `C2PMSG36_ADDR=0x00000000`.
  - `AM_PSP_SYSMSG1_GART=1` allocated sysmem at raw `0x80000000`, programmed high GART at `0x7fff00000000`, and confirmed msg1 readback for KDB.
  - Pre-KDB context was the intended high-GART setup: context0 enabled, page table base `0x01400001`, start `0x7fff00000000`, end `0x7fff1fffffff`, system aperture high `0x1fffc7ff`.
  - First KDB still dropped `C2PMSG35_BL` to `0x0` and timed out. No trusted advancement beyond KDB occurred under the all-ones guard.
  - Post-failure mailbox: `C2PMSG35_BL=0x00000000`, `C2PMSG36_ADDR=0x07fff000`, `C2PMSG81_SOS=0x00000000`.
  - Conclusion: high GART by itself is not sufficient. The earlier apparent KDB/SPL progress was likely contaminated by `0xffffffff` MMIO reads being accepted as ready.
- NBIO 7.9 `BIFC_*` access-path probe:
  - Added `indirect_rreg_pcie()` and opt-in `AM_NBIO_PCIE_BIFC=1` for writing `regBIFC_GFX_INT_MONITOR_MASK` and `regBIFC_DOORBELL_ACCESS_EN_PF` through the PCIE index/data path.
  - Added read-only/read-write `amd_repro.py` stages:
    - `nbio-status`: compares direct low NBIO reads, RSMU indirect reads, and PCIE-index indirect reads.
    - `nbio-bifc-pcie-write`: writes the two `BIFC_*` regs through PCIE index/data and reads them back.
    - `nbio-bifc-rsmu-write`: writes the same regs through RSMU index/data and reads them back.
  - Current PSP was already stuck from the high-GART test, but NBIO access probes are independent and completed.
  - Results:
    - `nbio-status`: both RSMU and PCIE reads report `BIFC_DOORBELL_ACCESS_EN_PF=0x0` and `BIFC_GFX_INT_MONITOR_MASK=0x0`.
    - PCIE write probe: wrote `0xfffff` / `0x7ff`, immediate readback remained `0x0`.
    - RSMU write probe: wrote `0xfffff` / `0x7ff`, immediate readback remained `0x0`.
  - Conclusion: the `BIFC_*` readback discrepancy is not a simple RSMU-vs-PCIE access-path bug. These registers may be write-ignored/write-only in this state, gated by another register, or irrelevant to the PSP KDB failure. Do not spend the next clean restart on `AM_NBIO_PCIE_BIFC=1` unless a Linux source comparison makes it meaningful.
- Linux PSP/init-order branch:
  - Linux `psp_v13_0_wait_for_bootloader_steady_state()` does a VMBX `C2PMSG33` precheck only for MP0 13.0.6/12/14/15; for this MP0 13.0.10 path it returns 0, so that is not the missing pre-KDB step.
  - Linux `psp_hw_start()` component order matches tinygrad for MP0 13.0.10: KDB, KDB-as-SPL table, SYS_DRV, SOC_DRV, INTF_DRV, DBG_DRV, RAS_DRV, SOS.
  - Linux `psp_sw_init()` can perform PSP memory training before `psp_hw_start()` and before KDB if the reserved VRAM training region is present/enabled. This is now the next clean experiment.
  - Added opt-in `AM_PSP_MEM_TRAIN=long|short` before KDB:
    - Computes Linux-style training offsets from VRAM size:
      - `c2p = ALIGN(vram_size - reserve_size - 1MB, 1MB)`
      - `p2c = vram_size - GDDR6_MEM_TRAINING_OFFSET`
      - default `reserve_size = 64KB`, override with `AM_PSP_MEM_TRAIN_RESERVE`.
    - Sends `PSP_BL__DRAM_LONG_TRAIN` or `PSP_BL__DRAM_SHORT_TRAIN` through `C2PMSG36/35`.
    - Rejects `0xffffffff` all-ones mailbox reads during the training wait.
    - For long training, saves and restores the bottom `BIST_MEM_TRAINING_ENCROACHED_SIZE` / 32MB of VRAM, matching Linux's warning that long training can overwrite bottom VRAM.
  - Validation: Python syntax check passes for `ip.py`, `amdev.py`, and `amd_repro.py`.
  - First live attempt failed before sending the training command because saving bottom 32MB VRAM as one bulk BAR read closed the TinyGPU connection. PSP mailbox remained clean after bridge restart.
  - The save/restore path now copies in chunks, defaulting to 1MB and tunable with `AM_PSP_MEM_TRAIN_COPY_CHUNK`.
  - Result after clean restart / clean mailbox:
    - Long training wrote `C2PMSG36=0x5fff` from `c2p=0x5fff00000`, wrote `C2PMSG35=0x100000` (`PSP_BL__DRAM_LONG_TRAIN`), and immediately observed `C2PMSG35=0x80000000`.
    - Bottom 32MB VRAM was restored and HDP flushed.
    - KDB then still failed exactly as before: `msg1_addr=0x8000100000`, `C2PMSG36=0x80001`, `C2PMSG35` dropped to `0x0`, and timed out.
    - Post-failure mailbox: `C2PMSG35_BL=0x00000000`, `C2PMSG36_ADDR=0x00080001`, `C2PMSG81_SOS=0x00000000`.
  - Conclusion: Linux-style long memory training does not fix the first KDB stall.
- Linux-like GTT primary-buffer branch:
  - Added `AM_PSP_SYSMSG1_GTT=1`, a cleaner normal-VMID0 system-memory path:
    - Allocates 2MB contiguous TinyGPU sysmem.
    - Chooses a 1MB-aligned 1MB sub-window.
    - Maps that window through the regular tinygrad VMID0 page table with `AddrSpace.SYS`, `uncached=True`, and `snooped=True`.
    - Passes the regular VMID0 mapped address to PSP, not the raw DMA address and not the hand-built high-GART aperture.
  - Result from clean mailbox:
    - Buffer raw sysmem `0x80000000`, `view_off=0`, mapped VA `0x200000000000`.
    - Pre-KDB context0 was the normal tinygrad VMID0 root page-table setup.
    - KDB readback passed.
    - KDB wrote `C2PMSG36=0x02000000` from `msg1_addr=0x200000000000`, wrote `C2PMSG35=0x80000`, then `C2PMSG35_BL` dropped to `0x0` and timed out.
    - Post-failure mailbox: `C2PMSG35_BL=0x00000000`, `C2PMSG36_ADDR=0x02000000`, `C2PMSG81_SOS=0x00000000`.
  - Conclusion: normal VMID0 GTT-style sysmem mapping with contiguous/snooped pages also does not fix KDB. This further weakens the msg1 placement/address hypothesis.

- Added `AM_PSP_STRONG_FLUSH=1` to test the stale write / weak ordering hypothesis on the default VRAM msg1 path.
  - `_prep_msg1` now optionally does repeated HDP flushes, short readbacks, mailbox reads, and a short settle before the bootloader command.
  - `_bootloader_load_component` optionally flushes after writing `C2PMSG36`, reads back `C2PMSG36` and `C2PMSG35`, and waits briefly before writing `C2PMSG35`.
  - Result from clean mailbox:
    - KDB readback passed.
    - Strong flush completed.
    - `C2PMSG36` read back as `0x80001`.
    - `C2PMSG35` read back ready as `0x80000000` immediately before the KDB command.
    - After writing `C2PMSG35=0x80000`, `C2PMSG35_BL` dropped to `0x0` and timed out.
    - Post-failure mailbox: `C2PMSG35_BL=0x00000000`, `C2PMSG36_ADDR=0x00080001`, `C2PMSG81_SOS=0x00000000`.
  - Conclusion: a simple CPU/HDP ordering issue is unlikely. The PSP is seeing the command transition, but not successfully completing the first KDB load.

- Added Linux-good PSP trace kit:
  - Script: `extra/amdpci/trace_amdgpu_psp.bt`.
  - Run on a Linux boot where `amdgpu` owns the RX 7900 XTX:

```text
sudo bpftrace extra/amdpci/trace_amdgpu_psp.bt | tee psp-linux-good.trace
```

  - Start the script before loading `amdgpu`, or unload/reload `amdgpu` while it is running.
  - Required output to compare against tinygrad:
    - `psp_hw_start` return.
    - Each `psp_v13_0_bootloader_load_component` command.
    - KDB `cmd=0x80000`.
    - Linux `fw_pri_mc_addr`.
    - Linux `c2p36 = fw_pri_mc_addr >> 20`.
    - KDB descriptor `size`.
    - `psp_v13_0_wait_for_bootloader` return after KDB.
  - This needs kernel BTF so bpftrace can dereference `struct psp_context` and `struct psp_bin_desc`.

Linux source comparison now established:

- `psp_sw_init` allocates `fw_pri_bo` as a 1MB, 1MB-aligned kernel BO. Domain is GTT by default unless SR-IOV or `debug_use_vram_fw_buf`.
- `psp_v13_0_bootloader_load_component`:
  - waits for bootloader,
  - zeroes the full 1MB `fw_pri_buf`,
  - copies the component bytes,
  - writes `C2PMSG36 = fw_pri_mc_addr >> 20`,
  - writes `C2PMSG35 = bl_cmd`,
  - waits for bootloader ready again.
- Our tinygrad implementation now matches that broad KDB sequence for VRAM, GTT-style sysmem, direct DMA, hand-built GART, and the strong-flush VRAM branch. The remaining unknown is the exact Linux BO address/domain and any lower-level setup that happens before PSP in Linux but not in tinygrad.

- Clean Linux-default-style GTT retest with full 1MB zeroing also failed.
  - Command used `AM_PSP_SYSMSG1_GTT=1 AM_PSP_ZERO_MSG1=1 AM_PSP_TRACE=1 AM_PSP_TRACE_REGS=1 AM_PSP_MSG1_READBACK=1`.
  - Clean starting mailbox: `C2PMSG35_BL=0x80000000`, `C2PMSG36_ADDR=0x00000000`.
  - Allocated 2MB contiguous TinyGPU sysmem, selected 1MB aligned view, mapped via normal VMID0 `AddrSpace.SYS`, uncached/snooped.
  - Buffer: raw sysmem `0x80000000`, `view_off=0`, mapped VA `0x200000000000`.
  - Full msg1 zeroed `1048576` bytes.
  - KDB readback passed.
  - Wrote `C2PMSG36=0x02000000`, `C2PMSG35=0x80000`.
  - `C2PMSG35_BL` dropped to `0x00000000` and timed out.
  - Post-failure mailbox: `C2PMSG35_BL=0x00000000`, `C2PMSG36_ADDR=0x02000000`, `C2PMSG81_SOS=0x00000000`.
  - Conclusion: the current GTT-style path, even with Linux-like 1MB zeroing, is ruled out. The next useful work is lower-level Linux pre-PSP setup comparison, not another `fw_pri_bo` placement variant.

- Linux lower-level pre-KDB source comparison:
  - Linux `soc21_common_hw_init` runs before PSP and does ASPM, NBIO `init_registers`, HDP remap, doorbell aperture enable, and SDMA doorbell range setup.
  - Tinygrad already approximates the relevant NBIO pieces:
    - strap clear was tested and strap was already `0x0`;
    - HDP remap was tested and did not fix KDB;
    - doorbell aperture is enabled;
    - BIFC writes were probed via RSMU and PCIE and read back `0x0`.
  - Linux `gmc_v11_0_gart_enable` runs before PSP and programs MMHUB GART aperture, system aperture, TLB/cache, system domain, identity aperture disable, VMID config, invalidation, HDP flush, fault default, and VMID0 TLB flush.
  - Tinygrad's high-GART calculation matches Linux's `AMDGPU_GART_PLACEMENT_HIGH` for GC 11 / 48-bit VM (`0x7fff00000000` for 512MB), so the GART base itself is not the discrepancy.
  - Linux `psp_sw_init` may run cold-boot two-stage memory training before allocating `fw_pri_bo`, based on PSP runtime DB boot config. If runtime DB is absent, Linux forces memory training.
  - We tested long memory training followed by default VRAM KDB, and GTT+zero without memory training. We have not yet tested the exact combined Linux-default order: long memory training, then 1MB-zeroed GTT KDB.

Next clean-reset experiment:

```text
PYTHONUNBUFFERED=1 REMOTE_TIMEOUT=30 REMOTE_RPC_TIMEOUT=60 REMOTE=127.0.0.1:6667 DEV=PCI+AMD AMD_REMOTE_ALLOC_CAP_MB=2 AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c \
AM_PSP_MEM_TRAIN=long AM_PSP_SYSMSG1_GTT=1 AM_PSP_ZERO_MSG1=1 AM_PSP_TRACE=1 AM_PSP_TRACE_REGS=1 AM_PSP_MSG1_READBACK=1 \
/Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python -c 'from tinygrad import Tensor; print((Tensor([1,2,3])+1).numpy().tolist())'
```

Combined Linux-default-order experiment result:

- Ran from clean mailbox with `AM_PSP_MEM_TRAIN=long AM_PSP_SYSMSG1_GTT=1 AM_PSP_ZERO_MSG1=1 AM_PSP_TRACE=1 AM_PSP_TRACE_REGS=1 AM_PSP_MSG1_READBACK=1`.
- GTT buffer: raw sysmem `0x80000000`, `view_off=0`, mapped VA `0x200000000000`.
- Long memory training completed:
  - saved bottom VRAM bytes `0x2000000`;
  - wrote `C2PMSG36=0x5fff`, `C2PMSG35=0x100000`;
  - observed `C2PMSG35=0x80000000`;
  - restored bottom VRAM bytes `0x2000000`.
- KDB then used zeroed GTT msg1:
  - full msg1 zeroed `1048576` bytes;
  - KDB readback passed;
  - wrote `C2PMSG36=0x02000000`, `C2PMSG35=0x80000`;
  - `C2PMSG35_BL` dropped to `0x00000000` and timed out.
- Post-failure mailbox: `C2PMSG35_BL=0x00000000`, `C2PMSG36_ADDR=0x02000000`, `C2PMSG81_SOS=0x00000000`.
- Conclusion: the final source-backed msg1/training combination is ruled out. The KDB failure is now very unlikely to be firmware parsing, msg1 placement, zeroing, memory training, or CPU/HDP ordering.

- Added read-oriented pre-KDB snapshot stage:
  - Stage: `extra/remote/amd_repro.py --stage psp-pre-kdb-snapshot`.
  - It avoids AMDDevice boot and does not send PSP bootloader commands.
  - It reads PCI config/capability list, BAR info, PSP mailboxes, NBIO direct regs, NBIO RSMU/PCIE indirect regs, and MMHUB 3.0.0 VMID0/system-aperture/protection-fault registers.
  - NBIO indirect reads necessarily write the NBIO index register before reading the data register; no PSP/GMC/NBIO state-programming writes are performed.
  - It attempts to report PSP runtime DB top-of-VRAM visibility, but the runtime DB bytes are only read if that offset is visible through BAR0. On this macOS/TinyGPU setup, top-of-VRAM is likely outside the visible BAR0 window.

Read-only snapshot command after clean GPU restart:

```text
REMOTE_TIMEOUT=5 REMOTE_RPC_TIMEOUT=10 \
/Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python /Users/julianabeleda/env/tinygrad-arkey/extra/remote/amd_repro.py 127.0.0.1:6667 --stage psp-pre-kdb-snapshot
```

Clean read-only snapshot result:

- Ran from clean mailbox and did not wedge PSP. Post-snapshot mailbox remained:
  - `C2PMSG35_BL=0x80000000`
  - `C2PMSG36_ADDR=0x00000000`
  - `C2PMSG81_SOS=0x00000000`
- PCI config:
  - vendor/device `0x744c1002`
  - command `0x0007`
  - status `0x0010`
  - class/rev `0x030000c8`
  - BAR0 size `0x10000000`
  - BAR2 size `0x200000`
  - BAR5 size `0x100000`
- PSP/NBIO:
  - PSP ready and clean.
  - `RCC_DEV0_EPF0_RCC_DOORBELL_APER_EN=0x0000000f` in read-only state.
  - `RCC_DEV0_EPF2_STRAP2=0x0`.
  - BIFC doorbell/int-monitor indirect reads still `0x0`.
  - HDP remap direct regs read `0x0` in this raw snapshot path.
- Runtime DB:
  - VRAM reported `0x5ff000000`; visible BAR0 only `0x10000000`.
  - top-of-VRAM runtime DB offset `0x5fef00000` is not visible through BAR0, so this stage cannot read runtime DB bytes on macOS/TinyGPU.
- Caveat:
  - Raw MMHUB reads in this snapshot returned `0xffffffff`, unlike AMDDevice pre-KDB trace where the generated register accessor reads MMHUB correctly.
  - The snapshot needs a follow-up implementation that instantiates enough discovery/register-offset context to read MMHUB through the same path as AMDDevice, but still avoids `AMDev.init_hw()` / PSP bootloader commands.

- Improved snapshot now includes a discovery-only `AMDev` register view:
  - It uses `object.__new__(AMDev)`, maps BAR0/BAR2/BAR5, runs `_run_discovery()` and `_build_regs()`, and does not call `init_sw()`, `init_hw()`, or PSP bootloader methods.
  - For remote 0x744c it defaults `AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c`.
  - PSP remained clean after the improved snapshot: `C2PMSG35_BL=0x80000000`, `C2PMSG36_ADDR=0x00000000`.
  - Real pre-KDB MMHUB state from discovery-backed reads:
    - `FB_LOCATION_BASE=0x00008000`, `FB_LOCATION_TOP=0x000085ff`.
    - `AGP_BASE=0x00000000`, `AGP_BOT=0x00000000`, `AGP_TOP=0x00000000`.
    - `SYSTEM_APERTURE_LOW=0x00200000`, `SYSTEM_APERTURE_HIGH=0x00217fff`.
    - `SYSTEM_APERTURE_DEFAULT_ADDR_LSB/MSB=0x0/0x0`.
    - `MMVM_CONTEXT0_CNTL=0x01fffe00`.
    - `CONTEXT0_PAGE_TABLE_BASE=0x0`, `START=0x0`, `END=0x0`.
    - `MMVM_L2_PROTECTION_FAULT_CNTL=0x3ffffffc`, `CNTL2=0x000a0000`, `STATUS=0x00000000`.
  - Interpretation:
    - The clean macOS pre-KDB MMHUB state is not the same as tinygrad's post-GMC-init pre-KDB state. Tinygrad later programs VMID0 page table/base/start/end before KDB.
    - No pre-existing MMHUB protection fault is visible in clean state.
    - AGP is not enabled in the clean macOS state.

- Added and ran `AM_PSP_BEFORE_GMC=1` experiment:
  - `AMDev.__init__` can now initialize `soc, ih, psp, gmc, smu` instead of `soc, gmc, ih, psp, smu` when the env var is set.
  - Purpose: test whether tinygrad's MMHUB/VMID0 programming before PSP KDB is causing the first-KDB failure.
  - Ran from clean mailbox with default VRAM msg1 and `AM_PSP_TRACE_REGS=1`.
  - Pre-KDB MMHUB matched clean snapshot state:
    - AGP `0/0/0`;
    - system aperture `0x00200000-0x00217fff`;
    - context0 page-table base/start/end all `0`;
    - `MMVM_CONTEXT0_CNTL=0x01fffe00`;
    - no visible protection fault status in prior snapshot.
  - KDB still failed:
    - KDB readback passed;
    - wrote `C2PMSG36=0x80001`, `C2PMSG35=0x80000`;
    - `C2PMSG35_BL` dropped to `0x00000000` and timed out;
    - post-failure `C2PMSG36_ADDR=0x00080001`, `C2PMSG81_SOS=0x00000000`.
  - Conclusion: tinygrad's pre-KDB MMHUB/VMID0 programming is not the cause of the first KDB failure on the default VRAM path.

- Added read-only PSP runtime DB dump stage:
  - Stage: `extra/remote/amd_repro.py --stage psp-runtime-db --sizes 0x1000`.
  - It builds the same discovery-only `AMDev` register view as the snapshot, sets `AM_REMOTE_SMALL_BAR_DISCOVERY=1`, and reads a bounded window at `vram_size - PSP_RUNTIME_DB_OFFSET` using `_read_vram()`.
  - It parses:
    - runtime DB header cookie/version;
    - directory entry count;
    - entry type/offset/size;
    - boot config bitmask/features for `PSP_RUNTIME_ENTRY_TYPE_BOOT_CONFIG`;
    - SCPM status for `PSP_RUNTIME_ENTRY_TYPE_PPTABLE_ERR_STATUS`.
  - It does not call `init_sw()`, `init_hw()`, or PSP bootloader methods.
  - Parser alignment fix: runtime DB directory entry list starts at offset 8 because `entry_count` is followed by padding.
  - Clean run result:
    - PSP stayed clean after the read: `C2PMSG35_BL=0x80000000`, `C2PMSG36_ADDR=0x00000000`.
    - `vram_size=0x5ff000000`, `db_off=0x5fef00000`.
    - Header `cookie=0x0ed5`, `version=0x0100`, valid.
    - Directory `entry_count=1`.
    - Only entry: `PSP_RUNTIME_ENTRY_TYPE_PPTABLE_ERR_STATUS`, offset `0x208`, size `0x4`.
    - Entry data `00000040`, parsed `scpm_status=0x40000000`.
    - No `PSP_RUNTIME_ENTRY_TYPE_BOOT_CONFIG` entry is present.
  - Interpretation:
    - Linux's `psp_get_runtime_db_entry(...BOOT_CONFIG...)` would return false on this boot, so Linux would force-enable memory training.
    - We already tested long memory training and the combined long-training + GTT + zeroed KDB path; runtime DB does not reveal an untested boot-config path.

Runtime DB command after clean restart:

```text
REMOTE_TIMEOUT=5 REMOTE_RPC_TIMEOUT=10 \
/Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python /Users/julianabeleda/env/tinygrad-arkey/extra/remote/amd_repro.py 127.0.0.1:6667 --stage psp-runtime-db --sizes 0x1000
```

Previous clean strong-flush experiment command:

```text
PYTHONUNBUFFERED=1 REMOTE_TIMEOUT=30 REMOTE_RPC_TIMEOUT=60 REMOTE=127.0.0.1:6667 DEV=PCI+AMD AMD_REMOTE_ALLOC_CAP_MB=2 AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c \
AM_PSP_STRONG_FLUSH=1 AM_PSP_TRACE=1 AM_PSP_TRACE_REGS=1 AM_PSP_MSG1_READBACK=1 \
/Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python -c 'from tinygrad import Tensor; print((Tensor([1,2,3])+1).numpy().tolist())'
```

Previous clean PSP memory-training experiment command:

```text
PYTHONUNBUFFERED=1 REMOTE_TIMEOUT=30 REMOTE_RPC_TIMEOUT=60 REMOTE=127.0.0.1:6667 DEV=PCI+AMD AMD_REMOTE_ALLOC_CAP_MB=2 AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c \
AM_PSP_MEM_TRAIN=long AM_PSP_TRACE=1 AM_PSP_TRACE_REGS=1 AM_PSP_MSG1_READBACK=1 \
/Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python -c 'from tinygrad import Tensor; print((Tensor([1,2,3])+1).numpy().tolist())'
```

Previous clean high-GART retest command, now that all-ones mailbox reads are rejected:

```text
PYTHONUNBUFFERED=1 REMOTE_TIMEOUT=30 REMOTE_RPC_TIMEOUT=60 REMOTE=127.0.0.1:6667 DEV=PCI+AMD AMD_REMOTE_ALLOC_CAP_MB=2 AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c \
AM_PSP_SYSMSG1_GART=1 AM_PSP_TRACE=1 AM_PSP_TRACE_REGS=1 AM_PSP_MSG1_READBACK=1 \
/Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python -c 'from tinygrad import Tensor; print((Tensor([1,2,3])+1).numpy().tolist())'
```

Previous clean HDP-remap experiment command:

```text
PYTHONUNBUFFERED=1 REMOTE_TIMEOUT=30 REMOTE_RPC_TIMEOUT=60 REMOTE=127.0.0.1:6667 DEV=PCI+AMD AMD_REMOTE_ALLOC_CAP_MB=2 AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c \
AM_NBIO_REMAP_HDP=1 AM_PSP_TRACE=1 AM_PSP_TRACE_REGS=1 \
/Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python -c 'from tinygrad import Tensor; print((Tensor([1,2,3])+1).numpy().tolist())'
```

Next clean low-GART experiment command after full GPU restart:

```text
REMOTE_TIMEOUT=5 REMOTE_RPC_TIMEOUT=30 REMOTE=127.0.0.1:6667 DEV=PCI+AMD AMD_REMOTE_ALLOC_CAP_MB=2 AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c \
AM_PSP_SYSMSG1_GART=1 AM_PSP_GART_LOW=1 AM_PSP_TRACE=1 AM_PSP_TRACE_REGS=1 AM_PSP_MSG1_READBACK=1 \
/Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python -c 'from tinygrad import Tensor; print((Tensor([1,2,3])+1).numpy().tolist())'
```

Previous clean experiment command:

```text
REMOTE_TIMEOUT=5 REMOTE=127.0.0.1:6667 DEV=PCI+AMD AMD_REMOTE_ALLOC_CAP_MB=2 AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c \
AM_NBIO_CLEAR_STRAP2=1 AM_PSP_TRACE=1 AM_PSP_TRACE_REGS=1 \
/Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python -c 'from tinygrad import Tensor; print((Tensor([1,2,3])+1).numpy().tolist())'
```

Next clean GART experiment command after full GPU restart:

```text
REMOTE_TIMEOUT=5 REMOTE=127.0.0.1:6667 DEV=PCI+AMD AMD_REMOTE_ALLOC_CAP_MB=2 AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c \
AM_PSP_SYSMSG1_GART=1 AM_PSP_TRACE=1 AM_PSP_TRACE_REGS=1 AM_PSP_MSG1_READBACK=1 \
/Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python -c 'from tinygrad import Tensor; print((Tensor([1,2,3])+1).numpy().tolist())'
```

12. If tensor sanity eventually passes, run Qwen 1.7B warmup with low max tokens before larger models.

13. If the GPU drops again, stop inference work and record the exact stage:

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
