# AMD Optimization Checklist

Use this checklist for work that changes the AMD remote path, TinyGPU bridge behavior, Q4_K inference path, or Radeon 7900 XTX benchmarking.

The goal is one active target: `tinygrad-arkey`.

## Progress

- [x] Active target is `tinygrad-arkey`.
- [x] Global root `/Users/julianabeleda/env/tinygrad` points at `tinygrad-arkey`.
- [x] AMD/ROCm/llama.cpp research note exists in `docs/`.
- [x] Runtime bridge health command exists.
- [x] Runtime bridge dirty-state gate exists.
- [x] Remote bench targets the AMD compute device path used by tinygrad.
- [x] Qwen 1.7B live inference baseline recorded on `PCI+AMD`.
- [x] LLM logs report prefill and decode remote pressure separately.
- [x] Q4_K baseline benchmark script exists.
- [x] Reproduced GPU dropout outside model inference.
- [x] Isolated first clear dropout trigger to repeated `16MB` TinyGPU `PrepareDMA` mappings.
- [x] Documented dropout investigation in `docs/amd-remote-dropout-investigation.md`.
- [x] Add remote-only AMD allocation cap below `16MB`.
- [x] Retest AMD boot with capped remote staging allocations.
- [ ] Retest Qwen 1.7B decode after allocation mitigation.
- [x] Add acknowledged remote sysmem writes and explicit invalid-handle errors.
- [x] Guard remote AMD small-BAR discovery from wedging TinyGPU.
- [x] Add RX 7900 XTX remote discovery profile to bypass unsafe small-BAR discovery.
- [x] Patch TinyGPU C server `CMD_MMIO_WRITE` to return an RPC response.
- [x] Build and ad-hoc sign patched TinyGPU app/dext for the No-SIP development path.
- [x] Disable SIP from Recovery on the Mac mini and install the patched No-SIP TinyGPU app/dext.
- [x] Retest BAR writes after patched TinyGPU app/dext activation.
- [x] Retest tensor sanity with `AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c` after GPU re-enumerates.
- [x] Reboot normally and confirm only patched TinyGPU dext processes are running.
- [ ] Investigate AMD PSP init failure: latest narrowed symptom is first KDB load dropping `C2PMSG35_BL` from ready to `0x0` and timing out with `BL not ready`.
- [x] Run `AM_PSP_SYSMSG1_GART=1` after the next clean GPU restart to test a Linux-style GART/MC address for PSP msg1.
- [x] Run `AM_PSP_SYSMSG1_GART=1 AM_PSP_GART_LOW=1` after the next clean GPU restart to test whether high GART MC placement is part of the instability.
- [x] Run `AM_NBIO_REMAP_HDP=1` after the next clean GPU restart to test Linux's pre-PSP HDP remap setup.
- [x] Probe NBIO 7.9 `BIFC_*` RSMU-vs-PCIE indirect access path.
- [x] Run `AM_PSP_MEM_TRAIN=long` after the next clean GPU restart to test Linux's pre-KDB memory-training step.
- [x] Run `AM_PSP_SYSMSG1_GTT=1` to test normal VMID0 contiguous/snooped system-memory mapping for PSP msg1.
- [ ] Retest Qwen warmup bridge dirty failure after GPU restart.
- [ ] Reduce decode-time `SYSMEM_READ`/`SYSMEM_WRITE` roundtrips.
- [ ] Add a benchmark gate for roundtrips/token regression checks.
- [ ] Prototype packed Q4_K_M fused dequant plus matvec for AMD/gfx1100.
- [ ] Compare fused path against the generic `ggml_data_to_tensor` baseline.

Latest baseline:

- Model: `Qwen3-1.7B`
- Device: `REMOTE=127.0.0.1:6667 DEV=PCI+AMD`
- Prefill: `256 tok/s`, `70.86 roundtrips/token`
- Decode: `33 tok/s`, `926.18 roundtrips/token`
- Decode pressure: `SYSMEM_READ:72499`, `SYSMEM_WRITE:24569`

Dropout investigation:

- Doc: `docs/amd-remote-dropout-investigation.md`
- Earlier trigger: repeated `16MB` TinyGPU `PrepareDMA` mappings.
- Prior narrowed failure node: mapped BAR reads succeeded, but mapped BAR writes wedged the TinyGPU bridge because the TinyGPU C server `CMD_MMIO_WRITE` path skipped the RPC response after successful writes.
- Latest live validation: patched `org.tinygrad.arkey.tinygpu.driver2` is installed, approved, and active on the No-SIP development path. 4-byte BAR writes to BAR0 and BAR2 now pass, and the RX 7900 XTX remains visible afterward. This validates the `CMD_MMIO_WRITE` response fix on live hardware.
- Current narrowed failure node: tinygrad AMD boot reaches PSP initialization. The most stable repro is first KDB bootloader load, where a clean ready mailbox accepts `PSP_BL__LOAD_KEY_DATABASE`, then `C2PMSG35_BL` drops from `0x80000000` to `0x0` and never returns ready. The older post-reboot `Connection closed` symptom remains history, but it is not the latest narrowed failure.
- Narrower PSP trace result: on a clean ready mailbox, the first bootloader component load (`PSP_FW_TYPE_PSP_KDB`, `PSP_BL__LOAD_KEY_DATABASE` / `0x80000`) writes C2PMSG36=`0x80001` and C2PMSG35=`0x80000`; C2PMSG35 then becomes `0x0` and never returns the ready bit. Current direct mailbox state is C2PMSG33=`0x80000000`, C2PMSG35=`0x00000000`, C2PMSG36=`0x00080001`, C2PMSG81=`0x00000000`, C2PMSG92=`0x00000000`.
- Linux amdgpu PSP v13 clears the full 1MB PSP primary firmware buffer before copying each bootloader component. tinygrad previously only overwrote the component payload plus padding. `AM_PSP_ZERO_MSG1=1` was added to test full-buffer zeroing on the next clean hardware-reset attempt.
- After a clean reboot/reset, `AM_PSP_ZERO_MSG1=1` did not recover KDB load; C2PMSG35 still dropped from `0x80000000` to `0x0` after writing `PSP_BL__LOAD_KEY_DATABASE`.
- `AM_PSP_SYSMSG1=1` was added to test the remote TinyGPU path with PSP `msg1` in system memory. The direct USB path already used sysmem, but the `REMOTE=...` client path used VRAM because `devfmt` is `remote:...`. The new sysmem path maps all returned pages instead of assuming a contiguous physical segment.
- After another clean reset, `AM_PSP_SYSMSG1=1` also failed at the first KDB bootloader load. It allocated sysmem msg1 at `0x200000000000` with 256 pages, wrote C2PMSG36=`0x02000000`, wrote C2PMSG35=`0x80000`, then C2PMSG35 stayed `0x0` until `BL not ready`. Bridge health stayed healthy and the GPU stayed visible.
- Linux source comparison found an MP0 13.0.10-specific PSP fatal-recovery quirk: read C2PMSG67, write it back plus `0x10`, then sleep 1000ms to trigger a PSP DRAM read and unhalt PSP during an MP1-triggered sync flood. `AM_PSP_FATAL_QUIRK=1` was added to test this before the first KDB bootloader load. `amd_repro.py --stage psp-status` now also reports C2PMSG64, C2PMSG67, C2PMSG90, and C2PMSG115.
- After a clean reset, `AM_PSP_FATAL_QUIRK=1` also failed at the first KDB bootloader load. C2PMSG67 changed from `0x0` to `0x10`, but KDB still wrote C2PMSG36=`0x80001`, wrote C2PMSG35=`0x80000`, then C2PMSG35 stayed `0x0` until `BL not ready`. Bridge health stayed healthy and the GPU stayed visible.
- `AM_PRE_PSP_MODE1_RESET=1` was added as the next reset/init experiment. On a non-partial boot it probes SMU liveness, clears PCI bus master, issues `smu.mode1_reset()`, restores bus master, then continues the normal SOC/GMC/IH/PSP/SMU initialization path. This mirrors tinygrad's existing malformed-state reset sequence but makes it testable before first KDB load on the clean remote path.
- After a clean reset, `AM_PRE_PSP_MODE1_RESET=1` was a no-op because `self.smu.is_smu_alive()` returned `False` before PSP init. KDB still dropped C2PMSG35 from `0x80000000` to `0x0` and timed out with `BL not ready`; bridge health stayed healthy and the GPU stayed visible.
- Memory mapping status: tinygrad is already using GPU page-table-backed mappings through `AMMemoryManager` and `AMPageTableEntry`, not a loose lookup table. VRAM mappings use `AddrSpace.PHYS` and `paddr2xgmi()`. The remote `AM_PSP_SYSMSG1=1` experiment maps all returned DriverKit/sysmem pages as `AddrSpace.SYS`. A hashmap/database would only be debug observability for VA-to-physical mappings; it would not change PSP consumption of `C2PMSG36_ADDR`.
- TinyGPU reset path is a real DriverKit PCI reset: `RemoteCmd.RESET` reaches TinyGPU `CMD_RESET`, which calls `TinyGPUDriver::ResetDevice()`. That calls `IOPCIDevice::Reset(kIOPCIDeviceResetTypeFunctionReset)` and falls back to `kIOPCIDeviceResetTypeHotReset`. Added `amd_repro.py --stage reset` so reset can be run as a standalone client before reconnecting for `psp-status`.
- Standalone TinyGPU reset did not recover PSP readiness from the KDB-stuck state. `amd_repro.py --stage reset` returned success in ~273ms, but a fresh-client `amd_repro.py --stage psp-status` still reported C2PMSG35=`0x00000000` and C2PMSG36=`0x00080001`.
- `AM_PSP_TRACE=1` traces PSP component loads and waits. `AM_PSP_MSG1_READBACK=1` verified KDB bytes were readable back from VRAM but did not recover boot. `amd_repro.py --stage psp-status` reads PSP mailbox registers without instantiating `AMDDevice`.
- Old signed dext state: after the normal reboot, `org.tinygrad.tinygpu.driver2` is still listed by `systemextensionsctl` as `activated disabled`, but `pgrep -afil tinygpu` only shows patched `org.tinygrad.arkey.tinygpu.driver2` processes.
- Root-cause candidate resolved for BAR writes: TinyGPU `CMD_MMIO_WRITE` response omission. Source is patched and upstream PR #16333 is open/mergeable. Local Xcode 26.5 builds the patched app after using local DerivedData/module-cache paths.
- Firmware framing: ASM2464PD is an active firmware-mediated bridge with an internal 8051 CPU, Program ROM, Program RAM, and XDATA. Treat this as relevant to the hypothesis, not as proof that the 8051 firmware is the root cause.
- Passed before earlier DMA trigger: BAR/MMIO checks, repeated `16KB` sysmem mappings, repeated `2MB` sysmem mappings.
- Passed in latest isolation: 4-byte reads from BAR0, BAR2, and BAR5.
- Current mitigation: remote-only AMD setup allocations default to a `2MB` cap through `AMD_REMOTE_ALLOC_CAP_MB`.
- Escape hatch: `AMD_REMOTE_ALLOC_CAP_MB=0` restores the previous `16MB` setup allocation behavior.
- Validation: capped AMD boot reached `gfx1100`, reported `has_sdma=True`, completed 16KB/2MB host allocations, and synchronized successfully.
- Qwen status: Qwen 1.7B warmup remains paused. The next blocker is AMD PSP initialization, not the previous bridge dirty `list index out of range` or BAR-write timeout.
- Bridge protocol update: remote `SYSMEM_WRITE` now returns an RPC response, and sysmem reads/writes report explicit invalid handle errors instead of raw `list index out of range`.
- Latest hardware state: the GPU dropped again at `2026-05-21 23:12:30`; Qwen retest is pending after physical restart.
- Small-BAR discovery guard: remote AMD now fails fast before the indirect VRAM MMIO path unless `AM_REMOTE_SMALL_BAR_DISCOVERY=1` is set.
- Discovery profile: `AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c` bypasses the unsafe indirect VRAM discovery read for RX 7900 XTX. Local syntax validation passes, but live tensor validation is blocked while macOS/TinyGPU reports zero AMD devices.
- Latest local state: the RX 7900 XTX is visible as external PCIe GPU `0x744c`; SIP is disabled; patched `org.tinygrad.arkey.tinygpu.driver2` is active and enabled; only patched TinyGPU dext processes are running. Remote health and BAR read/write checks pass after reboot. Tensor sanity fails during PSP init. The bridge can remain healthy on the `BL not ready` path, but the PSP mailbox is now stuck with C2PMSG35=`0x0`; TinyGPU `RESET` RPC did not restore readiness.
- Added `amd_repro.py --stage psp-fw` for non-hardware PSP SOS parsing. `psp_13_0_10_sos.bin` is header v2.0 with `ucode_array_offset=0x100` and 9 components. KDB is `PSP_FW_TYPE_PSP_KDB` at blob offset `0x100`, size `0x1d40` / `7488`, sha256 `f19238a9d2401673ddbc2d7a5eba1640afd2a524b63f83d1daa91ede38226632`. This matches Linux's packed PSP SOS v2 descriptor model of `ucode_array_offset + desc.offset_bytes`.
- After a fresh AMD GPU restart on 2026-05-23, clean `psp-status` showed C2PMSG35=`0x80000000` and C2PMSG36=`0x00000000`; remote health was healthy. Smallest tensor sanity with `AM_PSP_TRACE=1` reproduced the same KDB failure: KDB bytes=`7488`, C2PMSG36=`0x80001` from `msg1_addr=0x8000100000`, C2PMSG35=`0x80000`, then C2PMSG35 stayed `0x0` until `BL not ready`. Post-failure bridge health remained healthy.
- Linux PSP comparison found that the older `AM_PSP_SYSMSG1=1` experiment was not Linux-like: Linux passes a PSP primary buffer MC/GART address, while that tinygrad path passed a GPUVM VA. Added `amd_repro.py --stage psp-sysmem-probe`; default 1MB sysmem returned one span at `0x80104000` but not 1MB aligned, while `--contiguous` returned one 1MB-aligned span at `0x80000000` (`C2PMSG36=0x800`).
- Added `AM_PSP_SYSMSG1_DMA=1` as a clean-reset experiment. First attempt showed a 1MB contiguous DMA allocation was not necessarily 1MB-aligned, so the implementation now allocates 2MB and uses a 1MB-aligned sub-window. The retest reached PSP with `msg1_addr=0x81a00000`, wrote C2PMSG36=`0x81a`, wrote C2PMSG35=`0x80000`, then C2PMSG35 stayed `0x0` until `BL not ready`. Post-failure bridge health remained healthy.
- Next gate: avoid repeating the same clean-reset KDB test without a changed input. Firmware parsing is correct, zeroing did not help, GPUVM sysmem VA did not help, normal VMID0 contiguous/snooped GTT-style sysmem did not help, direct aligned TinyGPU DMA address did not help, fatal quirk did not help, SMU-dependent pre-PSP mode1 was a no-op, and `AM_NBIO_CLEAR_STRAP2=1` did not help because `regRCC_DEV0_EPF2_STRAP2` was already `0x0`. `AM_NBIO_REMAP_HDP=1` correctly programmed Linux's non-SRIOV HDP remap setup (`0x7f000` / `0x7f004`) before PSP, but still failed at first KDB with `C2PMSG35_BL=0x0`. `AM_PSP_STRONG_FLUSH=1` added repeated HDP flushes, msg1 readback, mailbox readback, and short delays around the default VRAM msg1 handoff; it still failed at first KDB after confirming `C2PMSG36=0x80001` and `C2PMSG35=0x80000000` immediately before the command. `AM_PSP_TRACE_REGS=1` captured the pre-KDB state. `_wait_for_bootloader()` now rejects all-ones MMIO reads as invalid instead of treating the ready bit as success. Under that guard, both HDP remap alone and high-GART msg1 fail at first KDB. The earlier apparent KDB/SPL progress was likely contaminated by `0xffffffff` MMIO reads. NBIO `BIFC_*` RSMU and PCIE write probes both read back `0x0`, so the visible discrepancy is not a simple indirect access-path bug. `AM_PSP_MEM_TRAIN=long` completed and returned ready, but KDB still failed. Qwen remains paused.
- Latest Linux-default-style GTT retest with full 1MB `AM_PSP_ZERO_MSG1=1` also failed from a clean mailbox. It wrote `C2PMSG36=0x02000000` and `C2PMSG35=0x80000`, then timed out with `C2PMSG35_BL=0x00000000`, `C2PMSG81_SOS=0x00000000`. Treat current `AM_PSP_SYSMSG1_GTT=1` as ruled out unless a new Linux trace shows a materially different GTT address/domain setup.
- Combined Linux cold-boot memory training plus Linux-default GTT placement failed from a clean mailbox. `AM_PSP_MEM_TRAIN=long` completed and returned ready, then `AM_PSP_SYSMSG1_GTT=1 AM_PSP_ZERO_MSG1=1` still failed first KDB with `C2PMSG35_BL=0x00000000`, `C2PMSG36_ADDR=0x02000000`, `C2PMSG81_SOS=0x00000000`. The source-backed msg1/training branch is closed.
- Read-only macOS fallback added: `amd_repro.py --stage psp-pre-kdb-snapshot` dumps PCI config/caps, BAR info, PSP mailboxes, NBIO direct/indirect state, and MMHUB 3.0.0 pre-KDB VMID0/system-aperture/protection-fault state without AMDDevice boot or PSP commands. Run it after a clean restart before any state-changing PSP test.
- Clean snapshot run did not wedge PSP and confirmed PCI/PSP/NBIO basics. Runtime DB is not visible through BAR0 (`off=0x5fef00000`, visible BAR0 `0x10000000`). Discovery-backed MMHUB snapshot now works without AMDDevice init: clean state has `CONTEXT0_PAGE_TABLE_BASE/START/END=0`, `MMVM_CONTEXT0_CNTL=0x01fffe00`, `MMVM_L2_PROTECTION_FAULT_STATUS=0`, `AGP_BASE/BOT/TOP=0`, and system aperture `0x00200000-0x00217fff`.
- `AM_PSP_BEFORE_GMC=1` ran PSP before GMC/MMHUB hardware programming, so KDB used the clean MMHUB state above. It still failed first KDB on default VRAM msg1 (`C2PMSG36=0x80001`, `C2PMSG35_BL=0x0`). This rules out tinygrad's pre-KDB VMID0/MMHUB programming as the cause for the VRAM path.
- Read-only runtime DB stage ran cleanly. Runtime DB exists (`cookie=0x0ed5`, version `0x0100`) with one entry: `PSP_RUNTIME_ENTRY_TYPE_PPTABLE_ERR_STATUS`, data `00000040`, parsed `scpm_status=0x40000000`. There is no `PSP_RUNTIME_ENTRY_TYPE_BOOT_CONFIG`, so Linux would force memory training on this boot. We already tested long memory training and long-training + GTT + zeroed KDB.
- Linux-good trace gate: `extra/amdpci/trace_amdgpu_psp.bt` is now the next useful data source. Run it under Linux with `sudo bpftrace extra/amdpci/trace_amdgpu_psp.bt | tee psp-linux-good.trace` before loading or while reloading `amdgpu`. Compare KDB `cmd=0x80000`, `fw_pri_mc_addr`, `c2p36`, descriptor size, and `wait_bl ret` against tinygrad's pre-KDB trace. Do not run another clean-reset KDB test until the next input is either a Linux-good trace difference or a newly identified Linux precondition.
- Causality boundary: a physical replug likely resets ASM2464PD firmware state, USB4 tunnel state, TinyGPU DriverKit state, PCIe link training, and the GPU endpoint. Do not claim the 8051 firmware is isolated as the stuck component without a narrower test.

## Before Editing

- Confirm checkout is `/Users/julianabeleda/env/tinygrad-arkey`.
- Confirm global root `/Users/julianabeleda/env/tinygrad` points at `tinygrad-arkey`.
- Check `git status --short` and do not overwrite unrelated user changes.
- Read the current plan or audit note before changing code.
- Identify the owning prefix from `tinygrad-coding-overrides.md`.

## Plan Gate

- State which slice the change belongs to:
  - Runtime stability and observability.
  - Remote roundtrip and residency reduction.
  - Q4_K baseline measurement.
  - Fused Q4_K kernel path.
- Name the exact delta from what already exists.
- Prefer measuring the current path before replacing it.
- Keep docs, runtime changes, benchmark scripts, and kernel changes in separate commits.

## Runtime Bridge Gate

- If `RemoteCmd` changes, restart the live remote bridge before testing.
- Keep `PING`, `PROBE`, and `HEALTH` available even when the bridge is dirty.
- After a device-level runtime failure, fail closed until `RESET` succeeds.
- Bench output must show:
  - `bridge health: healthy`
  - `health: healthy`
  - per-command stats
  - no failed RPCs
- If health is dirty, do not continue inference testing until the bridge is reset or restarted.

## Benchmark Gate

- Record the exact model, quantization, device, and command.
- Separate prefill and decode when possible.
- Track:
  - tokens/sec
  - roundtrips
  - roundtrips/token
  - host/device bytes
  - per-command latency
- For Q4_K work, keep a baseline from the generic `ggml_data_to_tensor` path before introducing a fused path.
- Do not compare against ROCm or llama.cpp without recording model, quant, batch, context length, and prompt length.

## Verification Gate

- Run syntax checks for changed Python files:

```text
python3 -m py_compile <changed-python-files>
```

- Run the remote health bench after bridge changes:

```text
REMOTE_TIMEOUT=3 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python /Users/julianabeleda/env/tinygrad-arkey/extra/remote/bench.py 127.0.0.1:6667 --skip-tensor
```

- For inference changes, run a small model first before moving up:
  - Qwen 1.7B or Qwen2.5-Coder 1.5B.
  - Then larger Qwen models only after the bridge stays healthy.
- If the GPU drops, record whether it happened during:
  - probe/open
  - BAR map
  - sysmem allocation
  - prefill
  - decode
  - model load

## Rebuild And Live Target Gate

- Restart the bridge from the global root path:

```text
cd /Users/julianabeleda/env/tinygrad
DEBUG=1 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python /Users/julianabeleda/env/tinygrad/extra/remote/serve.py 6667
```

- Confirm the running process uses `/Users/julianabeleda/env/tinygrad/extra/remote/serve.py`.
- Confirm the Python executable is from `/Users/julianabeleda/env/tinygrad-arkey/.venv`.
- Do not leave an old `tinygrad` checkout or upstream repo as the active server target.

## Commit Gate

- Use exactly one prefix from `tinygrad-coding-overrides.md`.
- Use `[docs]` for structure-only changes.
- Use `[runtime]` for bridge, AMD runtime, or device behavior changes.
- Use `[examples]` for standalone benchmark scripts in `extra/`.
- Keep the commit small and self-contained.
- Mention skipped verification explicitly if hardware is unavailable.

## Done Gate

- `git status --short` is clean.
- The branch is pushed to `JulianAbeleda/tinygrad-arkey` if the change should exist on GitHub.
- The live bridge has been restarted if runtime protocol or server behavior changed.
- The latest health bench result is recorded in the final handoff.
