# Session Handoff

## Stop Point: 2026-05-22 AMD Remote Runtime Work

This is the current handoff for the `tinygrad-arkey` AMD/TinyGPU work.

Use this file at the start of the next session before changing code or restarting Qwen.

## Active Target

- Local repo: `/Users/julianabeleda/env/tinygrad-arkey`
- Global root: `/Users/julianabeleda/env/tinygrad`
- Global root state: symlink to `/Users/julianabeleda/env/tinygrad-arkey`
- GitHub fork: `https://github.com/JulianAbeleda/tinygrad-arkey`
- Branch: `master`
- Latest pushed commit: `0c6301674 [docs] document remote AMD discovery profile`

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

The main blocker is still hardware/bridge stability. The RX 7900 XTX can disappear from the macOS PCIe tree under the TinyGPU/USB4 path. When it disappears, tinygrad sees zero AMD devices or a dirty bridge, but the root cause is below tinygrad.

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

## Latest Commits

```text
0c6301674 [docs] document remote AMD discovery profile
b18b8ffed [runtime] add remote AMD discovery profile
645851d1f [docs] track remote small-BAR discovery guard
b28f00ddf [runtime] guard remote AMD small-BAR discovery
8be401fd5 [runtime] acknowledge remote MMIO writes
6ad0dc357 [docs] track sysmem write acknowledgement
10ae6ca02 [runtime] acknowledge remote sysmem writes
11dad8f7d [docs] record AMD cap validation
b0b602e95 [runtime] tolerate remote socket buffer limits
a3ee33362 [docs] update AMD dropout mitigation status
0ccdfb0ee [runtime] cap remote AMD setup allocations
aa7246081 [docs] document AMD remote dropout investigation
```

## Latest Validation

Passed earlier:

- Capped AMD boot reached `gfx1100`.
- `has_sdma=True`.
- 16KB and 2MB host allocations completed.
- Synchronization completed.
- No `PrepareDMA size=16777216` during capped AMD boot.
- Tiny tensor sanity returned `[2, 3, 4]`.

Blocked now:

- Latest remote probe found zero AMD devices.
- `system_profiler SPDisplaysDataType` reported only the Apple M4 GPU.
- macOS logs showed repeated ACIO Gen2/3 link errors, then the AMD/UT4G PCIe tree was marked dead at `2026-05-22 00:29:01`.
- TinyGPU was force-closed after the PCIe tree died.

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

1. Confirm the RX 7900 XTX is visible again:

```text
system_profiler SPDisplaysDataType
```

2. Restart the bridge from the global root:

```text
cd /Users/julianabeleda/env/tinygrad
DEBUG=1 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python /Users/julianabeleda/env/tinygrad/extra/remote/serve.py 6667
```

3. Run the remote probe/health check:

```text
REMOTE_TIMEOUT=3 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python /Users/julianabeleda/env/tinygrad-arkey/extra/remote/bench.py 127.0.0.1:6667 --skip-tensor
```

4. If isolating the current bridge-write failure, run BAR reads before any writes:

```text
REMOTE_TIMEOUT=5 REMOTE_RPC_TIMEOUT=10 /Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python \
/Users/julianabeleda/env/tinygrad-arkey/extra/remote/amd_repro.py 127.0.0.1:6667 --stage bar-read --bars 0,2,5 --sizes 4 --offsets 0 --repeat 1
```

Known result: 4-byte reads from BAR0, BAR2, and BAR5 pass. 4-byte writes to BAR0 and BAR2 time out while the GPU remains visible. Instrumentation showed the Python bridge blocked at `MMIO_WRITE store-start` while waiting for the nested TinyGPU app RPC. The TinyGPU C server source is patched to send an acknowledgement for `CMD_MMIO_WRITE`, but validation requires rebuilding/reinstalling `TinyGPU.app`.

5. Run the smallest tensor sanity with the discovery profile:

```text
REMOTE_TIMEOUT=5 REMOTE=127.0.0.1:6667 DEV=PCI+AMD AMD_REMOTE_ALLOC_CAP_MB=2 AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c \
/Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python -c 'from tinygrad import Tensor; print((Tensor([1,2,3])+1).numpy().tolist())'
```

6. If tensor sanity passes, run Qwen 1.7B warmup with low max tokens before larger models.

7. If the GPU drops again, stop inference work and record the exact stage:

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
