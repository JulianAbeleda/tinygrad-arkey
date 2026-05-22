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
- [ ] Disable SIP from Recovery on the Mac mini and install the patched No-SIP TinyGPU app/dext.
- [ ] Retest BAR writes after patched TinyGPU app/dext activation.
- [ ] Retest tensor sanity with `AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c` after GPU re-enumerates.
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
- Current narrowed failure node: mapped BAR reads succeed, but mapped BAR writes can wedge the TinyGPU bridge. A 4-byte BAR0 write and a 4-byte BAR2 write both timed out while the RX 7900 XTX stayed enumerated.
- Root-cause candidate found: the TinyGPU C server `CMD_MMIO_WRITE` path skipped the RPC response after successful writes. Source is patched and upstream PR #16333 is open/mergeable. Local Xcode 26.5 builds the patched app after using local DerivedData/module-cache paths.
- Firmware framing: ASM2464PD is an active firmware-mediated bridge with an internal 8051 CPU, Program ROM, Program RAM, and XDATA. Treat this as relevant to the hypothesis, not as proof that the 8051 firmware is the root cause.
- Passed before earlier DMA trigger: BAR/MMIO checks, repeated `16KB` sysmem mappings, repeated `2MB` sysmem mappings.
- Passed in latest isolation: 4-byte reads from BAR0, BAR2, and BAR5.
- Current mitigation: remote-only AMD setup allocations default to a `2MB` cap through `AMD_REMOTE_ALLOC_CAP_MB`.
- Escape hatch: `AMD_REMOTE_ALLOC_CAP_MB=0` restores the previous `16MB` setup allocation behavior.
- Validation: capped AMD boot reached `gfx1100`, reported `has_sdma=True`, completed 16KB/2MB host allocations, and synchronized successfully.
- Qwen status: Qwen 1.7B warmup still fails, but the latest failure is a bridge dirty `list index out of range` without a macOS PCIe dead-device log.
- Bridge protocol update: remote `SYSMEM_WRITE` now returns an RPC response, and sysmem reads/writes report explicit invalid handle errors instead of raw `list index out of range`.
- Latest hardware state: the GPU dropped again at `2026-05-21 23:12:30`; Qwen retest is pending after physical restart.
- Small-BAR discovery guard: remote AMD now fails fast before the indirect VRAM MMIO path unless `AM_REMOTE_SMALL_BAR_DISCOVERY=1` is set.
- Discovery profile: `AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c` bypasses the unsafe indirect VRAM discovery read for RX 7900 XTX. Local syntax validation passes, but live tensor validation is blocked while macOS/TinyGPU reports zero AMD devices.
- Latest local state before reboot: the RX 7900 XTX is visible again as external PCIe GPU `0x744c`, but the active system extension is still the old signed `org.tinygrad.tinygpu.driver2`. The patched local app/dext uses `org.tinygrad.arkey.tinygpu.driver2` and is blocked by SIP/AMFI while ad-hoc signed.
- Reboot gate: on the Apple Silicon Mac mini, shut down, hold the physical power button until startup options load, choose Options, open Utilities > Terminal, run `csrutil disable`, then reboot normally. After that, run `./install_nosip.sh` from `extra/usbgpu/tbgpu/installer`.
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
