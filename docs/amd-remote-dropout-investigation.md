# AMD remote dropout investigation

This note tracks the Radeon RX 7900 XTX dropout seen while testing the tinygrad remote AMD path through TinyGPU/DriverKit.

## Hardware path

```text
Mac mini
  -> USB4 / Thunderbolt
  -> ADT-Link UT4G / TinyGPU DriverKit bridge
  -> AMD Radeon RX 7900 XTX
  -> tinygrad-arkey remote PCI runtime
```

The active tinygrad path is:

```text
tinygrad-arkey
  -> tinygrad AMD runtime
  -> RemotePCIDevice RPC
  -> extra/remote/serve.py
  -> TinyGPU DriverKit / UT4G
  -> RX 7900 XTX
```

## Symptom

The GPU can disappear from the macOS PCI tree while the USB4/UT4G layer remains visible or later re-enumerates. When this happens, tinygrad requests fail above the hardware layer as empty replies, dirty bridge state, or server-side device errors.

The relevant logs show macOS marking the downstream PCIe device tree dead:

```text
IOPCIFamily ... marking child ... 5:0:0(0x1002:0x744c) dead
DK: tinygpu-... force close
Thunderbolt PCI deactivating upstream/downstream path
```

The drop is not limited to the GPU function. The dead tree can include:

- GPU function `1002:744c`
- Audio/function device `1002:ab30`
- Other AMD functions such as `7446` and `7444`
- AMD bridge devices `1002:1478` / `1002:1479`
- ASMedia/UT4G bridge `1b21:2461`

This points to a USB4/Thunderbolt/PCIe/DriverKit link failure rather than a normal Python exception or LLM server error.

## Local evidence

The current repro tool is:

```text
/Users/julianabeleda/env/tinygrad-arkey/.venv/bin/python \
  /Users/julianabeleda/env/tinygrad-arkey/extra/remote/amd_repro.py \
  127.0.0.1:6667 --stage <stage>
```

Stages:

- `bars`: probe and BAR lifecycle checks.
- `remote-sysmem`: direct remote sysmem allocation/read/write checks.
- `amd-boot`: tinygrad AMD runtime initialization checks.
- `all`: run every stage.

Observed sequence after a fresh GPU restart:

- BAR/MMIO checks did not reproduce the drop.
- `remote-sysmem` with `16KB` allocations repeated 8 times passed.
- `remote-sysmem` with `2MB` allocations repeated 8 times passed.
- `remote-sysmem` with `16MB` allocations repeated 8 times returned OK at the RPC level, then triggered a delayed PCIe tree drop roughly 15 seconds later.

Representative local timing:

```text
22:49:32.032 PrepareDMA size=16777216 segs=1
22:49:33.103 PrepareDMA size=16777216 segs=1
22:49:48.838 IOPCIFamily ... marking child ... 0x1002:0x744c dead
22:49:48.841 DK: tinygpu-... force close
22:49:53.091 Found type 0 device ... 0x1002:0x744c
22:49:53.134 tinygpu: opened device ven=0x1002 dev=0x744c
```

The strongest current conclusion is that repeated `16MB` TinyGPU `PrepareDMA` mappings are the first clear local trigger. They can return success and still destabilize the PCIe chain shortly afterward.

## Current thesis

The dropout is probably caused by large remote DMA/staging mappings stressing the TinyGPU/DriverKit/USB4 path, not by Qwen generation directly.

Model load and inference can still expose the problem because the AMD runtime allocates large host-visible buffers and staging regions during boot, queue setup, or copy paths. The LLM workload is therefore a trigger for a lower-level transport failure.

## Mitigation plan

Short-term mitigation:

- Cap remote AMD host/staging allocations below `16MB`.
- Default to `2MB`, because repeated `2MB` mappings passed the repro.
- Preserve local KFD/PCI behavior; only apply the cap to `RemotePCIDevice` / TinyGPU remote paths.
- Retest staged repro before retesting Qwen.

Runtime knob:

- `AMD_REMOTE_ALLOC_CAP_MB=2` is the default for remote PCI AMD setup allocations.
- `AMD_REMOTE_ALLOC_CAP_MB=0` disables the cap and restores the previous `16MB` setup allocation behavior.
- Higher values such as `4` or `8` can be used for A/B testing if `2MB` is too small.

Likely code areas to inspect:

- `tinygrad/runtime/ops_amd.py`
- `tinygrad/runtime/support/hcq.py`
- `tinygrad/runtime/support/system.py`

Known allocation suspects:

- AMD kernargs buffer sizing.
- AMD PM4/AQL indirect buffer sizing.
- AMD compute queue ring sizing.
- AMD SDMA queue ring sizing.

Validation order:

1. Restart the bridge from `/Users/julianabeleda/env/tinygrad`.
2. Run `amd_repro.py --stage remote-sysmem` at `2MB`, `4MB`, `8MB`, and `16MB`.
3. Run `amd_repro.py --stage amd-boot`.
4. Watch bridge logs and confirm `PrepareDMA size=16777216` no longer appears during AMD boot with the default cap.
5. Run a small tensor sanity test on `REMOTE=127.0.0.1:6667 DEV=PCI+AMD`.
6. Retest Qwen 1.7B before moving to larger models.

## Online verification

Online evidence supports the broader failure class but does not confirm this exact TinyGPU trigger.

Confirmed broadly:

- USB4/Thunderbolt eGPU setups can disconnect or re-enumerate under load.
- DIY eGPU stability depends on power, cable quality, link training, OS support, firmware, and driver behavior.
- An enclosure or bridge layer can remain visible while the GPU function disappears.

Not confirmed publicly:

- A specific TinyGPU public bug where repeated `PrepareDMA size=16777216` mappings drop an RX 7900 XTX through UT4G.

So the current claim should be stated carefully:

> Public reports match the general USB4/eGPU dropout-under-load pattern. Our local logs are the stronger evidence for the specific `16MB` TinyGPU `PrepareDMA` trigger.

## Related docs

- `docs/amd-rocm-llamacpp-research.md`
- `structure/Development/amd-optimization-checklist.md`
