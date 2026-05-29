# AMD Linux PSP Good Trace

This captures the Linux-good PSP boot path for a Navi31/RX 7900 XTX installed
directly in a Linux PC PCIe slot. Use it to compare Linux `amdgpu` against the
Mac/TinyGPU failure where first KDB load writes `C2PMSG35=0x80000`, clears
ready, and never gets ready back.

## Preferred Capture

Use the Ubuntu checkout at:

```text
/home/ubuntu/tinygrad-arkey/tinygrad
```

Before capturing, make sure that checkout has the current trace tooling from
`JulianAbeleda/tinygrad-arkey` and preserve any local capture outputs. The trace
script must be the fixed-offset version because Ubuntu 6.8.0-117 did not expose
AMD module-private struct names to `bpftrace` even with BTF present.

Boot Linux with `amdgpu` blacklisted so the card is present but unbound:

```sh
modprobe.blacklist=amdgpu
```

Find the GPU BDF:

```sh
lspci -Dnn | grep -Ei '1002:744c|amd.*(vga|display|3d)'
```

Run the capture wrapper from this repo:

```sh
sudo extra/amdpci/capture_linux_psp_good_trace.sh --bind-bdf 0000:03:00.0
```

Replace `0000:03:00.0` with the RX 7900 XTX BDF. The script holds that device
unbound, loads `amdgpu` so PSP symbols exist, starts bpftrace, then binds the
GPU and saves the trace plus baseline files in `psp-linux-good-YYYYmmdd-HHMMSS`.

## Rebind Capture

If `amdgpu` is already loaded and the RX 7900 XTX is not driving the display:

```sh
sudo extra/amdpci/capture_linux_psp_good_trace.sh --rebind-bdf 0000:03:00.0
```

This unbinds the device, attaches bpftrace while the module symbols remain
visible, then binds the device again. Do not use this on the active display GPU.

## Ubuntu Preflight

Confirm UEFI/ReBAR state before capture:

```sh
[ -d /sys/firmware/efi ] && echo UEFI || echo Legacy
dmesg | grep -i 'Detected VRAM'
lspci -Dnn | grep -Ei '1002:744c|amd.*(vga|display|3d)'
```

Known good direction from the 2026-05-28 BIOS pass:

```text
UEFI
[drm] Detected VRAM RAM=24560M, BAR=32768M
```

If BAR falls back to `256M`, stop and fix BIOS/UEFI/ReBAR before taking a PSP
comparison trace.

If a bind/rebind attempt leaves the GPU unbound or logs an amdgpu probe failure,
stop and reboot before another capture. Do not stack a `--bind-bdf` attempt on
top of a failed `--rebind-bdf` run; the kernel/device state is no longer a clean
Linux-good baseline.

## Expected Files

- `psp-linux-good.trace`: PSP bootloader, memory-training, and selected GART
  mapping trace.
- `baseline.txt`: kernel, PCI, `amdgpu`, and BTF baseline.
- `dmesg-before.txt` / `dmesg-after.txt`: kernel log around the capture.
- `bpftrace.stderr`: bpftrace compile/attach errors, if any.
- `firmware-sha256.txt`: hashes for relevant AMD firmware blobs.
- `gpu-candidates.txt` and `lspci-*.txt`: PCI identity and topology.
- `psp-symbols-*.txt`: PSP/GART probe symbol visibility before and after setup.

## Compare Against TinyGPU Failure

Check the first `bl_load enter` line for KDB:

```text
cmd=0x80000 size=0x1d40 fw_pri_mc=... c2p36=...
```

The key comparison points are `fw_pri_mc`, `c2p36`, memory-training order, and
whether the matching `bl_load ret` / `wait_bl ret` returns `0`.

For the Linux-matching PSP primary firmware buffer, also check the selected
`gart_map` lines:

```text
gart_map enter offset=0x700000 pages=256 ... flags=... dma0=... dma_last=...
gart_map ret offset=0x700000 pages=256 ... pte0=... pte_last=...
```

Compare those PTE values and flags against the Mac/TinyGPU GART experiment
trace. A PTE mismatch should become the next Mac-side experiment; a match rules
out GART PTE flags/cacheability as the KDB blocker.

## Local Evidence

The 2026-05-27 Linux capture bundle is stored at:

```text
extra/amdpci/captures/psp-linux-good-bundle-20260527.tar.gz
```

SHA256:

```text
638e3930ac2527dce9cce2a8b2c21aab4498848fc2c2f9fbbf36ca645b476232
```

The first attempts in that bundle show why the trace script uses fixed offsets:
`bpftrace` could see kernel BTF but could not resolve AMD module-private struct
names. The fixed offsets came from `/sys/kernel/btf/amdgpu` on the Ubuntu host.
