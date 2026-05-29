# AMD Linux PSP Good Trace

This captures the Linux-good PSP boot path for a Navi31/RX 7900 XTX installed
directly in a Linux PC PCIe slot. Use it to compare Linux `amdgpu` against the
Mac/TinyGPU failure where first KDB load writes `C2PMSG35=0x80000`, clears
ready, and never gets ready back.

## Preferred Capture

Use the Ubuntu checkout at:

```text
/home/ubuntu/tinygrad-arkey
```

Before capturing, make sure that checkout has the current trace tooling from
`JulianAbeleda/tinygrad-arkey` and preserve any local capture outputs. The trace
script must be the fixed-offset version because Ubuntu 6.8.0-117 did not expose
AMD module-private struct names to `bpftrace` even with BTF present.

Boot Linux with `amdgpu` blacklisted so the card is present but unbound:

```sh
modprobe.blacklist=amdgpu
```

For repeated captures, install the helper GRUB entry once and select it from
the terminal for the next boot:

```sh
sudo extra/amdpci/linux_amdgpu_grub_switch.sh install
sudo extra/amdpci/linux_amdgpu_grub_switch.sh next-blacklist
sudo reboot
```

After the blacklisted run, select normal boot for the next reboot:

```sh
sudo extra/amdpci/linux_amdgpu_grub_switch.sh next-normal
sudo reboot
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

For the deeper pre-KDB comparison pass, add `--deep`:

```sh
sudo extra/amdpci/capture_linux_psp_good_trace.sh --deep --bind-bdf 0000:03:00.0
```

Deep mode generates a bpftrace script inside the output directory from the
symbols visible on that boot. It keeps the known-good PSP/GART probes and adds
optional PSP ring/TMR/helper and filtered register read/write probes when those
symbols are available.

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
- `psp-linux-good-deep.trace`: deep-mode trace with optional pre-KDB PSP,
  ring/TMR, MMHUB, and mailbox register events.
- `trace_amdgpu_psp_deep.generated.bt`: deep-mode bpftrace script generated
  from symbols visible in `/proc/kallsyms`.
- `psp-deep-generated-symbols.txt`: generated probe list and skipped optional
  probe classes.
- `linux-pre-kdb-key-events.txt`: focused grep output for KDB, mailbox,
  memory-training, GART, MMHUB, ring, and TMR events.
- `psp-linux-good-*.tar.gz` and `.sha256`: deep-mode archive and checksum
  generated after the trace completes.
- `mmhub-gart-snapshot.txt` / `mmhub-gart-snapshot.json`: read-only MMHUB
  and GART/context register snapshot, when `linux_mmhub_gart_snapshot.py` can
  read BAR5 after bind/rebind.
- `mmhub-gart-snapshot.err`: snapshot failure details, if the PSP trace succeeds
  but the register snapshot fails.
- `baseline.txt`: kernel, PCI, `amdgpu`, and BTF baseline.
- `dmesg-before.txt` / `dmesg-after.txt`: kernel log around the capture.
- `bpftrace.stderr`: bpftrace compile/attach errors, if any.
- `firmware-sha256.txt`: hashes for relevant AMD firmware blobs.
- `gpu-candidates.txt` and `lspci-*.txt`: PCI identity and topology.
- `psp-symbols-*.txt`: PSP/GART probe symbol visibility before and after setup.

## Read-Only MMHUB/GART Snapshot

The snapshot helper can also be run by itself on Ubuntu without unbinding the
GPU. Try this first on a normal boot before doing another blacklisted PSP
capture:

```sh
sudo python3 extra/amdpci/linux_mmhub_gart_snapshot.py \
  --bdf 0000:08:00.0 \
  --out linux-mmhub-gart-snapshot-$(date +%Y%m%d-%H%M%S)
```

Replace the BDF if `lspci -Dnn` reports a different RX 7900 XTX address. The
helper is read-only: it maps BAR5 and reads named MMHUB registers using the
same `gfx1100_744c` register metadata as tinygrad. It does not unbind, reset,
write PCI config, or write MMIO.

Compare `mmhub-gart-snapshot.txt` against a Mac/TinyGPU audit run using:

```text
AM_PSP_AUDIT_PRE_KDB=1 AM_PSP_PARITY_TRACE=1
```

The important values are context0 control/base/start/end, system aperture,
MMHUB L1/L2 controls, and protection-fault status/control. If the standalone
snapshot cannot read BAR5 while `amdgpu` owns the display GPU, rerun the normal
PSP capture from a blacklisted boot; the wrapper will attempt the same snapshot
after binding `amdgpu`.

## VRAM PSP Message Buffer Experiment

`AM_PSP_SYSMSG1_VRAM=1` uses a VRAM-backed PSP msg1 buffer on remote Linux
runs. `AM_PSP_SYSMSG1_VRAM_PADDR=0x...` can force the 1 MiB-aligned VRAM
physical address for that buffer. This is useful for avoiding low VRAM while
checking whether BAR0 writes/readbacks are stable before the KDB mailbox write.

For mailbox ordering experiments, `AM_PSP_MAILBOX_STRONG_ORDER=1` adds an HDP
flush and C2PMSG35/C2PMSG36 readbacks around the msg1 and component writes.
`AM_PSP_WAIT_TRACE_MS=100` emits periodic bootloader wait samples instead of
only tracing value changes.

`AM_PSP_TRACE_C2PMSG_DENSE=1` expands tinygrad post-compid/timeout PSP
snapshots from the sparse mailbox set to C2PMSG0..127. Pre-KDB tracing remains
sparse because dense C2PMSG reads may disturb the stable msg1 readback path. Use
it with `AM_PSP_PARITY_TRACE=1` when comparing against a Linux `--deep` capture;
the Linux wrapper also writes `linux-c2pmsg-events.txt` from the raw register
trace.

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

If PTE flags match, compare `mmhub-gart-snapshot.txt` against the Mac parity
trace before adding another KDB experiment. The next likely difference is
MMHUB/GART context setup, invalidation semantics, or another Linux pre-KDB init
side effect.

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
