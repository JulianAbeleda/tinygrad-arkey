#!/usr/bin/env python3
import argparse, os, subprocess, sys, time

from tinygrad.runtime.ops_amd import AMDDevice
from tinygrad.runtime.support.amd import AMD_RUNTIME_DEVICES
from tinygrad.runtime.support.system import RemotePCIDevice, System

def stamp(msg:str):
  print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

def mac_gpu_visible() -> bool:
  if sys.platform != "darwin": return True
  out = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, check=False).stdout
  return "Device ID: 0x744c" in out

def require_visible(stage:str):
  if not mac_gpu_visible(): raise RuntimeError(f"AMD GPU disappeared after {stage}")

def open_remote():
  devs = System.list_devices(0x1002, ((0xffff, AMD_RUNTIME_DEVICES),), 0)
  if not devs: raise RuntimeError("no AMD runtime device found")
  cl, name = devs[0]
  stamp(f"remote device {name}")
  return cl("SV", name)

def remote_bars(pci):
  stamp("cfg vendor/device")
  stamp(f"cfg0={pci.read_config(0, 4):#x} cmd={pci.read_config(4, 2):#x}")
  for bar in (0, 2, 5):
    stamp(f"map BAR{bar}")
    info = pci.bar_info(bar)
    view = pci.map_bar(bar, fmt="Q" if bar == 2 else "I" if bar == 5 else "B")
    stamp(f"BAR{bar} base={info[0]:#x} size={info[1]:#x} nbytes={view.nbytes:#x}")
    require_visible(f"BAR{bar}")
  mmio = pci.map_bar(5, fmt="I")
  stamp(f"memsize_reg={mmio[0xde3]:#x} vram={mmio[0xde3] << 20}")
  require_visible("MMIO memsize")

def remote_sysmem(pci, sizes:list[int], repeat:int):
  for sz in sizes:
    for i in range(repeat):
      stamp(f"alloc_sysmem size={sz} iter={i+1}/{repeat}")
      mem, paddrs = pci.alloc_sysmem(sz)
      stamp(f"alloc_sysmem ok paddrs={len(paddrs)}")
      if sz <= (8 << 20):
        mem[0:4] = b"\x11\x22\x33\x44"
        got = bytes(mem[0:4])
        stamp(f"sysmem rw got={got.hex()}")
      require_visible(f"alloc_sysmem {sz} iter {i+1}")

def remote_bar0_write(pci, offsets:list[int], sizes:list[int], repeat:int, readback:bool):
  stamp("cfg vendor/device")
  stamp(f"cfg0={pci.read_config(0, 4):#x} cmd={pci.read_config(4, 2):#x}")
  stamp("map BAR0")
  base, bar_size = pci.bar_info(0)
  bar0 = pci.map_bar(0, fmt="B")
  stamp(f"BAR0 base={base:#x} size={bar_size:#x} nbytes={bar0.nbytes:#x}")
  require_visible("BAR0")
  for off in offsets:
    for sz in sizes:
      if off + sz > bar0.nbytes: raise RuntimeError(f"BAR0 write out of range off={off:#x} size={sz:#x} nbytes={bar0.nbytes:#x}")
      data = bytes((i & 0xff for i in range(sz)))
      for i in range(repeat):
        stamp(f"bar0 write off={off:#x} size={sz:#x} iter={i+1}/{repeat}")
        bar0[off:off+sz] = data
        stamp("bar0 write ok")
        require_visible(f"bar0 write off={off:#x} size={sz:#x} iter {i+1}")
        if readback:
          got = bytes(bar0[off:off+sz])
          stamp(f"bar0 readback {'ok' if got == data else 'mismatch'}")
          if got != data: raise RuntimeError(f"BAR0 readback mismatch off={off:#x} size={sz:#x}")

def remote_bar0_read(pci, offsets:list[int], sizes:list[int], repeat:int):
  stamp("cfg vendor/device")
  stamp(f"cfg0={pci.read_config(0, 4):#x} cmd={pci.read_config(4, 2):#x}")
  stamp("map BAR0")
  base, bar_size = pci.bar_info(0)
  bar0 = pci.map_bar(0, fmt="B")
  stamp(f"BAR0 base={base:#x} size={bar_size:#x} nbytes={bar0.nbytes:#x}")
  require_visible("BAR0")
  for off in offsets:
    for sz in sizes:
      if off + sz > bar0.nbytes: raise RuntimeError(f"BAR0 read out of range off={off:#x} size={sz:#x} nbytes={bar0.nbytes:#x}")
      for i in range(repeat):
        stamp(f"bar0 read off={off:#x} size={sz:#x} iter={i+1}/{repeat}")
        got = bytes(bar0[off:off+sz])
        stamp(f"bar0 read ok first={got[:min(len(got), 16)].hex()}")
        require_visible(f"bar0 read off={off:#x} size={sz:#x} iter {i+1}")

def amd_boot_and_alloc(sizes:list[int], repeat:int):
  stamp("AMDDevice boot")
  dev = AMDDevice("AMD")
  stamp(f"AMDDevice ready arch={dev.arch} has_sdma={dev.has_sdma_queue}")
  require_visible("AMDDevice boot")
  for sz in sizes:
    for i in range(repeat):
      stamp(f"iface.alloc host size={sz} iter={i+1}/{repeat}")
      buf = dev.iface.alloc(sz, host=True)
      stamp(f"iface.alloc ok va={int(buf.va_addr):#x} size={buf.size}")
      require_visible(f"iface.alloc {sz} iter {i+1}")
  dev.synchronize()
  stamp("AMDDevice synchronize ok")

if __name__ == "__main__":
  p = argparse.ArgumentParser(description="Narrow AMD/TinyGPU dropout repro without LLM loading")
  p.add_argument("remote", nargs="?", default=os.environ.get("REMOTE", "127.0.0.1:6667"))
  p.add_argument("--stage", choices=("bars", "bar0-read", "bar0-write", "remote-sysmem", "amd-boot", "all"), default="all")
  p.add_argument("--sizes", default="16384,2097152,16777216", help="comma-separated allocation sizes")
  p.add_argument("--offsets", default="0", help="comma-separated BAR0 offsets for read/write stages")
  p.add_argument("--repeat", type=int, default=4)
  p.add_argument("--readback", action="store_true", help="read BAR0 writes back and compare")
  args = p.parse_args()

  os.environ["REMOTE"] = args.remote
  os.environ["DEV"] = "PCI+AMD"
  RemotePCIDevice.reset_stats()
  sizes = [int(x, 0) for x in args.sizes.split(",") if x]
  offsets = [int(x, 0) for x in args.offsets.split(",") if x]

  stamp(f"start remote={args.remote} stage={args.stage} sizes={sizes} repeat={args.repeat}")
  require_visible("start")
  pci = open_remote()
  if args.stage == "bar0-read":
    remote_bar0_read(pci, offsets, sizes, args.repeat)
  elif args.stage == "bar0-write":
    remote_bar0_write(pci, offsets, sizes, args.repeat, args.readback)
  else:
    remote_bars(pci)
  if args.stage in ("remote-sysmem", "all"): remote_sysmem(pci, sizes, args.repeat)
  if args.stage in ("amd-boot", "all"): amd_boot_and_alloc(sizes, args.repeat)
  stamp(f"done stats={RemotePCIDevice.stats()} commands={RemotePCIDevice.command_stats()}")
