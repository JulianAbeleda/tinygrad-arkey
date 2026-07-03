#!/usr/bin/env python3
import argparse, json, mmap, os, pathlib, struct, subprocess, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tinygrad.runtime.autogen.am import regs as am_regs
from tinygrad.runtime.autogen.am import navi_offsets

BASE_MMHUB_REGS = [
  "regMMMC_VM_FB_LOCATION_BASE", "regMMMC_VM_FB_LOCATION_TOP", "regMMMC_VM_AGP_BASE", "regMMMC_VM_AGP_BOT",
  "regMMMC_VM_AGP_TOP", "regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR", "regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR",
  "regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB", "regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB",
  "regMMMC_VM_MX_L1_TLB_CNTL", "regMMVM_L2_CNTL", "regMMVM_L2_CNTL2", "regMMVM_L2_CNTL3", "regMMVM_L2_CNTL4",
  "regMMVM_L2_CNTL5", "regMMVM_L2_BANK_SELECT_RESERVED_CID2", "regMMVM_L2_PROTECTION_FAULT_CNTL",
  "regMMVM_L2_PROTECTION_FAULT_CNTL2", "regMMVM_L2_PROTECTION_FAULT_STATUS",
  "regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32", "regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32",
  "regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32",
  "regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32", "regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32",
  "regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32", "regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32",
  "regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32",
]
CONTEXT_MMHUB_SUFFIXES = [
  "CNTL", "PAGE_TABLE_BASE_ADDR_LO32", "PAGE_TABLE_BASE_ADDR_HI32", "PAGE_TABLE_START_ADDR_LO32",
  "PAGE_TABLE_START_ADDR_HI32", "PAGE_TABLE_END_ADDR_LO32", "PAGE_TABLE_END_ADDR_HI32",
]
INVALIDATE_MMHUB_SUFFIXES = ["ADDR_RANGE_LO32", "ADDR_RANGE_HI32", "REQ", "ACK", "SEM"]
MMHUB_REGS = [
  *BASE_MMHUB_REGS,
  *(f"regMMVM_CONTEXT{i}_{suffix}" for i in range(16) for suffix in CONTEXT_MMHUB_SUFFIXES),
  *(f"regMMVM_INVALIDATE_ENG{i}_{suffix}" for i in range(18) for suffix in INVALIDATE_MMHUB_SUFFIXES),
]

PSP_C2PMSG_REGS = {
  "C2PMSG33_VMBX": 0x16061, "C2PMSG35_BL": 0x16063, "C2PMSG36_ADDR": 0x16064,
  "C2PMSG58_SOS_FW_VERSION": 0x1607a,
  "C2PMSG64_RING": 0x16080, "C2PMSG67_WPTR": 0x16083, "C2PMSG69_RING_LO": 0x16085,
  "C2PMSG70_RING_HI": 0x16086, "C2PMSG71_RING_SIZE": 0x16087, "C2PMSG73_SPI_DOORBELL": 0x16089,
  "C2PMSG81_SOS": 0x16091, "C2PMSG90_SMU": 0x1609a, "C2PMSG92_STATUS": 0x1609c,
  "C2PMSG101_GPCOM_CMD": 0x160a5, "C2PMSG102_GPCOM_LO": 0x160a6, "C2PMSG103_GPCOM_HI": 0x160a7,
  "C2PMSG115_SPI": 0x160b3, "C2PMSG116_SPI_ARG": 0x160b4, "C2PMSG127_RAS_CAP": 0x160bf,
}

PSP_DENSE_C2PMSG_REGS = {
  **{f"MP0_C2PMSG{i:03d}": 0x16040 + i for i in range(128)},
  **{f"MP1_C2PMSG{i:03d}": 0x16240 + i for i in range(128)},
}

PROFILE_IP = {
  "gfx1100_744c": {"mmhub": (3, 0, 0)},
}

def run(cmd:list[str]) -> str:
  return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()

def navi_bases(prefix:str) -> dict[int, tuple[int, ...]]:
  return {i: seg for i in range(7) if any(seg:=tuple(getattr(navi_offsets, f"{prefix}_BASE__INST{i}_SEG{s}", 0) for s in range(9)))}

def decode(fields:dict[str, tuple[int, int]], value:int) -> dict[str, int]:
  return {name: (value >> start) & ((1 << (end - start + 1)) - 1) for name, (start, end) in fields.items()}

def open_bar5(path:pathlib.Path, size:int) -> tuple[int, mmap.mmap, str]:
  try:
    fd = os.open(path, os.O_RDONLY | os.O_SYNC | os.O_CLOEXEC)
    return fd, mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ), "O_RDONLY"
  except OSError as first_err:
    if 'fd' in locals(): os.close(fd)
    fd = os.open(path, os.O_RDWR | os.O_SYNC | os.O_CLOEXEC)
    try:
      return fd, mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ), f"O_RDWR_PROT_READ fallback after {type(first_err).__name__}: {first_err}"
    except Exception:
      os.close(fd)
      raise

def read_resource_info(dev:pathlib.Path) -> list[tuple[int, int, int]]:
  rows = []
  for line in (dev/"resource").read_text().splitlines():
    start, end, flags = (int(x, 16) for x in line.split()[:3])
    rows.append((start, end, flags))
  return rows

def collect(bdf:str, profile:str) -> dict:
  if profile not in PROFILE_IP: raise ValueError(f"unknown profile {profile!r}")
  dev = pathlib.Path("/sys/bus/pci/devices") / bdf
  if not dev.is_dir(): raise FileNotFoundError(f"PCI device not found: {dev}")

  resources = read_resource_info(dev)
  _, bar5_end, _ = resources[5]
  bar5_start = resources[5][0]
  bar5_size = bar5_end - bar5_start + 1
  fd, bar5, open_mode = open_bar5(dev/"resource5", bar5_size)
  try:
    regs = getattr(am_regs, f"mmhub_{'_'.join(map(str, PROFILE_IP[profile]['mmhub']))}")
    bases = navi_bases("MMHUB")
    def read_bar5_regs(regs:dict[str, int]) -> list[dict]:
      out = []
      for name, addr in regs.items():
        byte_off = addr * 4
        if byte_off + 4 > len(bar5):
          out.append({"name": name, "dword_addr": addr, "out_of_range": True})
          continue
        out.append({"name": name, "dword_addr": addr, "value": struct.unpack_from("<I", bar5, byte_off)[0]})
      return out

    psp_entries = read_bar5_regs(PSP_C2PMSG_REGS)
    psp_dense_entries = read_bar5_regs(PSP_DENSE_C2PMSG_REGS)
    entries = []
    for name in MMHUB_REGS:
      if name not in regs:
        entries.append({"name": name, "missing": True})
        continue
      offset, segment, fields = regs[name]
      for inst, segs in sorted(bases.items()):
        if segment >= len(segs) or segs[segment] == 0:
          entries.append({"name": name, "instance": inst, "segment": segment, "missing_base": True})
          continue
        addr = segs[segment] + offset
        byte_off = addr * 4
        if byte_off + 4 > len(bar5):
          entries.append({"name": name, "instance": inst, "segment": segment, "dword_addr": addr, "out_of_range": True})
          continue
        value = struct.unpack_from("<I", bar5, byte_off)[0]
        entries.append({"name": name, "instance": inst, "segment": segment, "dword_addr": addr, "value": value, "fields": decode(fields, value)})
    return {
      "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
      "bdf": bdf,
      "profile": profile,
      "kernel": run(["uname", "-a"]),
      "lspci": run(["lspci", "-Dnnk", "-s", bdf]),
      "bar5": {"base": bar5_start, "size": bar5_size, "open_mode": open_mode},
      "resources": [{"bar": i, "base": s, "size": e - s + 1, "flags": f} for i, (s, e, f) in enumerate(resources)],
      "psp_registers": psp_entries,
      "psp_dense_registers": psp_dense_entries,
      "registers": entries,
    }
  finally:
    bar5.close()
    os.close(fd)

def write_outputs(snapshot:dict, out:pathlib.Path, name:str):
  out.mkdir(parents=True, exist_ok=True)
  (out/f"{name}.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
  lines = [
    f"captured_at {snapshot['captured_at']}",
    f"bdf {snapshot['bdf']}",
    f"profile {snapshot['profile']}",
    f"kernel {snapshot['kernel']}",
    "lspci begin",
    snapshot["lspci"],
    "lspci end",
    f"bar5 base={snapshot['bar5']['base']:#x} size={snapshot['bar5']['size']:#x} open_mode={snapshot['bar5']['open_mode']}",
  ]
  for reg in snapshot["psp_registers"]:
    if reg.get("out_of_range"):
      lines.append(f"psp {reg['name']} addr={reg['dword_addr']:#x} out_of_range")
    else:
      lines.append(f"psp {reg['name']} addr={reg['dword_addr']:#x} value={reg['value']:#010x}")
  for reg in snapshot["psp_dense_registers"]:
    if reg.get("out_of_range"):
      lines.append(f"psp-dense {reg['name']} addr={reg['dword_addr']:#x} out_of_range")
    else:
      lines.append(f"psp-dense {reg['name']} addr={reg['dword_addr']:#x} value={reg['value']:#010x}")
  for reg in snapshot["registers"]:
    if reg.get("missing"):
      lines.append(f"reg {reg['name']} missing")
    elif reg.get("missing_base"):
      lines.append(f"reg {reg['name']}[{reg['instance']}] segment={reg['segment']} missing_base")
    elif reg.get("out_of_range"):
      lines.append(f"reg {reg['name']}[{reg['instance']}] addr={reg['dword_addr']:#x} out_of_range")
    else:
      lines.append(f"reg {reg['name']}[{reg['instance']}] addr={reg['dword_addr']:#x} value={reg['value']:#010x}")
  (out/f"{name}.txt").write_text("\n".join(lines) + "\n")

def main():
  parser = argparse.ArgumentParser(description="Read-only Linux MMHUB/GART register snapshot for RX 7900 XTX PSP comparison")
  parser.add_argument("--bdf", required=True, help="PCI BDF, for example 0000:08:00.0")
  parser.add_argument("--out", required=True, help="Output directory")
  parser.add_argument("--profile", default="gfx1100_744c", choices=sorted(PROFILE_IP))
  parser.add_argument("--name", default="mmhub-gart-snapshot", help="Output basename without .txt/.json")
  args = parser.parse_args()
  snapshot = collect(args.bdf, args.profile)
  write_outputs(snapshot, pathlib.Path(args.out), args.name)

if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
    raise
