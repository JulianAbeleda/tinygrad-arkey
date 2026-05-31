#!/usr/bin/env python3
import argparse, ctypes, hashlib, os, struct, subprocess, sys, time

from tinygrad.helpers import fetch_fw, mv_address
from tinygrad.runtime.autogen.am import am, fw
from tinygrad.runtime.autogen.am import regs as am_regs
from tinygrad.runtime.ops_amd import AMDDevice
from tinygrad.runtime.support.am.amdev import AMDev
from tinygrad.runtime.support.amd import AMD_RUNTIME_DEVICES
from tinygrad.runtime.support.system import RemotePCIDevice, System

# The clean gate classifies VMBX/BL/ADDR/SOS; the remaining PSP registers are printed for operator context.
PSP_GATE_REGS = {
  "C2PMSG33_VMBX": 0x16061,
  "C2PMSG35_BL": 0x16063,
  "C2PMSG36_ADDR": 0x16064,
  "C2PMSG64_RING": 0x16080,
  "C2PMSG67_WPTR": 0x16083,
  "C2PMSG69_RING_LO": 0x16085,
  "C2PMSG70_RING_HI": 0x16086,
  "C2PMSG71_RING_SIZE": 0x16087,
  "C2PMSG81_SOS": 0x16091,
  "C2PMSG90_SMU": 0x1609a,
  "C2PMSG92_STATUS": 0x1609c,
  "C2PMSG115_SPI": 0x160b3,
}
PSP_CLEAN_GATE_ALLOWED_C2PMSG36 = {0x0, 0x5fff}

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

def remote_psp_sysmem_probe(pci, size:int, contiguous:bool):
  stamp(f"psp sysmem probe size={size:#x} contiguous={contiguous}")
  mem, paddrs = pci.alloc_sysmem(size, contiguous=contiguous)
  mem[0:16] = b"\xaa" * 16
  spans = []
  if paddrs:
    start = last = paddrs[0]
    for paddr in paddrs[1:]:
      if paddr == last + 0x1000:
        last = paddr
      else:
        spans.append((start, last + 0x1000 - start))
        start = last = paddr
    spans.append((start, last + 0x1000 - start))
  stamp(f"psp sysmem pages={len(paddrs)} spans={len(spans)} first={paddrs[0] if paddrs else 0:#x} first_align_1m={(paddrs[0] if paddrs else 0) & 0xfffff:#x}")
  for i, (start, span_size) in enumerate(spans[:16]):
    stamp(f"psp sysmem span[{i}] start={start:#x} size={span_size:#x} c2pmsg36={start >> 20:#x}")
  if len(spans) > 16: stamp(f"psp sysmem spans_truncated={len(spans)-16}")
  require_visible("psp-sysmem-probe")

def _bar_fmt(bar:int) -> str: return "Q" if bar == 2 else "I" if bar == 5 else "B"

def remote_bar_write(pci, bars:list[int], offsets:list[int], sizes:list[int], repeat:int, readback:bool):
  stamp("cfg vendor/device")
  stamp(f"cfg0={pci.read_config(0, 4):#x} cmd={pci.read_config(4, 2):#x}")
  for bar in bars:
    stamp(f"map BAR{bar}")
    base, bar_size = pci.bar_info(bar)
    view = pci.map_bar(bar, fmt=_bar_fmt(bar))
    byte_view = view.view(fmt="B")
    stamp(f"BAR{bar} base={base:#x} size={bar_size:#x} nbytes={byte_view.nbytes:#x} fmt={_bar_fmt(bar)}")
    require_visible(f"BAR{bar}")
    for off in offsets:
      for sz in sizes:
        if off + sz > byte_view.nbytes: raise RuntimeError(f"BAR{bar} write out of range off={off:#x} size={sz:#x} nbytes={byte_view.nbytes:#x}")
        data = bytes((i & 0xff for i in range(sz)))
        for i in range(repeat):
          stamp(f"bar{bar} write off={off:#x} size={sz:#x} iter={i+1}/{repeat}")
          byte_view[off:off+sz] = data
          stamp(f"bar{bar} write ok")
          require_visible(f"bar{bar} write off={off:#x} size={sz:#x} iter {i+1}")
          if readback:
            got = bytes(byte_view[off:off+sz])
            stamp(f"bar{bar} readback {'ok' if got == data else 'mismatch'}")
            if got != data: raise RuntimeError(f"BAR{bar} readback mismatch off={off:#x} size={sz:#x}")

def remote_bar_read(pci, bars:list[int], offsets:list[int], sizes:list[int], repeat:int):
  stamp("cfg vendor/device")
  stamp(f"cfg0={pci.read_config(0, 4):#x} cmd={pci.read_config(4, 2):#x}")
  for bar in bars:
    stamp(f"map BAR{bar}")
    base, bar_size = pci.bar_info(bar)
    view = pci.map_bar(bar, fmt=_bar_fmt(bar))
    byte_view = view.view(fmt="B")
    stamp(f"BAR{bar} base={base:#x} size={bar_size:#x} nbytes={byte_view.nbytes:#x} fmt={_bar_fmt(bar)}")
    require_visible(f"BAR{bar}")
    for off in offsets:
      for sz in sizes:
        if off + sz > byte_view.nbytes: raise RuntimeError(f"BAR{bar} read out of range off={off:#x} size={sz:#x} nbytes={byte_view.nbytes:#x}")
        for i in range(repeat):
          stamp(f"bar{bar} read off={off:#x} size={sz:#x} iter={i+1}/{repeat}")
          got = bytes(byte_view[off:off+sz])
          stamp(f"bar{bar} read ok first={got[:min(len(got), 16)].hex()}")
          require_visible(f"bar{bar} read off={off:#x} size={sz:#x} iter {i+1}")

def remote_psp_status(pci):
  stamp("map BAR5")
  view = pci.map_bar(5, fmt="I")
  for name, reg in PSP_GATE_REGS.items():
    stamp(f"psp {name} reg={reg:#x} val={view[reg]:#010x}")
  require_visible("psp-status")

def classify_psp_clean_gate(vals:dict[str, int]) -> tuple[str, list[str]]:
  reasons = []
  if vals.get("C2PMSG33_VMBX") == 0xffffffff or vals.get("C2PMSG35_BL") == 0xffffffff:
    return "DIRTY", ["PSP mailbox returned all-ones MMIO"]
  if vals.get("C2PMSG35_BL") == 0x0:
    return "DIRTY", ["bootloader mailbox is stuck at C2PMSG35_BL=0"]
  if vals.get("C2PMSG81_SOS", 0) != 0:
    return "DIRTY", [f"sOS is already alive/nonzero C2PMSG81_SOS={vals.get('C2PMSG81_SOS'):#010x}"]
  if vals.get("C2PMSG33_VMBX") != 0x80000000:
    reasons.append(f"unexpected C2PMSG33_VMBX={vals.get('C2PMSG33_VMBX', 0):#010x}")
  if vals.get("C2PMSG35_BL") != 0x80000000:
    reasons.append(f"unexpected C2PMSG35_BL={vals.get('C2PMSG35_BL', 0):#010x}")
  if vals.get("C2PMSG36_ADDR", 0) not in PSP_CLEAN_GATE_ALLOWED_C2PMSG36:
    reasons.append(f"suspicious nonzero C2PMSG36_ADDR={vals.get('C2PMSG36_ADDR'):#010x}")
  return ("UNKNOWN", reasons) if reasons else ("CLEAN", ["PSP mailbox is at pre-KDB ready baseline"])

def remote_psp_clean_gate(pci) -> int:
  stamp("clean-gate begin")
  try:
    stamp(f"gate cfg0={pci.read_config(0, 4):#x} cmd={pci.read_config(4, 2):#x}")
    stamp("gate map BAR5")
    view = pci.map_bar(5, fmt="I")
    stamp(f"gate memsize_reg={view[0xde3]:#x} vram={view[0xde3] << 20:#x}")
    vals = {name: view[reg] for name, reg in PSP_GATE_REGS.items()}
    for name, reg in PSP_GATE_REGS.items(): stamp(f"gate psp {name} reg={reg:#x} val={vals[name]:#010x}")
    require_visible("psp-clean-gate")
  except Exception as e:
    stamp(f"gate error={type(e).__name__}: {e}")
    print("DIRTY: full hardware restart required", flush=True)
    return 1

  status, reasons = classify_psp_clean_gate(vals)
  for reason in reasons: stamp(f"gate {status.lower()} reason={reason}")
  if status == "CLEAN":
    print("CLEAN: safe to run audit/real KDB attempt", flush=True)
    return 0
  if status == "DIRTY":
    print("DIRTY: full hardware restart required", flush=True)
    return 1
  print("UNKNOWN: do not trust this state", flush=True)
  return 2

def remote_nbio_status(pci):
  direct_regs = {
    "BIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL": 301,
    "BIF_BX0_REMAP_HDP_REG_FLUSH_CNTL": 302,
    "BIF_BX_PF0_GPU_HDP_FLUSH_REQ": 262,
    "BIF_BX_PF0_GPU_HDP_FLUSH_DONE": 263,
    "RCC_DEV0_EPF0_RCC_DOORBELL_APER_EN": 192,
  }
  indirect_regs = {
    "RCC_DEV0_EPF2_STRAP2": 53506,
    "BIFC_DOORBELL_ACCESS_EN_PF": 53102,
    "BIFC_GFX_INT_MONITOR_MASK": 59565,
  }
  stamp("map BAR5")
  view = pci.map_bar(5, fmt="I")
  for name, reg in direct_regs.items():
    stamp(f"nbio direct {name} reg={reg:#x} val={view[reg]:#010x}")
  for name, reg in indirect_regs.items():
    view[0] = reg * 4
    stamp(f"nbio rsmu {name} reg={reg:#x} val={view[1]:#010x}")
  for name, reg in indirect_regs.items():
    view[14] = reg * 4
    stamp(f"nbio pcie {name} reg={reg:#x} val={view[15]:#010x}")
  require_visible("nbio-status")

def _dump_cfg(pci):
  stamp("snapshot cfg")
  for off, size, name in [(0x00, 4, "vendor_device"), (0x04, 2, "command"), (0x06, 2, "status"), (0x08, 4, "class_rev"),
                          (0x10, 4, "bar0_lo"), (0x14, 4, "bar0_hi"), (0x18, 4, "bar2_lo"), (0x1c, 4, "bar2_hi"),
                          (0x24, 4, "bar5"), (0x34, 1, "cap_ptr")]:
    stamp(f"cfg {name} off={off:#04x} size={size} val={pci.read_config(off, size):#010x}")
  cap = pci.read_config(0x34, 1) & ~0x3
  seen = set()
  for i in range(32):
    if cap == 0 or cap in seen: break
    seen.add(cap)
    hdr = pci.read_config(cap, 2)
    cap_id, nxt = hdr & 0xff, (hdr >> 8) & ~0x3
    data = pci.read_config(cap, 4)
    stamp(f"cfg cap[{i}] off={cap:#04x} id={cap_id:#04x} next={nxt:#04x} data={data:#010x}")
    cap = nxt

def _dump_direct_regs(view, group:str, regs:dict[str, int]):
  for name, reg in regs.items():
    try:
      stamp(f"snapshot {group} {name} reg={reg:#x} val={view[reg]:#010x}")
    except Exception as e:
      stamp(f"snapshot {group} {name} reg={reg:#x} read_failed={e}")

def _dump_indirect_regs(view, group:str, regs:dict[str, int], index_reg:int, data_reg:int):
  for name, reg in regs.items():
    try:
      view[index_reg] = reg * 4
      stamp(f"snapshot {group} {name} reg={reg:#x} val={view[data_reg]:#010x}")
    except Exception as e:
      stamp(f"snapshot {group} {name} reg={reg:#x} read_failed={e}")

def _reg_table_regs(table:dict[str, tuple], names:list[str]) -> dict[str, int]:
  return {name: table[name][0] for name in names if name in table}

def _c2pmsg_dense_regs(prefix:str, base:int) -> dict[str, int]:
  return {f"{prefix}_C2PMSG{i:03d}": base + i for i in range(128)}

def _open_discovery_only_amdev(pci):
  if getattr(pci, "is_remote", False):
    os.environ.setdefault("AM_REMOTE_DISCOVERY_PROFILE", "gfx1100_744c")
  dev = object.__new__(AMDev)
  dev.pci_dev, dev.devfmt = pci, pci.pcibus
  dev.vram, dev.doorbell64, dev.mmio = pci.map_bar(0), pci.map_bar(2, fmt='Q'), pci.map_bar(5, fmt='I')
  dev._run_discovery()
  dev._build_regs()
  return dev

def _dump_amdev_regs(dev, group:str, names:list[str]):
  for name in names:
    if not hasattr(dev, name):
      stamp(f"snapshot {group} {name} missing")
      continue
    reg = dev.reg(name)
    for inst in range(len(reg.addr)):
      try:
        stamp(f"snapshot {group} {name}[{inst}] reg={reg.addr[inst]:#x} val={reg.read(inst=inst):#010x}")
      except Exception as e:
        stamp(f"snapshot {group} {name}[{inst}] reg={reg.addr[inst]:#x} read_failed={e}")

def remote_psp_pre_kdb_snapshot(pci):
  stamp("snapshot begin read-only pre-kdb")
  _dump_cfg(pci)
  for bar in (0, 2, 5):
    try:
      base, size = pci.bar_info(bar)
      stamp(f"snapshot bar{bar} base={base:#x} size={size:#x}")
    except Exception as e:
      stamp(f"snapshot bar{bar} info_failed={e}")

  stamp("snapshot map BAR5")
  view = pci.map_bar(5, fmt="I")
  psp_regs = {
    "C2PMSG33_VMBX": 0x16061, "C2PMSG35_BL": 0x16063, "C2PMSG36_ADDR": 0x16064,
    "C2PMSG58_SOS_FW_VERSION": 0x1607a,
    "C2PMSG64_RING": 0x16080, "C2PMSG67_WPTR": 0x16083, "C2PMSG69_RING_LO": 0x16085,
    "C2PMSG70_RING_HI": 0x16086, "C2PMSG71_RING_SIZE": 0x16087, "C2PMSG73_SPI_DOORBELL": 0x16089,
    "C2PMSG81_SOS": 0x16091, "C2PMSG90_SMU": 0x1609a, "C2PMSG92_STATUS": 0x1609c,
    "C2PMSG101_GPCOM_CMD": 0x160a5, "C2PMSG102_GPCOM_LO": 0x160a6, "C2PMSG103_GPCOM_HI": 0x160a7,
    "C2PMSG115_SPI": 0x160b3, "C2PMSG116_SPI_ARG": 0x160b4, "C2PMSG127_RAS_CAP": 0x160bf,
  }
  nbio_direct = {
    "BIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL": 301, "BIF_BX0_REMAP_HDP_REG_FLUSH_CNTL": 302,
    "BIF_BX_PF0_GPU_HDP_FLUSH_REQ": 262, "BIF_BX_PF0_GPU_HDP_FLUSH_DONE": 263,
    "RCC_DEV0_EPF0_RCC_DOORBELL_APER_EN": 192,
  }
  nbio_indirect = {
    "RCC_DEV0_EPF2_STRAP2": 53506, "RCC_DEV0_EPF2_STRAP20": 53524,
    "BIFC_DOORBELL_ACCESS_EN_PF": 53102, "BIFC_GFX_INT_MONITOR_MASK": 59565,
  }
  mmhub_names = [
    "regMMMC_VM_FB_LOCATION_BASE", "regMMMC_VM_FB_LOCATION_TOP", "regMMMC_VM_AGP_BASE", "regMMMC_VM_AGP_BOT",
    "regMMMC_VM_AGP_TOP", "regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR", "regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR",
    "regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB", "regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB",
    "regMMMC_VM_MX_L1_TLB_CNTL", "regMMVM_L2_CNTL", "regMMVM_L2_CNTL2", "regMMVM_L2_CNTL3", "regMMVM_L2_CNTL4",
    "regMMVM_L2_CNTL5", "regMMVM_L2_PROTECTION_FAULT_CNTL", "regMMVM_L2_PROTECTION_FAULT_CNTL2",
    "regMMVM_L2_PROTECTION_FAULT_STATUS", "regMMVM_CONTEXT0_CNTL", "regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32",
    "regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32", "regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32",
    "regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32", "regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32",
    "regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32", "regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32",
    "regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32", "regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32",
    "regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32", "regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32",
    "regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32",
  ]
  _dump_direct_regs(view, "psp", psp_regs)
  _dump_direct_regs(view, "mp0-c2pmsg-dense", _c2pmsg_dense_regs("MP0", 0x16040))
  _dump_direct_regs(view, "mp1-c2pmsg-dense", _c2pmsg_dense_regs("MP1", 0x16240))
  _dump_direct_regs(view, "nbio-direct", nbio_direct)
  _dump_indirect_regs(view, "nbio-rsmu", nbio_indirect, 0, 1)
  _dump_indirect_regs(view, "nbio-pcie", nbio_indirect, 14, 15)
  _dump_direct_regs(view, "mmhub", _reg_table_regs(am_regs.mmhub_3_0_0, mmhub_names))
  try:
    dev = _open_discovery_only_amdev(pci)
    stamp(f"snapshot discovery-only ipver gc={dev.ip_ver.get(am.GC_HWIP)} mmhub={dev.ip_ver.get(am.MMHUB_HWIP)} nbio={dev.ip_ver.get(am.NBIO_HWIP)} mp0={dev.ip_ver.get(am.MP0_HWIP)}")
    _dump_amdev_regs(dev, "mmhub-discovery", mmhub_names)
  except Exception as e:
    stamp(f"snapshot discovery-only failed={type(e).__name__}: {e}")

  memsize = view[0xde3] << 20
  bar0_base, bar0_size = pci.bar_info(0)
  stamp(f"snapshot vram memsize={memsize:#x} visible_bar0_size={bar0_size:#x}")
  db_off = memsize - 0x100000
  if db_off + 0x100 <= bar0_size:
    bar0 = pci.map_bar(0, fmt="B")
    stamp(f"snapshot runtime_db visible off={db_off:#x} first256={bytes(bar0[db_off:db_off+0x100]).hex()}")
  else:
    stamp(f"snapshot runtime_db not_visible off={db_off:#x} visible_bar0_size={bar0_size:#x}")
  require_visible("psp-pre-kdb-snapshot")
  stamp("snapshot end")

def remote_psp_runtime_db(pci, size:int):
  stamp("runtime-db begin read-only")
  os.environ.setdefault("AM_REMOTE_DISCOVERY_PROFILE", "gfx1100_744c")
  os.environ.setdefault("AM_REMOTE_SMALL_BAR_DISCOVERY", "1")
  dev = _open_discovery_only_amdev(pci)
  size = min(max(size, 0x200), am.PSP_RUNTIME_DB_SIZE_IN_BYTES)
  db_off = dev.vram_size - am.PSP_RUNTIME_DB_OFFSET
  stamp(f"runtime-db vram_size={dev.vram_size:#x} db_off={db_off:#x} read_size={size:#x}")
  blob = dev._read_vram(db_off, (size + 3) & ~3)
  cookie, version = struct.unpack_from("<HH", blob, 0)
  stamp(f"runtime-db header cookie={cookie:#06x} version={version:#06x} valid={cookie == am.PSP_RUNTIME_DB_COOKIE_ID}")
  stamp(f"runtime-db first256={blob[:0x100].hex()}")
  if cookie != am.PSP_RUNTIME_DB_COOKIE_ID:
    require_visible("psp-runtime-db")
    stamp("runtime-db end no-db")
    return

  entry_count = struct.unpack_from("<H", blob, 4)[0]
  stamp(f"runtime-db directory entry_count={entry_count}")
  if entry_count >= am.PSP_RUNTIME_DB_DIAG_ENTRY_MAX_COUNT:
    stamp(f"runtime-db invalid entry_count={entry_count}")
  for i in range(min(entry_count, am.PSP_RUNTIME_DB_DIAG_ENTRY_MAX_COUNT)):
    ent_off = 8 + i * 8
    if ent_off + 8 > len(blob):
      stamp(f"runtime-db entry[{i}] directory_truncated")
      break
    entry_type, offset, ent_size = struct.unpack_from("<IHH", blob, ent_off)
    type_name = am.enum_psp_runtime_entry_type.get(entry_type, f"UNKNOWN_{entry_type}")
    stamp(f"runtime-db entry[{i}] type={type_name}({entry_type}) offset={offset:#x} size={ent_size:#x}")
    if offset + ent_size > len(blob):
      stamp(f"runtime-db entry[{i}] data_not_in_read_window")
      continue
    data = blob[offset:offset + ent_size]
    stamp(f"runtime-db entry[{i}] data={data[:min(len(data), 64)].hex()}")
    if entry_type == am.PSP_RUNTIME_ENTRY_TYPE_BOOT_CONFIG and ent_size >= 8:
      bitmask, reserved = struct.unpack_from("<II", data, 0)
      feats = [name for bit, name in am.enum_psp_runtime_boot_cfg_feature.items() if bitmask & bit]
      stamp(f"runtime-db boot_cfg bitmask={bitmask:#010x} reserved={reserved:#010x} features={','.join(feats) if feats else 'none'}")
    elif entry_type == am.PSP_RUNTIME_ENTRY_TYPE_PPTABLE_ERR_STATUS and ent_size >= 4:
      status = struct.unpack_from("<I", data, 0)[0]
      stamp(f"runtime-db scpm_status={status:#010x}")
  require_visible("psp-runtime-db")
  stamp("runtime-db end")

def remote_nbio_bifc_pcie_write(pci):
  regs = {
    "BIFC_DOORBELL_ACCESS_EN_PF": (53102, 0xfffff),
    "BIFC_GFX_INT_MONITOR_MASK": (59565, 0x7ff),
  }
  stamp("map BAR5")
  view = pci.map_bar(5, fmt="I")
  for name, (reg, val) in regs.items():
    view[14] = reg * 4
    before = view[15]
    view[15] = val
    view[14] = reg * 4
    after = view[15]
    stamp(f"nbio pcie-write {name} reg={reg:#x} before={before:#010x} write={val:#010x} after={after:#010x}")
  require_visible("nbio-bifc-pcie-write")

def remote_nbio_bifc_rsmu_write(pci):
  regs = {
    "BIFC_DOORBELL_ACCESS_EN_PF": (53102, 0xfffff),
    "BIFC_GFX_INT_MONITOR_MASK": (59565, 0x7ff),
  }
  stamp("map BAR5")
  view = pci.map_bar(5, fmt="I")
  for name, (reg, val) in regs.items():
    view[0] = reg * 4
    before = view[1]
    view[1] = val
    view[0] = reg * 4
    after = view[1]
    stamp(f"nbio rsmu-write {name} reg={reg:#x} before={before:#010x} write={val:#010x} after={after:#010x}")
  require_visible("nbio-bifc-rsmu-write")

def remote_reset(pci):
  stamp("reset device")
  pci.reset()
  stamp("reset ok")
  require_visible("reset")

def psp_fw_dump(fname:str):
  blob = memoryview(bytearray(fetch_fw("amdgpu", fname, fw.hashes[fname])))
  chdr = am.struct_common_firmware_header.from_address(mv_address(blob))
  hdr_t = getattr(am, f"struct_psp_firmware_header_v{chdr.header_version_major}_{chdr.header_version_minor}")
  hdr = hdr_t.from_address(mv_address(blob))
  stamp(f"fw file={fname} bytes={len(blob)} sha256={hashlib.sha256(blob).hexdigest()}")
  stamp(f"fw header version={chdr.header_version_major}.{chdr.header_version_minor} ucode_array_offset={hdr.header.ucode_array_offset_bytes:#x} bins={hdr.psp_fw_bin_count}")
  for fw_i in range(hdr.psp_fw_bin_count):
    desc = am.struct_psp_fw_bin_desc.from_address(ctypes.addressof(hdr.psp_fw_bin) + fw_i * ctypes.sizeof(am.struct_psp_fw_bin_desc))
    start = hdr.header.ucode_array_offset_bytes + desc.offset_bytes
    data = blob[start:start+desc.size_bytes]
    name = am.enum_psp_fw_type.get(desc.fw_type, f"UNKNOWN_{desc.fw_type}")
    stamp(f"fw[{fw_i}] type={name}({desc.fw_type}) offset={start:#x} desc_offset={desc.offset_bytes:#x} size={desc.size_bytes:#x} sha256={hashlib.sha256(data).hexdigest()}")
    if desc.fw_type == am.PSP_FW_TYPE_PSP_KDB:
      stamp(f"kdb load_command={am.PSP_BL__LOAD_KEY_DATABASE:#x} first16={bytes(data[:16]).hex()} last16={bytes(data[-16:]).hex()}")

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
  p.add_argument("--stage", choices=("bars", "bar-read", "bar-write", "bar0-read", "bar0-write", "psp-fw", "psp-status", "psp-clean-gate", "psp-pre-kdb-snapshot", "psp-runtime-db", "nbio-status", "nbio-bifc-pcie-write", "nbio-bifc-rsmu-write", "psp-sysmem-probe", "reset", "remote-sysmem", "amd-boot", "all"), default="all")
  p.add_argument("--fw", default="psp_13_0_10_sos.bin", help="PSP firmware file for psp-fw stage")
  p.add_argument("--sizes", default="16384,2097152,16777216", help="comma-separated allocation sizes")
  p.add_argument("--bars", default="0", help="comma-separated BAR indexes for read/write stages")
  p.add_argument("--offsets", default="0", help="comma-separated BAR offsets for read/write stages")
  p.add_argument("--contiguous", action="store_true", help="request contiguous sysmem for psp-sysmem-probe")
  p.add_argument("--repeat", type=int, default=4)
  p.add_argument("--readback", action="store_true", help="read BAR writes back and compare")
  args = p.parse_args()

  os.environ["REMOTE"] = args.remote
  os.environ["DEV"] = "PCI+AMD"
  RemotePCIDevice.reset_stats()
  sizes = [int(x, 0) for x in args.sizes.split(",") if x]
  bars = [int(x, 0) for x in args.bars.split(",") if x]
  offsets = [int(x, 0) for x in args.offsets.split(",") if x]

  stamp(f"start remote={args.remote} stage={args.stage} sizes={sizes} repeat={args.repeat}")
  if args.stage == "psp-fw":
    psp_fw_dump(args.fw)
    sys.exit(0)
  if args.stage == "psp-clean-gate":
    if not mac_gpu_visible():
      print("DIRTY: full hardware restart required", flush=True)
      sys.exit(1)
    try:
      pci = open_remote()
    except Exception as e:
      stamp(f"gate remote open error={type(e).__name__}: {e}")
      print("DIRTY: full hardware restart required", flush=True)
      sys.exit(1)
    sys.exit(remote_psp_clean_gate(pci))
  require_visible("start")
  pci = open_remote()
  if args.stage in ("bar-read", "bar0-read"):
    remote_bar_read(pci, [0] if args.stage == "bar0-read" else bars, offsets, sizes, args.repeat)
  elif args.stage in ("bar-write", "bar0-write"):
    remote_bar_write(pci, [0] if args.stage == "bar0-write" else bars, offsets, sizes, args.repeat, args.readback)
  elif args.stage == "psp-status":
    remote_psp_status(pci)
  elif args.stage == "psp-pre-kdb-snapshot":
    remote_psp_pre_kdb_snapshot(pci)
  elif args.stage == "psp-runtime-db":
    remote_psp_runtime_db(pci, sizes[0])
  elif args.stage == "nbio-status":
    remote_nbio_status(pci)
  elif args.stage == "nbio-bifc-pcie-write":
    remote_nbio_bifc_pcie_write(pci)
  elif args.stage == "nbio-bifc-rsmu-write":
    remote_nbio_bifc_rsmu_write(pci)
  elif args.stage == "psp-sysmem-probe":
    remote_psp_sysmem_probe(pci, sizes[0], args.contiguous)
  elif args.stage == "reset":
    remote_reset(pci)
  else:
    remote_bars(pci)
  if args.stage in ("remote-sysmem", "all"): remote_sysmem(pci, sizes, args.repeat)
  if args.stage in ("amd-boot", "all"): amd_boot_and_alloc(sizes, args.repeat)
  stamp(f"done stats={RemotePCIDevice.stats()} commands={RemotePCIDevice.command_stats()}")
