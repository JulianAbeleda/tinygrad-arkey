from __future__ import annotations
import ctypes, collections, dataclasses, functools, hashlib, array
from tinygrad.helpers import mv_address, getenv, DEBUG, lo32, hi32, fetch_fw
from tinygrad.runtime.autogen import pci
from tinygrad.runtime.autogen.am import am, fw
from tinygrad.runtime.support.amd import AMDReg, import_module, import_asic_regs
from tinygrad.runtime.support.memory import TLSFAllocator, MemoryManager, AddrSpace
from tinygrad.runtime.support.system import PCIDevice
from tinygrad.runtime.support.am.ip import AM_Experiment, AM_IP, AM_SOC, AM_GMC, AM_IH, AM_PSP, AM_SMU, AM_GFX, AM_SDMA

AM_DEBUG = getenv("AM_DEBUG", 0)

@dataclasses.dataclass
class AMRegister(AMDReg):
  adev:AMDev

  def read(self, inst=0): return self.adev.rreg(self.addr[inst])
  def read_bitfields(self, inst=0) -> dict[str, int]: return self.decode(self.read(inst=inst))

  def write(self, _am_val:int=0, inst=0, **kwargs): self.adev.wreg(self.addr[inst], _am_val | self.encode(**kwargs))

  def update(self, inst=0, **kwargs): self.write(self.read(inst=inst) & ~self.fields_mask(*kwargs.keys()), inst=inst, **kwargs)

class AMFirmware:
  def __init__(self, adev):
    self.adev = adev
    def fmt_ver(hwip): return '_'.join(map(str, adev.ip_ver[hwip]))

    # Load SOS firmware
    self.sos_fw = {}

    blob, sos_hdr = self.load_fw(f"psp_{fmt_ver(am.MP0_HWIP)}_sos.bin", versioned_header='struct_psp_firmware_header')
    fw_bin = sos_hdr.psp_fw_bin

    for fw_i in range(sos_hdr.psp_fw_bin_count):
      fw_bin_desc = am.struct_psp_fw_bin_desc.from_address(ctypes.addressof(fw_bin) + fw_i * ctypes.sizeof(am.struct_psp_fw_bin_desc))
      ucode_start_offset = fw_bin_desc.offset_bytes + sos_hdr.header.ucode_array_offset_bytes
      self.sos_fw[fw_bin_desc.fw_type] = blob[ucode_start_offset:ucode_start_offset+fw_bin_desc.size_bytes]

    # Load other fw
    self.ucode_start: dict[str, int] = {}
    self.descs: list[tuple[list[int], memoryview]] = []

    # SMU firmware
    if adev.ip_ver[am.MP1_HWIP] != (13,0,12):
      blob, hdr = self.load_fw(f"smu_{fmt_ver(am.MP1_HWIP)}.bin", versioned_header="struct_smc_firmware_header")
      if self.adev.ip_ver[am.GC_HWIP] >= (11,0,0):
        self.smu_psp_desc = self.desc(blob, hdr.v1_0.header.ucode_array_offset_bytes, hdr.v1_0.header.ucode_size_bytes, am.GFX_FW_TYPE_SMU)
      else:
        p2stables = (am.struct_smc_soft_pptable_entry * hdr.pptable_count).from_buffer(blob[hdr.pptable_entry_offset:])
        for p2stable in p2stables:
          if p2stable.id == (__P2S_TABLE_ID_X:=0x50325358):
            self.descs += [self.desc(blob, p2stable.ppt_offset_bytes, p2stable.ppt_size_bytes, am.GFX_FW_TYPE_P2S_TABLE)]

    # SDMA firmware
    blob, hdr = self.load_fw(f"sdma_{fmt_ver(am.SDMA0_HWIP)}.bin", versioned_header="struct_sdma_firmware_header")
    if hdr.header.header_version_major == 1:
      self.descs += [self.desc(blob, hdr.header.ucode_array_offset_bytes, hdr.header.ucode_size_bytes, am.GFX_FW_TYPE_SDMA0,
                               am.GFX_FW_TYPE_SDMA1, am.GFX_FW_TYPE_SDMA2, am.GFX_FW_TYPE_SDMA3)]
    elif hdr.header.header_version_major == 2:
      self.descs += [self.desc(blob, hdr.ctl_ucode_offset, hdr.ctl_ucode_size_bytes, am.GFX_FW_TYPE_SDMA_UCODE_TH1)]
      self.descs += [self.desc(blob, hdr.header.ucode_array_offset_bytes, hdr.ctx_ucode_size_bytes, am.GFX_FW_TYPE_SDMA_UCODE_TH0)]
    else: self.descs += [self.desc(blob, hdr.header.ucode_array_offset_bytes, hdr.ucode_size_bytes, am.GFX_FW_TYPE_SDMA_UCODE_TH0)]

    # PFP, ME, MEC firmware
    for (fw_name, fw_cnt) in ([('PFP', 1), ('ME', 1)] if self.adev.ip_ver[am.GC_HWIP] >= (12,0,0) else []) + [('MEC', 1)]:
      blob, hdr = self.load_fw(f"gc_{fmt_ver(am.GC_HWIP)}_{fw_name.lower()}.bin", versioned_header="struct_gfx_firmware_header")

      ucode_off = hdr.header.ucode_array_offset_bytes
      if hdr.header.header_version_major == 1:
        # Code
        self.descs += [self.desc(blob, ucode_off, hdr.header.ucode_size_bytes - hdr.jt_size * 4, getattr(am, f'GFX_FW_TYPE_CP_{fw_name}'))]
        # JT
        self.descs += [self.desc(blob, ucode_off + hdr.jt_offset * 4, hdr.jt_size * 4, getattr(am, f'GFX_FW_TYPE_CP_{fw_name}_ME1'))]
      else:
        # Code
        self.descs += [self.desc(blob, ucode_off, hdr.ucode_size_bytes, getattr(am, f'GFX_FW_TYPE_RS64_{fw_name}'))]
        # Stack
        stack_fws = [getattr(am, f'GFX_FW_TYPE_RS64_{fw_name}_P{fwnum}_STACK') for fwnum in range(fw_cnt)]
        self.descs += [self.desc(blob, hdr.data_offset_bytes, hdr.data_size_bytes, *stack_fws)]
        self.ucode_start[fw_name] = hdr.ucode_start_addr_lo | (hdr.ucode_start_addr_hi << 32)

    # IMU firmware
    if self.adev.ip_ver[am.GC_HWIP] >= (11,0,0):
      blob, hdr = self.load_fw(f"gc_{fmt_ver(am.GC_HWIP)}_imu.bin", am.struct_imu_firmware_header_v1_0)
      imu_i_off, imu_i_sz, imu_d_sz = hdr.header.ucode_array_offset_bytes, hdr.imu_iram_ucode_size_bytes, hdr.imu_dram_ucode_size_bytes
      self.descs += [self.desc(blob, imu_i_off, imu_i_sz, am.GFX_FW_TYPE_IMU_I), self.desc(blob, imu_i_off+imu_i_sz, imu_d_sz, am.GFX_FW_TYPE_IMU_D)]

    # RLC firmware
    blob, hdr0, hdr1, hdr2, hdr3 = self.load_fw(f"gc_{fmt_ver(am.GC_HWIP)}_rlc.bin", am.struct_rlc_firmware_header_v2_0,
      am.struct_rlc_firmware_header_v2_1, am.struct_rlc_firmware_header_v2_2, am.struct_rlc_firmware_header_v2_3)

    if hdr0.header.header_version_minor == 1:
      for mem,fmem in [('LIST_SRM_CNTL', 'list_cntl'), ('LIST_GPM_MEM', 'list_gpm'), ('LIST_SRM_MEM', 'list_srm')]:
        off, sz = getattr(hdr1, f'save_restore_{fmem}_offset_bytes'), getattr(hdr1, f'save_restore_{fmem}_size_bytes')
        self.descs += [self.desc(blob, off, sz, getattr(am, f'GFX_FW_TYPE_RLC_RESTORE_{mem}'))]

    if hdr0.header.header_version_minor >= 2:
      for mem,fmem in [('IRAM', 'iram'), ('DRAM_BOOT', 'dram')]:
        off, sz = getattr(hdr2, f'rlc_{fmem}_ucode_offset_bytes'), getattr(hdr2, f'rlc_{fmem}_ucode_size_bytes')
        self.descs += [self.desc(blob, off, sz, getattr(am, f'GFX_FW_TYPE_RLC_{mem}'))]

    if hdr0.header.header_version_minor == 3:
      for mem in ['P', 'V']:
        off, sz = getattr(hdr3, f'rlc{mem.lower()}_ucode_offset_bytes'), getattr(hdr3, f'rlc{mem.lower()}_ucode_size_bytes')
        self.descs += [self.desc(blob, off, sz, getattr(am, f'GFX_FW_TYPE_RLC_{mem}'))]

    self.descs += [self.desc(blob, hdr0.header.ucode_array_offset_bytes, hdr0.header.ucode_size_bytes, am.GFX_FW_TYPE_RLC_G)]

  def load_fw(self, fname:str, *headers, versioned_header:str|None=None):
    blob = memoryview(bytearray(fetch_fw("amdgpu", fname, fw.hashes[fname])))
    if AM_DEBUG >= 1: print(f"am {self.adev.devfmt}: loading firmware {fname}: {hashlib.sha256(blob).hexdigest()}")
    if versioned_header:
      chdr = am.struct_common_firmware_header.from_address(mv_address(blob))
      headers += (getattr(am, versioned_header + f"_v{chdr.header_version_major}_{chdr.header_version_minor}"),)
    return tuple([blob] + [hdr.from_address(mv_address(blob)) for hdr in headers])

  def desc(self, blob:memoryview, offset:int, size:int, *types:int) -> tuple[list[int], memoryview]: return (list(types), blob[offset:offset+size])

class AMPageTableEntry:
  def __init__(self, adev, paddr, lv): self.adev, self.paddr, self.lv, self.entries = adev, paddr, lv, adev.vram.view(paddr, 0x1000, fmt='Q')

  def set_entry(self, entry_id:int, paddr:int, table=False, uncached=False, aspace=AddrSpace.PHYS, snooped=False, frag=0, valid=True):
    is_sys = aspace is AddrSpace.SYS
    if aspace is AddrSpace.PHYS: paddr = self.adev.paddr2xgmi(paddr)
    if paddr & self.adev.gmc.address_space_mask != paddr: raise ValueError(f"Invalid physical address {paddr:#x}")
    self.entries[entry_id] = self.adev.gmc.get_pte_flags(self.lv, table, frag, uncached, is_sys, snooped, valid) | (paddr & 0x0000FFFFFFFFF000)

  def entry(self, entry_id:int) -> int: return self.entries[entry_id]
  def valid(self, entry_id:int) -> bool: return (self.entries[entry_id] & am.AMDGPU_PTE_VALID) != 0
  def address(self, entry_id:int) -> int:
    if self.entries[entry_id] & am.AMDGPU_PTE_SYSTEM != 0: raise ValueError("should not be system address")
    return self.adev.xgmi2paddr(self.entries[entry_id] & 0x0000FFFFFFFFF000)
  def is_page(self, entry_id:int) -> bool: return self.lv == am.AMDGPU_VM_PTB or self.adev.gmc.is_pte_huge_page(self.lv, self.entries[entry_id])
  def supports_huge_page(self, paddr:int): return self.lv >= am.AMDGPU_VM_PDB2

class AMMemoryManager(MemoryManager):
  va_allocator = TLSFAllocator((1 << 44), base=0x200000000000) # global for all devices.

  def on_range_mapped(self):
    # Invalidate TLB after mappings.
    self.dev.gmc.flush_tlb(ip='GC', vmid=0)
    self.dev.gmc.flush_tlb(ip='MM', vmid=0)

class AMDev:
  Version = 0xA0000008

  def __init__(self, pci_dev:PCIDevice, reset_mode=False):
    self.pci_dev, self.devfmt = pci_dev, pci_dev.pcibus
    self._init_trace_raw("before-map-bars")
    if AM_Experiment.trace_map_bar0_last():
      self.mmio = self.pci_dev.map_bar(5, fmt='I')
      self._init_trace_raw("after-map-bar5-first")
      self.doorbell64 = self.pci_dev.map_bar(2, fmt='Q')
      self._init_trace_raw("after-map-bar2")
      self.vram = self.pci_dev.map_bar(0)
      self._init_trace_raw("after-map-bar0-last")
    elif AM_Experiment.trace_map_bar5_first():
      self.mmio = self.pci_dev.map_bar(5, fmt='I')
      self._init_trace_raw("after-map-bar5-first")
      self.vram = self.pci_dev.map_bar(0)
      self._init_trace_raw("after-map-bar0")
      self.doorbell64 = self.pci_dev.map_bar(2, fmt='Q')
      self._init_trace_raw("after-map-bar2")
    else:
      self.vram = self.pci_dev.map_bar(0)
      self._init_trace_raw("after-map-bar0")
      self.doorbell64 = self.pci_dev.map_bar(2, fmt='Q')
      self._init_trace_raw("after-map-bar2")
      self.mmio = self.pci_dev.map_bar(5, fmt='I')
      self._init_trace_raw("after-map-bar5")

    self._init_trace_raw("before-run-discovery")
    self._run_discovery()
    self._init_trace_raw("after-run-discovery")
    self._init_trace_raw("before-build-regs")
    self._build_regs()
    self._init_trace_regs("after-build-regs")

    # AM boot Process:
    # The GPU being passed can be in one of several states: 1. Not initialized. 2. Initialized by amdgpu. 3. Initialized by AM.
    # The 1st and 2nd states require a full GPU setup since their states are unknown. The 2nd state also requires a mode1 reset to
    # reinitialize all components.
    #
    # The 3rd state can be set up partially to optimize boot time. In this case, only the GFX and SDMA IPs need to be initialized.
    # To enable this, AM uses a separate boot memory that is guaranteed not to be overwritten. This physical memory is utilized for
    # all blocks that are initialized only during the initial AM boot.
    # To determine if the GPU is in the third state, AM uses regSCRATCH_REG7 as a flag.
    # To determine if the previous AM session finalized correctly, AM uses regSCRATCH_REG6 as a flag.
    self.is_booting = True # During boot only boot memory can be allocated. This flag is to validate this.
    self._init_trace_regs("before-init-sw")
    self.init_sw(smi_dev=False)
    self._init_trace_regs("after-init-sw")

    self.partial_boot = (self.reg("regSCRATCH_REG7").read() == AMDev.Version) and (getenv("AM_RESET", 0) != 1)
    self._init_trace_regs(f"after-partial-boot-check partial_boot={int(self.partial_boot)}")
    if self.partial_boot and (self.reg("regSCRATCH_REG6").read() != 0 or self.reg(self.gmc.pf_status_reg("GC")).read() != 0):
      if DEBUG >= 2: print(f"am {self.devfmt}: Malformed state. Issuing a full reset.")
      self.partial_boot = False
      self._init_trace_regs("after-malformed-partial-boot")

    # Init hw for IP blocks where it is needed
    if not self.partial_boot:
      if getenv("AM_PRE_PSP_MODE1_RESET", 0):
        smu_alive = self.smu.is_smu_alive()
        if DEBUG >= 2: print(f"am {self.devfmt}: pre-PSP mode1 reset requested smu_alive={smu_alive}")
        if smu_alive:
          self.pci_dev.write_config_flush(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) & ~pci.PCI_COMMAND_MASTER, 2)
          self.smu.mode1_reset()
          self.pci_dev.write_config_flush(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)
      if self.psp.is_sos_alive() and self.smu.is_smu_alive():
        self._init_trace_regs("before-sos-alive-reset")
        self.pci_dev.write_config_flush(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) & ~pci.PCI_COMMAND_MASTER, 2)
        if self.is_hive():
          if reset_mode: return # in reset mode, do not raise
          raise RuntimeError("Malformed state. Use extra/hardware/amdpci/hive_reset.py to reset the hive")
        self.smu.mode1_reset()
        self._init_trace_regs("after-sos-alive-reset")
      self._init_trace_regs("before-bus-master-enable")
      self.pci_dev.write_config_flush(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)
      self._init_trace_regs("after-bus-master-enable")
      if getenv("AM_PSP_BEFORE_GMC", 0):
        if DEBUG >= 2: print(f"am {self.devfmt}: AM_PSP_BEFORE_GMC=1, initializing PSP before GMC")
        self.init_hw(self.soc, self.ih, self.psp, self.gmc, self.smu)
      else:
        self.init_hw(self.soc, self.gmc, self.ih, self.psp, self.smu)

    # Booting done
    self.is_booting = False
    self._init_trace_regs("after-booting-flag-clear")

    # Re-initialize main blocks
    self.init_hw(self.gfx, self.sdma)

    if (max_power:=getenv("AM_POWER_LIMIT", 0.0)) > 0:
      self.smu.set_power_limit(max_power)
      self.smu.set_clocks(level=None)
    else: self.smu.set_clocks(level=-1) # last level, max perf.
    for ip in [self.soc, self.gfx]: ip.set_clockgating_state()
    self.reg("regSCRATCH_REG7").write(AMDev.Version)
    self.reg("regSCRATCH_REG6").write(1) # set initialized state.
    if DEBUG >= 2: print(f"am {self.devfmt}: boot done")

  def _init_trace_enabled(self) -> bool:
    return bool(AM_Experiment.gmc_init_trace())

  def _init_trace_regs(self, label:str):
    if not self._init_trace_enabled(): return
    vals = []
    for reg in ["regMMVM_INVALIDATE_ENG17_SEM", "regMMVM_INVALIDATE_ENG17_REQ", "regMMVM_INVALIDATE_ENG17_ACK",
                "regMMVM_L2_BANK_SELECT_RESERVED_CID2", "regMP0_SMN_C2PMSG_35", "regMP0_SMN_C2PMSG_36", "regMP0_SMN_C2PMSG_81"]:
      if not hasattr(self, reg):
        vals.append(f"{reg}=missing")
        continue
      try: vals += self._init_trace_read_reg(self.reg(reg), reg)
      except Exception as e: vals.append(f"{reg}=read_failed:{type(e).__name__}")
    print(f"am {self.devfmt}: AMDev init {label} " + " ".join(vals), flush=True)

  def _init_trace_read_reg(self, reg, name:str, inst:int=0) -> list[str]:
    val = reg.read(inst=inst)
    vals = [f"{name}={val:#010x}"]
    if name == "regMMVM_INVALIDATE_ENG17_SEM" and val & 0x1:
      reg.write(0, inst=inst)
      vals.append(f"{name}_released=1")
    return vals

  def _init_trace_raw(self, label:str):
    if not self._init_trace_enabled(): return
    if not hasattr(self, "mmio"):
      print(f"am {self.devfmt}: AMDev init {label} BAR5=unmapped", flush=True)
      return
    vals = []
    for reg, name in [(0x16063, "C2PMSG35"), (0x16064, "C2PMSG36"), (0x16091, "C2PMSG81")]:
      try: vals.append(f"{name}={self.mmio[reg]:#010x}")
      except Exception as e: vals.append(f"{name}=read_failed:{type(e).__name__}")
    profile = getattr(self.pci_dev, "is_remote", False) and getenv("AM_REMOTE_DISCOVERY_PROFILE", "") == "gfx1100_744c"
    if profile and not hasattr(self, "regs_offset"):
      from tinygrad.runtime.autogen.am import navi_offsets
      self.regs_offset = {am.MMHUB_HWIP: {0: tuple(getattr(navi_offsets, f"MMHUB_BASE__INST0_SEG{s}", 0) for s in range(9))}}
      self.ip_ver = {am.MMHUB_HWIP: (3,0,0)}
    if hasattr(self, "regs_offset") and hasattr(self, "ip_ver") and am.MMHUB_HWIP in self.ip_ver:
      try:
        mmhub_regs = import_asic_regs("mmhub", self.ip_ver[am.MMHUB_HWIP],
                                      cls=functools.partial(AMRegister, adev=self, bases=self.regs_offset[am.MMHUB_HWIP]))
        for reg in ["regMMVM_INVALIDATE_ENG17_SEM", "regMMVM_INVALIDATE_ENG17_REQ", "regMMVM_INVALIDATE_ENG17_ACK",
                    "regMMVM_L2_BANK_SELECT_RESERVED_CID2"]:
          vals += self._init_trace_read_reg(mmhub_regs[reg], reg)
      except Exception as e:
        vals.append(f"MMHUB_raw=read_failed:{type(e).__name__}")
    if profile and set(getattr(self, "ip_ver", {})) == {am.MMHUB_HWIP}:
      del self.regs_offset, self.ip_ver
    print(f"am {self.devfmt}: AMDev init {label} " + " ".join(vals), flush=True)

  def init_sw(self, smi_dev=False):
    self.smi_dev, self.is_err_state = smi_dev, False

    # Memory manager & firmware
    self._init_trace_regs("init-sw-entry")
    self.mm = AMMemoryManager(self, self.vram_size - self.reserved_vram_size, boot_size=(32 << 20), pt_t=AMPageTableEntry, va_shifts=[12, 21, 30, 39],
      va_bits=48, first_lv=am.AMDGPU_VM_PDB2, va_base=AMMemoryManager.va_allocator.base, reserve_ptable=not self.large_bar,
      palloc_ranges=[(1 << (i + 12), (2 << 20) if i >= 9 else 0x1000) for i in range(9 * (3 - am.AMDGPU_VM_PDB2), -1, -1)])
    self._init_trace_regs("after-memory-manager")
    self.fw = AMFirmware(self)
    self._init_trace_regs("after-firmware")

    # Initialize IP blocks
    self.soc:AM_SOC = AM_SOC(self)
    self.gmc:AM_GMC = AM_GMC(self)
    self.ih:AM_IH = AM_IH(self)
    self.psp:AM_PSP = AM_PSP(self)
    self.smu:AM_SMU = AM_SMU(self)
    self.gfx:AM_GFX = AM_GFX(self)
    self.sdma:AM_SDMA = AM_SDMA(self)

    # Init sw for all IP blocks
    for ip in [self.soc, self.gmc, self.ih, self.psp, self.smu, self.gfx, self.sdma]:
      name = ip.__class__.__name__
      self._init_trace_regs(f"before-{name}.init_sw")
      ip.init_sw()
      self._init_trace_regs(f"after-{name}.init_sw")

  def init_hw(self, *blocks:AM_IP):
    for ip in blocks:
      name = ip.__class__.__name__
      self._init_trace_regs(f"before-{name}.init_hw")
      ip.init_hw()
      self._init_trace_regs(f"after-{name}.init_hw")
      if DEBUG >= 2: print(f"am {self.devfmt}: {ip.__class__.__name__} initialized")

  def fini(self):
    if DEBUG >= 2: print(f"am {self.devfmt}: Finalizing")
    for ip in [self.sdma, self.gfx]: ip.fini_hw()
    self.smu.set_clocks(level=0)
    self.ih.interrupt_handler()
    self.reg("regSCRATCH_REG6").write(self.is_err_state) # set finalized state.

  def recover(self, force=False) -> bool:
    if not force and not self.is_err_state: return False
    if DEBUG >= 3: print(f"am {self.devfmt}: Start recovery")
    self.ih.interrupt_handler()
    self.gfx.reset_mec()
    self.is_err_state = False
    if DEBUG >= 3: print(f"am {self.devfmt}: Recovery complete")
    return True

  def is_hive(self) -> bool: return self.gmc.xgmi_seg_sz > 0

  def paddr2mc(self, paddr:int) -> int: return self.gmc.mc_base + paddr
  def paddr2xgmi(self, paddr:int) -> int: return self.gmc.paddr_base + paddr
  def xgmi2paddr(self, xgmi_paddr:int) -> int: return xgmi_paddr - self.gmc.paddr_base

  def reg(self, reg:str) -> AMRegister: return self.__dict__[reg]

  def rreg(self, reg:int) -> int:
    val = self.indirect_rreg(reg) if reg >= len(self.mmio) else self.mmio[reg]
    if AM_DEBUG >= 4 and getattr(self, '_prev_rreg', None) != (reg, val): print(f"am {self.devfmt}: Reading register {reg:#x} with value {val:#x}")
    self._prev_rreg = (reg, val)
    return val

  def wreg(self, reg:int, val:int):
    if AM_DEBUG >= 4: print(f"am {self.devfmt}: Writing register {reg:#x} with value {val:#x}")
    if reg >= len(self.mmio): self.indirect_wreg(reg, val)
    else: self.mmio[reg] = val

  def wreg_pair(self, reg_base:str, lo_suffix:str, hi_suffix:str, val:int, inst:int=0):
    self.reg(f"{reg_base}{lo_suffix}").write(lo32(val), inst=inst)
    self.reg(f"{reg_base}{hi_suffix}").write(hi32(val), inst=inst)

  def indirect_rreg(self, reg:int) -> int:
    self.reg("regBIF_BX_PF0_RSMU_INDEX").write(reg * 4)
    return self.reg("regBIF_BX_PF0_RSMU_DATA").read()

  def indirect_wreg(self, reg:int, val:int):
    self.reg("regBIF_BX_PF0_RSMU_INDEX").write(reg * 4)
    self.reg("regBIF_BX_PF0_RSMU_DATA").write(val)

  def indirect_wreg_pcie(self, reg:int, val:int, aid:int=0):
    reg_addr = reg * 4 + ((((aid & 0b11) << 32) | (1 << 34)) if aid > 0 else 0)
    self.reg("regBIF_BX0_PCIE_INDEX2").write(lo32(reg_addr))
    if hi32(reg_addr) > 0: self.reg("regBIF_BX0_PCIE_INDEX2_HI").write(hi32(reg_addr) & 0xff)
    self.reg("regBIF_BX0_PCIE_DATA2").write(val)
    if hi32(reg_addr) > 0: self.reg("regBIF_BX0_PCIE_INDEX2_HI").write(0)

  def indirect_rreg_pcie(self, reg:int, aid:int=0) -> int:
    reg_addr = reg * 4 + ((((aid & 0b11) << 32) | (1 << 34)) if aid > 0 else 0)
    self.reg("regBIF_BX0_PCIE_INDEX2").write(lo32(reg_addr))
    if hi32(reg_addr) > 0: self.reg("regBIF_BX0_PCIE_INDEX2_HI").write(hi32(reg_addr) & 0xff)
    val = self.reg("regBIF_BX0_PCIE_DATA2").read()
    if hi32(reg_addr) > 0: self.reg("regBIF_BX0_PCIE_INDEX2_HI").write(0)
    return val

  def _read_vram(self, addr, size) -> bytes:
    if addr % 4 != 0 or size % 4 != 0: raise ValueError(f"Invalid address {addr:#x} or size {size:#x}")
    if getattr(self.pci_dev, "is_remote", False) and not getenv("AM_REMOTE_SMALL_BAR_DISCOVERY", 0):
      raise RuntimeError("remote AMD small-BAR discovery is disabled because the indirect VRAM MMIO path can wedge TinyGPU. "
                         "Set AM_REMOTE_SMALL_BAR_DISCOVERY=1 to force the unsafe path.")
    res = []
    for caddr in range(addr, addr + size, 4):
      self.wreg(0x06, caddr >> 31)
      self.wreg(0x00, (caddr & 0x7FFFFFFF) | 0x80000000)
      res.append(self.rreg(0x01))
    return bytes(array.array('I', res))

  def _write_vram(self, addr:int, data:bytes, *, allow_remote_sparse:bool=False):
    # Keep remote indirect writes bounded to a single page; larger writes can close the TinyGPU RPC connection.
    remote_sparse = allow_remote_sparse and len(data) <= 0x1000
    if getattr(self.pci_dev, "is_remote", False) and not remote_sparse and not getenv("AM_REMOTE_UNSAFE_INDIRECT_VRAM_WRITE", 0):
      raise RuntimeError("remote AMD indirect VRAM writes are disabled because this path can close the TinyGPU RPC connection. "
                         "Set AM_REMOTE_UNSAFE_INDIRECT_VRAM_WRITE=1 to force the unsafe path.")
    if addr % 4 != 0 or len(data) % 4 != 0: raise ValueError(f"Invalid address {addr:#x} or size {len(data):#x}")
    vals = array.array('I')
    vals.frombytes(data)
    for caddr, val in zip(range(addr, addr + len(data), 4), vals):
      self.wreg(0x06, caddr >> 31)
      self.wreg(0x00, (caddr & 0x7FFFFFFF) | 0x80000000)
      self.wreg(0x01, val)

  def _load_remote_discovery_profile(self, profile:str):
    if profile != "gfx1100_744c": raise RuntimeError(f"unknown AM_REMOTE_DISCOVERY_PROFILE={profile!r}")
    from tinygrad.runtime.autogen.am import navi_offsets

    # Some register metadata references segments beyond navi_offsets' explicit bases.
    # Missing segment bases are padded as zero, matching the raw register offsets used for NBIO doorbell registers.
    def bases(prefix:str) -> dict[int, tuple]:
      return {i: seg for i in range(7) if any(seg:=tuple(getattr(navi_offsets, f"{prefix}_BASE__INST{i}_SEG{s}", 0) for s in range(9)))}

    self.large_bar = False
    # NBIO_HWIP/NBIF_HWIP alias the same hw block; real discovery fills both keys, so the profile must too
    # (AM boot reads NBIO_HWIP, AMDDevice reads NBIF_HWIP).
    self.regs_offset = collections.defaultdict(dict, {
      am.GC_HWIP: bases("GC"), am.HDP_HWIP: bases("HDP"), am.MMHUB_HWIP: bases("MMHUB"), am.NBIO_HWIP: bases("NBIO"),
      am.NBIF_HWIP: bases("NBIO"),
      am.MP0_HWIP: bases("MP0"), am.MP1_HWIP: bases("MP1"), am.OSSSYS_HWIP: bases("OSSSYS"), am.SDMA0_HWIP: bases("SDMA0")})
    # IP versions confirmed against this card's real discovery table (pre-bl ipver trace, 2026-06-10):
    # nbio=(4,3,0) gc=(11,0,0) mp0=(13,0,0). The previous MP0/MP1 (13,0,10) selected wrong-ASIC signed
    # PSP/SMU firmware, which the PSP bootloader silently rejected (first-KDB "BL not ready" hang).
    self.ip_ver = {am.GC_HWIP: (11,0,0), am.HDP_HWIP: (6,0,0), am.MMHUB_HWIP: (3,0,0), am.NBIO_HWIP: (4,3,0),
                   am.NBIF_HWIP: (4,3,0),
                   am.MP0_HWIP: (13,0,0), am.MP1_HWIP: (13,0,0), am.OSSSYS_HWIP: (6,0,0), am.SDMA0_HWIP: (6,0,0)}
    self.gc_info = am.struct_gc_info_v1_0()
    self.gc_info.header.table_id, self.gc_info.header.version_major, self.gc_info.header.version_minor = am.GC, 1, 0
    self.gc_info.header.size = ctypes.sizeof(self.gc_info)
    self.gc_info.gc_num_se, self.gc_info.gc_num_wgp0_per_sa, self.gc_info.gc_num_wgp1_per_sa = 6, 2, 2
    self.gc_info.gc_wave_size, self.gc_info.gc_max_waves_per_simd, self.gc_info.gc_max_scratch_slots_per_cu = 32, 16, 32
    self.gc_info.gc_lds_size, self.gc_info.gc_num_sa_per_se, self.gc_info.gc_num_rb_per_se = 64, 2, 4
    self.reserved_vram_size = 64 << 20
    if DEBUG >= 1: print(f"am {self.devfmt}: using remote discovery profile {profile}")

  def _run_discovery(self):
    # NOTE: Fixed register to query memory size without known ip bases to find the discovery table.
    #       The table is located at the end of VRAM - 64KB and is 10KB in size.
    mmRCC_CONFIG_MEMSIZE = 0xde3
    self.vram_size = self.rreg(mmRCC_CONFIG_MEMSIZE) << 20
    self.large_bar = self.vram.nbytes >= self.vram_size
    if getattr(self.pci_dev, "is_remote", False) and (profile:=getenv("AM_REMOTE_DISCOVERY_PROFILE", "")):
      return self._load_remote_discovery_profile(profile)
    tmr_offset, tmr_size = self.vram_size - (64 << 10), (10 << 10)

    disc_tbl = self.vram.view(tmr_offset, tmr_size)[:] if self.large_bar else self._read_vram(tmr_offset, tmr_size)
    self.bhdr = am.struct_binary_header.from_buffer(bytearray(disc_tbl))
    ihdr = am.struct_ip_discovery_header.from_address(ctypes.addressof(self.bhdr) + self.bhdr.table_list[am.IP_DISCOVERY].offset)
    assert self.bhdr.binary_signature == am.BINARY_SIGNATURE and ihdr.signature == am.DISCOVERY_TABLE_SIGNATURE, "discovery signatures mismatch"

    self.regs_offset:dict[int, dict[int, tuple]] = collections.defaultdict(dict)
    self.ip_ver:dict[int, tuple[int, int, int]] = {}

    for num_die in range(ihdr.num_dies):
      dhdr = am.struct_die_header.from_address(ctypes.addressof(self.bhdr) + ihdr.die_info[num_die].die_offset)

      ip_offset = ctypes.addressof(self.bhdr) + ctypes.sizeof(dhdr) + ihdr.die_info[num_die].die_offset
      for _ in range(dhdr.num_ips):
        ip = am.struct_ip_v4.from_address(ip_offset)
        ba = ((ctypes.c_uint64 if ihdr.base_addr_64_bit else ctypes.c_uint32) * ip.num_base_address).from_address(ip_offset + 8)
        for hw_ip in range(1, am.MAX_HWIP):
          if hw_ip in am.hw_id_map and am.hw_id_map[hw_ip] == ip.hw_id:
            self.regs_offset[hw_ip][ip.instance_number] = tuple(list(ba))
            self.ip_ver[hw_ip] = (ip.major, ip.minor, ip.revision)

        ip_offset += 8 + (8 if ihdr.base_addr_64_bit else 4) * ip.num_base_address

    gc_info = am.struct_gc_info_v1_0.from_address(gc_addr:=ctypes.addressof(self.bhdr) + self.bhdr.table_list[am.GC].offset)
    self.gc_info = getattr(am, f"struct_gc_info_v{gc_info.header.version_major}_{gc_info.header.version_minor}").from_address(gc_addr)
    self.reserved_vram_size = (384 << 20) if self.ip_ver[am.GC_HWIP][:2] in {(9,4), (9,5)} else (64 << 20)

  @functools.cached_property
  def hwid_names(self) -> dict[int, str]: return {v:k.removesuffix('_HWID') for k,v in vars(am).items() if k.endswith('_HWID') and isinstance(v, int)}

  def _ip_module(self, prefix:str, hwip): return import_module(prefix, self.ip_ver[hwip])

  def _build_regs(self):
    mods = [("mp", am.MP0_HWIP), ("hdp", am.HDP_HWIP), ("gc", am.GC_HWIP), ("mmhub", am.MMHUB_HWIP), ("osssys", am.OSSSYS_HWIP),
      ("nbio" if self.ip_ver[am.GC_HWIP] < (12,0,0) else "nbif", am.NBIO_HWIP)]
    if self.ip_ver[am.SDMA0_HWIP] in {(4,4,2), (4,4,4)}: mods += [("sdma", am.SDMA0_HWIP)]

    for prefix, hwip in mods:
      self.__dict__.update(import_asic_regs(prefix, self.ip_ver[hwip], cls=functools.partial(AMRegister, adev=self, bases=self.regs_offset[hwip])))
    self.__dict__.update(import_asic_regs('mp', (11, 0, 0), cls=functools.partial(AMRegister, adev=self, bases=self.regs_offset[am.MP1_HWIP])))
