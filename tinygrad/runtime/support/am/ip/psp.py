from tinygrad.runtime.support.am.ip.common import *

class AM_PSP(AM_IP):
  def init_sw(self):
    self.reg_pref = "regMP0_SMN_C2PMSG" if self.adev.ip_ver[am.MP0_HWIP] < (14,0,0) else "regMPASP_SMN_C2PMSG"
    self.msg1_kind = "vram"

    use_vram_msg1 = getattr(self.adev.pci_dev, "is_remote", False) and getenv("AM_PSP_SYSMSG1_VRAM", 0)
    if use_vram_msg1:
      if (msg1_paddr := AM_Experiment.vram_msg1_paddr()) is not None:
        if msg1_paddr % am.PSP_1_MEG != 0: raise ValueError(f"AM_PSP_SYSMSG1_VRAM_PADDR must be 1MB aligned, got {msg1_paddr:#x}")
        if msg1_paddr < 0 or msg1_paddr + am.PSP_1_MEG > self.adev.vram_size:
          raise ValueError(f"AM_PSP_SYSMSG1_VRAM_PADDR={msg1_paddr:#x} is outside VRAM size {self.adev.vram_size:#x}")
        self.msg1_paddr = msg1_paddr
        self.msg1_kind = "vram-forced"
      else:
        self.msg1_paddr = self.adev.mm.palloc(am.PSP_1_MEG, align=am.PSP_1_MEG, zero=False, boot=True)
      self.msg1_addr, self.msg1_view = self.adev.paddr2mc(self.msg1_paddr), self.adev.vram.view(self.msg1_paddr, am.PSP_1_MEG, 'B')
      self.msg1_paddrs = [self.msg1_paddr + i * 0x1000 for i in range(am.PSP_1_MEG // 0x1000)]
      self._trace(f"msg1 vram paddr={self.msg1_paddr:#x} addr={self.msg1_addr:#x} bytes={self.msg1_view.nbytes}")
    elif getattr(self.adev.pci_dev, "is_remote", False) and getenv("AM_PSP_SYSMSG1_GTT", 0):
      raw_view, paddrs = self.adev.pci_dev.alloc_sysmem(2 * am.PSP_1_MEG, contiguous=True)
      if len(paddrs) != 2 * am.PSP_1_MEG // 0x1000: raise ValueError(f"expected 2MB sysmem pages, got {len(paddrs)}")
      if not all(paddr == paddrs[0] + i * 0x1000 for i, paddr in enumerate(paddrs)): raise ValueError("PSP sysmem GTT buffer is not contiguous")
      view_off = (-paddrs[0]) % am.PSP_1_MEG
      if view_off + am.PSP_1_MEG > raw_view.nbytes: raise ValueError(f"failed to find aligned 1MB PSP window in 2MB GTT buffer: {paddrs[0]:#x}")
      self.msg1_view = raw_view.view(view_off, am.PSP_1_MEG)
      self.msg1_addr = self.adev.mm.alloc_vaddr(size=self.msg1_view.nbytes, align=am.PSP_1_MEG)
      self.adev.mm.map_range(self.msg1_addr, self.msg1_view.nbytes, [(paddr, 0x1000) for paddr in paddrs[view_off // 0x1000:][:am.PSP_1_MEG // 0x1000]],
                             AddrSpace.SYS, uncached=True, snooped=True, boot=True)
      self.msg1_kind = "sysmem-gtt"
      self.msg1_paddrs = paddrs[view_off // 0x1000:][:am.PSP_1_MEG // 0x1000]
      self._trace(f"msg1 sysmem gtt raw={paddrs[0]:#x} view_off={view_off:#x} va={self.msg1_addr:#x} pages={len(paddrs)} bytes={self.msg1_view.nbytes}")
    elif getenv("AM_PSP_SYSMSG1_GART", 0):
      raw_view, paddrs = self.adev.pci_dev.alloc_contiguous_sysmem(am.PSP_1_MEG) if getenv("AM_PSP_SYSMSG1_GART_CONTIG", 0) else \
                         self.adev.pci_dev.alloc_sysmem(am.PSP_1_MEG)
      if len(paddrs) != am.PSP_1_MEG // 0x1000: raise ValueError(f"expected 1MB sysmem pages, got {len(paddrs)}")
      if getenv("AM_PSP_SYSMSG1_GART_CONTIG", 0) and not all(paddr == paddrs[0] + i * 0x1000 for i, paddr in enumerate(paddrs)):
        raise ValueError("PSP sysmem GART buffer is not contiguous")
      view_off = 0
      if AM_Experiment.sysmsg1_gart_sort_paddrs():
        order = sorted(range(len(paddrs)), key=paddrs.__getitem__)
        raw_first, raw_last = paddrs[0], paddrs[-1]
        paddrs = [paddrs[i] for i in order]
        raw_view = AM_ReorderedMsg1View(raw_view, order)
        self._trace(f"msg1 sysmem gart sorted paddr order raw_first={raw_first:#x} raw_last={raw_last:#x} "
                    f"sorted_first={paddrs[0]:#x} sorted_last={paddrs[-1]:#x}")
      self.msg1_view = raw_view
      self.msg1_addr = self.adev.gmc.gart_start
      self.msg1_gart_args = (paddrs, view_off, am.PSP_1_MEG)
      self.msg1_kind = "sysmem-gart"
      self.msg1_paddrs = paddrs
      self._trace(f"msg1 sysmem gart raw={paddrs[0]:#x} view_off={view_off:#x} addr={self.msg1_addr:#x} pages={len(paddrs)} bytes={self.msg1_view.nbytes}")
    elif getattr(self.adev.pci_dev, "is_remote", False) and getenv("AM_PSP_SYSMSG1_DMA", 0):
      raw_view, paddrs = self.adev.pci_dev.alloc_sysmem(2 * am.PSP_1_MEG, contiguous=True)
      if len(paddrs) != 2 * am.PSP_1_MEG // 0x1000: raise ValueError(f"expected 2MB sysmem pages, got {len(paddrs)}")
      if not all(paddr == paddrs[0] + i * 0x1000 for i, paddr in enumerate(paddrs)): raise ValueError("PSP sysmem DMA buffer is not contiguous")
      view_off = (-paddrs[0]) % am.PSP_1_MEG
      if view_off + am.PSP_1_MEG > raw_view.nbytes: raise ValueError(f"failed to find aligned 1MB PSP window in 2MB DMA buffer: {paddrs[0]:#x}")
      self.msg1_view = raw_view.view(view_off, am.PSP_1_MEG)
      self.msg1_addr = paddrs[0] + view_off
      self.msg1_kind = "sysmem-dma"
      self.msg1_paddrs = paddrs[view_off // 0x1000:][:am.PSP_1_MEG // 0x1000]
      self._trace(f"msg1 sysmem dma raw={paddrs[0]:#x} view_off={view_off:#x} addr={self.msg1_addr:#x} pages={len(paddrs)} bytes={self.msg1_view.nbytes}")
    elif self.adev.devfmt.startswith("usb:") or (getattr(self.adev.pci_dev, "is_remote", False) and getenv("AM_PSP_SYSMSG1", 0)):
      self.msg1_view, paddrs = self.adev.pci_dev.alloc_sysmem(am.PSP_1_MEG)
      self.msg1_addr = self.adev.mm.alloc_vaddr(size=self.msg1_view.nbytes, align=am.PSP_1_MEG)
      self.adev.mm.map_range(self.msg1_addr, self.msg1_view.nbytes, [(paddr, 0x1000) for paddr in paddrs], AddrSpace.SYS,
                             uncached=True, boot=True)
      self.msg1_kind = "sysmem-va"
      self.msg1_paddrs = paddrs
      self._trace(f"msg1 sysmem va addr={self.msg1_addr:#x} pages={len(paddrs)} bytes={self.msg1_view.nbytes}")
    else:
      self.msg1_paddr = self.adev.mm.palloc(am.PSP_1_MEG, align=am.PSP_1_MEG, zero=False, boot=True)
      self.msg1_addr, self.msg1_view = self.adev.paddr2mc(self.msg1_paddr), self.adev.vram.view(self.msg1_paddr, am.PSP_1_MEG, 'B')
      self.msg1_paddrs = [self.msg1_paddr + i * 0x1000 for i in range(am.PSP_1_MEG // 0x1000)]

    self.cmd_paddr = self.adev.mm.palloc(am.PSP_CMD_BUFFER_SIZE, zero=False, boot=True)
    self.fence_paddr = self.adev.mm.palloc(am.PSP_FENCE_BUFFER_SIZE, zero=True, boot=True)

    self.ring_size = 0x10000
    self.ring_paddr = self.adev.mm.palloc(self.ring_size, zero=False, boot=True)

    self.max_tmr_size, self.tmr_size = 0x1300000, 0
    self.boot_time_tmr = self.adev.ip_ver[am.MP0_HWIP] in {(13,0,6), (13,0,14), (14,0,2), (14,0,3)}
    self.autoload_tmr = self.adev.ip_ver[am.MP0_HWIP] not in {(13,0,6), (13,0,14)}
    self.tmr_paddr = self.adev.mm.palloc(self.max_tmr_size, align=am.PSP_TMR_ALIGNMENT, zero=False, boot=True) if not self.boot_time_tmr else 0
    self.vram_size = self.adev.gmc.fb_end - self.adev.gmc.fb_base + 0x1000000

  def init_hw(self):
    spl_key = am.PSP_FW_TYPE_PSP_SPL if self.adev.ip_ver[am.MP0_HWIP] >= (14,0,0) else am.PSP_FW_TYPE_PSP_KDB
    sos_components = [(am.PSP_FW_TYPE_PSP_KDB, am.PSP_BL__LOAD_KEY_DATABASE), (spl_key, am.PSP_BL__LOAD_TOS_SPL_TABLE),
      (am.PSP_FW_TYPE_PSP_SYS_DRV, am.PSP_BL__LOAD_SYSDRV), (am.PSP_FW_TYPE_PSP_SOC_DRV, am.PSP_BL__LOAD_SOCDRV),
      (am.PSP_FW_TYPE_PSP_INTF_DRV, am.PSP_BL__LOAD_INTFDRV), (am.PSP_FW_TYPE_PSP_DBG_DRV, am.PSP_BL__LOAD_DBGDRV),
      (am.PSP_FW_TYPE_PSP_RAS_DRV, am.PSP_BL__LOAD_RASDRV), (am.PSP_FW_TYPE_PSP_SOS, am.PSP_BL__LOAD_SOSDRV)]

    if not self.is_sos_alive():
      if getenv("AM_PSP_FATAL_QUIRK", 0): self._fatal_error_recovery_quirk()
      if (mem_train:=getenv("AM_PSP_MEM_TRAIN", "")): self._memory_training(mem_train)
      if hasattr(self, "msg1_gart_args"):
        self.msg1_addr = self.adev.gmc.setup_psp_gart(*self.msg1_gart_args)
        self._trace(f"msg1 gart programmed addr={self.msg1_addr:#x}")
      if AM_Experiment.msg1_visibility_probe(): self._msg1_visibility_probe()
      self._trace_pre_bootloader_regs()
      if AM_Experiment.sos_fw_inventory_audit():
        window = AM_Experiment.sos_fw_inventory_audit_bytes()
        if window < 0 or window > 512: raise ValueError(f"AM_PSP_SOS_FW_INVENTORY_AUDIT_BYTES={window} is outside 0..512")
        used = {fw: compid for fw, compid in sos_components}
        self._trace(f"sos fw inventory begin entries={len(self.adev.fw.sos_fw)} used={len(used)} spl_key={am.enum_psp_fw_type.get(spl_key, spl_key)}")
        for fw, data in sorted(self.adev.fw.sos_fw.items()):
          fw_name = am.enum_psp_fw_type.get(fw, fw)
          used_compid = used.get(fw)
          used_s = "none" if used_compid is None else f"{used_compid:#x}"
          self._trace(f"sos fw inventory fw={fw_name} id={fw:#x} used_compid={used_s} size={len(data):#x} "
                      f"sha256={hashlib.sha256(data).hexdigest()} first{window}={data[:window].hex()} last{window}={data[-window:].hex()}")
        self._trace("sos fw inventory end")
        if AM_Experiment.sos_fw_inventory_audit_stop(): raise RuntimeError("AM_PSP_SOS_FW_INVENTORY_AUDIT_STOP stopped before bootloader component loads")
      if AM_Experiment.kdb_header_audit():
        data = self.adev.fw.sos_fw.get(am.PSP_FW_TYPE_PSP_KDB)
        if data is None: raise RuntimeError("AM_PSP_KDB_HEADER_AUDIT requested but PSP_FW_TYPE_PSP_KDB is absent")
        window = AM_Experiment.kdb_header_audit_bytes()
        if window < 0 or window > len(data): raise ValueError(f"AM_PSP_KDB_HEADER_AUDIT_BYTES={window:#x} is outside 0..{len(data):#x}")
        words = [int.from_bytes(data[i:i + 4], "little") for i in range(0, window & ~3, 4)]
        self._trace(f"KDB header audit size={len(data):#x} sha256={hashlib.sha256(data).hexdigest()} window={window:#x}")
        for base in range(0, len(words), 8):
          chunk = ",".join(f"{w:#010x}" for w in words[base:base + 8])
          self._trace(f"KDB header audit dwords off={base * 4:#x} {chunk}")
        needles = {0x640, 0x1700, 0x1710, 0x1d40, len(data)}
        for off in range(0, len(data) - 3, 4):
          val = int.from_bytes(data[off:off + 4], "little")
          if val in needles or (0 < val < len(data) and val % 0x10 == 0):
            self._trace(f"KDB header audit candidate field_off={off:#x} val={val:#x}")
        if AM_Experiment.kdb_header_audit_stop(): raise RuntimeError("AM_PSP_KDB_HEADER_AUDIT_STOP stopped before bootloader component loads")
      if AM_Experiment.kdb_record_audit():
        data = self.adev.fw.sos_fw.get(am.PSP_FW_TYPE_PSP_KDB)
        if data is None: raise RuntimeError("AM_PSP_KDB_RECORD_AUDIT requested but PSP_FW_TYPE_PSP_KDB is absent")
        start, stride, window = AM_Experiment.kdb_record_audit_start(), AM_Experiment.kdb_record_audit_stride(), AM_Experiment.kdb_record_audit_bytes()
        if start < 0 or start >= len(data): raise ValueError(f"AM_PSP_KDB_RECORD_AUDIT_START={start:#x} is outside KDB size {len(data):#x}")
        if stride <= 0 or stride > len(data): raise ValueError(f"AM_PSP_KDB_RECORD_AUDIT_STRIDE={stride:#x} is outside 1..{len(data):#x}")
        if window < 0 or window > 512: raise ValueError(f"AM_PSP_KDB_RECORD_AUDIT_BYTES={window} is outside 0..512")
        self._trace(f"KDB record audit size={len(data):#x} sha256={hashlib.sha256(data).hexdigest()} start={start:#x} stride={stride:#x} window={window:#x}")
        idx, off = 0, start
        while off < len(data):
          rec = data[off:min(off + stride, len(data))]
          dwords = [int.from_bytes(rec[i:i + 4], "little") for i in range(0, min(len(rec), 0x40) & ~3, 4)]
          self._trace(f"KDB record audit rec={idx} off={off:#x} size={len(rec):#x} sha256={hashlib.sha256(rec).hexdigest()} "
                      f"first{window}={rec[:window].hex()} last{window}={rec[-window:].hex()}")
          self._trace(f"KDB record audit rec={idx} dwords={','.join(f'{w:#010x}' for w in dwords)}")
          idx, off = idx + 1, off + stride
        if AM_Experiment.kdb_record_audit_stop(): raise RuntimeError("AM_PSP_KDB_RECORD_AUDIT_STOP stopped before bootloader component loads")
      for fw, compid in sos_components: self._bootloader_load_component(fw, compid)
      if AM_Experiment.bl_pipeline_count():
        reg81 = self.adev.reg(f"{self.reg_pref}_81")
        self._trace(f"bootloader pipeline before sOS wait reg81={reg81.addr[0]:#x} val={reg81.read():#x}")
      self._sos_final_state_audit("before-wait")
      if (delay_ms := AM_Experiment.sos_wait_delay_ms()):
        if delay_ms < 0 or delay_ms > 10000: raise ValueError(f"AM_PSP_SOS_WAIT_DELAY_MS={delay_ms} is outside 0..10000")
        self._trace(f"sOS wait delay ms={delay_ms}")
        time.sleep(delay_ms / 1000)
        if AM_Experiment.bl_pipeline_count():
          reg81 = self.adev.reg(f"{self.reg_pref}_81")
          self._trace(f"bootloader pipeline after sOS wait delay reg81={reg81.addr[0]:#x} val={reg81.read():#x}")
        self._sos_final_state_audit("after-delay")
      try:
        wait_cond(self.is_sos_alive, value=True, msg="sOS failed to start")
      except Exception:
        self._sos_final_state_audit("wait-exception")
        raise
      if AM_Experiment.bl_pipeline_count():
        reg81 = self.adev.reg(f"{self.reg_pref}_81")
        self._trace(f"bootloader pipeline sOS alive reg81={reg81.addr[0]:#x} val={reg81.read():#x}")
      self._sos_final_state_audit("alive")

    self._ring_create()
    if am.PSP_FW_TYPE_PSP_TOC in self.adev.fw.sos_fw: self._tmr_init()

    # SMU fw should be loaded before TMR.
    if hasattr(self.adev.fw, 'smu_psp_desc'): self._load_ip_fw_cmd(*self.adev.fw.smu_psp_desc)
    if not self.boot_time_tmr or not self.autoload_tmr: self._tmr_load_cmd()

    for psp_desc in self.adev.fw.descs: self._load_ip_fw_cmd(*psp_desc)

    if self.adev.ip_ver[am.GC_HWIP] >= (11,0,0): self._rlc_autoload_cmd()
    else: self._load_ip_fw_cmd([am.GFX_FW_TYPE_REG_LIST], self.adev.fw.sos_fw[am.PSP_FW_TYPE_PSP_RL])

  def is_sos_alive(self): return self.adev.reg(f"{self.reg_pref}_81").read() != 0x0

  def _trace_enabled(self) -> bool:
    return getenv("AM_PSP_TRACE", 0) or getenv("AM_PSP_PARITY_TRACE", 0) or AM_Experiment.kdb_fail_capture() or \
      AM_Experiment.mailbox_visibility() or AM_Experiment.kdb_order_barrier() or AM_Experiment.bl_payload_audit() or \
      AM_Experiment.pre_kdb_gart_audit() or AM_Experiment.pre_kdb_linux_final_invalidate() or \
      AM_Experiment.pre_kdb_linux_mmhub_window() or AM_Experiment.pre_kdb_cid2_audit() or \
      AM_Experiment.msg1_full_audit() or AM_Experiment.fw_pri_equiv_audit()

  def _trace(self, msg:str):
    if self._trace_enabled(): print(f"am {self.adev.devfmt}: PSP {msg}", flush=True)

  def _trace_reg(self, name:str, inst:int|None=None):
    if not hasattr(self.adev, name): return
    try:
      reg = self.adev.reg(name)
      val = reg.read() if inst is None else reg.read(inst=inst)
      self._trace(f"reg {name}{'' if inst is None else f'[{inst}]'}={val:#010x}")
      if name.startswith("regMMVM_INVALIDATE_ENG") and name.endswith("_SEM") and val & 0x1:
        reg.write(0) if inst is None else reg.write(0, inst=inst)
        self._trace(f"reg {name}{'' if inst is None else f'[{inst}]'} released after diagnostic read")
    except Exception as e:
      self._trace(f"reg {name}{'' if inst is None else f'[{inst}]'} read failed: {e}")

  def _trace_c2pmsg_regs(self, dense=False):
    regs = range(128) if dense else (33, 35, 36, 64, 67, 69, 70, 71, 81, 90, 92, 115)
    for reg in [f"{self.reg_pref}_{x}" for x in regs]: self._trace_reg(reg)

  def _trace_mmhub_gart_regs(self, inst:int):
    base_regs = [
      "regMMMC_VM_AGP_BASE", "regMMMC_VM_AGP_BOT", "regMMMC_VM_AGP_TOP", "regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR",
      "regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR", "regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB",
      "regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB", "regMMMC_VM_MX_L1_TLB_CNTL", "regMMVM_L2_CNTL",
      "regMMVM_L2_CNTL2", "regMMVM_L2_CNTL3", "regMMVM_L2_CNTL4", "regMMVM_L2_CNTL5",
      "regMMVM_L2_BANK_SELECT_RESERVED_CID2", "regMMVM_L2_PROTECTION_FAULT_CNTL", "regMMVM_L2_PROTECTION_FAULT_CNTL2",
      "regMMVM_L2_PROTECTION_FAULT_STATUS", "regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32",
      "regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32", "regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32",
      "regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32", "regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32",
      "regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32", "regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32",
      "regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32",
    ]
    context_suffixes = [
      "CNTL", "PAGE_TABLE_BASE_ADDR_LO32", "PAGE_TABLE_BASE_ADDR_HI32", "PAGE_TABLE_START_ADDR_LO32",
      "PAGE_TABLE_START_ADDR_HI32", "PAGE_TABLE_END_ADDR_LO32", "PAGE_TABLE_END_ADDR_HI32",
    ]
    invalidate_suffixes = ["ADDR_RANGE_LO32", "ADDR_RANGE_HI32", "REQ", "ACK", "SEM"]
    for reg in [
      *base_regs,
      *(f"regMMVM_CONTEXT{i}_{suffix}" for i in range(16) for suffix in context_suffixes),
      *(f"regMMVM_INVALIDATE_ENG{i}_{suffix}" for i in range(18) for suffix in invalidate_suffixes),
    ]:
      self._trace_reg(reg, inst=inst)

  def _trace_pre_bootloader_regs(self):
    if not (getenv("AM_PSP_TRACE_REGS", 0) or getenv("AM_PSP_PARITY_TRACE", 0)): return
    self._trace(f"pre-bl ipver nbio={self.adev.ip_ver[am.NBIO_HWIP]} gc={self.adev.ip_ver[am.GC_HWIP]} mp0={self.adev.ip_ver[am.MP0_HWIP]}")
    self._trace(f"pre-bl gmc fb_base={self.adev.gmc.fb_base:#x} fb_end={self.adev.gmc.fb_end:#x} mc_base={self.adev.gmc.mc_base:#x} "
                f"gart={self.adev.gmc.gart_start:#x}-{self.adev.gmc.gart_end:#x} vmhubs={self.adev.gmc.vmhubs}")
    self._trace(f"pre-bl msg1 kind={self.msg1_kind} addr={self.msg1_addr:#x} c2p36={self.msg1_addr >> 20:#x} size={self.msg1_view.nbytes:#x}")
    for reg in ["regRCC_DEV0_EPF2_STRAP2", "regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN", "regBIFC_GFX_INT_MONITOR_MASK",
                "regBIFC_DOORBELL_ACCESS_EN_PF", "regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL", "regMMMC_VM_FB_LOCATION_BASE",
                "regMMMC_VM_FB_LOCATION_TOP"]:
      self._trace_reg(reg)
    for i in range(self.adev.gmc.vmhubs): self._trace_mmhub_gart_regs(i)
    self._trace_c2pmsg_regs()

  def _trace_bootloader_snapshot(self, label:str):
    if not getenv("AM_PSP_PARITY_TRACE", 0): return
    self._trace(f"parity snapshot {label} begin")
    self._trace_c2pmsg_regs(dense=AM_Experiment.trace_c2pmsg_dense())
    for i in range(getattr(self.adev.gmc, "vmhubs", 0)): self._trace_mmhub_gart_regs(i)
    self._trace(f"parity snapshot {label} end")

  def _pre_kdb_gart_audit(self, label:str):
    if not AM_Experiment.pre_kdb_gart_audit(): return
    self.adev.gmc.flush_hdp()
    paddrs = getattr(self, "msg1_paddrs", [])
    contiguous = bool(paddrs) and all(paddr == paddrs[0] + i * 0x1000 for i, paddr in enumerate(paddrs))
    self._trace(f"pre-KDB GART audit {label} begin")
    self._trace(f"pre-KDB GART audit msg1 kind={self.msg1_kind} addr={self.msg1_addr:#x} c2p36={self.msg1_addr >> 20:#x} "
                f"bytes={self.msg1_view.nbytes:#x} pages={len(paddrs)} contiguous={int(contiguous)} "
                f"first_paddr={(paddrs[0] if paddrs else 0):#x} last_paddr={(paddrs[-1] if paddrs else 0):#x}")
    if (info := getattr(self, "msg1_gart_info", None)) is not None:
      gart_table, gart_page, page_count = info
      first_pte, last_pte = gart_table[gart_page], gart_table[gart_page + page_count - 1]
      self._trace(f"pre-KDB GART audit pte gart_page={gart_page:#x} pages={page_count:#x} "
                  f"pte0={first_pte:#018x} pte_last={last_pte:#018x}")
    self._trace_c2pmsg_regs(dense=False)
    for i in range(getattr(self.adev.gmc, "vmhubs", 0)): self._trace_mmhub_gart_regs(i)
    self._trace(f"pre-KDB GART audit {label} end")

  def _fmt_reg_value(self, val:int|None) -> str: return "unreadable" if val is None else f"{val:#010x}"

  def _read_reg_value(self, name:str, inst:int|None=None) -> int|None:
    try:
      reg = self.adev.reg(name)
      return reg.read() if inst is None else reg.read(inst=inst)
    except Exception:
      return None

  def _fw_pri_equiv_audit(self, label:str, padded_data:bytes):
    if not AM_Experiment.fw_pri_equiv_audit(): return
    self.adev.gmc.flush_hdp()
    full_data = bytes(self.msg1_view[:self.msg1_view.nbytes])
    tail = full_data[len(padded_data):]
    tail_nonzero_count = sum(x != 0 for x in tail)
    paddrs = getattr(self, "msg1_paddrs", [])
    contiguous = bool(paddrs) and all(paddr == paddrs[0] + i * 0x1000 for i, paddr in enumerate(paddrs))
    first_paddrs = ",".join(f"{paddr:#x}" for paddr in paddrs[:4])
    last_paddrs = ",".join(f"{paddr:#x}" for paddr in paddrs[-4:])
    c2p35 = self._read_reg_value(f"{self.reg_pref}_35")
    c2p36 = self._read_reg_value(f"{self.reg_pref}_36")
    c2p81 = self._read_reg_value(f"{self.reg_pref}_81")
    self._trace(f"fw_pri equivalence {label} msg1 kind={self.msg1_kind} fw_pri_mc={self.msg1_addr:#x} c2p36={self.msg1_addr >> 20:#x} "
                f"bytes={self.msg1_view.nbytes:#x} padded_size={len(padded_data):#x} "
                f"padded_sha256={hashlib.sha256(padded_data).hexdigest()} full_sha256={hashlib.sha256(full_data).hexdigest()} "
                f"tail_zero={int(tail_nonzero_count == 0)} tail_nonzero_count={tail_nonzero_count} "
                f"C2PMSG35={self._fmt_reg_value(c2p35)} C2PMSG36={self._fmt_reg_value(c2p36)} C2PMSG81={self._fmt_reg_value(c2p81)}")
    self._trace(f"fw_pri equivalence {label} paddrs pages={len(paddrs)} contiguous={int(contiguous)} "
                f"first_paddr={(paddrs[0] if paddrs else 0):#x} last_paddr={(paddrs[-1] if paddrs else 0):#x} "
                f"first4={first_paddrs} last4={last_paddrs}")
    if (info := getattr(self, "msg1_gart_info", None)) is not None:
      gart_table, gart_page, page_count = info
      first_pte, last_pte = gart_table[gart_page], gart_table[gart_page + page_count - 1]
      flag_names = [(am.AMDGPU_PTE_VALID, "VALID"), (am.AMDGPU_PTE_SYSTEM, "SYSTEM"), (am.AMDGPU_PTE_SNOOPED, "SNOOPED"),
                    (am.AMDGPU_PTE_EXECUTABLE, "EXEC"), (am.AMDGPU_PTE_READABLE, "READ"), (am.AMDGPU_PTE_WRITEABLE, "WRITE")]
      first_flags = ",".join(name for bit, name in flag_names if first_pte & bit)
      self._trace(f"fw_pri equivalence {label} pte gart_page={gart_page:#x} pages={page_count:#x} "
                  f"pte0={first_pte:#018x} pte_last={last_pte:#018x} flags={first_flags} "
                  f"mtype={(first_pte & am.AMDGPU_PTE_MTYPE_NV10_MASK) >> 48:#x}")
    for inst in range(getattr(self.adev.gmc, "vmhubs", 0)):
      fault_reg = self.adev.gmc.pf_status_reg("MM") if hasattr(self.adev.gmc, "pf_status_reg") else "regMMVM_L2_PROTECTION_FAULT_STATUS"
      vals = {
        "req": self._read_reg_value("regMMVM_INVALIDATE_ENG17_REQ", inst),
        "ack": self._read_reg_value("regMMVM_INVALIDATE_ENG17_ACK", inst),
        "sem": self._read_reg_value("regMMVM_INVALIDATE_ENG17_SEM", inst),
        "cid2": self._read_reg_value("regMMVM_L2_BANK_SELECT_RESERVED_CID2", inst),
        "fault": self._read_reg_value(fault_reg, inst),
      }
      self._trace(f"fw_pri equivalence {label} mmhub inst={inst} "
                  f"req={self._fmt_reg_value(vals['req'])} ack={self._fmt_reg_value(vals['ack'])} "
                  f"sem={self._fmt_reg_value(vals['sem'])} cid2={self._fmt_reg_value(vals['cid2'])} "
                  f"fault={self._fmt_reg_value(vals['fault'])}")

  def _sos_final_state_audit(self, label:str):
    if not AM_Experiment.sos_final_state_audit() and not AM_Experiment.bl_boundary_audit(): return
    parts = []
    for idx in [35, 36, 64, 67, 81, 90, 92, 115]:
      if not hasattr(self.adev, f"{self.reg_pref}_{idx}"): continue
      reg = self.adev.reg(f"{self.reg_pref}_{idx}")
      parts.append(f"{idx}=0x{reg.read():08x}")
    self._trace(f"sOS final state audit {label} {' '.join(parts)}")

  def _kdb_fail_capture_snapshot(self, label:str):
    self._trace(f"kdb fail capture {label} begin")
    self._trace_c2pmsg_regs(dense=True)
    for i in range(getattr(self.adev.gmc, "vmhubs", 0)): self._trace_mmhub_gart_regs(i)
    self._trace(f"kdb fail capture {label} end")

  def _kdb_fail_capture_sample(self, reg35, reg36):
    max_ms, max_reads = AM_Experiment.kdb_fail_capture_ms(), AM_Experiment.kdb_fail_capture_reads()
    focus = [35, 36, 64, 67, 81, 90, 92, 115]
    regs = [(idx, self.adev.reg(f"{self.reg_pref}_{idx}")) for idx in focus if hasattr(self.adev, f"{self.reg_pref}_{idx}")]
    start, last35, reads = time.perf_counter(), None, 0
    self._trace(f"kdb fail capture sample begin max_ms={max_ms} max_reads={max_reads}")
    while reads < max_reads:
      elapsed_ms = (time.perf_counter() - start) * 1000
      if reads and elapsed_ms >= max_ms: break
      val35 = reg35.read()
      reads += 1
      if reads <= 16 or val35 != last35 or val35 & 0x80000000:
        self._trace(f"kdb fail capture reg35 sample read={reads} elapsed_ms={elapsed_ms:.3f} val={val35:#010x}")
      if reads == 1 or reads % 16 == 0 or val35 != last35 or val35 & 0x80000000:
        vals = " ".join(f"{idx}={reg.read():#010x}" for idx, reg in regs)
        self._trace(f"kdb fail capture focus read={reads} elapsed_ms={elapsed_ms:.3f} {vals}")
      last35 = val35
      if val35 != 0xffffffff and val35 & 0x80000000: break
    elapsed_ms = (time.perf_counter() - start) * 1000
    vals = " ".join(f"{idx}={reg.read():#010x}" for idx, reg in regs)
    self._trace(f"kdb fail capture sample end reads={reads} elapsed_ms={elapsed_ms:.3f} reg36_written={reg36.addr[0]:#x} {vals}")

  def _mailbox_visibility_sample(self, label:str, reg35, reg36):
    if not AM_Experiment.mailbox_visibility(): return
    reads, delay_us = AM_Experiment.mailbox_visibility_reads(), AM_Experiment.mailbox_visibility_delay_us()
    if reads <= 0: return
    if reads > 4096: raise ValueError(f"AM_PSP_MAILBOX_VIS_READS={reads} is too large")
    if delay_us < 0: raise ValueError(f"AM_PSP_MAILBOX_VIS_DELAY_US={delay_us} must be non-negative")
    focus = [35, 36, 64, 67, 81, 90, 92, 115]
    regs = [(idx, self.adev.reg(f"{self.reg_pref}_{idx}")) for idx in focus if hasattr(self.adev, f"{self.reg_pref}_{idx}")]
    start = time.perf_counter()
    self._trace(f"mailbox vis {label} begin reads={reads} delay_us={delay_us} hdp_flush={AM_Experiment.mailbox_visibility_hdp_flush()}")
    for read in range(1, reads + 1):
      if AM_Experiment.mailbox_visibility_hdp_flush(): self.adev.gmc.flush_hdp()
      elapsed_ms = (time.perf_counter() - start) * 1000
      vals = " ".join(f"{idx}={reg.read():#010x}" for idx, reg in regs)
      self._trace(f"mailbox vis {label} read={read} elapsed_ms={elapsed_ms:.3f} {vals}")
      if delay_us: time.sleep(delay_us / 1_000_000)
    self._trace(f"mailbox vis {label} end elapsed_ms={(time.perf_counter() - start) * 1000:.3f}")

  def _msg1_visibility_probe(self):
    probe_len = min(0x1000, self.msg1_view.nbytes)
    original = bytes(self.msg1_view[:probe_len])
    pattern = bytes(((i * 37 + 0x5a) & 0xff) for i in range(probe_len))
    paddrs = getattr(self, "msg1_paddrs", [])
    contiguous = bool(paddrs) and all(paddr == paddrs[0] + i * 0x1000 for i, paddr in enumerate(paddrs))
    self._trace(f"msg1 vis probe begin kind={self.msg1_kind} addr={self.msg1_addr:#x} c2p36={self.msg1_addr >> 20:#x} "
                f"bytes={self.msg1_view.nbytes:#x} probe_len={probe_len:#x} pages={len(paddrs)} "
                f"first_paddr={(paddrs[0] if paddrs else 0):#x} last_paddr={(paddrs[-1] if paddrs else 0):#x} contiguous={int(contiguous)}")
    try:
      self.msg1_view[:probe_len] = pattern
      self.adev.gmc.flush_hdp()
      readback = bytes(self.msg1_view[:probe_len])
      if readback != pattern:
        first_bad = next((i for i, (got, exp) in enumerate(zip(readback, pattern)) if got != exp), -1)
        raise RuntimeError(f"PSP msg1 visibility probe mismatch first_bad={first_bad:#x} "
                           f"expected={pattern[first_bad]:#x} actual={readback[first_bad]:#x}")
      self._trace(f"msg1 vis probe ok first={readback[:16].hex()} last={readback[-16:].hex()}")
    finally:
      self.msg1_view[:probe_len] = original
      self.adev.gmc.flush_hdp()

  def _sync_msg1_sysmem(self, label:str, size:int|None=None, force:bool=False):
    if not (force or AM_Experiment.msg1_sysmem_sync()): return
    sync = getattr(self.msg1_view, "sync", None)
    if sync is None:
      self._trace(f"msg1 sysmem sync {label} skipped kind={self.msg1_kind} no-sync-method")
      return
    view = self.msg1_view
    if size is not None and size != self.msg1_view.nbytes and hasattr(self.msg1_view, "view"):
      view = self.msg1_view.view(0, size)
      sync = getattr(view, "sync", sync)
    invalidate = bool(AM_Experiment.msg1_sysmem_sync_invalidate())
    sync(invalidate=invalidate)
    self._trace(f"msg1 sysmem sync {label} kind={self.msg1_kind} bytes={getattr(view, 'nbytes', size or 0):#x} invalidate={int(invalidate)}")

  def _sync_msg1_primary(self, label:str):
    if not AM_Experiment.msg1_primary_sync(): return
    self._sync_msg1_sysmem(f"primary-{label}", self.msg1_view.nbytes, force=True)
    self.adev.gmc.flush_hdp()
    data = bytes(self.msg1_view[:self.msg1_view.nbytes])
    self._trace(f"msg1 primary sync {label} kind={self.msg1_kind} bytes={len(data):#x} sha256={hashlib.sha256(data).hexdigest()} "
                f"first16={data[:16].hex()} last16={data[-16:].hex()}")

  def _msg1_full_audit(self, label:str, padded_size:int):
    if not AM_Experiment.msg1_full_audit(): return
    data = bytes(self.msg1_view[:self.msg1_view.nbytes])
    tail = data[padded_size:]
    tail_nonzero_count = sum(x != 0 for x in tail)
    padded_tail = data[max(0, padded_size - 16):padded_size]
    self._trace(f"msg1 full audit {label} kind={self.msg1_kind} bytes={len(data):#x} padded_size={padded_size:#x} "
                f"sha256={hashlib.sha256(data).hexdigest()} tail_zero={int(tail_nonzero_count == 0)} "
                f"tail_nonzero_count={tail_nonzero_count} first16={data[:16].hex()} "
                f"padded_last16={padded_tail.hex()} tail_first16={tail[:16].hex()} last16={data[-16:].hex()}")

  def _kdb_order_barrier(self, label:str, expected:bytes, reg35=None, reg36=None):
    if not AM_Experiment.kdb_order_barrier(): return
    size, sample_len = len(expected), min(len(expected), 4096)
    self._sync_msg1_sysmem(f"order-{label}", size, force=True)
    self.adev.gmc.flush_hdp()
    readback = bytes(self.msg1_view[:sample_len])
    if readback != expected[:sample_len]:
      first_bad = next((i for i, (got, exp) in enumerate(zip(readback, expected[:sample_len])) if got != exp), -1)
      raise RuntimeError(f"KDB order barrier msg1 mismatch {label} first_bad={first_bad:#x} expected={expected[first_bad]:#x} actual={readback[first_bad]:#x}")
    checksum = sum(readback) & 0xffffffff
    self._trace(f"KDB order barrier {label} msg1 bytes={size:#x} sample={sample_len:#x} checksum={checksum:#010x} "
                f"first={readback[:16].hex()} last={readback[-16:].hex()}")
    if (info := getattr(self, "msg1_gart_info", None)) is not None:
      gart_table, gart_page, page_count = info
      first_pte, last_pte = gart_table[gart_page], gart_table[gart_page + page_count - 1]
      self._trace(f"KDB order barrier {label} pte0={first_pte:#018x} pte_last={last_pte:#018x} pages={page_count:#x}")
    if reg35 is not None and reg36 is not None:
      self._trace(f"KDB order barrier {label} regs reg35={reg35.read():#010x} reg36={reg36.read():#010x}")

  def _wait_for_bootloader(self):
    reg = self.adev.reg(f"{self.reg_pref}_35")
    start = time.perf_counter()
    last_val = None
    reads = 0
    first_zero_ms = None
    next_trace_ms = 0
    trace_every_ms = AM_Experiment.wait_trace_ms()
    while (time.perf_counter() - start) * 1000 < 10000:
      elapsed_ms = (time.perf_counter() - start) * 1000
      val = reg.read()
      reads += 1
      # Linux-good traces show C2PMSG35 can pass through 0 before the ready value.
      if val == 0 and first_zero_ms is None: first_zero_ms = elapsed_ms
      if self._trace_enabled() and (val != last_val or (trace_every_ms and elapsed_ms >= next_trace_ms)):
        self._trace(f"wait BL reg35={reg.addr[0]:#x} val={val:#x} elapsed_ms={elapsed_ms:.3f} reads={reads}")
        if trace_every_ms: next_trace_ms = elapsed_ms + trace_every_ms
      last_val = val
      if val != 0xffffffff and (val == 0x80000000 if AM_Experiment.exact_bootloader_wait() else val & 0x80000000):
        self._trace(f"wait BL ready elapsed_ms={(time.perf_counter() - start) * 1000:.3f} reads={reads} first_zero_ms={first_zero_ms}")
        return 0x80000000
    self._trace(f"wait BL timeout elapsed_ms={(time.perf_counter() - start) * 1000:.3f} reads={reads} first_zero_ms={first_zero_ms} last_val={last_val:#x}")
    self._trace_bootloader_snapshot("wait-timeout")
    raise TimeoutError(f"BL not ready. Timed out after 10000 ms, condition not met: {last_val & 0x80000000 if last_val is not None else None} != 2147483648")

  def _fatal_error_recovery_quirk(self):
    if self.adev.ip_ver[am.MP0_HWIP] != (13,0,10): return
    reg = self.adev.reg(f"{self.reg_pref}_67")
    val = reg.read()
    self._trace(f"fatal quirk reg67={reg.addr[0]:#x} old={val:#x} new={val + 0x10:#x}")
    reg.write(val + 0x10)
    time.sleep(1.0)

  def _pre_kdb_invalidate_burst(self, count:int):
    if count <= 0: return
    if count > 256: raise ValueError(f"AM_PSP_PRE_KDB_INVALIDATE_BURST={count} is too large")
    self._trace(f"pre-KDB invalidate burst count={count}")
    for i in range(count):
      self.adev.gmc.flush_hdp()
      for inst in range(self.adev.gmc.vmhubs):
        self.adev.reg("regMMVM_INVALIDATE_ENG17_REQ").write(0xf80001, inst=inst)
        self.adev.reg("regMMVM_INVALIDATE_ENG17_SEM").write(0x0, inst=inst)
        self.adev.reg("regMMVM_L2_BANK_SELECT_RESERVED_CID2").write(0x12104010, inst=inst)
        if self._trace_enabled():
          ack = self.adev.reg("regMMVM_INVALIDATE_ENG17_ACK").read(inst=inst)
          sem = self.adev.reg("regMMVM_INVALIDATE_ENG17_SEM").read(inst=inst)
          cid2 = self.adev.reg("regMMVM_L2_BANK_SELECT_RESERVED_CID2").read(inst=inst)
          fault = self.adev.reg(self.adev.gmc.pf_status_reg("MM")).read(inst=inst)
          self._trace(f"pre-KDB invalidate burst pass={i} inst={inst} ack={ack:#010x} sem={sem:#010x} cid2={cid2:#010x} fault={fault:#010x}")

  def _pre_kdb_linux_final_invalidate(self):
    if not AM_Experiment.pre_kdb_linux_final_invalidate(): return
    cid2_val = AM_Experiment.pre_kdb_linux_final_cid2()
    self.adev.gmc.flush_hdp()
    self._trace(f"pre-KDB linux final invalidate cid2={cid2_val:#010x}")
    for inst in range(self.adev.gmc.vmhubs):
      self.adev.reg("regMMVM_INVALIDATE_ENG17_REQ").write(0xf80001, inst=inst)
      self.adev.reg("regMMVM_INVALIDATE_ENG17_SEM").write(0x0, inst=inst)
      self.adev.reg("regMMVM_L2_BANK_SELECT_RESERVED_CID2").write(cid2_val, inst=inst)
      if self._trace_enabled():
        ack = self.adev.reg("regMMVM_INVALIDATE_ENG17_ACK").read(inst=inst)
        sem = self.adev.reg("regMMVM_INVALIDATE_ENG17_SEM").read(inst=inst)
        cid2 = self.adev.reg("regMMVM_L2_BANK_SELECT_RESERVED_CID2").read(inst=inst)
        fault = self.adev.reg(self.adev.gmc.pf_status_reg("MM")).read(inst=inst)
        self._trace(f"pre-KDB linux final invalidate inst={inst} ack={ack:#010x} sem={sem:#010x} cid2={cid2:#010x} fault={fault:#010x}")
    self.adev.gmc.flush_hdp()

  def _pre_kdb_linux_mmhub_window(self):
    if not AM_Experiment.pre_kdb_linux_mmhub_window(): return
    self.adev.gmc.flush_hdp()
    # Linux-good trace window immediately before the first KDB mailbox write. These are raw MMHUB
    # register dword addresses from the trace, kept as an isolated experiment so normal init stays untouched.
    writes = [
      (0x1a740, 0x01fffe01),
      (0x1a712, 0xffffffff), (0x1a713, 0x0000000f), (0x1a714, 0x00000000), (0x1a715, 0x00000000),
      (0x1a716, 0x00000000), (0x1a717, 0x00000000),
      (0x1a741, 0x01fffe07), (0x1a7cd, 0x00000000), (0x1a7ce, 0x00000000),
      (0x1a7ed, 0xffffffff), (0x1a7ee, 0x0000000f),
      (0x1a742, 0x01fffe07), (0x1a7cf, 0x00000000), (0x1a7d0, 0x00000000),
      (0x1a7ef, 0xffffffff), (0x1a7f0, 0x0000000f),
      (0x1a743, 0x01fffe07), (0x1a7d1, 0x00000000), (0x1a7d2, 0x00000000),
      (0x1a7f1, 0xffffffff), (0x1a7f2, 0x0000000f),
      (0x1a744, 0x01fffe07), (0x1a7d3, 0x00000000), (0x1a7d4, 0x00000000),
      (0x1a7f3, 0xffffffff), (0x1a7f4, 0x0000000f),
      (0x1a745, 0x01fffe07), (0x1a7d5, 0x00000000), (0x1a7d6, 0x00000000),
      (0x1a7f5, 0xffffffff), (0x1a7f6, 0x0000000f),
      (0x1a746, 0x01fffe07), (0x1a7d7, 0x00000000), (0x1a7d8, 0x00000000),
      (0x1a7f7, 0xffffffff), (0x1a7f8, 0x0000000f),
      (0x1a747, 0x01fffe07), (0x1a7d9, 0x00000000), (0x1a7da, 0x00000000),
      (0x1a7f9, 0xffffffff), (0x1a7fa, 0x0000000f),
      (0x1a748, 0x01fffe07), (0x1a7db, 0x00000000), (0x1a7dc, 0x00000000),
      (0x1a7fb, 0xffffffff), (0x1a7fc, 0x0000000f),
      (0x1a749, 0x01fffe07), (0x1a7dd, 0x00000000), (0x1a7de, 0x00000000),
      (0x1a7fd, 0xffffffff), (0x1a7fe, 0x0000000f),
      (0x1a74a, 0x01fffe07), (0x1a7df, 0x00000000), (0x1a7e0, 0x00000000),
      (0x1a7ff, 0xffffffff), (0x1a800, 0x0000000f),
      (0x1a74b, 0x01fffe07), (0x1a7e1, 0x00000000), (0x1a7e2, 0x00000000),
      (0x1a74c, 0x01fffe07), (0x1a7e3, 0x00000000), (0x1a7e4, 0x00000000),
      (0x1a74d, 0x01fffe07), (0x1a7e5, 0x00000000), (0x1a7e6, 0x00000000),
      (0x1a74e, 0x01fffe07), (0x1a7e7, 0x00000000), (0x1a7e8, 0x00000000),
      (0x1a74f, 0x01fffe07), (0x1a7e9, 0x00000000), (0x1a7ea, 0x00000000),
      *[(r, v) for reg in range(0x1a787, 0x1a7ab, 2) for r, v in ((reg, 0xffffffff), (reg + 1, 0x0000001f))],
      (0x1a708, 0x3ffffffc), (0x1a774, 0x00f80001), (0x1a762, 0x00000000), (0x1a71b, 0x12104010),
    ]
    self._trace(f"pre-KDB linux MMHUB window begin raw_writes={len(writes)}")
    for reg, val in writes: self.adev.wreg(reg, val)
    if self._trace_enabled():
      vals = []
      for reg in (0x1a740, 0x1a741, 0x1a74e, 0x1a74f, 0x1a708, 0x1a774, 0x1a786, 0x1a762, 0x1a71b):
        try: vals.append(f"{reg:#x}={self.adev.rreg(reg):#010x}")
        except Exception as e: vals.append(f"{reg:#x}=ERR:{e}")
      self._trace(f"pre-KDB linux MMHUB window readback {' '.join(vals)}")
    self.adev.gmc.flush_hdp()
    self._trace("pre-KDB linux MMHUB window end")

  def _pre_kdb_cid2_audit(self):
    if not AM_Experiment.pre_kdb_cid2_audit(): return
    self.adev.gmc.flush_hdp()
    self._trace("pre-KDB CID2 audit begin")

    def sample(inst:int, label:str):
      req = self.adev.reg("regMMVM_INVALIDATE_ENG17_REQ").read(inst=inst)
      ack = self.adev.reg("regMMVM_INVALIDATE_ENG17_ACK").read(inst=inst)
      sem = self.adev.reg("regMMVM_INVALIDATE_ENG17_SEM").read(inst=inst)
      cid2 = self.adev.reg("regMMVM_L2_BANK_SELECT_RESERVED_CID2").read(inst=inst)
      fault = self.adev.reg(self.adev.gmc.pf_status_reg("MM")).read(inst=inst)
      self._trace(f"pre-KDB CID2 audit {label} inst={inst} req={req:#010x} ack={ack:#010x} sem={sem:#010x} cid2={cid2:#010x} fault={fault:#010x}")

    for inst in range(self.adev.gmc.vmhubs):
      sample(inst, "initial")
      self.adev.reg("regMMVM_L2_BANK_SELECT_RESERVED_CID2").write(0x12104010, inst=inst)
      sample(inst, "after-cid2-12104010")
      self.adev.reg("regMMVM_L2_BANK_SELECT_RESERVED_CID2").write(0x10104010, inst=inst)
      sample(inst, "after-cid2-10104010")
      self.adev.reg("regMMVM_INVALIDATE_ENG17_REQ").write(0xf80001, inst=inst)
      self.adev.reg("regMMVM_INVALIDATE_ENG17_SEM").write(0x0, inst=inst)
      self.adev.reg("regMMVM_L2_BANK_SELECT_RESERVED_CID2").write(0x12104010, inst=inst)
      sample(inst, "after-req-sem-cid2-12104010")
      self.adev.reg("regMMVM_L2_BANK_SELECT_RESERVED_CID2").write(0x12104010, inst=inst)
      self.adev.reg("regMMVM_INVALIDATE_ENG17_REQ").write(0xf80001, inst=inst)
      self.adev.reg("regMMVM_INVALIDATE_ENG17_SEM").write(0x0, inst=inst)
      sample(inst, "after-cid2-req-sem-12104010")
    self.adev.gmc.flush_hdp()
    self._trace("pre-KDB CID2 audit end")

  def _memory_training_offsets(self) -> tuple[int, int, int]:
    reserve_size = getenv("AM_PSP_MEM_TRAIN_RESERVE", 64 << 10)
    c2p = ((self.vram_size - reserve_size - am.PSP_1_MEG + am.PSP_1_MEG - 1) // am.PSP_1_MEG) * am.PSP_1_MEG
    p2c = self.vram_size - am.GDDR6_MEM_TRAINING_OFFSET
    return c2p, p2c, reserve_size

  def _memory_training(self, mode:str):
    if self.adev.ip_ver[am.MP0_HWIP][0] != 13: return
    mode = mode.lower()
    if mode not in {"long", "short"}: raise ValueError(f"AM_PSP_MEM_TRAIN must be long or short, got {mode!r}")
    c2p, p2c, reserve_size = self._memory_training_offsets()
    msg = am.PSP_BL__DRAM_LONG_TRAIN if mode == "long" else am.PSP_BL__DRAM_SHORT_TRAIN
    self._trace(f"mem train mode={mode} msg={msg:#x} c2p={c2p:#x} p2c={p2c:#x} reserve={reserve_size:#x} vram_size={self.vram_size:#x}")

    saved_bottom = None
    if mode == "long":
      chunk = getenv("AM_PSP_MEM_TRAIN_COPY_CHUNK", am.PSP_1_MEG)
      saved_bottom = bytearray()
      for off in range(0, am.BIST_MEM_TRAINING_ENCROACHED_SIZE, chunk):
        saved_bottom += self.adev.vram.view(off, min(chunk, am.BIST_MEM_TRAINING_ENCROACHED_SIZE - off), 'B')[:]
      self._trace(f"mem train saved bottom vram bytes={len(saved_bottom):#x}")

    reg36, reg35 = self.adev.reg(f"{self.reg_pref}_36"), self.adev.reg(f"{self.reg_pref}_35")
    reg36.write(c2p >> 20)
    reg35.write(msg)
    self._trace(f"mem train write reg36={c2p >> 20:#x} reg35={msg:#x}")

    start = time.perf_counter()
    last_val = None
    while (time.perf_counter() - start) * 1000 < 3000:
      val = reg35.read()
      if self._trace_enabled() and val != last_val:
        self._trace(f"mem train wait reg35={reg35.addr[0]:#x} val={val:#x}")
        last_val = val
      if val != 0xffffffff and val & 0x80000000: break
    else:
      raise TimeoutError(f"PSP memory training {mode} timed out, last reg35={last_val}")

    if saved_bottom is not None:
      chunk = getenv("AM_PSP_MEM_TRAIN_COPY_CHUNK", am.PSP_1_MEG)
      for off in range(0, len(saved_bottom), chunk):
        self.adev.vram.view(off, min(chunk, len(saved_bottom) - off), 'B')[:] = saved_bottom[off:off + chunk]
      self.adev.gmc.flush_hdp()
      self._trace(f"mem train restored bottom vram bytes={len(saved_bottom):#x}")

  def _prep_msg1(self, data:memoryview):
    if len(data) > self.msg1_view.nbytes: raise ValueError(f"msg1 buffer is too small {len(data):#x} > {self.msg1_view.nbytes:#x}")
    padded_data = pad_bytes(bytes(data) + b'\x00' * 4, 16) # HACK: apple's memcpy requires 16-bytes alignment
    if getenv("AM_PSP_ZERO_MSG1", 0):
      self.msg1_view[:] = b'\x00' * self.msg1_view.nbytes
      self._trace(f"msg1 zeroed bytes={self.msg1_view.nbytes}")
    self.msg1_view[:len(padded_data)] = padded_data
    if getenv("AM_PSP_MSG1_READBACK", 0):
      readback = bytes(self.msg1_view[:len(padded_data)])
      if readback != padded_data:
        first_bad = next((i for i, (got, exp) in enumerate(zip(readback, padded_data)) if got != exp), -1)
        mismatch_count = sum(got != exp for got, exp in zip(readback, padded_data))
        exp = padded_data[first_bad] if first_bad >= 0 else None
        got = readback[first_bad] if first_bad >= 0 else None
        raise RuntimeError("PSP msg1 readback mismatch "
          f"bytes={len(readback)} mismatches={mismatch_count} first_bad={first_bad:#x} "
          f"expected={exp:#x} actual={got:#x} "
          f"expected_first32={padded_data[:32].hex()} actual_first32={readback[:32].hex()} "
          f"expected_last32={padded_data[-32:].hex()} actual_last32={readback[-32:].hex()}")
      self._trace(f"msg1 readback ok bytes={len(readback)} first={readback[:16].hex()} last={readback[-16:].hex()}")
    self._sync_msg1_primary("prep")
    self._sync_msg1_sysmem("prep", len(padded_data))
    self.adev.gmc.flush_hdp()
    self._msg1_full_audit("prep", len(padded_data))
    if getenv("AM_PSP_STRONG_FLUSH", 0):
      for i in range(3):
        self.adev.gmc.flush_hdp()
        if getenv("AM_PSP_MSG1_READBACK", 0):
          rb = bytes(self.msg1_view[:min(len(padded_data), 4096)])
          if rb != padded_data[:len(rb)]: raise RuntimeError(f"PSP msg1 strong readback mismatch pass={i}")
        _ = self.adev.reg(f"{self.reg_pref}_35").read()
      time.sleep(0.001)
      self._trace("msg1 strong flush complete")
    return padded_data

  def _kdb_payload_audit(self, payload:bytes, padded_data:bytes):
    if not AM_Experiment.kdb_payload_audit(): return
    window = AM_Experiment.kdb_payload_audit_bytes()
    if window < 0 or window > 512: raise ValueError(f"AM_PSP_KDB_PAYLOAD_AUDIT_BYTES={window} is outside 0..512")
    words = [f"{int.from_bytes(padded_data[i:i + 4], 'little'):#010x}" for i in range(0, min(len(padded_data), 64), 4)]
    self._trace(f"KDB payload audit payload_size={len(payload):#x} padded_size={len(padded_data):#x} "
                f"payload_sha256={hashlib.sha256(payload).hexdigest()} padded_sha256={hashlib.sha256(padded_data).hexdigest()}")
    self._trace(f"KDB payload audit bytes first{window}={padded_data[:window].hex()} last{window}={padded_data[-window:].hex()}")
    self._trace(f"KDB payload audit dwords_le first{len(words)}={','.join(words)}")

  def _bootloader_payload_audit(self, fw:int, compid:int, payload:bytes, padded_data:bytes):
    if not AM_Experiment.bl_payload_audit(): return
    window = AM_Experiment.bl_payload_audit_bytes()
    if window < 0 or window > 512: raise ValueError(f"AM_PSP_BL_PAYLOAD_AUDIT_BYTES={window} is outside 0..512")
    fw_name = am.enum_psp_fw_type.get(fw, fw)
    words = [f"{int.from_bytes(padded_data[i:i + 4], 'little'):#010x}" for i in range(0, min(len(padded_data), 64), 4)]
    self._trace(f"bootloader payload audit fw={fw_name} compid={compid:#x} payload_size={len(payload):#x} padded_size={len(padded_data):#x} "
                f"payload_sha256={hashlib.sha256(payload).hexdigest()} padded_sha256={hashlib.sha256(padded_data).hexdigest()}")
    self._trace(f"bootloader payload audit bytes fw={fw_name} compid={compid:#x} "
                f"first{window}={padded_data[:window].hex()} last{window}={padded_data[-window:].hex()}")
    self._trace(f"bootloader payload audit dwords_le fw={fw_name} compid={compid:#x} first{len(words)}={','.join(words)}")

  def _bootloader_metadata_audit(self, fw:int, compid:int, raw_data:bytes, payload:bytes, padded_data:bytes, source:str, offset:int):
    if not AM_Experiment.bl_metadata_audit(): return
    window = AM_Experiment.bl_metadata_audit_bytes()
    if window < 0 or window > 512: raise ValueError(f"AM_PSP_BL_METADATA_AUDIT_BYTES={window} is outside 0..512")
    fw_name = am.enum_psp_fw_type.get(fw, fw)
    self._trace(f"bootloader metadata audit fw={fw_name} compid={compid:#x} source={source} "
                f"raw_size={len(raw_data):#x} selected_offset={offset:#x} selected_size={len(payload):#x} "
                f"padded_size={len(padded_data):#x} raw_sha256={hashlib.sha256(raw_data).hexdigest()} "
                f"payload_sha256={hashlib.sha256(payload).hexdigest()} padded_sha256={hashlib.sha256(padded_data).hexdigest()}")
    self._trace(f"bootloader metadata audit bytes fw={fw_name} compid={compid:#x} "
                f"raw_first{window}={raw_data[:window].hex()} raw_last{window}={raw_data[-window:].hex()} "
                f"payload_first{window}={payload[:window].hex()} payload_last{window}={payload[-window:].hex()}")

  def _bootloader_load_component(self, fw:int, compid:int):
    if fw not in self.adev.fw.sos_fw: return 0

    if AM_Experiment.linux_pre_bl_status():
      reg81 = self.adev.reg(f"{self.reg_pref}_81")
      self._trace(f"linux pre-bl status reg81={reg81.addr[0]:#x} val={reg81.read():#x}")
    if getattr(self, "_skip_next_bootloader_prewait", False):
      self._skip_next_bootloader_prewait = False
      pipeline_name = "KDB pipeline" if AM_Experiment.kdb_pipeline_seq() else "bootloader pipeline"
      self._trace(f"{pipeline_name} skip prewait for fw={am.enum_psp_fw_type.get(fw, fw)} compid={compid:#x}")
    else:
      self._wait_for_bootloader()

    if DEBUG >= 2: print(f"am {self.adev.devfmt}: loading sos component: {am.enum_psp_fw_type.get(fw)}")
    raw_data = self.adev.fw.sos_fw[fw]
    data, source, selected_offset = raw_data, "raw", 0
    kdb_skip_compids = {am.PSP_BL__LOAD_KEY_DATABASE, am.PSP_BL__LOAD_TOS_SPL_TABLE}
    if fw == am.PSP_FW_TYPE_PSP_KDB and compid in kdb_skip_compids:
      if (slice_off := AM_Experiment.kdb_slice_offset()) is not None:
        slice_size = AM_Experiment.kdb_slice_size()
        end = slice_off + slice_size if slice_size is not None else len(raw_data)
        if slice_off < 0 or end < slice_off or end > len(raw_data):
          raise ValueError(f"AM_PSP_KDB_SLICE_OFFSET={slice_off:#x} AM_PSP_KDB_SLICE_SIZE={slice_size} exceeds KDB bytes={len(raw_data):#x}")
        self._trace(f"KDB slice offset={slice_off:#x} size={end - slice_off:#x} old_size={len(raw_data):#x}")
        data = data[slice_off:end]
        source, selected_offset = "kdb-slice", slice_off
      elif skip := AM_Experiment.kdb_skip_prefix():
        if skip >= len(raw_data): raise ValueError(f"AM_PSP_KDB_SKIP_PREFIX={skip:#x} exceeds KDB bytes={len(raw_data):#x}")
        self._trace(f"KDB skip prefix bytes={skip:#x} old_size={len(raw_data):#x} new_size={len(raw_data) - skip:#x}")
        data = data[skip:]
        source, selected_offset = "kdb-skip-prefix", skip

    self._trace(f"load component fw={am.enum_psp_fw_type.get(fw, fw)} compid={compid:#x} bytes={len(data)}")
    padded_preview = pad_bytes(bytes(data) + b'\x00' * 4, 16)
    self._bootloader_metadata_audit(fw, compid, bytes(raw_data), bytes(data), padded_preview, source, selected_offset)
    if AM_Experiment.bl_metadata_audit_stop():
      audited = getattr(self, "_bl_metadata_audited", 0) + 1
      self._bl_metadata_audited = audited
      stop_after = AM_Experiment.bl_metadata_audit_stop_after()
      if audited >= stop_after:
        raise RuntimeError(f"AM_PSP_BL_METADATA_AUDIT_STOP stopped before msg1/mailbox writes after {audited} components")
      self._skip_next_bootloader_prewait = True
      return 0
    padded_data = self._prep_msg1(data)
    self._bootloader_payload_audit(fw, compid, bytes(data), padded_data)
    if fw == am.PSP_FW_TYPE_PSP_KDB: self._kdb_payload_audit(bytes(data), padded_data)
    if fw == am.PSP_FW_TYPE_PSP_KDB and AM_Experiment.audit_pre_kdb():
      self._kdb_order_barrier("audit-pre-mailbox", padded_data)
      self._trace_bootloader_snapshot("audit-pre-kdb")
      raise RuntimeError("AM_PSP_AUDIT_PRE_KDB stopped before KDB mailbox writes")
    if fw == am.PSP_FW_TYPE_PSP_KDB:
      self._pre_kdb_invalidate_burst(AM_Experiment.pre_kdb_invalidate_burst())
      self._pre_kdb_gart_audit("pre-mailbox")
      if AM_Experiment.pre_kdb_gart_audit_stop():
        raise RuntimeError("AM_PSP_PRE_KDB_GART_AUDIT_STOP stopped before KDB mailbox writes")
      self._pre_kdb_cid2_audit()
      if AM_Experiment.pre_kdb_cid2_audit_stop():
        raise RuntimeError("AM_PSP_PRE_KDB_CID2_AUDIT_STOP stopped before KDB mailbox writes")
      self._pre_kdb_linux_mmhub_window()
      self._pre_kdb_linux_final_invalidate()
      self._fw_pri_equiv_audit("pre-mailbox", padded_data)
    reg36, reg35 = self.adev.reg(f"{self.reg_pref}_36"), self.adev.reg(f"{self.reg_pref}_35")
    if AM_Experiment.mailbox_strong_order():
      self.adev.gmc.flush_hdp()
      self._trace(f"mailbox before-reg36 reg35={reg35.read():#x} reg36={reg36.read():#x}")
    kdb_fail_capture = fw == am.PSP_FW_TYPE_PSP_KDB and AM_Experiment.kdb_fail_capture()
    if kdb_fail_capture and AM_Experiment.kdb_fail_capture_pre_command(): self._kdb_fail_capture_snapshot("pre-command")
    if fw == am.PSP_FW_TYPE_PSP_KDB: self._mailbox_visibility_sample("pre-reg36", reg35, reg36)
    if fw == am.PSP_FW_TYPE_PSP_KDB: self._kdb_order_barrier("pre-reg36", padded_data, reg35, reg36)
    if compid == am.PSP_BL__LOAD_SOSDRV: self._sos_final_state_audit("pre-reg36")
    self._trace(f"write msg1 kind={self.msg1_kind} reg36={reg36.addr[0]:#x} val={self.msg1_addr >> 20:#x} msg1_addr={self.msg1_addr:#x}")
    reg36.write(self.msg1_addr >> 20)
    if compid == am.PSP_BL__LOAD_SOSDRV: self._sos_final_state_audit("post-reg36")
    if fw == am.PSP_FW_TYPE_PSP_KDB: self._mailbox_visibility_sample("post-reg36", reg35, reg36)
    if fw == am.PSP_FW_TYPE_PSP_KDB: self._kdb_order_barrier("post-reg36", padded_data, reg35, reg36)
    if AM_Experiment.mailbox_strong_order() or getenv("AM_PSP_STRONG_FLUSH", 0):
      self.adev.gmc.flush_hdp()
      rb36 = reg36.read()
      rb35 = reg35.read()
      self._trace(f"pre-compid strong barrier reg36={rb36:#x} reg35={rb35:#x}")
      time.sleep(0.001)
    self._trace(f"write compid reg35={reg35.addr[0]:#x} val={compid:#x}")
    reg35.write(compid)
    if fw == am.PSP_FW_TYPE_PSP_KDB and AM_Experiment.kdb_order_barrier(): self.adev.gmc.flush_hdp()
    self._trace(f"write compid done val={compid:#x}")
    if compid == am.PSP_BL__LOAD_SOSDRV: self._sos_final_state_audit("post-compid")
    if fw == am.PSP_FW_TYPE_PSP_KDB: self._mailbox_visibility_sample("post-compid", reg35, reg36)
    if kdb_fail_capture: self._kdb_fail_capture_sample(reg35, reg36)
    if AM_Experiment.mailbox_strong_order():
      self._trace(f"mailbox post-compid reg35={reg35.read():#x} reg36={reg36.read():#x}")
    self._trace_bootloader_snapshot(f"post-compid-{compid:#x}")

    if bl_pipeline_count := AM_Experiment.bl_pipeline_count():
      if bl_pipeline_count < 0 or bl_pipeline_count > 16: raise ValueError(f"AM_PSP_BL_PIPELINE_COUNT={bl_pipeline_count} is outside 0..16")
      pipeline_done = getattr(self, "_bl_pipeline_done", 0)
      if pipeline_done < bl_pipeline_count:
        delay_us = AM_Experiment.bl_pipeline_delay_us()
        if delay_us < 0 or delay_us > 100000: raise ValueError(f"AM_PSP_BL_PIPELINE_DELAY_US={delay_us} is outside 0..100000")
        self._trace(f"bootloader pipeline continue after delay_us={delay_us} count={pipeline_done + 1}/{bl_pipeline_count} "
                    f"fw={am.enum_psp_fw_type.get(fw, fw)} compid={compid:#x}")
        if delay_us: time.sleep(delay_us / 1_000_000)
        self._bl_pipeline_done = pipeline_done + 1
        self._skip_next_bootloader_prewait = True
        return 0

    if fw == am.PSP_FW_TYPE_PSP_KDB and AM_Experiment.kdb_pipeline_seq():
      pipeline_count = AM_Experiment.kdb_pipeline_count()
      if pipeline_count < 0 or pipeline_count > 16: raise ValueError(f"AM_PSP_KDB_PIPELINE_COUNT={pipeline_count} is outside 0..16")
      pipeline_done = getattr(self, "_kdb_pipeline_done", 0)
      if pipeline_done >= pipeline_count: return self._wait_for_bootloader()
      delay_us = AM_Experiment.kdb_pipeline_delay_us()
      if delay_us < 0 or delay_us > 100000: raise ValueError(f"AM_PSP_KDB_PIPELINE_DELAY_US={delay_us} is outside 0..100000")
      self._trace(f"KDB pipeline continue after delay_us={delay_us} count={pipeline_done + 1}/{pipeline_count}")
      if delay_us: time.sleep(delay_us / 1_000_000)
      self._kdb_pipeline_done = pipeline_done + 1
      self._skip_next_bootloader_prewait = True
      return 0

    if compid == am.PSP_BL__LOAD_SOSDRV: return 0
    try:
      if AM_Experiment.bl_boundary_audit(): self._sos_final_state_audit(f"boundary-before-wait-{compid:#x}")
      ret = self._wait_for_bootloader()
      if AM_Experiment.bl_boundary_audit(): self._sos_final_state_audit(f"boundary-after-wait-{compid:#x}")
      return ret
    except Exception:
      if AM_Experiment.bl_boundary_audit(): self._sos_final_state_audit(f"boundary-wait-exception-{compid:#x}")
      if kdb_fail_capture: self._kdb_fail_capture_snapshot("wait-exception")
      raise

  def _tmr_init(self):
    # Load TOC and calculate TMR size
    self._prep_msg1(fwm:=self.adev.fw.sos_fw[am.PSP_FW_TYPE_PSP_TOC])
    self.tmr_size = self._load_toc_cmd(len(fwm)).resp.tmr_size
    if self.tmr_size > self.max_tmr_size: raise ValueError(f"TMR size is too large {self.tmr_size:#x} > {self.max_tmr_size:#x}")

  def _ring_create(self):
    # If the ring is already created, destroy it
    if self.adev.reg(f"{self.reg_pref}_71").read() != 0:
      self.adev.reg(f"{self.reg_pref}_64").write(am.GFX_CTRL_CMD_ID_DESTROY_RINGS)

      # There might be handshake issue with hardware which needs delay
      time.sleep(0.02)

    # Wait until the sOS is ready
    wait_cond(lambda: self.adev.reg(f"{self.reg_pref}_64").read() & 0x80000000, value=0x80000000, msg="sOS not ready")

    self.adev.wreg_pair(self.reg_pref, "_69", "_70", self.adev.paddr2mc(self.ring_paddr))
    self.adev.reg(f"{self.reg_pref}_71").write(self.ring_size)
    self.adev.reg(f"{self.reg_pref}_64").write(am.PSP_RING_TYPE__KM << 16)

    # There might be handshake issue with hardware which needs delay
    time.sleep(0.02)

    wait_cond(lambda: self.adev.reg(f"{self.reg_pref}_64").read() & 0x8000FFFF, value=0x80000000, msg="sOS ring not created")

  def _ring_submit(self, cmd:am.struct_psp_gfx_cmd_resp) -> am.struct_psp_gfx_cmd_resp:
    msg = am.struct_psp_gfx_rb_frame(fence_value=(prev_wptr:=self.adev.reg(f"{self.reg_pref}_67").read()) + 1,
      cmd_buf_addr_lo=lo32(self.adev.paddr2mc(self.cmd_paddr)), cmd_buf_addr_hi=hi32(self.adev.paddr2mc(self.cmd_paddr)),
      fence_addr_lo=lo32(self.adev.paddr2mc(self.fence_paddr)), fence_addr_hi=hi32(self.adev.paddr2mc(self.fence_paddr)))

    self.adev.vram.view(self.cmd_paddr, ctypes.sizeof(cmd))[:] = memoryview(cmd).cast('B')
    self.adev.vram.view(self.ring_paddr + prev_wptr * 4, ctypes.sizeof(msg))[:] = memoryview(msg).cast('B')

    # Move the wptr
    self.adev.reg(f"{self.reg_pref}_67").write(prev_wptr + ctypes.sizeof(am.struct_psp_gfx_rb_frame) // 4)

    wait_cond(lambda: self.adev.vram.view(self.fence_paddr, 4, 'I')[0], value=msg.fence_value, msg="sOS ring not responding")

    resp = type(cmd).from_buffer(bytearray(self.adev.vram.view(self.cmd_paddr, ctypes.sizeof(cmd))[:]))
    if resp.resp.status != 0: raise RuntimeError(f"PSP command failed {resp.cmd_id} {resp.resp.status}")

    return resp

  def _load_ip_fw_cmd(self, fw_types:list[int], fw_bytes:memoryview):
    self._prep_msg1(fw_bytes)
    for fw_type in fw_types:
      if DEBUG >= 2: print(f"am {self.adev.devfmt}: loading fw: {am.enum_psp_gfx_fw_type.get(fw_type)}")
      cmd = am.struct_psp_gfx_cmd_resp(cmd_id=am.GFX_CMD_ID_LOAD_IP_FW)
      cmd.cmd.cmd_load_ip_fw.fw_phy_addr_hi, cmd.cmd.cmd_load_ip_fw.fw_phy_addr_lo = data64(self.msg1_addr)
      cmd.cmd.cmd_load_ip_fw.fw_size = len(fw_bytes)
      cmd.cmd.cmd_load_ip_fw.fw_type = fw_type
      self._ring_submit(cmd)

  def _tmr_load_cmd(self) -> am.struct_psp_gfx_cmd_resp:
    tmr_paddr = self.adev.paddr2xgmi(self.tmr_paddr) if self.tmr_paddr else 0

    cmd = am.struct_psp_gfx_cmd_resp(cmd_id=am.GFX_CMD_ID_SETUP_TMR)
    cmd.cmd.cmd_setup_tmr.buf_phy_addr_hi, cmd.cmd.cmd_setup_tmr.buf_phy_addr_lo = data64(self.adev.paddr2mc(self.tmr_paddr) if self.tmr_paddr else 0)
    cmd.cmd.cmd_setup_tmr.system_phy_addr_hi, cmd.cmd.cmd_setup_tmr.system_phy_addr_lo = data64(tmr_paddr)
    cmd.cmd.cmd_setup_tmr.bitfield.virt_phy_addr = 1
    cmd.cmd.cmd_setup_tmr.buf_size = self.tmr_size if self.tmr_paddr else 0
    return self._ring_submit(cmd)

  def _load_toc_cmd(self, toc_size:int) -> am.struct_psp_gfx_cmd_resp:
    cmd = am.struct_psp_gfx_cmd_resp(cmd_id=am.GFX_CMD_ID_LOAD_TOC)
    cmd.cmd.cmd_load_toc.toc_phy_addr_hi, cmd.cmd.cmd_load_toc.toc_phy_addr_lo = data64(self.msg1_addr)
    cmd.cmd.cmd_load_toc.toc_size = toc_size
    return self._ring_submit(cmd)

  def _spatial_partition_cmd(self, mode):
    cmd = am.struct_psp_gfx_cmd_resp(cmd_id=am.GFX_CMD_ID_SRIOV_SPATIAL_PART)
    cmd.cmd.cmd_spatial_part.mode = mode
    return self._ring_submit(cmd)

  def _rlc_autoload_cmd(self): return self._ring_submit(am.struct_psp_gfx_cmd_resp(cmd_id=am.GFX_CMD_ID_AUTOLOAD_RLC))
