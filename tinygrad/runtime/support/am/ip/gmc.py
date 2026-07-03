from tinygrad.runtime.support.am.ip.common import *

class AM_GMC(AM_IP):
  def init_sw(self):
    self.vmhubs = len(self.adev.regs_offset[am.MMHUB_HWIP])

    # XGMI (for supported systems)
    self.xgmi_phys_id = self.adev.regMMMC_VM_XGMI_LFB_CNTL.read_bitfields()['pf_lfb_region'] if hasattr(self.adev, 'regMMMC_VM_XGMI_LFB_CNTL') else 0
    self.xgmi_seg_sz = self.adev.regMMMC_VM_XGMI_LFB_SIZE.read_bitfields()['pf_lfb_size']<<24 if hasattr(self.adev, 'regMMMC_VM_XGMI_LFB_SIZE') else 0

    self.paddr_base = self.xgmi_phys_id * self.xgmi_seg_sz

    self.fb_base = (self.adev.regMMMC_VM_FB_LOCATION_BASE.read() & 0xFFFFFF) << 24
    self.fb_end = (self.adev.regMMMC_VM_FB_LOCATION_TOP.read() & 0xFFFFFF) << 24
    self.gart_size = 512 << 20
    self.gart_start = 0 if getenv("AM_PSP_GART_LOW", 0) else ((0x0000800000000000 - self.gart_size) & ~0xffffffff)
    self.gart_end = self.gart_start + self.gart_size - 1

    # Memory controller aperture
    self.mc_base = self.fb_base + self.paddr_base

    # VM aperture
    self.vm_base = self.adev.mm.va_base
    self.vm_end = min(self.vm_base + (1 << self.adev.mm.va_bits) - 1, 0x7fffffffffff)

    self.trans_futher = self.adev.ip_ver[am.GC_HWIP] < (10, 0, 0)

    # mi3xx has 48-bit, others have 44-bit address space
    self.address_space_mask = (1 << (48 if self.adev.ip_ver[am.GC_HWIP][:2] in {(9,4), (9,5)} else 44)) - 1

    self.memscratch_xgmi_paddr = self.adev.paddr2xgmi(self.adev.mm.palloc(0x1000, zero=False, boot=True))
    self.dummy_page_xgmi_paddr = self.adev.paddr2xgmi(self.adev.mm.palloc(0x1000, zero=False, boot=True))

    # MM hub is inited before any tlb flushes and is still valid during partial_boot, so set it to true
    self.hub_initted = {"MM": True, "GC": False}

    self.pf_status_reg = lambda ip: f"reg{ip}VM_L2_PROTECTION_FAULT_STATUS{'_LO32' if self.adev.ip_ver[am.GC_HWIP] >= (12,0,0) else ''}"

  def init_hw(self): self.init_hub("MM", inst_cnt=self.vmhubs)

  def _psp_trace_enabled(self) -> bool:
    return getattr(self.adev, "psp", None) is not None and self.adev.psp._trace_enabled()

  def _trace_psp_gart_pte(self, label:str, gart_table_paddr:int, pt_base:int, gart_table, paddrs:list[int], start_page:int, gart_page:int,
                          page_count:int, msg1_off:int):
    if not self._psp_trace_enabled(): return
    first_pte, last_pte = gart_table[gart_page], gart_table[gart_page + page_count - 1]
    flag_names = [(am.AMDGPU_PTE_VALID, "VALID"), (am.AMDGPU_PTE_SYSTEM, "SYSTEM"), (am.AMDGPU_PTE_SNOOPED, "SNOOPED"),
                  (am.AMDGPU_PTE_EXECUTABLE, "EXEC"), (am.AMDGPU_PTE_READABLE, "READ"), (am.AMDGPU_PTE_WRITEABLE, "WRITE")]
    first_flags = ",".join(name for bit, name in flag_names if first_pte & bit)
    self.adev.psp._trace(f"gart {label} table_paddr={gart_table_paddr:#x} pt_base={pt_base:#x} msg1_off={msg1_off:#x} "
                         f"gart_page={gart_page:#x} paddr0={paddrs[start_page]:#x} paddr_last={paddrs[start_page + page_count - 1]:#x} "
                         f"pte0={first_pte:#018x} pte_last={last_pte:#018x} flags={first_flags} "
                         f"mtype={(first_pte & am.AMDGPU_PTE_MTYPE_NV10_MASK) >> 48:#x}")

  def _strong_invalidate_psp_gart(self, gart_table_paddr:int, pt_base:int, gart_table, paddrs:list[int], start_page:int, gart_page:int,
                                  page_count:int, msg1_off:int):
    self.flush_hdp()
    self._trace_psp_gart_pte("strong invalidate", gart_table_paddr, pt_base, gart_table, paddrs, start_page, gart_page, page_count, msg1_off)
    self.flush_tlb("MM", 0)
    self.flush_hdp()
    # Diagnostic sampling after a second VMID0 invalidate; 1ms was enough for macOS/TinyGPU audit reads and is not a readiness guarantee.
    time.sleep(0.001)
    if self._psp_trace_enabled():
      for inst in range(self.vmhubs):
        ack = self.adev.reg("regMMVM_INVALIDATE_ENG17_ACK").read(inst=inst)
        fault = self.adev.reg(self.pf_status_reg("MM")).read(inst=inst)
        self.adev.psp._trace(f"gart strong invalidate inst={inst} ack={ack:#010x} fault={fault:#010x}")

  def _trace_psp_gart_setup_regs(self, label:str):
    if not (AM_Experiment.gart_setup_trace() and self._psp_trace_enabled()): return
    for inst in range(self.vmhubs):
      vals = []
      for reg in ["regMMVM_INVALIDATE_ENG17_SEM", "regMMVM_INVALIDATE_ENG17_REQ", "regMMVM_INVALIDATE_ENG17_ACK",
                  "regMMVM_L2_BANK_SELECT_RESERVED_CID2", self.pf_status_reg("MM")]:
        with contextlib.suppress(Exception): vals += self._trace_read_reg(reg, inst)
      self.adev.psp._trace(f"gart setup {label} inst={inst} " + " ".join(vals))

  def setup_psp_gart(self, paddrs:list[int], view_off:int, size:int) -> int:
    self._trace_psp_gart_setup_regs("entry")
    if size <= 0 or view_off % 0x1000 != 0 or size % 0x1000 != 0:
      raise ValueError(f"invalid PSP GART window view_off={view_off:#x} size={size:#x}")
    msg1_off = AM_Experiment.gart_msg1_offset()
    if msg1_off % 0x1000 != 0 or msg1_off + size > self.gart_size:
      raise ValueError(f"invalid PSP GART msg1 offset={msg1_off:#x} size={size:#x}")
    table_size = self.gart_size // 0x1000 * 8
    if AM_Experiment.gart_table_top():
      # Linux amdgpu placed the successful PSP GART table at 0x5feb00000 on this 24GB RX 7900 XTX.
      gart_table_paddr = AM_Experiment.gart_table_addr(self.adev.vram_size - 0x1500000)
      if gart_table_paddr % 0x1000 != 0 or gart_table_paddr + table_size > self.adev.vram_size:
        raise ValueError(f"invalid PSP GART table paddr={gart_table_paddr:#x} size={table_size:#x} vram={self.adev.vram_size:#x}")
      if getattr(self.adev.pci_dev, "is_remote", False) and gart_table_paddr + table_size > self.adev.vram.nbytes and \
         not AM_Experiment.gart_table_sparse() and not getenv("AM_REMOTE_UNSAFE_INDIRECT_VRAM_WRITE", 0):
        raise RuntimeError(f"AM_PSP_GART_TABLE_TOP=1 needs to write PSP GART table paddr={gart_table_paddr:#x} "
                           f"size={table_size:#x}, but remote BAR0 maps only {self.adev.vram.nbytes:#x} bytes. "
                           "Run this experiment from a Linux path with full VRAM BAR access, or explicitly set "
                           "AM_REMOTE_UNSAFE_INDIRECT_VRAM_WRITE=1 to force the guarded indirect write path.")
      gart_table = [0] * (table_size // 8)
      self._trace_psp_gart_setup_regs("after-top-table-list")
    else:
      self._trace_psp_gart_setup_regs("before-table-palloc")
      gart_table_paddr = self.adev.mm.palloc(table_size, align=0x1000, zero=True, boot=True)
      self._trace_psp_gart_setup_regs("after-table-palloc")
      gart_table = self.adev.vram.view(gart_table_paddr, table_size, 'Q')
      self._trace_psp_gart_setup_regs("after-table-view")
    flags = am.AMDGPU_PTE_VALID | am.AMDGPU_PTE_SYSTEM | am.AMDGPU_PTE_EXECUTABLE | am.AMDGPU_PTE_READABLE | \
            am.AMDGPU_PTE_WRITEABLE | am.AMDGPU_PTE_MTYPE_NV10(0, self.adev.soc.module.MTYPE_UC)
    if AM_Experiment.gart_snooped(): flags |= am.AMDGPU_PTE_SNOOPED
    start_page = view_off // 0x1000
    gart_page = msg1_off // 0x1000
    page_count = size // 0x1000
    if start_page + page_count > len(paddrs):
      raise ValueError(f"invalid PSP GART source pages start={start_page:#x} count={page_count:#x} len={len(paddrs):#x}")
    self._trace_psp_gart_setup_regs("before-pte-fill")
    for i, paddr in enumerate(paddrs[start_page:start_page + page_count]):
      gart_table[gart_page + i] = (paddr & 0x0000FFFFFFFFF000) | flags
      if i in (0, page_count - 1): self._trace_psp_gart_setup_regs(f"after-pte-{i}")
    self._trace_psp_gart_setup_regs("after-pte-fill")
    if AM_Experiment.gart_table_top():
      if AM_Experiment.gart_table_sparse():
        sparse_off, sparse_count = gart_page * 8, page_count
        sparse_data = array.array('Q', gart_table[gart_page:gart_page + page_count]).tobytes()
        self.adev.psp._trace(f"gart sparse table write paddr={gart_table_paddr + sparse_off:#x} entries={sparse_count}")
        self.adev._write_vram(gart_table_paddr + sparse_off, sparse_data, allow_remote_sparse=True)
      else:
        self.adev._write_vram(gart_table_paddr, array.array('Q', gart_table).tobytes())
    self._trace_psp_gart_setup_regs("after-table-write")
    self.flush_hdp()
    self._trace_psp_gart_setup_regs("after-hdp-flush")

    pt_base = self.adev.paddr2xgmi(gart_table_paddr) | am.AMDGPU_PTE_VALID
    self._trace_psp_gart_pte("pte", gart_table_paddr, pt_base, gart_table, paddrs, start_page, gart_page, page_count, msg1_off)
    self.adev.psp.msg1_gart_info = (gart_table, gart_page, page_count)
    linux_context = AM_Experiment.gart_linux_context()
    for inst in range(self.vmhubs):
      aperture_low = AM_Experiment.gart_aperture_low()
      aperture_high = AM_Experiment.gart_aperture_high()
      default_addr = AM_Experiment.gart_default_addr()
      fault_default_addr = AM_Experiment.gart_fault_default_addr()
      if aperture_low is not None:
        self.adev.reg("regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR").write(aperture_low, inst=inst)
      elif not linux_context:
        self.adev.reg("regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR").write(min(self.fb_base, self.gart_start) >> 18, inst=inst)
      if aperture_high is not None:
        self.adev.reg("regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR").write(aperture_high, inst=inst)
      elif not linux_context:
        self.adev.reg("regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR").write(max(self.fb_end, self.gart_end) >> 18, inst=inst)
      if default_addr is not None:
        self.adev.wreg_pair("regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR", "_LSB", "_MSB", default_addr, inst=inst)
      if fault_default_addr is not None:
        self.adev.wreg_pair("regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR", "_LO32", "_HI32", fault_default_addr, inst=inst)
      self._trace_psp_gart_setup_regs("after-aperture")
      self.adev.wreg_pair("regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR", "_LO32", "_HI32", pt_base, inst=inst)
      self.adev.wreg_pair("regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR", "_LO32", "_HI32", self.gart_start >> 12, inst=inst)
      self.adev.wreg_pair("regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR", "_LO32", "_HI32", self.gart_end >> 12, inst=inst)
      self._trace_psp_gart_setup_regs("after-context0-table")
      # Linux amdgpu on RX 7900 XTX used this CONTEXT0_CNTL value in the successful PSP KDB trace.
      if linux_context: self.adev.reg("regMMVM_CONTEXT0_CNTL").write(0x01fffe01, inst=inst)
      else: self.adev.reg("regMMVM_CONTEXT0_CNTL").write(enable_context=1, page_table_depth=0, retry_permission_or_invalid_page_fault=0, inst=inst)
      self._trace_psp_gart_setup_regs("after-context0-cntl")
      if AM_Experiment.gart_linux_full_context():
        self.adev.reg("regMMMC_VM_MX_L1_TLB_CNTL").write(0x1859, inst=inst)
        self.adev.reg("regMMVM_L2_BANK_SELECT_RESERVED_CID2").write(0x12104010, inst=inst)
        self._trace_psp_gart_setup_regs("after-linux-full-base")
        for vmid in range(1, 16):
          self.adev.reg(f"regMMVM_CONTEXT{vmid}_CNTL").write(0x01fffe07, inst=inst)
          self.adev.wreg_pair(f"regMMVM_CONTEXT{vmid}_PAGE_TABLE_START_ADDR", "_LO32", "_HI32", 0x0, inst=inst)
          self.adev.wreg_pair(f"regMMVM_CONTEXT{vmid}_PAGE_TABLE_END_ADDR", "_LO32", "_HI32", 0xfffffffff, inst=inst)
          self._trace_psp_gart_setup_regs(f"after-context{vmid}")
    self._trace_psp_gart_setup_regs("before-flush-tlb")
    self.flush_tlb("MM", 0)
    if AM_Experiment.gart_linux_full_context():
      for inst in range(self.vmhubs): self.adev.reg("regMMVM_L2_BANK_SELECT_RESERVED_CID2").write(0x12104010, inst=inst)
    if AM_Experiment.gart_strong_invalidate():
      self._strong_invalidate_psp_gart(gart_table_paddr, pt_base, gart_table, paddrs, start_page, gart_page, page_count, msg1_off)
      if AM_Experiment.gart_linux_full_context():
        for inst in range(self.vmhubs): self.adev.reg("regMMVM_L2_BANK_SELECT_RESERVED_CID2").write(0x12104010, inst=inst)
    msg1_addr = self.gart_start + msg1_off
    return msg1_addr

  def flush_hdp(self): self.adev.wreg(self.adev.reg("regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL").read() // 4, 0x0)

  def _tlb_trace_enabled(self) -> bool:
    return bool(AM_Experiment.tlb_trace())

  def _tlb_trace_context(self) -> str:
    parts = []
    frame = None
    with contextlib.suppress(Exception): frame = sys._getframe(2)
    while frame is not None and len(parts) < 4:
      code = frame.f_code
      parts.append(f"{code.co_name}:{frame.f_lineno}")
      frame = frame.f_back
    return " <- ".join(parts)

  def _tlb_trace(self, msg:str):
    if not self._tlb_trace_enabled(): return
    if self._psp_trace_enabled(): self.adev.psp._trace(f"tlb {msg}")
    else: print(f"am {self.adev.devfmt}: TLB {msg}", flush=True)

  def _trace_read_reg(self, reg:str, inst:int) -> list[str]:
    val = self.adev.reg(reg).read(inst=inst)
    vals = [f"{reg}={val:#010x}"]
    if reg == "regMMVM_INVALIDATE_ENG17_SEM" and val & 0x1:
      self.adev.reg(reg).write(0, inst=inst)
      vals.append(f"{reg}_released=1")
    return vals

  def _tlb_trace_regs(self, label:str, ip:Literal["MM", "GC"], inst:int):
    if not self._tlb_trace_enabled(): return
    regs = [f"reg{ip}VM_INVALIDATE_ENG17_REQ", f"reg{ip}VM_INVALIDATE_ENG17_ACK"]
    if ip == "MM": regs += ["regMMVM_INVALIDATE_ENG17_SEM", "regMMVM_L2_BANK_SELECT_RESERVED_CID2", self.pf_status_reg("MM")]
    vals = []
    for reg in regs:
      try: vals += self._trace_read_reg(reg, inst)
      except Exception as e: vals.append(f"{reg}=read_failed:{type(e).__name__}")
    self._tlb_trace(f"{label} ip={ip} inst={inst} " + " ".join(vals))

  def _gmc_init_trace(self, msg:str):
    if not AM_Experiment.gmc_init_trace(): return
    if self._psp_trace_enabled(): self.adev.psp._trace(f"gmc init {msg}")
    else: print(f"am {self.adev.devfmt}: GMC init {msg}", flush=True)

  def _gmc_init_trace_regs(self, label:str, ip:Literal["MM", "GC"], inst:int):
    if not AM_Experiment.gmc_init_trace(): return
    regs = [f"reg{ip}VM_INVALIDATE_ENG17_REQ", f"reg{ip}VM_INVALIDATE_ENG17_ACK", f"reg{ip}VM_L2_CNTL",
            f"reg{ip}VM_L2_CNTL2", f"reg{ip}VM_CONTEXT0_CNTL"]
    if ip == "MM": regs += ["regMMVM_INVALIDATE_ENG17_SEM", "regMMVM_L2_BANK_SELECT_RESERVED_CID2", self.pf_status_reg("MM")]
    vals = []
    for reg in regs:
      try: vals += self._trace_read_reg(reg, inst)
      except Exception as e: vals.append(f"{reg}=read_failed:{type(e).__name__}")
    self._gmc_init_trace(f"{label} ip={ip} inst={inst} " + " ".join(vals))

  def flush_tlb(self, ip:Literal["MM", "GC"], vmid, flush_type=0):
    self.flush_hdp()

    # Can't issue TLB invalidation if the hub isn't initialized.
    if not self.hub_initted[ip]: return

    self._tlb_trace(f"begin ip={ip} vmid={vmid} flush_type={flush_type} caller={self._tlb_trace_context()}")
    for inst in range(self.adev.gmc.vmhubs if ip == "MM" else self.adev.gfx.xccs):
      self._tlb_trace_regs("before-wait", ip, inst)
      if ip == "MM":
        try:
          wait_cond(lambda: self.adev.regMMVM_INVALIDATE_ENG17_SEM.read(inst=inst) & 0x1, value=1, msg="mm flush_tlb timeout")
        except TimeoutError:
          self._tlb_trace_regs("sem-timeout", ip, inst)
          raise
        self._tlb_trace_regs("after-sem", ip, inst)

      self.adev.reg(f"reg{ip}VM_INVALIDATE_ENG17_REQ").write(flush_type=flush_type, per_vmid_invalidate_req=(1 << vmid), invalidate_l2_ptes=1,
        invalidate_l2_pde0=1, invalidate_l2_pde1=1, invalidate_l2_pde2=1, invalidate_l1_ptes=1, clear_protection_fault_status_addr=0, inst=inst)
      self._tlb_trace_regs("after-req", ip, inst)

      try:
        wait_cond(lambda: self.adev.reg(f"reg{ip}VM_INVALIDATE_ENG17_ACK").read(inst=inst) & (1 << vmid), value=(1 << vmid), msg="flush_tlb timeout")
      except TimeoutError:
        self._tlb_trace_regs("ack-timeout", ip, inst)
        raise
      self._tlb_trace_regs("after-ack", ip, inst)

      if ip == "MM": self.adev.regMMVM_INVALIDATE_ENG17_SEM.write(0x0, inst=inst)
      if ip == "MM": self._tlb_trace_regs("after-sem-release", ip, inst)
      if self.adev.ip_ver[am.GC_HWIP] >= (11,0,0) and ip == "MM":
        self.adev.regMMVM_L2_BANK_SELECT_RESERVED_CID2.update(reserved_cache_private_invalidation=1, inst=inst)

        # Read back the register to ensure the invalidation is complete
        self.adev.regMMVM_L2_BANK_SELECT_RESERVED_CID2.read(inst=inst)
        self._tlb_trace_regs("after-cid2-readback", ip, inst)

  def enable_vm_addressing(self, page_table, ip:Literal["MM", "GC"], vmid, inst):
    self.adev.wreg_pair(f"reg{ip}VM_CONTEXT{vmid}_PAGE_TABLE_START_ADDR", "_LO32", "_HI32", self.vm_base >> 12, inst=inst)
    self.adev.wreg_pair(f"reg{ip}VM_CONTEXT{vmid}_PAGE_TABLE_END_ADDR", "_LO32", "_HI32", self.vm_end >> 12, inst=inst)
    self.adev.wreg_pair(f"reg{ip}VM_CONTEXT{vmid}_PAGE_TABLE_BASE_ADDR", "_LO32", "_HI32", self.adev.paddr2xgmi(page_table.paddr) | 1, inst=inst)

    fault_flags = {f'{x}_protection_fault_enable_interrupt':1 for x in ['pde0', 'dummy_page', 'range', 'valid', 'read', 'write', 'execute']}
    en_def_flags = {f'{x}_protection_fault_enable_default':1 for x in ['pde0', 'dummy_page', 'range', 'valid', 'read', 'write', 'execute']}
    self.adev.reg(f"reg{ip}VM_CONTEXT{vmid}_CNTL").write(0x1800000, **fault_flags, **en_def_flags, enable_context=1,
      page_table_depth=((2 if self.trans_futher else 3) - page_table.lv), page_table_block_size=9 if self.trans_futher else 0, inst=inst)

  def init_hub(self, ip:Literal["MM", "GC"], inst_cnt:int):
    # Init system apertures
    for inst in range(inst_cnt):
      self._gmc_init_trace_regs("entry", ip, inst)
      self.adev.reg(f"reg{ip}MC_VM_AGP_BASE").write(0, inst=inst)
      self.adev.reg(f"reg{ip}MC_VM_AGP_BOT").write(0xffffffffffff >> 24, inst=inst) # disable AGP
      self.adev.reg(f"reg{ip}MC_VM_AGP_TOP").write(0, inst=inst)
      self._gmc_init_trace_regs("after-agp", ip, inst)

      self.adev.reg(f"reg{ip}MC_VM_SYSTEM_APERTURE_LOW_ADDR").write(self.fb_base >> 18, inst=inst)
      self.adev.reg(f"reg{ip}MC_VM_SYSTEM_APERTURE_HIGH_ADDR").write(self.fb_end >> 18, inst=inst)
      self.adev.wreg_pair(f"reg{ip}MC_VM_SYSTEM_APERTURE_DEFAULT_ADDR", "_LSB", "_MSB", self.memscratch_xgmi_paddr >> 12, inst=inst)
      self.adev.wreg_pair(f"reg{ip}VM_L2_PROTECTION_FAULT_DEFAULT_ADDR", "_LO32", "_HI32", self.dummy_page_xgmi_paddr >> 12, inst=inst)
      self._gmc_init_trace_regs("after-aperture", ip, inst)

      self.adev.reg(f"reg{ip}VM_L2_PROTECTION_FAULT_CNTL2").update(active_page_migration_pte_read_retry=1, inst=inst)
      self._gmc_init_trace_regs("after-fault-cntl", ip, inst)

      # Init TLB and cache
      self.adev.reg(f"reg{ip}MC_VM_MX_L1_TLB_CNTL").update(enable_l1_tlb=1, system_access_mode=3, enable_advanced_driver_model=1,
        system_aperture_unmapped_access=0, mtype=self.adev.soc.module.MTYPE_UC, inst=inst)
      self._gmc_init_trace_regs("after-l1-tlb", ip, inst)

      self.adev.reg(f"reg{ip}VM_L2_CNTL").update(enable_l2_cache=1, enable_default_page_out_to_system_memory=1,
        l2_pde0_cache_tag_generation_mode=0, pde_fault_classification=0, context1_identity_access_mode=1, identity_mode_fragment_size=0,
        enable_l2_fragment_processing=int(self.adev.ip_ver[am.GC_HWIP] < (10,0,0)), inst=inst)
      self._gmc_init_trace_regs("after-l2-cntl", ip, inst)
      self._gmc_init_trace_regs("before-l2-cntl2", ip, inst)
      self.adev.reg(f"reg{ip}VM_L2_CNTL2").update(invalidate_all_l1_tlbs=1, invalidate_l2_cache=1, inst=inst)
      self._gmc_init_trace_regs("after-l2-cntl2", ip, inst)
      self.adev.reg(f"reg{ip}VM_L2_CNTL3").write(l2_cache_4k_associativity=1, l2_cache_bigk_associativity=1,
        bank_select=12 if self.trans_futher else 9, l2_cache_bigk_fragment_size=9 if self.trans_futher else 6, inst=inst)
      self.adev.reg(f"reg{ip}VM_L2_CNTL4").write(l2_cache_4k_partition_count=1, inst=inst)
      if self.adev.ip_ver[am.GC_HWIP] >= (10,0,0): self.adev.reg(f"reg{ip}VM_L2_CNTL5").write(walker_priority_client_id=0x1ff, inst=inst)
      self._gmc_init_trace_regs("after-l2-cntl3-5", ip, inst)

      self._gmc_init_trace_regs("before-context0", ip, inst)
      self.enable_vm_addressing(self.adev.mm.root_page_table, ip, vmid=0, inst=inst)
      self._gmc_init_trace_regs("after-context0", ip, inst)

      # Disable identity aperture
      self.adev.wreg_pair(f"reg{ip}VM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR", "_LO32", "_HI32", 0xfffffffff, inst=inst)
      self.adev.wreg_pair(f"reg{ip}VM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR", "_LO32", "_HI32", 0x0, inst=inst)
      self.adev.wreg_pair(f"reg{ip}VM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET", "_LO32", "_HI32", 0x0, inst=inst)
      self._gmc_init_trace_regs("after-identity-aperture", ip, inst)

      for eng_i in range(18): self.adev.wreg_pair(f"reg{ip}VM_INVALIDATE_ENG{eng_i}_ADDR_RANGE", "_LO32", "_HI32", 0x1fffffffff, inst=inst)
      self._gmc_init_trace_regs("after-invalidate-ranges", ip, inst)
    self.hub_initted[ip] = True
    for inst in range(inst_cnt): self._gmc_init_trace_regs("after-hub-initted", ip, inst)

  @functools.cache  # pylint: disable=method-cache-max-size-none
  def get_pte_flags(self, pte_lv, is_table, frag, uncached, system, snooped, valid, extra=0):
    extra |= (am.AMDGPU_PTE_SYSTEM * system) | (am.AMDGPU_PTE_SNOOPED * snooped) | (am.AMDGPU_PTE_VALID * valid) | am.AMDGPU_PTE_FRAG(frag)
    if not is_table: extra |= (am.AMDGPU_PTE_WRITEABLE | am.AMDGPU_PTE_READABLE | am.AMDGPU_PTE_EXECUTABLE)
    if self.adev.ip_ver[am.GC_HWIP] >= (12,0,0):
      extra |= am.AMDGPU_PTE_MTYPE_GFX12(0, self.adev.soc.module.MTYPE_UC if uncached else 0)
      extra |= (am.AMDGPU_PDE_PTE_GFX12 if not is_table and pte_lv != am.AMDGPU_VM_PTB else (am.AMDGPU_PTE_IS_PTE if not is_table else 0))
    elif self.adev.ip_ver[am.GC_HWIP] >= (10,0,0):
      extra |= am.AMDGPU_PTE_MTYPE_NV10(0, self.adev.soc.module.MTYPE_UC if uncached else 0)
      extra |= (am.AMDGPU_PDE_PTE if not is_table and pte_lv != am.AMDGPU_VM_PTB else 0)
    else:
      extra |= am.AMDGPU_PTE_MTYPE_VG10(0, self.adev.soc.module.MTYPE_UC if uncached else 0)
      if is_table and pte_lv == am.AMDGPU_VM_PDB1: extra |= am.AMDGPU_PDE_BFS(0x9)
      if is_table and pte_lv == am.AMDGPU_VM_PDB0: extra |= am.AMDGPU_PTE_TF
      if not is_table and pte_lv not in {am.AMDGPU_VM_PTB, am.AMDGPU_VM_PDB0}: extra |= am.AMDGPU_PDE_PTE
    return extra
  def is_pte_huge_page(self, pte_lv, pte):
    if self.adev.ip_ver[am.GC_HWIP] < (10,0,0): return (pte & am.AMDGPU_PDE_PTE) if pte_lv != am.AMDGPU_VM_PDB0 else not (pte & am.AMDGPU_PTE_TF)
    return pte & (am.AMDGPU_PDE_PTE_GFX12 if self.adev.ip_ver[am.GC_HWIP] >= (12,0,0) else am.AMDGPU_PDE_PTE)
