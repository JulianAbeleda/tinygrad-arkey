from tinygrad.runtime.support.am.ip.common import *

class PSPDiagnostics:
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
