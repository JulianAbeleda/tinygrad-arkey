from tinygrad.runtime.support.am.ip.common import *
from tinygrad.runtime.support.am.ip.psp_diag import PSPDiagnostics
from tinygrad.runtime.support.am.ip.psp_mem import alloc_aligned_sysmem_window, map_sysmem_view

class AM_PSP(PSPDiagnostics, AM_IP):
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
      self.msg1_view, self.msg1_paddrs, raw_paddr, view_off, raw_pages = \
        alloc_aligned_sysmem_window(self.adev.pci_dev, am.PSP_1_MEG, am.PSP_1_MEG, "GTT")
      self.msg1_addr = map_sysmem_view(self.adev.mm, self.msg1_view, self.msg1_paddrs, am.PSP_1_MEG, snooped=True)
      self.msg1_kind = "sysmem-gtt"
      self._trace(f"msg1 sysmem gtt raw={raw_paddr:#x} view_off={view_off:#x} va={self.msg1_addr:#x} pages={raw_pages} bytes={self.msg1_view.nbytes}")
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
      self.msg1_view, self.msg1_paddrs, raw_paddr, view_off, raw_pages = \
        alloc_aligned_sysmem_window(self.adev.pci_dev, am.PSP_1_MEG, am.PSP_1_MEG, "DMA")
      self.msg1_addr = raw_paddr + view_off
      self.msg1_kind = "sysmem-dma"
      self._trace(f"msg1 sysmem dma raw={raw_paddr:#x} view_off={view_off:#x} addr={self.msg1_addr:#x} pages={raw_pages} bytes={self.msg1_view.nbytes}")
    elif self.adev.devfmt.startswith("usb:") or (getattr(self.adev.pci_dev, "is_remote", False) and getenv("AM_PSP_SYSMSG1", 0)):
      self.msg1_view, paddrs = self.adev.pci_dev.alloc_sysmem(am.PSP_1_MEG)
      self.msg1_addr = map_sysmem_view(self.adev.mm, self.msg1_view, paddrs, am.PSP_1_MEG)
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
