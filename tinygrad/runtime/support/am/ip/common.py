import array, ctypes, hashlib, time, contextlib, functools, sys
from typing import Literal
from tinygrad.helpers import to_mv, data64, lo32, hi32, DEBUG, wait_cond, pad_bytes, getbits, getenv
from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.support.amd import import_soc
from tinygrad.runtime.support.memory import AddrSpace

def _env_int(name:str, default:int=0) -> int:
  raw = getenv(name, "")
  return int(raw, 0) if raw else default

def _env_optional_int(name:str) -> int|None:
  raw = getenv(name, "")
  return int(raw, 0) if raw else None

_AM_EXPERIMENT_NAMES = frozenset("""audit_pre_kdb gart_msg1_offset gart_strong_invalidate gart_linux_context gart_linux_full_context
  gart_table_top gart_table_sparse gart_table_addr gart_snooped kdb_skip_prefix kdb_slice_offset kdb_slice_size gart_aperture_low
  gart_aperture_high gart_default_addr gart_fault_default_addr exact_bootloader_wait vram_msg1_paddr mailbox_strong_order wait_trace_ms
  trace_c2pmsg_dense pre_kdb_invalidate_burst pre_kdb_linux_final_invalidate pre_kdb_linux_final_cid2 pre_kdb_linux_mmhub_window
  pre_kdb_cid2_audit pre_kdb_cid2_audit_stop pre_kdb_gart_audit pre_kdb_gart_audit_stop fw_pri_equiv_audit linux_pre_bl_status
  msg1_visibility_probe kdb_fail_capture kdb_fail_capture_pre_command kdb_fail_capture_ms kdb_fail_capture_reads mailbox_visibility
  mailbox_visibility_reads mailbox_visibility_delay_us mailbox_visibility_hdp_flush msg1_sysmem_sync msg1_sysmem_sync_invalidate
  msg1_primary_sync msg1_full_audit kdb_order_barrier kdb_payload_audit kdb_payload_audit_bytes bl_payload_audit bl_payload_audit_bytes
  bl_metadata_audit bl_metadata_audit_bytes bl_metadata_audit_stop bl_metadata_audit_stop_after sos_fw_inventory_audit
  sos_fw_inventory_audit_bytes sos_fw_inventory_audit_stop kdb_header_audit kdb_header_audit_bytes kdb_header_audit_stop
  kdb_record_audit kdb_record_audit_start kdb_record_audit_stride kdb_record_audit_bytes kdb_record_audit_stop sos_wait_delay_ms
  sos_final_state_audit bl_boundary_audit sysmsg1_gart_sort_paddrs kdb_pipeline_seq kdb_pipeline_count kdb_pipeline_delay_us
  bl_pipeline_count bl_pipeline_delay_us tlb_trace gart_setup_trace gmc_init_trace trace_map_bar5_first trace_map_bar0_last""".split())
_AM_EXPERIMENT_OPTIONAL = frozenset("kdb_slice_offset kdb_slice_size gart_aperture_low gart_aperture_high gart_default_addr gart_fault_default_addr vram_msg1_paddr".split())
_AM_EXPERIMENT_KEYS = {"exact_bootloader_wait":"AM_PSP_EXACT_BL_WAIT", "vram_msg1_paddr":"AM_PSP_SYSMSG1_VRAM_PADDR",
                       "msg1_visibility_probe":"AM_PSP_MSG1_VIS_PROBE", "mailbox_visibility":"AM_PSP_MAILBOX_VIS",
                       "mailbox_visibility_reads":"AM_PSP_MAILBOX_VIS_READS", "mailbox_visibility_delay_us":"AM_PSP_MAILBOX_VIS_DELAY_US",
                       "mailbox_visibility_hdp_flush":"AM_PSP_MAILBOX_VIS_HDP_FLUSH"}
_AM_EXPERIMENT_DEFAULTS = {"gart_snooped":1, "pre_kdb_linux_final_cid2":0x12104010, "kdb_fail_capture_pre_command":1,
  "kdb_fail_capture_ms":20, "kdb_fail_capture_reads":256, "mailbox_visibility_reads":8, "kdb_payload_audit_bytes":64,
  "bl_payload_audit_bytes":64, "bl_metadata_audit_bytes":64, "bl_metadata_audit_stop_after":1, "sos_fw_inventory_audit_bytes":64,
  "kdb_header_audit_bytes":0x200, "kdb_record_audit_start":0x150, "kdb_record_audit_stride":0x150,
  "kdb_record_audit_bytes":64, "kdb_pipeline_count":1, "kdb_pipeline_delay_us":900, "bl_pipeline_delay_us":900}

class _AMExperimentMeta(type):
  def __getattr__(cls, name:str):
    if name not in _AM_EXPERIMENT_NAMES: raise AttributeError(name)
    key = _AM_EXPERIMENT_KEYS.get(name, "AM_PSP_" + name.upper())
    def read(default=_AM_EXPERIMENT_DEFAULTS.get(name, 0)):
      return _env_optional_int(key) if name in _AM_EXPERIMENT_OPTIONAL else _env_int(key, default)
    return read

class AM_Experiment(metaclass=_AMExperimentMeta): pass
class AM_IP:
  def __init__(self, adev): self.adev = adev
  def init_sw(self): pass # Prepare sw/allocations for this IP
  def init_hw(self): pass # Initialize hw for this IP
  def fini_hw(self): pass # Finalize hw for this IP
  def set_clockgating_state(self): pass # Set clockgating state for this IP

class AM_ReorderedMsg1View:
  def __init__(self, raw_view, order:list[int], offset:int=0, size:int|None=None):
    self.raw_view, self.order, self.offset = raw_view, order, offset
    self.nbytes = len(order) * 0x1000 - offset if size is None else size

  def _raw_offset(self, logical:int) -> int:
    absolute = self.offset + logical
    page, page_off = divmod(absolute, 0x1000)
    return self.order[page] * 0x1000 + page_off

  def __getitem__(self, idx):
    if isinstance(idx, slice):
      start, stop, step = idx.indices(self.nbytes)
      if step != 1: return bytes(self[i] for i in range(start, stop, step))
      chunks = []
      cur = start
      while cur < stop:
        absolute = self.offset + cur
        page, page_off = divmod(absolute, 0x1000)
        n = min(stop - cur, 0x1000 - page_off)
        raw_off = self.order[page] * 0x1000 + page_off
        chunks.append(bytes(self.raw_view[raw_off:raw_off + n]))
        cur += n
      return b"".join(chunks)
    return self.raw_view[self._raw_offset(idx)]

  def __setitem__(self, idx, val):
    if isinstance(idx, slice):
      start, stop, step = idx.indices(self.nbytes)
      if step != 1: raise ValueError("AM_ReorderedMsg1View only supports contiguous slice writes")
      if len(val) != stop - start: raise ValueError(f"slice write size mismatch {len(val)} != {stop - start}")
      cur, src_off = start, 0
      while cur < stop:
        absolute = self.offset + cur
        page, page_off = divmod(absolute, 0x1000)
        n = min(stop - cur, 0x1000 - page_off)
        raw_off = self.order[page] * 0x1000 + page_off
        self.raw_view[raw_off:raw_off + n] = val[src_off:src_off + n]
        cur, src_off = cur + n, src_off + n
    else:
      self.raw_view[self._raw_offset(idx)] = val

  def view(self, offset:int=0, size:int|None=None, fmt=None):
    return AM_ReorderedMsg1View(self.raw_view, self.order, self.offset + offset, size)

  def sync(self, invalidate=False):
    if hasattr(self.raw_view, "sync"): self.raw_view.sync(invalidate=invalidate)
