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

class AM_Experiment:
  @staticmethod
  def audit_pre_kdb() -> int: return _env_int("AM_PSP_AUDIT_PRE_KDB")
  @staticmethod
  def gart_msg1_offset() -> int: return _env_int("AM_PSP_GART_MSG1_OFFSET")
  @staticmethod
  def gart_strong_invalidate() -> int: return _env_int("AM_PSP_GART_STRONG_INVALIDATE")
  @staticmethod
  def gart_linux_context() -> int: return _env_int("AM_PSP_GART_LINUX_CONTEXT")
  @staticmethod
  def gart_linux_full_context() -> int: return _env_int("AM_PSP_GART_LINUX_FULL_CONTEXT")
  @staticmethod
  def gart_table_top() -> int: return _env_int("AM_PSP_GART_TABLE_TOP")
  @staticmethod
  def gart_table_sparse() -> int: return _env_int("AM_PSP_GART_TABLE_SPARSE")
  @staticmethod
  def gart_table_addr(default:int) -> int: return _env_int("AM_PSP_GART_TABLE_ADDR", default)
  @staticmethod
  def gart_snooped() -> int: return _env_int("AM_PSP_GART_SNOOPED", 1)
  @staticmethod
  def kdb_skip_prefix() -> int: return _env_int("AM_PSP_KDB_SKIP_PREFIX")
  @staticmethod
  def kdb_slice_offset() -> int|None: return _env_optional_int("AM_PSP_KDB_SLICE_OFFSET")
  @staticmethod
  def kdb_slice_size() -> int|None: return _env_optional_int("AM_PSP_KDB_SLICE_SIZE")
  @staticmethod
  def gart_aperture_low() -> int|None: return _env_optional_int("AM_PSP_GART_APERTURE_LOW")
  @staticmethod
  def gart_aperture_high() -> int|None: return _env_optional_int("AM_PSP_GART_APERTURE_HIGH")
  @staticmethod
  def gart_default_addr() -> int|None: return _env_optional_int("AM_PSP_GART_DEFAULT_ADDR")
  @staticmethod
  def gart_fault_default_addr() -> int|None: return _env_optional_int("AM_PSP_GART_FAULT_DEFAULT_ADDR")
  @staticmethod
  def exact_bootloader_wait() -> int: return _env_int("AM_PSP_EXACT_BL_WAIT")
  @staticmethod
  def vram_msg1_paddr() -> int|None: return _env_optional_int("AM_PSP_SYSMSG1_VRAM_PADDR")
  @staticmethod
  def mailbox_strong_order() -> int: return _env_int("AM_PSP_MAILBOX_STRONG_ORDER")
  @staticmethod
  def wait_trace_ms() -> int: return _env_int("AM_PSP_WAIT_TRACE_MS")
  @staticmethod
  def trace_c2pmsg_dense() -> int: return _env_int("AM_PSP_TRACE_C2PMSG_DENSE")
  @staticmethod
  def pre_kdb_invalidate_burst() -> int: return _env_int("AM_PSP_PRE_KDB_INVALIDATE_BURST")
  @staticmethod
  def pre_kdb_linux_final_invalidate() -> int: return _env_int("AM_PSP_PRE_KDB_LINUX_FINAL_INVALIDATE")
  @staticmethod
  def pre_kdb_linux_final_cid2() -> int: return _env_int("AM_PSP_PRE_KDB_LINUX_FINAL_CID2", 0x12104010)
  @staticmethod
  def pre_kdb_linux_mmhub_window() -> int: return _env_int("AM_PSP_PRE_KDB_LINUX_MMHUB_WINDOW")
  @staticmethod
  def pre_kdb_cid2_audit() -> int: return _env_int("AM_PSP_PRE_KDB_CID2_AUDIT")
  @staticmethod
  def pre_kdb_cid2_audit_stop() -> int: return _env_int("AM_PSP_PRE_KDB_CID2_AUDIT_STOP")
  @staticmethod
  def pre_kdb_gart_audit() -> int: return _env_int("AM_PSP_PRE_KDB_GART_AUDIT")
  @staticmethod
  def pre_kdb_gart_audit_stop() -> int: return _env_int("AM_PSP_PRE_KDB_GART_AUDIT_STOP")
  @staticmethod
  def fw_pri_equiv_audit() -> int: return _env_int("AM_PSP_FW_PRI_EQUIV_AUDIT")
  @staticmethod
  def linux_pre_bl_status() -> int: return _env_int("AM_PSP_LINUX_PRE_BL_STATUS")
  @staticmethod
  def msg1_visibility_probe() -> int: return _env_int("AM_PSP_MSG1_VIS_PROBE")
  @staticmethod
  def kdb_fail_capture() -> int: return _env_int("AM_PSP_KDB_FAIL_CAPTURE")
  @staticmethod
  def kdb_fail_capture_pre_command() -> int: return _env_int("AM_PSP_KDB_FAIL_CAPTURE_PRE_COMMAND", 1)
  @staticmethod
  def kdb_fail_capture_ms() -> int: return _env_int("AM_PSP_KDB_FAIL_CAPTURE_MS", 20)
  @staticmethod
  def kdb_fail_capture_reads() -> int: return _env_int("AM_PSP_KDB_FAIL_CAPTURE_READS", 256)
  @staticmethod
  def mailbox_visibility() -> int: return _env_int("AM_PSP_MAILBOX_VIS")
  @staticmethod
  def mailbox_visibility_reads() -> int: return _env_int("AM_PSP_MAILBOX_VIS_READS", 8)
  @staticmethod
  def mailbox_visibility_delay_us() -> int: return _env_int("AM_PSP_MAILBOX_VIS_DELAY_US")
  @staticmethod
  def mailbox_visibility_hdp_flush() -> int: return _env_int("AM_PSP_MAILBOX_VIS_HDP_FLUSH")
  @staticmethod
  def msg1_sysmem_sync() -> int: return _env_int("AM_PSP_MSG1_SYSMEM_SYNC")
  @staticmethod
  def msg1_sysmem_sync_invalidate() -> int: return _env_int("AM_PSP_MSG1_SYSMEM_SYNC_INVALIDATE")
  @staticmethod
  def msg1_primary_sync() -> int: return _env_int("AM_PSP_MSG1_PRIMARY_SYNC")
  @staticmethod
  def msg1_full_audit() -> int: return _env_int("AM_PSP_MSG1_FULL_AUDIT")
  @staticmethod
  def kdb_order_barrier() -> int: return _env_int("AM_PSP_KDB_ORDER_BARRIER")
  @staticmethod
  def kdb_payload_audit() -> int: return _env_int("AM_PSP_KDB_PAYLOAD_AUDIT")
  @staticmethod
  def kdb_payload_audit_bytes() -> int: return _env_int("AM_PSP_KDB_PAYLOAD_AUDIT_BYTES", 64)
  @staticmethod
  def bl_payload_audit() -> int: return _env_int("AM_PSP_BL_PAYLOAD_AUDIT")
  @staticmethod
  def bl_payload_audit_bytes() -> int: return _env_int("AM_PSP_BL_PAYLOAD_AUDIT_BYTES", 64)
  @staticmethod
  def bl_metadata_audit() -> int: return _env_int("AM_PSP_BL_METADATA_AUDIT")
  @staticmethod
  def bl_metadata_audit_bytes() -> int: return _env_int("AM_PSP_BL_METADATA_AUDIT_BYTES", 64)
  @staticmethod
  def bl_metadata_audit_stop() -> int: return _env_int("AM_PSP_BL_METADATA_AUDIT_STOP")
  @staticmethod
  def bl_metadata_audit_stop_after() -> int: return _env_int("AM_PSP_BL_METADATA_AUDIT_STOP_AFTER", 1)
  @staticmethod
  def sos_fw_inventory_audit() -> int: return _env_int("AM_PSP_SOS_FW_INVENTORY_AUDIT")
  @staticmethod
  def sos_fw_inventory_audit_bytes() -> int: return _env_int("AM_PSP_SOS_FW_INVENTORY_AUDIT_BYTES", 64)
  @staticmethod
  def sos_fw_inventory_audit_stop() -> int: return _env_int("AM_PSP_SOS_FW_INVENTORY_AUDIT_STOP")
  @staticmethod
  def kdb_header_audit() -> int: return _env_int("AM_PSP_KDB_HEADER_AUDIT")
  @staticmethod
  def kdb_header_audit_bytes() -> int: return _env_int("AM_PSP_KDB_HEADER_AUDIT_BYTES", 0x200)
  @staticmethod
  def kdb_header_audit_stop() -> int: return _env_int("AM_PSP_KDB_HEADER_AUDIT_STOP")
  @staticmethod
  def kdb_record_audit() -> int: return _env_int("AM_PSP_KDB_RECORD_AUDIT")
  @staticmethod
  def kdb_record_audit_start() -> int: return _env_int("AM_PSP_KDB_RECORD_AUDIT_START", 0x150)
  @staticmethod
  def kdb_record_audit_stride() -> int: return _env_int("AM_PSP_KDB_RECORD_AUDIT_STRIDE", 0x150)
  @staticmethod
  def kdb_record_audit_bytes() -> int: return _env_int("AM_PSP_KDB_RECORD_AUDIT_BYTES", 64)
  @staticmethod
  def kdb_record_audit_stop() -> int: return _env_int("AM_PSP_KDB_RECORD_AUDIT_STOP")
  @staticmethod
  def sos_wait_delay_ms() -> int: return _env_int("AM_PSP_SOS_WAIT_DELAY_MS")
  @staticmethod
  def sos_final_state_audit() -> int: return _env_int("AM_PSP_SOS_FINAL_STATE_AUDIT")
  @staticmethod
  def bl_boundary_audit() -> int: return _env_int("AM_PSP_BL_BOUNDARY_AUDIT")
  @staticmethod
  def sysmsg1_gart_sort_paddrs() -> int: return _env_int("AM_PSP_SYSMSG1_GART_SORT_PADDRS")
  @staticmethod
  def kdb_pipeline_seq() -> int: return _env_int("AM_PSP_KDB_PIPELINE_SEQ")
  @staticmethod
  def kdb_pipeline_count() -> int: return _env_int("AM_PSP_KDB_PIPELINE_COUNT", 1)
  @staticmethod
  def kdb_pipeline_delay_us() -> int: return _env_int("AM_PSP_KDB_PIPELINE_DELAY_US", 900)
  @staticmethod
  def bl_pipeline_count() -> int: return _env_int("AM_PSP_BL_PIPELINE_COUNT")
  @staticmethod
  def bl_pipeline_delay_us() -> int: return _env_int("AM_PSP_BL_PIPELINE_DELAY_US", 900)
  @staticmethod
  def tlb_trace() -> int: return _env_int("AM_PSP_TLB_TRACE")
  @staticmethod
  def gart_setup_trace() -> int: return _env_int("AM_PSP_GART_SETUP_TRACE")
  @staticmethod
  def gmc_init_trace() -> int: return _env_int("AM_PSP_GMC_INIT_TRACE")
  @staticmethod
  def trace_map_bar5_first() -> int: return _env_int("AM_PSP_TRACE_MAP_BAR5_FIRST")
  @staticmethod
  def trace_map_bar0_last() -> int: return _env_int("AM_PSP_TRACE_MAP_BAR0_LAST")

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
