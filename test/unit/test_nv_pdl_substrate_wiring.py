"""Hermetic pin for the native NV PDL wiring (S4 substrate half).

llama's overlap is single-stream PDL: the producer fires
`cudaTriggerProgrammaticLaunchCompletion` at kernel start and the consumer
calls `cudaGridDependencySynchronize` before reading producer output. The
native QMD v05 equivalent (proven on silicon in the 08-17 probe) is: producer
QMD `arrive_at_latch` + program pre-exit, consumer QMD `wait_on_latch`, and a
`griddepcontrol.wait` (SASS ACQBULK) at the top of the consumer kernel. This
test pins the two wiring halves -- the renderer emission and the exec-path
QMD arming -- so a future refactor cannot silently drop either.

Both are env-gated and name-pinned; the default (empty lists) must leave every
kernel source and every QMD byte-identical, which the no-match cases assert.
"""
import os

from tinygrad.helpers import Context, JIT_BATCH_SIZE, NV_FLASH_LOAD_SCHEDULE
from tinygrad.llm.model import Transformer
from tinygrad.renderer.cuda import _nv_min_blocks_source, _nv_pdl_body, _nv_pdl_match
from tinygrad.runtime.ops_nv import QMD, NVComputeQueue, _nv_pdl_arm_pair


def _queue() -> NVComputeQueue:
  return NVComputeQueue(queue_idx=0)


def test_match_exact_and_prefix():
  spec = frozenset(["exact_name", "prefix:E_"])
  assert _nv_pdl_match("exact_name", spec)
  assert _nv_pdl_match("E_1187_16_4", spec)
  assert not _nv_pdl_match("r_1187_16_4", spec)
  assert not _nv_pdl_match("exact_name_suffix", spec)


def test_min_blocks_source_is_exact_name_gated_and_default_closed(monkeypatch):
  source = 'extern "C" __global__ void __launch_bounds__(128) flash_target() {}'
  monkeypatch.delenv("NV_MIN_BLOCKS_PROGRAMS", raising=False)
  assert _nv_min_blocks_source("flash_target", source) == source
  monkeypatch.setenv("NV_MIN_BLOCKS_PROGRAMS", "flash_other,prefix:flash_vec_")
  assert _nv_min_blocks_source("flash_target", source) == source
  monkeypatch.setenv("NV_MIN_BLOCKS_PROGRAMS", "flash_target")
  assert "__launch_bounds__(128, 1)" in _nv_min_blocks_source("flash_target", source)


def test_flash_load_schedule_context_only_marks_score_program(monkeypatch):
  monkeypatch.delenv("NV_MIN_BLOCKS_PROGRAMS", raising=False)
  score = 'extern "C" __global__ void __launch_bounds__(128) flash_vec_llama_score_pv_32_128_8_widekv16() {}'
  combine = 'extern "C" __global__ void __launch_bounds__(128) flash_fused_gmax_combine_f16_32_128_s8_lw128() {}'
  with Context(NV_FLASH_LOAD_SCHEDULE=1):
    assert "__launch_bounds__(128, 1)" in _nv_min_blocks_source("flash_vec_llama_score_pv_32_128_8_widekv16", score)
    assert _nv_min_blocks_source("flash_fused_gmax_combine_f16_32_128_s8_lw128", combine) == combine


def test_flash_load_schedule_capture_scope_is_closed_and_restores_context():
  model = object.__new__(Transformer)
  model._decode_flash_load_schedule_promoted = True
  before = (NV_FLASH_LOAD_SCHEDULE.value, JIT_BATCH_SIZE.value)
  with model._decode_flash_load_schedule_substrate(True):
    assert (NV_FLASH_LOAD_SCHEDULE.value, JIT_BATCH_SIZE.value) == (1, 33)
  assert (NV_FLASH_LOAD_SCHEDULE.value, JIT_BATCH_SIZE.value) == before
  with model._decode_flash_load_schedule_substrate(False):
    assert (NV_FLASH_LOAD_SCHEDULE.value, JIT_BATCH_SIZE.value) == before


def test_renderer_consumer_gets_wait_at_top(monkeypatch):
  monkeypatch.setenv("NV_PDL_CONSUMER_PROGRAMS", "prefix:E_")
  monkeypatch.setenv("NV_PDL_PRODUCER_PROGRAMS", "")
  body = _nv_pdl_body("E_1187_16_4", ["  float x = 1.0;", "  buf[0] = x;"])
  assert body[0] == '  asm volatile("griddepcontrol.wait;");'
  assert body[-1] == "  buf[0] = x;"


def test_renderer_producer_gets_launch_at_end(monkeypatch):
  monkeypatch.setenv("NV_PDL_CONSUMER_PROGRAMS", "")
  monkeypatch.setenv("NV_PDL_PRODUCER_PROGRAMS", "prefix:q4k_g3_lanemap")
  body = _nv_pdl_body("q4k_g3_lanemap_gemv_1024_4096", ["  out[i] = v;"])
  assert body[-1] == '  asm volatile("griddepcontrol.launch_dependents;");'
  assert body[0] == "  out[i] = v;"


def test_renderer_producer_trigger_start_emits_launch_at_top(monkeypatch):
  monkeypatch.setenv("NV_PDL_CONSUMER_PROGRAMS", "")
  monkeypatch.setenv("NV_PDL_PRODUCER_PROGRAMS", "prefix:q4k_g3_lanemap")
  monkeypatch.setenv("NV_PDL_TRIGGER_POSITION", "start")
  body = _nv_pdl_body("q4k_g3_lanemap_gemv_1024_4096", ["  out[i] = v;"])
  assert body[0] == '  asm volatile("griddepcontrol.launch_dependents;");'
  assert body[-1] == "  out[i] = v;"


def test_renderer_trigger_unset_matches_end_byte_for_byte(monkeypatch):
  monkeypatch.setenv("NV_PDL_CONSUMER_PROGRAMS", "")
  monkeypatch.setenv("NV_PDL_PRODUCER_PROGRAMS", "prefix:q4k_g3_lanemap")
  monkeypatch.delenv("NV_PDL_TRIGGER_POSITION", raising=False)
  default = _nv_pdl_body("q4k_g3_lanemap_gemv_1024_4096", ["  float x = 1.0;", "  out[i] = x;"])
  monkeypatch.setenv("NV_PDL_TRIGGER_POSITION", "end")
  explicit_end = _nv_pdl_body("q4k_g3_lanemap_gemv_1024_4096", ["  float x = 1.0;", "  out[i] = x;"])
  assert default == explicit_end == ["  float x = 1.0;", "  out[i] = x;",
                                     '  asm volatile("griddepcontrol.launch_dependents;");']


def test_renderer_default_is_byte_identical(monkeypatch):
  monkeypatch.setenv("NV_PDL_CONSUMER_PROGRAMS", "")
  monkeypatch.setenv("NV_PDL_PRODUCER_PROGRAMS", "")
  body = ["  float x = 1.0;", "  buf[0] = x;"]
  assert _nv_pdl_body("E_1187_16_4", list(body)) == body


def _fake_dev():
  class _Iface:
    compute_class = 0xcdc0  # Blackwell compute class (>= BLACKWELL_COMPUTE_A)
  class _Dev:
    iface = _Iface()
  return _Dev()


def test_exec_arms_latch_pair_fields(monkeypatch):
  monkeypatch.setenv("NV_PDL_PRODUCER_PROGRAMS", "prefix:q4k_g3_lanemap")
  monkeypatch.setenv("NV_PDL_CONSUMER_PROGRAMS", "prefix:E_")
  monkeypatch.setenv("NV_PDL_LATCH_ID", "7")
  dev = _fake_dev()
  active = QMD(dev=dev)
  new = QMD(dev=dev)
  assert _nv_pdl_arm_pair(active, new, "q4k_g3_lanemap_gemv_1024_4096", "E_1187_16_4")
  assert active.read("arrive_at_latch_valid") == 1
  assert active.read("arrive_at_latch_id") == 7
  assert active.read("enable_program_pre_exit") == 1
  assert active.read("pre_exit_at_last_cta_launch") == 1
  assert new.read("wait_on_latch_valid") == 1
  assert new.read("wait_on_latch_id") == 7


def test_exec_no_match_leaves_qmds_untouched(monkeypatch):
  monkeypatch.setenv("NV_PDL_PRODUCER_PROGRAMS", "prefix:q4k_g3_lanemap")
  monkeypatch.setenv("NV_PDL_CONSUMER_PROGRAMS", "prefix:E_")
  dev = _fake_dev()
  active = QMD(dev=dev)
  new = QMD(dev=dev)
  assert not _nv_pdl_arm_pair(active, new, "r_16_4_1187", "r_16_8")
  assert active.read("arrive_at_latch_valid") == 0
  assert new.read("wait_on_latch_valid") == 0


def test_exec_default_disabled_is_byte_identical(monkeypatch):
  monkeypatch.setenv("NV_PDL_PRODUCER_PROGRAMS", "")
  monkeypatch.setenv("NV_PDL_CONSUMER_PROGRAMS", "")
  dev = _fake_dev()
  active = QMD(dev=dev)
  new = QMD(dev=dev)
  assert not _nv_pdl_arm_pair(active, new, "q4k_g3_lanemap_gemv_1024_4096", "E_1187_16_4")
  assert active.read("arrive_at_latch_valid") == 0
  assert new.read("wait_on_latch_valid") == 0


def test_queue_tracks_active_program_name():
  q = _queue()
  assert q.active_prg_name is None
