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

from tinygrad.renderer.cuda import _nv_pdl_body, _nv_pdl_match
from tinygrad.runtime.ops_nv import QMD, NVComputeQueue, _nv_pdl_arm_pair


def _queue() -> NVComputeQueue:
  return NVComputeQueue(queue_idx=0)


def test_match_exact_and_prefix():
  spec = frozenset(["exact_name", "prefix:E_"])
  assert _nv_pdl_match("exact_name", spec)
  assert _nv_pdl_match("E_1187_16_4", spec)
  assert not _nv_pdl_match("r_1187_16_4", spec)
  assert not _nv_pdl_match("exact_name_suffix", spec)


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
