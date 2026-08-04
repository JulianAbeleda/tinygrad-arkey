"""Hermetic CPU-only tests for the Phase 0 NV multi-queue probe construction seam.

These tests pin the contracts Phase 0 must hold without any RM/GPU hardware (no
Device["NV"], no probe run): the ordered construction op-plan builder and the
fake-RM executor that drives it. Covered here:

  (a) every NVA06F control targets the raw channel handle rm_alloc returned, never
      a GPFifo wrapper;
  (b) mode "group" schedules each fresh KEPLER_CHANNEL_GROUP_A and never
      dev.channel_group;
  (c) mode "ctxshare" order is CTXSHARE_ALLOC -> CHANNEL_ALLOC -> NVA06F_BIND ->
      NVA06F_GPFIFO_SCHEDULE -> NVA06C_GPFIFO_SCHEDULE;
  (d) the R1 verdict is a pure function: anchored hash equality AND the declared
      max-error bound, never np.allclose;
  (e) arm payloads are JSON-safe and carry the documented schema.

The live GPU question (whether the corrected construction co-schedules) is not
testable here; it is answered by the probe arms themselves, which are isolated
timed subprocesses.
"""
import importlib.util, json, pathlib

from tinygrad.runtime import ops_nv

PROBE = pathlib.Path(__file__).resolve().parents[2] / "extra" / "llm_research" / "decode" / "nv_multi_queue_probe.py"
_spec = importlib.util.spec_from_file_location("nv_multi_queue_probe", PROBE)
probe = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(probe)


class FakeArea:
  va_addr = 0x200000000000
  meta = type("Meta", (), {"hMemory": 0x7777})()


class FakeRM:
  """RM-shaped device slice with recorded calls; touches no hardware."""
  def __init__(self):
    self.nvdevice, self.channel_group, self.ctxshare, self.vaspace = 0x100, 0x200, 0x300, 0x400
    self.next_handle = 0x5000
    self.alloc_calls: list[tuple] = []
    self.alloced: dict[int, tuple] = {}
    self.control_calls: list[tuple] = []
    self.channel_alloc_calls: list[tuple] = []
    self.iface = self

  def alloc(self, size, **kwargs):
    self.alloc_calls.append((size, kwargs))
    return FakeArea()

  def rm_alloc(self, parent, clss, params):
    h = self.next_handle
    self.next_handle += 1
    self.alloced[h] = (parent, clss, params)
    return h

  def rm_control(self, obj, cmd, params):
    self.control_calls.append((obj, cmd, params))
    return params

  def _new_gpu_fifo(self, area, ctxshare, group, offset=0, entries=0x400, compute=False, video=False,
                    debugger=True, engine_type=None):
    h = self.next_handle
    self.next_handle += 1
    self.channel_alloc_calls.append((ctxshare, group, engine_type))
    # A real GPFifo wrapper, so tests prove controls take .handle, never the wrapper.
    return ops_nv.GPFifo(ring=None, gpput=None, entries_count=entries, token=0x1234, handle=h)


def collect(fake, mode, engines):
  rm_ops = []
  fifos, errors = probe.extra_gpfifos(fake, engines, mode=mode, on_rm_op=rm_ops.append)
  return fifos, errors, rm_ops


def test_nva06f_controls_target_raw_channel_handle_never_gpfifo_wrapper():
  fake = FakeRM()
  fifos, errors, rm_ops = collect(fake, "ctxshare", [0, 4])
  assert len(fifos) == 2 and not errors
  nva06f = [op for op in rm_ops if op["op"] in ("NVA06F_BIND", "NVA06F_GPFIFO_SCHEDULE")]
  assert len(nva06f) == 4
  assert {op["channel"] for op in nva06f} == {f.handle for f in fifos}
  for obj, cmd, _ in fake.control_calls:
    assert isinstance(obj, int) and not isinstance(obj, ops_nv.GPFifo), \
      "NVA06F/NVA06C controls must take the raw RM handle, never a GPFifo wrapper"
    if cmd in (ops_nv.nv_gpu.NVA06F_CTRL_CMD_BIND, ops_nv.nv_gpu.NVA06F_CTRL_CMD_GPFIFO_SCHEDULE):
      assert obj in {f.handle for f in fifos}
  assert all(isinstance(f.handle, int) for f in fifos)


def test_group_mode_schedules_the_fresh_group_never_boot_group():
  fake = FakeRM()
  fifos, errors, rm_ops = collect(fake, "group", [0, 0])
  assert len(fifos) == 2 and not errors
  group_allocs = [h for h, (parent, clss, _) in fake.alloced.items() if clss == ops_nv.nv_gpu.KEPLER_CHANNEL_GROUP_A]
  assert len(group_allocs) == 2
  assert all(parent == fake.nvdevice for _, (parent, clss, _) in fake.alloced.items() if clss == ops_nv.nv_gpu.KEPLER_CHANNEL_GROUP_A)
  nva06c = [op for op in rm_ops if op["op"] == "NVA06C_GPFIFO_SCHEDULE"]
  assert len(nva06c) == 2
  assert all(op["group"] != fake.channel_group for op in nva06c)
  assert {op["group"] for op in nva06c} == set(group_allocs)
  assert all(obj != fake.channel_group for obj, _, _ in fake.control_calls), \
    "mode group must never schedule dev.channel_group"


def test_ctxshare_mode_op_order():
  fake = FakeRM()
  fifos, errors, rm_ops = collect(fake, "ctxshare", [0])
  assert len(fifos) == 1 and not errors
  assert [op["op"] for op in rm_ops] == ["CTXSHARE_ALLOC", "CHANNEL_ALLOC", "NVA06F_BIND",
                                         "NVA06F_GPFIFO_SCHEDULE", "NVA06C_GPFIFO_SCHEDULE"]
  assert rm_ops[0]["group"] == fake.channel_group
  assert rm_ops[-1]["group"] == fake.channel_group


def test_shared_mode_matches_control_arm_construction():
  fake = FakeRM()
  fifos, errors, rm_ops = collect(fake, "shared", [0, 0])
  assert len(fifos) == 2 and not errors
  assert [op["op"] for op in rm_ops] == ["CHANNEL_ALLOC", "CHANNEL_ALLOC", "NVA06C_GPFIFO_SCHEDULE"]
  assert rm_ops[-1]["group"] == fake.channel_group
  assert all(op["group"] == fake.channel_group for op in rm_ops)


def test_failed_channel_step_is_recorded_and_channel_excluded():
  class FlakyRM(FakeRM):
    def __init__(self):
      super().__init__()
      self._bind_attempts = 0

    def rm_control(self, obj, cmd, params):
      if cmd == ops_nv.nv_gpu.NVA06F_CTRL_CMD_BIND:
        self._bind_attempts += 1
        if self._bind_attempts == 1: raise RuntimeError("bind rejected")
      return super().rm_control(obj, cmd, params)

  fake = FlakyRM()
  rm_ops = []
  fifos, errors = probe.extra_gpfifos(fake, [0, 0], mode="ctxshare", on_rm_op=rm_ops.append)
  assert len(fifos) == 1, "the channel whose bind failed must be excluded from execution"
  assert errors and errors[0]["op"] == "NVA06F_BIND"
  assert [op["op"] for op in rm_ops] == ["CTXSHARE_ALLOC", "CHANNEL_ALLOC", "NVA06F_BIND",
                                         "CTXSHARE_ALLOC", "CHANNEL_ALLOC", "NVA06F_BIND",
                                         "NVA06F_GPFIFO_SCHEDULE", "NVA06C_GPFIFO_SCHEDULE"]


def test_construction_plan_is_pure_and_symbolic():
  plan = probe.build_construction_plan("group", [0, 4])
  assert [op["op"] for op in plan] == ["CHANNEL_GROUP_ALLOC", "CTXSHARE_ALLOC", "CHANNEL_ALLOC", "NVA06F_BIND",
                                       "NVA06F_GPFIFO_SCHEDULE", "NVA06C_GPFIFO_SCHEDULE"] * 2
  assert plan[0]["group"] == "group:0" and plan[6]["group"] == "group:1"
  assert [op["engine_type"] for op in plan] == [0] * 6 + [4] * 6
  assert all(op["requires_channel"] == op["engine_index"] for op in plan if op["op"] in ("NVA06F_BIND", "NVA06C_GPFIFO_SCHEDULE"))


def test_r1_verdict_requires_hash_match_and_max_error_bound():
  ok = ("h1", "h2")
  assert probe.r1_contract_pass(ok, ok, (1e-6, 2e-6), (1.0, 1.0), tol=1e-3)
  assert probe.r1_contract_pass(ok, ok, (0.0, 1e-3), (2.0, 1.0), tol=1e-3)  # exactly at the bound
  assert not probe.r1_contract_pass(("h1", "wrong"), ok, (0.0, 0.0), (1.0, 1.0), tol=1e-3), \
    "hash mismatch must fail even with zero error"
  assert not probe.r1_contract_pass(ok, ok, (2e-3, 1e-6), (1.0, 1.0), tol=1e-3), \
    "error above the declared bound must fail even with matching hashes"


def test_r2_serial_contract_uses_tolerance_not_equality():
  assert probe.serial_contract_ok(100.0, 101.9, max_pct=2.0, max_abs_us=10.0)  # pct 1.9 <= 2
  assert probe.serial_contract_ok(100.0, 105.0, max_pct=2.0, max_abs_us=10.0)  # abs 5 <= 10
  assert not probe.serial_contract_ok(100.0, 85.0, max_pct=2.0, max_abs_us=10.0)  # pct 17.6, abs 15


def test_arm_payload_schema_round_trips_and_carries_expected_keys():
  payload = probe.arm_payload_schema("ctxshare", [0, 0], n=1 << 20, matmul=2048, grid_div=4)
  payload["rm_ops"].append({"op": "NVA06F_BIND", "kind": "NVA06F", "group": 0x200, "channel": 0x5001,
                            "engine_type": 0, "status": "ok", "error": None})
  row = probe.experiment_row("R1", "cross-gpfifo semaphore dep (hash + max-error contract)",
                             [(1.0, 3.0), (2.0, 5.0)], {"mode": "ctxshare", "exit_code": 0, "timed_out": False})
  payload["experiments"].append(row)
  back = json.loads(json.dumps(payload))
  assert back["schema"] == "tinygrad.nv_multi_queue_probe.v2"
  assert back["mode"] == "ctxshare"
  for key in ("n", "matmul", "engines", "grid_div", "gpfifo_engine_types", "rm_ops",
              "construction_errors", "errors", "experiments", "arm"):
    assert key in back, f"arm payload missing key {key}"
  assert set(back["arm"]) == {"mode", "exit_code", "timed_out"}
  assert set(back["experiments"][0]) >= {"name", "status", "check", "timestamps_us", "span_us",
                                         "node_sum_us", "overlap", "arm"}
