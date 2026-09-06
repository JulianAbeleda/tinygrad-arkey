from extra.llm_research.decode.decode_runtime_overhead import _token_evidence


def test_disabled_reduce_output_trace_skips_graph_walk():
  from tinygrad.callify import _trace_reduce_output_markers
  # An invalid graph input makes any attempted UOp.sink/toposort walk fail.
  _trace_reduce_output_markers((object(),), "disabled")


def test_complete_token_ids_are_opt_in_for_cross_runtime_correctness():
  compact = _token_evidence([1, 2, 3])
  complete = _token_evidence([1, 2, 3], include_token_ids=True)
  assert "token_ids" not in compact
  assert complete["token_ids"] == [1, 2, 3]
  assert complete["sha256"] == compact["sha256"]


def test_atomic_evidence_serializes_symbolic_uop_and_rejects_unknown(tmp_path):
  import json, pytest
  from tinygrad import UOp
  from extra.llm_research.decode.decode_runtime_overhead import _atomic_json
  path = tmp_path / "symbolic.json"
  symbolic = UOp.variable("evidence_bound", 0, 8)
  _atomic_json(path, {"bound":symbolic})
  assert json.loads(path.read_text())["bound"] == {
    "kind":"symbolic_uop_unresolved", "key":symbolic.key.hex(), "op":symbolic.op.name,
    "dtype":str(symbolic.dtype), "expression":str(symbolic), "arg":repr(symbolic.arg), "bound_value":None}
  with pytest.raises(TypeError): _atomic_json(path, {"unsupported":object()})


def test_observed_jits_include_both_slots_and_flash_variants():
  from types import SimpleNamespace as NS
  from extra.llm_research.decode.decode_runtime_overhead import _decode_jits, _used_decode_jits
  pair = [NS(cnt=2, captured=object()), NS(cnt=2, captured=object())]
  model = NS(rollout_jit=NS(cnt=0, captured=None), rollout_greedy_pingpong_jits_flash_live={64:pair},
             prefill_jit=NS(cnt=9, captured=object()))
  before = {name:jit.cnt for name,jit in _decode_jits(model).items()}
  pair[0].cnt += 1
  pair[1].cnt += 1
  assert list(_used_decode_jits(model, before)) == [
    'rollout_greedy_pingpong_jits_flash_live[64][0]', 'rollout_greedy_pingpong_jits_flash_live[64][1]']


def test_capture_warms_and_observes_both_production_slots(monkeypatch):
  from types import SimpleNamespace as NS
  from extra.llm_research.decode import decode_runtime_overhead as mod
  pair = [NS(cnt=0, captured=None), NS(cnt=0, captured=None)]
  model = NS(_decode_feedback_pingpong_promoted=True, rollout_greedy_pingpong_jits=pair,
             reset_generation_state=lambda:None)
  def generation():
    for index in range(6):
      jit = pair[index % 2]
      jit.cnt += 1
      if jit.cnt >= 2: jit.captured = object()
      yield 1
  monkeypatch.setattr(mod, '_prefill', lambda *args:(generation(), 1))
  _, selected, count = mod._capture_decode_graph(model, [1]*128, 32, 3, False)
  assert count == 6
  assert len(selected) == 2
  assert all(jit.cnt == 3 and jit.captured is not None for jit in selected.values())


def test_captured_program_evidence_records_identity_geometry_and_order():
  from types import SimpleNamespace as NS
  from tinygrad import dtypes, UOp
  from tinygrad.uop.ops import Ops, ProgramInfo
  from extra.llm_research.decode.decode_runtime_overhead import _captured_program_evidence
  body, device = UOp(Ops.SINK), UOp(Ops.DEVICE, arg="CPU")
  source, binary = UOp(Ops.SOURCE, arg="kernel source"), UOp(Ops.BINARY, arg=b"kernel binary")
  program = UOp(Ops.PROGRAM, dtypes.void, src=(body, device, UOp(Ops.LINEAR), source, binary),
                arg=ProgramInfo("captured_test", global_size=(7, 2, 1), local_size=(32, 1, 1)))
  captured = NS(linear=UOp(Ops.LINEAR, src=(program.call(), program.call())))
  rows = _captured_program_evidence(NS(captured=captured))
  assert [row["ordinal"] for row in rows] == [0, 1]
  assert all(row["program_name"] == "captured_test" and row["global_size"] == [7, 2, 1] for row in rows)
  assert len({row["source_sha256"] for row in rows}) == len({row["binary_sha256"] for row in rows}) == 1
  assert all(row["source_text"] == "kernel source" for row in rows)


def test_nv_gpu_state_requires_and_names_every_field(monkeypatch):
  from extra.llm_research.decode import decode_runtime_overhead as mod
  values=[str(i) for i in range(len(mod.NV_STATE_FIELDS))]
  monkeypatch.setattr(mod.subprocess, "check_output", lambda *args, **kwargs: ", ".join(values))
  assert mod._nv_gpu_state() == dict(zip(mod.NV_STATE_FIELDS, values))


def test_measure_w_gpu_state_brackets_only_timed_decode_window(monkeypatch):
  from extra.llm_research.decode import decode_runtime_overhead as mod
  events=[]
  class Gen:
    def __next__(self): events.append("token"); return 7
    def close(self): events.append("close")
  class Dev:
    def synchronize(self): events.append("sync")
  monkeypatch.setattr(mod, "_reset", lambda model:events.append("reset"))
  monkeypatch.setattr(mod, "_prefill", lambda *args:(events.append("prefill") or Gen(), 3))
  states=iter(({"point":"before"}, {"point":"after"}))
  monkeypatch.setattr(mod, "_nv_gpu_state", lambda:(events.append("state") or next(states)))
  _, _, tokens, prelude, before, after = mod._measure_w(object(), Dev(), [1], 1, 2, capture_gpu_state=True)
  assert events == ["reset", "prefill", "sync", "state", "token", "token", "state", "close"]
  assert tokens == [7, 7] and prelude == 3 and before["point"] == "before" and after["point"] == "after"
