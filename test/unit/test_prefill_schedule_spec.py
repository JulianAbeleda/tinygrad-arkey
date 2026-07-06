from types import SimpleNamespace

from extra.qk.prefill_schedule_spec import (
  PIPELINE_TARGET_SUBSTRATE, PrefillGEMMScheduleSpec, describe_prefill_schedule,
  emit_prefill_gemm_from_spec, prefill_pipe_role_selective_generated_pure_search_proof)


def _schedule_spec(route_family: str, *, out_f: int = 4096, in_f: int = 4096, reloc: bool = True) -> PrefillGEMMScheduleSpec:
  return PrefillGEMMScheduleSpec(
    m=512, n=out_f, k=in_f, route_family=route_family, tile_m=128, tile_n=128, tile_k=32,
    waves_m=2, waves_n=2, wm=4, wn=4, pipe_tm=2, pipe_tn=2, pipeline_depth=2, threads=128,
    dbuf=1, plra=0, plrab=0, pad=16, leanaddr=0, role="ffn_down", reloc=reloc)


def test_describe_prefill_schedule_keeps_role_policy_in_spec():
  from extra.qk import prefill_graph_gemm_route as route
  route._resolve_schedule.cache_clear()

  spec_pipe = describe_prefill_schedule(4096, 4096, role="attn_kv")
  assert spec_pipe.route_family == "pipe"
  assert spec_pipe.role == "attn_kv"

  spec_lds = describe_prefill_schedule(12288, 5120, role="ffn_gate_up")
  assert spec_lds.route_family == "lds"
  assert spec_lds.role == "ffn_gate_up"
  assert spec_lds.protected_roles == ("ffn_gate_up",)


def test_emit_prefill_gemm_from_spec_targets_expected_wmma_builders(monkeypatch):
  from extra.qk import prefill_graph_gemm_route as route

  calls = []

  def fake_pipe(m, n, k, tm, tn):
    calls.append(("pipe", m, n, k, tm, tn))
    return ("pipe_ins",)

  def fake_lds(m, n, k, waves_m, waves_n, wm, wn, bk, pad, dbuf, PLRA, PLRAB, LEANADDR):
    calls.append(("lds", m, n, k, waves_m, waves_n, wm, wn, bk, pad, dbuf, PLRA, PLRAB, LEANADDR))
    return ("lds_ins",)

  fake_ref = SimpleNamespace(build_gemm_pipe=fake_pipe, build_gemm_lds2=fake_lds)
  monkeypatch.setattr(route, "ref", fake_ref)

  pipe_out = emit_prefill_gemm_from_spec(_schedule_spec("pipe"))
  assert calls == [("pipe", 512, 4096, 4096, 2, 2)]
  assert pipe_out[0] == ("pipe_ins",)
  assert pipe_out[4] == 128
  assert pipe_out[5] == "prefill_gen_sched_gemm_512_4096_4096"

  calls.clear()
  lds_out = emit_prefill_gemm_from_spec(_schedule_spec("lds", reloc=False))
  assert calls[0][0] == "lds"
  assert lds_out[0] == ("lds_ins",)
  assert lds_out[4] == 128


def test_prefill_pipe_role_selective_generated_pure_search_proof_points_to_shaped_wmma_rangeify():
  proof = prefill_pipe_role_selective_generated_pure_search_proof()
  assert proof["route_id"] == "prefill_pipe_role_selective_generated"
  assert proof["is_pure"] is False
  assert proof["blocker"] == "Ops.INS"
  assert PIPELINE_TARGET_SUBSTRATE == tuple(proof["target_lowering_substrate"]["target"])
  assert "tinygrad.schedule.rangeify" in proof["target_lowering_substrate"]["path"]
  assert proof["target_lowering_substrate"]["goal"] == "backend-owned matrix instructions via Tinygrad IR"
  assert proof["executing_surface"]["writer"] == "extra/qk/prefill_graph_gemm_route.py::route_pf16_graph_gemm"
