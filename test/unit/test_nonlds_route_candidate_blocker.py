import pytest
from extra.qk.prefill_schedule_spec import describe_prefill_schedule
from extra.qk.wmma_pipe_spec import extract_wmma_pipe_spec, lower_wmma_pipe_spec, build_wmma_pipe_diagnostic_lowering_report

def test_attn_qo_lean_route_surface_is_pipe_and_lowerer_is_explicitly_blocked():
  spec = describe_prefill_schedule(4096, 4096, role="attn_qo")
  pipe = extract_wmma_pipe_spec(spec)
  assert pipe is not None and (pipe.m, pipe.n, pipe.k, pipe.role) == (512, 4096, 4096, "attn_qo")
  with pytest.raises(NotImplementedError): lower_wmma_pipe_spec(pipe)

def test_attn_qo_diagnostic_is_generated_but_not_route_bound():
  pipe = extract_wmma_pipe_spec(describe_prefill_schedule(4096, 4096, role="attn_qo"))
  report = build_wmma_pipe_diagnostic_lowering_report(pipe)
  assert report["route_bound"] is False
  assert report["uses_hand_pipe_oracle"] is False
  assert report["mvp_structure_ok"] is True
