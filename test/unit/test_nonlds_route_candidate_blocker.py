import pytest
from extra.qk.prefill_schedule_spec import describe_prefill_schedule
from extra.qk.wmma_pipe_spec import extract_wmma_pipe_spec, lower_wmma_pipe_spec, build_wmma_pipe_diagnostic_lowering_report, pipe_candidate_context, build_wmma_pipe_ir, attach_pipe_candidate_context, WMMAPipeOp
from tinygrad.uop.ops import UOp, Ops

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

def test_pipe_candidate_context_preserves_identity_and_buffer_neutral_payload():
  pipe = extract_wmma_pipe_spec(describe_prefill_schedule(4096, 4096, role="attn_qo"))
  ctx = pipe_candidate_context(pipe, "a" * 64)
  assert ctx.canonical_identity == "a" * 64 and ctx.geometry is None
  payload = dict(ctx.pipeline)
  assert payload["schema"] == "wmma_pipe_ir.v1"
  assert payload["role"] == "attn_qo" and payload["shape"] == (512, 4096, 4096)
  assert payload["provenance"] == "compiler_owned_typed_pipe_ir"

def test_typed_pipe_ir_carries_lifecycle_without_native_isa():
  pipe = extract_wmma_pipe_spec(describe_prefill_schedule(4096, 4096, role="attn_qo"))
  ir = build_wmma_pipe_ir(pipe)
  assert ir.shape == (512, 4096, 4096) and ir.stages == 2
  assert ir.loads_per_stage == 8 and ir.provenance == "compiler_owned_typed_pipe_ir"

def test_pipe_context_attaches_to_ordinary_sink():
  pipe = extract_wmma_pipe_spec(describe_prefill_schedule(4096, 4096, role="attn_qo"))
  sink = UOp.sink()
  out = attach_pipe_candidate_context(sink, pipe_candidate_context(pipe, "b" * 64))
  assert out.op is Ops.SINK and out.arg.candidate_context.canonical_identity == "b" * 64

def test_pipe_op_contract_is_typed_and_compiler_owned():
  pipe = extract_wmma_pipe_spec(describe_prefill_schedule(4096, 4096, role="attn_qo"))
  op = WMMAPipeOp(build_wmma_pipe_ir(pipe), 0, 1, 2, (256, 4, 1), (256, 1, 1))
  assert op.wait_scope == "per_stage" and op.resource_owner == "compiler"
  assert op.resource_estimate()["lds_bytes"] > 0 and op.resource_estimate()["vgpr"] is None

def test_pipe_op_rejects_lifecycle_mismatch():
  pipe = extract_wmma_pipe_spec(describe_prefill_schedule(4096, 4096, role="attn_qo"))
  with pytest.raises(ValueError): WMMAPipeOp(build_wmma_pipe_ir(pipe), 0, 1, 2, (1, 1, 1), (1, 1, 1), wait_vmcnt=1)
