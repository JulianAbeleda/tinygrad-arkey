import pytest

from tinygrad import Tensor, dtypes
from tinygrad.engine.realize import compile_linear
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops

from extra.qk.layout import Q6_K_BLOCK_BYTES, q6_k_reference
from extra.qk.q6k_wmma_prefill_spec import describe_q6k_wmma_prefill, emit_q6k_wmma_prefill


def _packed_q6k(n:int, k:int) -> Tensor:
  # Deterministic valid blocks: codes/scales vary, while d is finite and small.
  blocks = n * k // 256
  raw = bytearray((i * 29 + 7) & 0xff for i in range(blocks * Q6_K_BLOCK_BYTES))
  d = Tensor([0.03125], dtype=dtypes.float16).bitcast(dtypes.uint8).numpy().tolist()
  for b in range(blocks): raw[b*Q6_K_BLOCK_BYTES+208:b*Q6_K_BLOCK_BYTES+210] = bytes(d)
  return Tensor(list(raw), dtype=dtypes.uint8)


def test_q6k_wmma_prefill_spec_is_typed_and_bounded():
  spec = describe_q6k_wmma_prefill(16, 32, 256, role="ffn_down")
  assert spec.to_json()["bounds"] == {"m": 512, "n": 4096, "k": 12288}
  assert spec.packed_bytes == 32 * Q6_K_BLOCK_BYTES
  with pytest.raises(ValueError, match="exceeds bounded maximum"):
    describe_q6k_wmma_prefill(16, 8192, 256, role="ffn_down")
  with pytest.raises(ValueError, match="unsupported role"):
    describe_q6k_wmma_prefill(16, 16, 256, role="model_specific")  # type: ignore[arg-type]


def test_q6k_wmma_admits_real_14b_ffn_down_role_shape():
  spec = describe_q6k_wmma_prefill(512, 4096, 12288, role="ffn_down")
  assert spec.admission()["admitted"] is True
  assert spec.packed_bytes == 4096 * 12288 // 256 * Q6_K_BLOCK_BYTES
  assert spec.to_json()["stage_boundary"] == "Q6_K bytes -> fp16 weights -> WMMA"


def test_q6k_wmma_admission_is_staged_and_explicitly_corrected():
  spec = describe_q6k_wmma_prefill(16, 16, 256)
  gate = spec.admission()
  assert gate["admitted"] is True
  assert gate["route"] == "staged_dequant_then_fp16_wmma"
  assert gate["quant_correction"] == "d * scale_i8 * (code_u6 - 32)"
  assert spec.to_json()["stage_boundary"] == "Q6_K bytes -> fp16 weights -> WMMA"
  assert spec.admission(fused=True)["admitted"] is False
  assert any("no legal gfx1100 WMMA lowering" in e for e in spec.admission(fused=True)["errors"])


def test_q6k_wmma_rejects_unknown_target_before_dispatch():
  spec = describe_q6k_wmma_prefill(16, 16, 256)
  bad = spec.__class__(m=16, n=16, k=256, role="test", target="amd_unknown")
  assert bad.admission()["admitted"] is False
  with pytest.raises(ValueError, match="admission failed"):
    emit_q6k_wmma_prefill(_packed_q6k(16, 256), Tensor.empty(16, 256, dtype=dtypes.float16), bad)


def test_q6k_wmma_prefill_bounded_correctness():
  spec = describe_q6k_wmma_prefill(16, 16, 256, role="test")
  packed, x = _packed_q6k(spec.n, spec.k), Tensor.randn(spec.m, spec.k).cast(dtypes.float16)
  got = emit_q6k_wmma_prefill(packed, x, spec).numpy()
  weight = q6_k_reference(packed, spec.n * spec.k).reshape(spec.n, spec.k).cast(dtypes.float16)
  expected = x.matmul(weight.transpose(), dtype=dtypes.float32).numpy()
  # fp16 WMMA accumulation order is intentionally allowed to differ from the CPU scheduler's reduction order.
  assert got == pytest.approx(expected, rel=5e-2, abs=6e-1)


def test_q6k_wmma_prefill_schedule_is_generated_contraction():
  spec = describe_q6k_wmma_prefill(16, 16, 256, role="test")
  out = emit_q6k_wmma_prefill(_packed_q6k(spec.n, spec.k), Tensor.empty(spec.m, spec.k, dtype=dtypes.float16), spec)
  linear = out.schedule_linear()
  topo = linear.toposort()
  sinks = [u for u in topo if u.op is Ops.SINK]
  assert any(u.op is Ops.REDUCE for u in topo), "fp16 matmul must retain a generated K contraction"
  assert any(getattr(u.arg, "name", None) == spec.kernel_name for u in sinks), "named scheduler-owned contraction missing"
  assert not any(u.op is Ops.CUSTOM for u in topo), "primitive must not contain a hand/custom kernel"


def test_q6k_dequant_once_compile_has_wmma_and_explicit_materialization_boundary():
  """This diagnostic proves TC compatibility, not fused-tile production eligibility."""
  spec = describe_q6k_wmma_prefill(16, 16, 256, role="test")
  with Context(DEV="AMD:ISA:gfx1100"):
    packed = Tensor.empty(spec.packed_bytes, dtype=dtypes.uint8)
    out = emit_q6k_wmma_prefill(packed, Tensor.empty(spec.m, spec.k, dtype=dtypes.float16), spec)
    compiled = compile_linear(out.schedule_linear())
  programs = [u.src[0] for u in compiled.src if u.op is Ops.CALL and u.src and u.src[0].op is Ops.PROGRAM]
  sources = [next((u.arg for u in program.src if u.op is Ops.SOURCE), "") for program in programs]
  assert len(programs) == 2, "dequant-once must remain visibly separate from its WMMA contraction"
  assert sum("wmma_f32_16x16x16_f16" in source for source in sources) == 1
