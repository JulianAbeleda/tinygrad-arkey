import json
import pytest

from tinygrad import dtypes
from tinygrad.codegen.late.reduce_output import emit_reduce_output_rope_kv_cache
from tinygrad.llm.producer_kv_cache_sink import ProducerKVCacheSinkAdmission, producer_kv_cache_sink_call
from tinygrad.llm.model_route_plan import (decode_producer_kv_cache_sink_promoted,
  load_decode_producer_kv_cache_sink_promotion)
from tinygrad.uop.ops import AxisType, Ops, ReduceOutputSpec, UOp


def _spec(**kwargs):
  return ReduceOutputSpec(8, 128, 1e-6, dtypes.float32, warps=8, lanes=32, per_lane=4,
                          epilogue="rope", **kwargs)


@pytest.mark.parametrize("cache_dtype", (dtypes.float16, dtypes.float32))
def test_exact_k_terminal_cache_sink_body(cache_dtype):
  emit = emit_reduce_output_rope_kv_cache(_spec(), dtypes.float32, dtypes.float16, cache_dtype, 1024)
  body = emit(UOp.placeholder((2, 1, 8, 1024, 128), cache_dtype, 0),
              UOp.placeholder((1, 1, 1024), dtypes.float32, 1),
              UOp.placeholder((128,), dtypes.float16, 2),
              UOp.placeholder((1, 1, 1024), dtypes.float32, 3),
              UOp.placeholder((1024, 128), dtypes.float32, 4))
  topo = body.toposort()
  assert body.arg.name == "reduce_output_rmsnorm_rope_kv_cache_8_128"
  assert sum(u.op is Ops.BARRIER for u in topo) == 1
  assert sum(u.op is Ops.STORE for u in topo) == 7  # accumulator x2, smem, K x2, V x2
  assert any(u.op is Ops.DEFINE_VAR and u.arg[0] == "start_pos" for u in topo)
  global_rows = [u for u in topo if u.op is Ops.RANGE and u.arg == (0, AxisType.GLOBAL)]
  assert len(global_rows) == 1 and global_rows[0].src[0].arg == 8


def test_cache_sink_emitter_fails_closed_outside_exact_contract():
  with pytest.raises(ValueError):
    emit_reduce_output_rope_kv_cache(
      ReduceOutputSpec(32, 128, 1e-6, dtypes.float32, warps=32, lanes=32, per_lane=4, epilogue="rope"),
      dtypes.float32, dtypes.float16, dtypes.float16, 1024)
  with pytest.raises(ValueError):
    emit_reduce_output_rope_kv_cache(_spec(), dtypes.float32, dtypes.float16, dtypes.int8, 1024)
  with pytest.raises(ValueError):
    emit_reduce_output_rope_kv_cache(_spec(), dtypes.float32, dtypes.float16, dtypes.float16, 0)


def test_cache_sink_admission_and_closed_default():
  assert ProducerKVCacheSinkAdmission(0).block_index == 0
  with pytest.raises(ValueError): ProducerKVCacheSinkAdmission(-1)
  with pytest.raises(ValueError): ProducerKVCacheSinkAdmission(True)
  # Admission is checked before any Tensor boundary is inspected.
  assert producer_kv_cache_sink_call(None, None, None, None, None, None, 1024) is None


def test_cache_sink_shipped_target_policy_and_rollback():
  enabled = lambda _name, default=0: default
  disabled = lambda name, default=0: 1 if name == "TINYGRAD_PRODUCER_KV_CACHE_SINK_DISABLE" else default
  assert decode_producer_kv_cache_sink_promoted(("NV", "sm_120"), enabled)
  assert not decode_producer_kv_cache_sink_promoted(("AMD", "gfx1100"), enabled)
  assert not decode_producer_kv_cache_sink_promoted(("CUDA", "sm_120"), enabled)
  assert not decode_producer_kv_cache_sink_promoted(("NV", "sm_120"), disabled)


def test_cache_sink_policy_loader_is_closed_default(tmp_path):
  path = tmp_path/"policy.json"
  path.write_text(json.dumps({"schema":"boltbeam.route_policy.v1",
                              "promoted_targets":[{"backend":"NV", "architecture":"sm_120"}]}))
  assert load_decode_producer_kv_cache_sink_promotion(str(path)) == frozenset({("NV", "sm_120")})
  path.write_text(json.dumps({"schema":"boltbeam.route_policy.v1"}))
  assert load_decode_producer_kv_cache_sink_promotion(str(path)) == frozenset()
