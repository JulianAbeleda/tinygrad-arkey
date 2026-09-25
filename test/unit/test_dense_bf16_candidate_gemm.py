import json
from pathlib import Path

import pytest

from tinygrad import Tensor, dtypes
from tinygrad.llm import dense_candidate_gemm as dense
from tinygrad.llm.prefill_candidate_runtime import NV_ARTIFACT, expand_compact_candidate_set, promoted_candidate_set

ROOT = Path(__file__).parents[2]
NV = {"backend":"NV", "arch":"sm_120", "wave_size":32}
NEMOTRON_ROLES = {"ssm_in":(17536, 3136), "ssm_out":(3200, 7680), "attn_q":(5120, 3136), "attn_kv":(1024, 3136),
                  "attn_o":(3200, 5120), "ffn_up":(12544, 3136), "ffn_down":(3200, 12544), "output":(131072, 3136)}


def _raw(): return json.loads(dense.DENSE_BF16_ARTIFACT.read_text())


def test_dense_bf16_artifact_reuses_the_promoted_nv_schedule_verbatim():
  raw, nv = _raw(), json.loads(NV_ARTIFACT.read_text())
  assert raw["target"] == nv["target"] == NV and raw["route_id"] == nv["route_id"]
  assert raw["template"]["schedule"] == nv["template"]["schedule"]
  assert raw["template"]["static_constraints"] == nv["template"]["static_constraints"]
  assert raw["template"]["dtypes"] == {"a":"bf16", "b":"bf16", "accumulator":"fp32", "c":"fp32"}


def test_dense_bf16_artifact_is_reproduced_by_the_mint():
  from extra.llm_research.mint_typed_candidate_template import mint_dense_bf16
  assert mint_dense_bf16() == _raw()


def test_dense_bf16_registry_covers_exact_tile_padded_nemotron_rows():
  registry = dense.dense_bf16_registry(**NV)
  keys = {(a.normalized_payload["workload"]["role"], *(a.normalized_payload["workload"]["shape"][x] for x in "mnk"))
          for a in registry.admissions}
  assert keys == {(role, m, n, k) for role, (n, k) in NEMOTRON_ROLES.items() for m in (128, 256, 384, 512)
                  if role != "attn_kv" or m == 512}   # attn_kv lost to the safe TC path below 512 rows
  assert dense.promoted_row_chunks(registry) == (128, 256, 384, 512)
  assert registry.get("ffn_down", (512, 3136, 12544), NV) is None   # unpadded N is never admitted
  assert dense.dense_bf16_registry("NV", "sm_89", 32) is None and dense.dense_bf16_registry("AMD", "gfx1100", 32) is None


def test_dense_bf16_artifact_identity_drift_fails_closed():
  raw = _raw()
  raw["entries"][0]["shape"]["m"] = 640
  with pytest.raises(ValueError, match="identity drifted"): expand_compact_candidate_set(raw, **NV)
  raw = _raw()
  raw["template"]["schedule"]["tile"]["k"] = 64
  with pytest.raises(ValueError, match="identity drifted"): expand_compact_candidate_set(raw, **NV)


def test_qwen_compact_artifact_still_requires_its_four_rows():
  assert len(promoted_candidate_set(**NV).entries) == 4
  raw = json.loads(NV_ARTIFACT.read_text())
  raw["entries"] = raw["entries"][:3]
  with pytest.raises(ValueError, match="must contain 4 rows"): expand_compact_candidate_set(raw, **NV, row_count=4)


def test_route_declines_without_a_promoted_target_or_exact_row(monkeypatch):
  x, w = Tensor.ones(4, 3136), Tensor.ones(5120, 3136, dtype=dtypes.bfloat16)
  monkeypatch.setattr(dense, "device_target", lambda device: {"backend":"CPU", "arch":"x86_64", "wave_size":None})
  assert dense.route_dense_bf16(x, w, "attn_q", 5120) is None
  monkeypatch.setattr(dense, "device_target", lambda device: dict(NV))
  assert dense.route_dense_bf16(x, w, "attn_q", 5120, min_rows=8) is None          # below the declared row floor
  assert dense.route_dense_bf16(x, w.cast(dtypes.half), "attn_q", 5120) is None    # fp16 weight: not this artifact
  assert dense.route_dense_bf16(x, Tensor.ones(5000, 3136, dtype=dtypes.bfloat16), "attn_q", 5000) is None  # no exact row
  assert dense.route_dense_bf16(Tensor.ones(4, 1000), w, "attn_q", 5120) is None   # K mismatch


def test_route_plans_padded_row_chunks_through_exact_rows(monkeypatch):
  monkeypatch.setattr(dense, "device_target", lambda device: dict(NV))
  installed = []
  monkeypatch.setattr(dense, "_install", lambda admission, m, n, k: installed.append((admission.normalized_payload["workload"]["role"], m, n, k)))
  out = dense.route_dense_bf16(Tensor.ones(1, 700, 3136), Tensor.ones(3200, 3136, dtype=dtypes.bfloat16), "ffn_down_absent", 3136)
  assert out is None and installed == []
  out = dense.route_dense_bf16(Tensor.ones(1, 700, 3136), Tensor.ones(1024, 3136, dtype=dtypes.bfloat16), "attn_kv", 1024)
  assert out is None and installed == []   # the 188-row tail pads to an unpromoted attn_kv row, so the whole projection declines
  out = dense.route_dense_bf16(Tensor.ones(1, 700, 3136), Tensor.ones(5120, 3136, dtype=dtypes.bfloat16), "attn_q", 5120)
  assert out is not None and out.shape == (1, 700, 5120) and out.dtype == dtypes.float
  assert installed == [("attn_q", 512, 5120, 3136), ("attn_q", 256, 5120, 3136)]


def test_padded_product_is_materialized_before_the_slice(monkeypatch):
  # A lazy slice of the padded product would shrink the GEMM to the unpromoted (m, n) shape.
  from tinygrad.uop.ops import Ops
  monkeypatch.setattr(dense, "device_target", lambda device: dict(NV))
  monkeypatch.setattr(dense, "_install", lambda *args: None)
  out = dense.route_dense_bf16(Tensor.ones(1, 100, 7680), Tensor.ones(3200, 7680, dtype=dtypes.bfloat16), "ssm_out", 3136)
  assert out.shape == (1, 100, 3136)
  assert any(u.op is Ops.CONTIGUOUS and u.shape == (128, 3200) for u in out.uop.toposort())
