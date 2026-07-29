"""CPU-only characterization of observable LLM route/runtime contracts.

These assertions intentionally freeze values and failure boundaries rather than
where their owners live; the consolidation may move the public entry points.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tinygrad import UOp
from tinygrad.llm.model_route_plan import (build_model_route_plan, load_qk_route_policy, qk_route_policy_selected,
  qk_route_policy_selects_q4k_g3, qk_route_policy_selects_q6k_generated)
from tinygrad.llm.admission import immutable_prefill_policy, prefill_concrete_kv_auto_decision
from tinygrad.llm.prefill_attachments import PrefillDirectPackedBinding, PrefillRouteAttachment
from tinygrad.llm.prefill_route_observer import (PrefillRouteExecution, notify_prefill_route, notify_prefill_route_execution, observe_prefill_route_executions,
  observe_prefill_routes, prefill_route_scope)
from tinygrad.llm.prefill_candidate_runtime import automatic_promoted_prefill_graph_policy
from tinygrad.llm.decode_routes import decode_route_mode, should_use_flash_decode


ROOT = Path(__file__).parents[2]
ARTIFACT = ROOT / "tinygrad/llm/generated/prefill_wmma_lds_dbuf_candidate_set.json"


def _env(values): return lambda key, default=None: values.get(key, default)


def test_static_q4_q6_plan_contract_is_fixed_for_representative_roles():
  meta = {"tensor_infos": [
    ("blk.0.ffn_gate.weight", (4096, 12288), 12, 0),
    ("blk.0.ffn_down.weight", (12288, 4096), 12, 1),
    ("blk.0.attn_v.weight", (4096, 1024), 14, 2),
    ("output.weight", (4096, 151936), 14, 3),
  ]}
  plan = build_model_route_plan(meta)
  assert {entry.name: (entry.module_path, entry.quant_label, entry.rows, entry.cols, entry.role,
                       entry.parts, entry.opts, entry.family, entry.kernel_mode) for entry in plan} == {
    "blk.0.ffn_gate.weight": ("blk.0.ffn_gate", "Q4_K", 12288, 4096, "ffn_gate", 1, ("LOCAL:0:64",), "q4_k_packed_u32", "partial"),
    "blk.0.ffn_down.weight": ("blk.0.ffn_down", "Q4_K", 4096, 12288, "ffn_down", 4, ("LOCAL:0:32",), "q4_k_packed_u32", "partial"),
    "blk.0.attn_v.weight": ("blk.0.attn_v", "Q6_K", 1024, 4096, "attn_v", 4, ("LOCAL:0:32",), "q6_k_packed_u16", "partial"),
    "output.weight": ("output", "Q6_K", 151936, 4096, "output", 1, ("LOCAL:0:64",), "q6_k_packed_u16", "partial"),
  }


def test_serialized_policy_query_contract(tmp_path):
  path = tmp_path / "policy.json"
  path.write_text(json.dumps({"schema": "boltbeam.route_policy.v1", "run": "fixture", "routes": [
    {"selected_route": "decode_q4k_g3_generated", "route_params": {"BUBBLEBEAM_FUTURESIGHT": "1"}, "shape": {"rows": 32, "cols": 1024}},
    {"selected_route": "decode_q6k_coop_generated", "route_params": {"DECODE_Q6K_GENERATED": "1"}, "shape": {"rows": 16, "cols": 256}},
  ]}))
  policy = load_qk_route_policy(str(path))
  assert qk_route_policy_selected("decode_q4k_g3_generated", policy=policy)
  assert qk_route_policy_selects_q4k_g3(32, 1024, policy=policy)
  assert qk_route_policy_selects_q6k_generated(16, 256, policy=policy)
  assert not qk_route_policy_selects_q4k_g3(33, 1024, policy=policy)


@pytest.mark.parametrize(("values", "expected"), [
  ({}, "auto"), ({"TINYGRAD_DECODE_ROUTE": "sdpa"}, "fp16"),
  ({"FLASH_DECODE": "true"}, "flash"), ({"FLASH_DECODE": "off"}, "fp16"),
])
def test_decode_environment_mode_contract(values, expected):
  assert decode_route_mode(_env(values)) == expected


def test_decode_invalid_mode_and_symbolic_context_threshold_contract():
  with pytest.raises(ValueError, match="TINYGRAD_DECODE_ROUTE"):
    decode_route_mode(_env({"TINYGRAD_DECODE_ROUTE": "wat"}))
  start = UOp.variable("start", 0, 4096).bind(510)
  assert not should_use_flash_decode(start, 1, getenv_fn=_env({"FLASH_DECODE_THRESHOLD": 512}))
  assert should_use_flash_decode(start, 1, use_flash=True, getenv_fn=_env({"FLASH_DECODE_THRESHOLD": 9999}))
  assert not should_use_flash_decode(start, 2, use_flash=True, getenv_fn=_env({}))


def test_prefill_policy_and_observer_sequence_contract():
  policy = immutable_prefill_policy({"strategy": "FULL_RESIDENT_OVERLAY", "candidate_id": "candidate", "routes": {"q": "route"}})
  with pytest.raises(TypeError): policy["routes"]["q"] = "other"
  assert prefill_concrete_kv_auto_decision(False, False)[0] is False
  assert prefill_concrete_kv_auto_decision(False, True)[0] is True
  attachment = PrefillRouteAttachment("q", "route", "blk.0.attn_q.weight", policy, {"backend": "AMD"}, "owner")
  binding = PrefillDirectPackedBinding("q", "prefill", "attn_qo", (512, 4096, 4096))
  assert (attachment.invocation_id, attachment.route_id, binding.shape) == ("q", "route", (512, 4096, 4096))
  linear, events = SimpleNamespace(), []
  execution = PrefillRouteExecution("q", "route", "candidate", "program", False)
  with observe_prefill_routes(lambda value: events.append(("route", value))), observe_prefill_route_executions(
      lambda value, event: events.append(("execution", value, event))):
    notify_prefill_route(linear)
    with prefill_route_scope():
      notify_prefill_route(linear); notify_prefill_route_execution(linear, execution)
    notify_prefill_route_execution(linear, execution)
  assert events == [("route", linear), ("execution", linear, execution)]


def test_promoted_policy_identity_and_generated_artifact_hash_contract():
  from tinygrad.llm.prefill_candidate_runtime import promoted_candidate_set
  candidate_set = promoted_candidate_set().to_json()
  rows = [{"invocation_id": f"candidate-{i}", "candidate_controlled": True,
           "role": entry["payload"]["workload"]["role"], "shape": entry["payload"]["workload"]["shape"]}
          for i, entry in enumerate(candidate_set["entries"])]
  inventory = {"inventory_identity": "inventory:sha256:" + "a" * 64, "rows": rows}
  policy = automatic_promoted_prefill_graph_policy(inventory, {"backend": "AMD", "architecture": "gfx1100", "capabilities": {"wave_size": 32}})
  assert policy["candidate_id"] == "prefill_wmma_lds_dbuf_generated"
  assert policy["graph_gemm"]["candidate_set_identity"] == "candidate_set:sha256:2783d3ebb084e465d733cc161aa31485ac3f8ec45ff5c3aa4c1790795e852847"
  assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == "cae4c8d3d259bcd0355600e240e09f80afb8adf90a33ff0aec847b2ee358d91f"
