#!/usr/bin/env python3
"""End-to-end classification gate for the generated Q4_K/Q8_1 prefill candidate.

This gate intentionally treats the current 14B smoke fail-fast as a classified result, not a promotion. Passing means:
descriptor selection works, numeric parity passes, AMD int8 WMMA codegen passes, and the canonical 14B smoke reaches
the generated route before being blocked by the known Tensor graph explosion guard.
"""
from __future__ import annotations

import json, os, pathlib, subprocess, sys
from typing import Any

from tinygrad.llm.generated_candidates import select_generated_candidate
from tinygrad.llm.quant_specs import activation_spec, quant_spec
from tinygrad.llm.runtime_specs import RuntimeOpSpec

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = "/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf"
EXPECTED_CANDIDATE = "quant_linear_prefill.q4k_int8_wmma_tensor_substrate"
EXPECTED_TILED_CANDIDATE = "quant_linear_prefill.q4k_int8_wmma_tiled_substrate"
GRAPH_BLOCKER = "PREFILL_Q4K_Q8=wmma Tensor-substrate blocked for full-model shape"
TILED_BLOCKER = "PREFILL_Q4K_Q8=wmma_tiled is not implemented for full route shape"


def _run(argv:list[str], *, env:dict[str, str]|None=None, timeout:int=180) -> dict[str, Any]:
  child_env = {**os.environ, "PYTHONPATH": str(ROOT), **(env or {})}
  r = subprocess.run([sys.executable, *argv], cwd=str(ROOT), env=child_env, capture_output=True, text=True, timeout=timeout)
  return {"argv": [sys.executable, *argv], "returncode": r.returncode,
          "stdout_tail": r.stdout[-6000:], "stderr_tail": r.stderr[-6000:]}


def _candidate_selection() -> dict[str, Any]:
  op = RuntimeOpSpec("QuantizedLinear", "prefill", "ffn_gate_up", {"M": 512, "N": 17408, "K": 5120},
                     quant_spec("Q4_K").tensor_spec(), activation_spec("Q8_1").activation_spec(),
                     lowering_strategy="iu8_wmma_grouped_dot")
  sel = select_generated_candidate(op, preferred=(EXPECTED_CANDIDATE,))
  return sel.to_json()


def _tiled_candidate_selection() -> dict[str, Any]:
  op = RuntimeOpSpec("QuantizedLinear", "prefill", "ffn_gate_up", {"M": 512, "N": 17408, "K": 5120},
                     quant_spec("Q4_K").tensor_spec(), activation_spec("Q8_1").activation_spec(),
                     lowering_strategy="iu8_wmma_tiled_grouped_dot")
  sel = select_generated_candidate(op, preferred=(EXPECTED_TILED_CANDIDATE,))
  return sel.to_json()


def _smoke_classification(smoke:dict[str, Any]) -> dict[str, Any]:
  text = smoke["stdout_tail"] + "\n" + smoke["stderr_tail"]
  graph_blocked = GRAPH_BLOCKER in text and "RAW groups*m*n=" in text
  route_reached = "PREFILL_Q4K_Q8=wmma" in text
  return {"route_reached": route_reached, "graph_explosion_guard": graph_blocked,
          "class": "blocked.graph_explosion" if route_reached and graph_blocked else "unknown"}


def _tiled_smoke_classification(smoke:dict[str, Any]) -> dict[str, Any]:
  text = smoke["stdout_tail"] + "\n" + smoke["stderr_tail"]
  route_reached = TILED_BLOCKER in text
  explicit_blocked = route_reached and "planned kernel=" in text and "prevents fallthrough" in text
  return {"route_reached": route_reached, "explicit_no_fallthrough_guard": explicit_blocked,
          "class": "blocked.full_route_lowering_missing" if explicit_blocked else "unknown"}


def build() -> dict[str, Any]:
  candidate = _candidate_selection()
  tiled_candidate = _tiled_candidate_selection()
  parity = _run(["extra/qk/prefill_mmq_parity_gate.py"], env={"DEV": "PYTHON"}, timeout=120)
  from extra.qk.int8_wmma_codegen_gate import build as codegen_build
  from extra.qk.q4k_wmma_tiled_lowering_feasibility import build as tiled_lowering_build
  from extra.qk.q4k_wmma_tiled_microgate import build as tiled_microgate_build
  from extra.qk.q4k_wmma_tiled_surface_gate import build as tiled_surface_build
  from extra.qk.q4k_wmma_tiled_no_hand_kernel_gate import build as tiled_no_hand_build
  from extra.qk.q4k_wmma_tiled_lifecycle_gate import build as tiled_lifecycle_build
  from extra.qk.q4k_wmma_tiled_role_shape_exec_gate import build as tiled_role_shape_exec_build
  from extra.qk.q4k_wmma_tiled_role_shape_gate import build as tiled_role_shape_build
  codegen = codegen_build()
  tiled_lowering = tiled_lowering_build()
  tiled_microgate = tiled_microgate_build()
  tiled_surface = tiled_surface_build()
  tiled_no_hand = tiled_no_hand_build()
  tiled_lifecycle = tiled_lifecycle_build(surface=tiled_surface, microgate=tiled_microgate)
  tiled_role_shape_exec = tiled_role_shape_exec_build(lifecycle=tiled_lifecycle)
  tiled_role_shape = tiled_role_shape_build()
  smoke = _run(["extra/qk/bench.py", "--model", MODEL, "--prefill", "--prefill-mode", "smoke"],
               env={"PREFILL_Q4K_Q8": "wmma", "DEVICE_IN_FUNCTION_BUG": "1", "ALLOW_DEVICE_USAGE": "1"},
               timeout=180)
  tiled_smoke = _run(["extra/qk/bench.py", "--model", MODEL, "--prefill", "--prefill-mode", "smoke"],
                     env={"PREFILL_Q4K_Q8": "wmma_tiled", "DEVICE_IN_FUNCTION_BUG": "1", "ALLOW_DEVICE_USAGE": "1"},
                     timeout=180)
  smoke_class = _smoke_classification(smoke)
  tiled_smoke_class = _tiled_smoke_classification(tiled_smoke)
  selected_ok = candidate["status"] == "selected" and candidate["candidate"]["candidate_id"] == EXPECTED_CANDIDATE
  tiled_selected_ok = tiled_candidate["status"] == "selected" and tiled_candidate["candidate"]["candidate_id"] == EXPECTED_TILED_CANDIDATE
  parity_ok = parity["returncode"] == 0 and "MMQ parity gate PASS" in parity["stdout_tail"]
  codegen_ok = codegen["verdict"] == "INT8_WMMA_CODEGEN_PASS"
  tiled_lowering_ok = tiled_lowering["verdict"] == "Q4K_WMMA_TILED_LOWERING_FEASIBLE"
  tiled_microgate_ok = tiled_microgate["verdict"] == "Q4K_WMMA_TILED_MICROGATE_PASS"
  tiled_surface_ok = tiled_surface["verdict"] == "Q4K_WMMA_TILED_SURFACE_TC_MATCHER_SELECTED"
  tiled_no_hand_ok = tiled_no_hand["verdict"] == "Q4K_WMMA_TILED_NO_HAND_KERNEL_PASS"
  tiled_lifecycle_classified = tiled_lifecycle["verdict"] == "Q4K_WMMA_TILED_LIFECYCLE_BLOCKED_MULTI_TILE_LOWERING"
  tiled_role_shape_exec_classified = tiled_role_shape_exec["verdict"] == "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_BLOCKED_LIFECYCLE"
  tiled_role_shape_ok = tiled_role_shape["verdict"] == "Q4K_WMMA_TILED_ROLE_SHAPES_BLOCKED_FULL_ROUTE"
  smoke_ok = smoke_class["class"] == "blocked.graph_explosion"
  tiled_smoke_ok = tiled_smoke_class["class"] == "blocked.full_route_lowering_missing"
  old_ok = selected_ok and parity_ok and codegen_ok and smoke_ok
  tiled_ok = tiled_selected_ok and tiled_lowering_ok and tiled_microgate_ok and tiled_surface_ok and tiled_no_hand_ok and \
    tiled_lifecycle_classified and tiled_role_shape_exec_classified and tiled_role_shape_ok and tiled_smoke_ok
  verdict = "GENERATED_Q4K_PREFILL_E2E_TILED_BLOCKED_FULL_ROUTE" if old_ok and tiled_ok else \
    "GENERATED_Q4K_PREFILL_E2E_BLOCKED_GRAPH_EXPLOSION" if old_ok else "GENERATED_Q4K_PREFILL_E2E_FAIL"
  return {"schema": "generated_q4k_prefill_e2e_gate.v1",
          "scope": "Q4_K/Q8_1 generated prefill candidate selection, parity, AMD WMMA codegen, and 14B smoke classification",
          "verdict": verdict,
          "candidate_selection": candidate,
          "tiled_candidate_selection": tiled_candidate,
          "parity": {"ok": parity_ok, **parity},
          "codegen": codegen,
          "tiled_lowering": {"ok": tiled_lowering_ok, **tiled_lowering},
          "tiled_microgate": {"ok": tiled_microgate_ok, **tiled_microgate},
          "tiled_surface": {"ok": tiled_surface_ok, **tiled_surface},
          "tiled_no_hand_kernel": {"ok": tiled_no_hand_ok, **tiled_no_hand},
          "tiled_lifecycle": {"ok": False, "classified": tiled_lifecycle_classified, **tiled_lifecycle},
          "tiled_role_shape_exec": {"ok": False, "classified": tiled_role_shape_exec_classified, **tiled_role_shape_exec},
          "tiled_role_shape": {"ok": tiled_role_shape_ok, **tiled_role_shape},
          "smoke": {"ok": smoke_ok, "classification": smoke_class, **smoke},
          "tiled_smoke": {"ok": tiled_smoke_ok, "classification": tiled_smoke_class, **tiled_smoke},
          "blocker": "direct tiled full-role scheduler/codegen lowering missing; one-tile tiled WMMA is correct and old Tensor route remains graph-explosion blocked" if verdict == "GENERATED_Q4K_PREFILL_E2E_TILED_BLOCKED_FULL_ROUTE" else
                     "full-model Tensor graph explosion in group_tensor_matmul_v0; needs fused/tiled generated emitter" if smoke_ok else "unclassified"}


if __name__ == "__main__":
  print(json.dumps(build(), indent=2))
