#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, re
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def read_text(rel: str) -> str:
  path = ROOT / rel
  return path.read_text() if path.exists() else ""


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def has(rel: str, pattern: str) -> bool:
  return re.search(pattern, read_text(rel)) is not None


def main() -> int:
  bb5a1 = read_json("bench/amd-broad-backend-roadmap/bb5a1_pipeline_ir_result.json", {})
  bb5a2 = read_json("bench/amd-broad-backend-roadmap/bb5a2_double_buffer_lds_result.json", {})
  layer1 = read_json("bench/amd-broad-backend-roadmap/bb5a2_lds_stage_plan_result.json", {})
  layer2 = read_json("bench/amd-broad-backend-roadmap/bb5a2_lowering_hook_result.json", {})
  layer3 = read_json("bench/amd-broad-backend-roadmap/bb5a2_render_isa_evidence_result.json", {})
  current = {
    "pipeline_ir_pass": bb5a1.get("verdict") == "PASS_PIPELINE_IR_SURFACE" and bool(bb5a1.get("gate_pass")),
    "bb5a2_current_verdict": bb5a2.get("verdict"),
    "ops_stage_exists": has("tinygrad/uop/__init__.py", r"STAGE") or has("tinygrad/uop/ops.py", r"Ops\.STAGE"),
    "bufferize_opts_exists": has("tinygrad/schedule/indexing.py", r"class BufferizeOpts"),
    "define_local_exists": has("tinygrad/uop/__init__.py", r"DEFINE_LOCAL"),
    "elf_scans_define_local": has("tinygrad/renderer/amd/elf.py", r"DEFINE_LOCAL.*lds_size|lds_size.*DEFINE_LOCAL"),
    "pipeline_stage_meta_exists": has("tinygrad/renderer/amd/schedule.py", r"class AMDPipelineStageMeta"),
    "lds_stage_plan_exists": has("tinygrad/renderer/amd/schedule.py", r"class AMDLDSStagePlan"),
    "lds_stage_plan_pass": layer1.get("verdict") == "PASS_LDS_STAGE_PLAN" and bool(layer1.get("gate_pass")),
    "define_local_lowering_hook_exists": has("tinygrad/renderer/amd/schedule.py", r"lower_lds_stage_plan_to_define_locals"),
    "define_local_lowering_hook_pass": layer2.get("verdict") == "PASS_DEFINE_LOCAL_LOWERING_HOOK" and bool(layer2.get("gate_pass")),
    "render_elf_lds_evidence_pass": layer3.get("verdict") == "PASS_RENDER_ELF_LDS_EVIDENCE" and bool(layer3.get("gate_pass")),
    "postrange_consumes_pipeline": has("tinygrad/codegen/opt/postrange.py", r"AMDPipelineStageMeta|AMDLDSStagePlan|DOUBLE_BUFFER|PREFETCH|PIPELINE"),
    "rangeify_consumes_pipeline": has("tinygrad/schedule/rangeify.py", r"AMDPipelineStageMeta|AMDLDSStagePlan|DOUBLE_BUFFER|PREFETCH|PIPELINE"),
    "renderer_consumes_pipeline": has("tinygrad/renderer/llvmir.py", r"AMDPipelineStageMeta|AMDLDSStagePlan|pipeline_stage") or
                                  has("tinygrad/renderer/amd/elf.py", r"AMDPipelineStageMeta|AMDLDSStagePlan|pipeline_stage"),
  }
  layers = [
    {
      "id": "layer_1_stage_to_lds_plan",
      "target_files": ["tinygrad/renderer/amd/schedule.py"],
      "deliverables": ["AMDLDSStagePlan", "lds_stage_plan_from_pipeline", "lds_stage_plan_dump"],
      "current_state": "pass" if current["lds_stage_plan_pass"] else "present" if current["lds_stage_plan_exists"] else "missing",
      "minimum_pass": "two slots, deterministic offsets, alias-safe, required_local_bytes recorded",
    },
    {
      "id": "layer_2_postrange_rangeify_lowering",
      "target_files": ["tinygrad/codegen/opt/postrange.py", "tinygrad/schedule/rangeify.py"],
      "deliverables": ["gated lowering hook", "two durable DEFINE_LOCAL slots or non-foldable offsets"],
      "current_state": "pass" if current["define_local_lowering_hook_pass"] else
                       "present" if current["define_local_lowering_hook_exists"] or current["postrange_consumes_pipeline"] or current["rangeify_consumes_pipeline"] else "missing",
      "minimum_pass": "lowered UOps preserve lds_slot=0/1 through local-buffer cleanup",
    },
    {
      "id": "layer_3_render_isa_evidence",
      "target_files": ["tinygrad/renderer/amd/elf.py", "extra/qk_amd_bb5a2_real_lds_lowering_probe.py"],
      "deliverables": ["LDS size confirmation", "source/hash/ISA non-byte-identical evidence"],
      "current_state": "pass" if current["render_elf_lds_evidence_pass"] else "present" if current["renderer_consumes_pipeline"] else "missing",
      "minimum_pass": "AMD render/assembly sees two-slot LDS structure and differs from serialized baseline",
    },
  ]
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.2_solution_scope",
    "schema": "amd_bb5a2_solution_scope_v1",
    "scope_doc": "docs/amd-broad-backend-bb5a2-real-lds-lowering-solution-scope-20260619.md",
    "verdict": "BB5A2_SOLUTION_SCOPED_REAL_LOWERING_REQUIRED",
    "gate_pass": True,
    "default_behavior_changed": False,
    "current_state": current,
    "solution_layers": layers,
    "required_probe": {
      "script": "extra/qk_amd_bb5a2_real_lds_lowering_probe.py",
      "artifact": "bench/amd-broad-backend-roadmap/bb5a2_real_lds_lowering_result.json",
      "minimum_pass": [
        "input pipeline IR passes",
        "LDS stage plan has two slots and alias_safe=true",
        "lowered UOps expose two durable local slots or offsets",
        "AMD render path sees the two-slot structure",
        "source/hash/ISA differs from serialized baseline",
        "default_behavior_changed=false",
      ],
    },
    "next_action": "Integrate the gated LDS plan/lowering path into real postrange or AMD renderer lowering." if current["render_elf_lds_evidence_pass"] else
                   "Implement Layer 3 renderer/ISA evidence for the lowered two-slot LDS structure." if current["define_local_lowering_hook_pass"] else
                   "Implement Layer 2 gated postrange/rangeify lowering hook." if current["lds_stage_plan_pass"] else
                   "Implement Layer 1 AMDLDSStagePlan, then the BB-5a.2 real LDS lowering probe before adding the gated lowering hook.",
  }
  write_json("bb5a2_solution_scope.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a2_solution_scope.json",
    "verdict": result["verdict"],
    "layers": [x["id"] for x in layers],
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
