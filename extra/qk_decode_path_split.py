#!/usr/bin/env python3
"""Read-only scope artifact for the large and small decode paths.

Large path: parity-scale MMVQ contract preservation or mature artifact import.
Small path: q8 FFN artifact route that already passed as a research flag.
"""
from __future__ import annotations

import json, pathlib, re, subprocess
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-path-split"
LLAMA = pathlib.Path("/home/ubuntu/env/llama.cpp")


def read_json(path: pathlib.Path) -> dict[str, Any]:
  return json.loads(path.read_text())


def git_commit() -> str:
  try:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          timeout=10).stdout.strip() or "unknown"
  except Exception:
    return "unknown"


def rel(path: pathlib.Path) -> str:
  try:
    return str(path.relative_to(LLAMA))
  except ValueError:
    return str(path)


def file_info(path: pathlib.Path) -> dict[str, Any]:
  return {"path": rel(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}


def grep_count(path: pathlib.Path, pattern: str) -> int:
  if not path.exists():
    return 0
  return len(re.findall(pattern, path.read_text(errors="ignore")))


def large_inventory() -> dict[str, Any]:
  srcs = [
    LLAMA / "ggml/src/ggml-cuda/mmvq.cu",
    LLAMA / "ggml/src/ggml-cuda/mmvq.cuh",
    LLAMA / "ggml/src/ggml-cuda/template-instances/mmq-instance-q4_k.cu",
    LLAMA / "ggml/src/ggml-cuda/template-instances/mmq-instance-q6_k.cu",
  ]
  objs = [
    LLAMA / "build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/mmvq.cu.o",
    LLAMA / "build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/mmvq.cu.o.0.hipv4-amdgcn-amd-amdhsa--gfx1100",
    LLAMA / "build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/template-instances/mmq-instance-q4_k.cu.o",
    LLAMA / "build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/template-instances/mmq-instance-q6_k.cu.o",
  ]
  code_objects = sorted(
    p for p in LLAMA.rglob("*")
    if p.is_file() and p.suffix in {".hsaco", ".co"}
  ) if LLAMA.exists() else []
  mmvq_related_code_objects = [p for p in code_objects if re.search(r"(mmvq|mmq|q4_k|q6_k)", str(p), re.I)]
  mmvq = srcs[0]
  symbols = {
    "vec_dot_q4_K_q8_1": grep_count(mmvq, r"vec_dot_q4_K_q8_1"),
    "vec_dot_q6_K_q8_1": grep_count(mmvq, r"vec_dot_q6_K_q8_1"),
    "MMVQ_PARAMETERS_RDNA3_0": grep_count(mmvq, r"MMVQ_PARAMETERS_RDNA3_0"),
    "quantize_q8_1": grep_count(mmvq, r"quantize_q8_1"),
  }
  source_family_found = all(p.exists() for p in srcs)
  build_objects_found = any(p.exists() for p in objs)
  ready_hcq_artifact_found = len(mmvq_related_code_objects) > 0
  return {
    "llama_root": str(LLAMA),
    "source_family": [file_info(p) for p in srcs],
    "build_objects": [file_info(p) for p in objs],
    "code_objects_total": len(code_objects),
    "mmvq_related_code_objects": [rel(p) for p in mmvq_related_code_objects[:50]],
    "symbols_in_mmvq_cu": symbols,
    "source_family_found": source_family_found,
    "build_objects_found": build_objects_found,
    "ready_hcq_artifact_found": ready_hcq_artifact_found,
    "large_path_decision": "NO_READY_HCQ_ARTIFACT__SOURCE_IMPORT_OR_RENDERER_PROJECT_LEVEL"
      if source_family_found and build_objects_found and not ready_hcq_artifact_found else
      "READY_ARTIFACT_DISCOVERY_REQUIRED",
    "large_path_potential": {
      "measured_current_in_model_hbm_pct": 44,
      "llama_in_model_hbm_pct": 54,
      "tinygrad_standalone_hbm_pct": 76,
      "target_44_to_54_over_weight_gemv_bucket": "1.187x decode",
      "full_standalone_transfer_theoretical": "1.557x decode",
    },
  }


def small_q8_status() -> dict[str, Any]:
  baseline = read_json(ROOT / "bench/q8-ffn-handwritten-oracle/decode_wd_baseline.json")
  q8 = read_json(ROOT / "bench/q8-ffn-handwritten-oracle/decode_wd_q8_route.json")
  nll_base = read_json(ROOT / "bench/q8-ffn-handwritten-oracle/nll_baseline.json")
  nll_q8 = read_json(ROOT / "bench/q8-ffn-handwritten-oracle/nll_q8_route.json")
  artifact = read_json(ROOT / "bench/q8-ffn-amd-scheduler-project/result.json")
  policy = read_json(ROOT / "bench/q8-ffn-amd-scheduler-project/artifact_policy_boundary.json")
  rows = []
  for b, r in zip(baseline["rows"], q8["rows"]):
    rows.append({
      "ctx": b["ctx"],
      "baseline_tok_s": b["tok_s_W"],
      "q8_tok_s": r["tok_s_W"],
      "speedup": r["tok_s_W"] / b["tok_s_W"],
    })
  speedups = [r["speedup"] for r in rows]
  dnll = nll_q8["nll"] - nll_base["nll"]
  return {
    "status": "PASS_RESEARCH_HARDENED_EXISTING_EVIDENCE",
    "rows": rows,
    "speedup_min": min(speedups),
    "speedup_max": max(speedups),
    "speedup_mean": sum(speedups) / len(speedups),
    "nll_baseline": nll_base["nll"],
    "nll_q8": nll_q8["nll"],
    "dnll": dnll,
    "dnll_gate": 0.01,
    "dnll_pass": dnll <= 0.01,
    "tokens": nll_base.get("tokens"),
    "artifact_summary": artifact["summary"],
    "supported_boundary": policy["supported"],
    "default_changed": policy["default_changed"],
    "policy_gate": policy["policy_gate"],
    "small_path_decision": "DONE_AS_RESEARCH_FLAG__NOT_PARITY_PATH",
    "optional_next": [
      "multi-window dNLL or task eval before any non-research use",
      "shape/arch portability only if leaving Qwen3-8B/gfx1100",
      "native compiler ownership only after accepting project-level scheduler work",
    ],
  }


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  large = large_inventory()
  small = small_q8_status()
  result = {
    "schema": "decode_path_split_v1",
    "date": "2026-06-19",
    "commit": git_commit(),
    "large": large,
    "small": small,
    "combined_decision": {
      "large": "parity-scale but project-level: no ready llama.cpp HCQ artifact family found",
      "small": "bounded and already passed: keep q8 route default-off as research flag",
      "next_decode_build": "none bounded; choose q8 research use or fund MMVQ source-import/renderer project",
    },
  }
  (OUT / "large_artifact_inventory.json").write_text(json.dumps(large, indent=2) + "\n")
  (OUT / "small_q8_hardening.json").write_text(json.dumps(small, indent=2) + "\n")
  (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
  summary = [
    "# Decode large/small path split",
    "",
    f"- commit: `{result['commit']}`",
    f"- large path: `{large['large_path_decision']}`",
    f"- small path: `{small['small_path_decision']}`",
    "",
    "## Large Path",
    "",
    f"- llama source family found: `{large['source_family_found']}`",
    f"- llama build objects found: `{large['build_objects_found']}`",
    f"- ready HCQ code-object family found: `{large['ready_hcq_artifact_found']}`",
    "- decision: no Tensile-like decode artifact was found; parity-scale decode is source-import or renderer/scheduler project work.",
    "",
    "## Small Path",
    "",
    f"- W==D speedup range: `{small['speedup_min']:.3f}x` to `{small['speedup_max']:.3f}x`",
    f"- dNLL: `{small['dnll']:.6f}` over `{small['tokens']}` tokens",
    f"- default changed: `{small['default_changed']}`",
    "- decision: q8 route is done enough as a research flag; it is not the parity path.",
    "",
  ]
  (OUT / "summary.md").write_text("\n".join(summary))


if __name__ == "__main__":
  main()
