"""Static provenance audit for the default NV pp512 production route.

This module deliberately performs no imports of the model/runtime and no GPU work.
It is a conservative source audit: unknown production bindings are violations.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "tinygrad/llm/model.py"

FAMILIES = {
  "q4_gate_up": "nv_llama_packed_q4k_pp512_binding.py",
  "qkv": "nv_qkv_packed_pp512_binding.py",
  "q4_attention_o": "nv_llama_packed_q4k_o_pp512_binding.py",
  "q4_ffn_down": "nv_llama_packed_q4k_down_pp512_binding.py",
  "q6_ffn_down": "nv_llama_packed_q6k_down_pp512_binding.py",
  "flash_attention": "nv_llama_fattn_mma_pp512_binding.py",
  "q6_vocabulary": "nv_llama_q6k_vocab_pp512_binding.py",
}

def _classification(path: Path) -> tuple[str, list[str]]:
  text = path.read_text()
  violations: list[str] = []
  try:
    tree = ast.parse(text, filename=str(path))
  except SyntaxError as e:
    return "unknown", [f"syntax_error:{e.lineno}:{e.offset}"]
  imports_llama = any(isinstance(n, ast.ImportFrom) and n.module and "nv_llama_" in n.module
                      or isinstance(n, ast.Import) and any("nv_llama_" in a.name for a in n.names)
                      for n in ast.walk(tree))
  reads_cubin = ".cubin" in text and ("read_bytes" in text or "ARTIFACTS" in text or "CUBIN" in text)
  has_source_compile = "NVRTCCompiler" in text or "NVCCCompiler" in text
  has_native_identity = "native_nv_program" in text or "KernelProgramProvenance" in text or "uop_program" in text
  if imports_llama: violations.append("imports_nv_llama_binding")
  if reads_cubin: violations.append("reads_cubin_artifact")
  if not has_native_identity: violations.append("missing_tinygrad_provenance_identity")
  if reads_cubin or imports_llama: kind = "llama_extracted"
  elif has_source_compile: kind = "external_source_compiled"
  elif has_native_identity: kind = "tinygrad_generated"
  else: kind = "unknown"
  return kind, violations

def audit() -> dict:
  model_text = MODEL.read_text()
  selected = "NV_LLAMA_FULL_PACKED_PP512" in model_text and "getenv(\"NV_LLAMA_FULL_PACKED_PP512\", 1)" in model_text
  families = {}
  violations = []
  for name, filename in FAMILIES.items():
    path = Path(__file__).with_name(filename)
    kind, issues = _classification(path) if path.is_file() else ("unknown", ["missing_binding"])
    entry = {"binding": str(path.relative_to(ROOT)), "classification": kind, "violations": issues}
    families[name] = entry
    if selected and issues: violations.extend(f"{name}:{x}" for x in issues)
  return {"schema": "nv_all_native_provenance_gate/v1", "route": "NV pp512 default", "default_selected": selected,
          "families": families, "violations": sorted(violations), "all_native": not violations}

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=Path, help="write JSON report to this path")
  ap.add_argument("--require-all-native", action="store_true")
  args = ap.parse_args()
  report = audit()
  payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
  if args.out: args.out.write_text(payload)
  else: print(payload, end="")
  return 1 if args.require_all_native and not report["all_native"] else 0

if __name__ == "__main__":
  raise SystemExit(main())
