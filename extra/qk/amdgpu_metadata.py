"""Neutral AMDGPU code-object metadata reader used by generated artifacts."""
from __future__ import annotations
import re, shutil, subprocess, tempfile
from pathlib import Path
from typing import Any

def _tool() -> str:
  for name in ("/opt/rocm/llvm/bin/llvm-readelf", "llvm-readelf-21", "llvm-readelf-20", "llvm-readelf"):
    if Path(name).is_file() or shutil.which(name): return name
  raise FileNotFoundError("no llvm-readelf available")

def parse_amdgpu_metadata(binary: bytes) -> dict[str, Any]:
  tool = _tool()
  with tempfile.NamedTemporaryFile(suffix=".hsaco") as f:
    f.write(binary); f.flush()
    text = subprocess.run((tool, "--notes", f.name), check=True, capture_output=True, text=True).stdout
  def integer(field: str) -> int:
    match = re.search(rf"^\s*\.{re.escape(field)}:\s*(\d+)\s*$", text, re.MULTILINE)
    if not match: raise ValueError(f"AMDGPU metadata missing .{field}")
    return int(match.group(1))
  def string(field: str) -> str:
    match = re.search(rf"^\s*\.{re.escape(field)}:\s*(.+?)\s*$", text, re.MULTILINE)
    if not match: raise ValueError(f"AMDGPU metadata missing .{field}")
    return match.group(1).strip()
  target = re.search(r"^amdhsa\.target:\s*(\S+)\s*$", text, re.MULTILINE)
  if not target: raise ValueError("AMDGPU metadata missing amdhsa.target")
  return {"vgpr": integer("vgpr_count"), "sgpr": integer("sgpr_count"),
          "vgpr_spills": integer("vgpr_spill_count"), "sgpr_spills": integer("sgpr_spill_count"),
          "lds_bytes": integer("group_segment_fixed_size"),
          "scratch_bytes": integer("private_segment_fixed_size"),
          "wavefront_size": integer("wavefront_size"),
          "dynamic_stack": string("uses_dynamic_stack").lower() == "true",
          "symbol": string("symbol"), "target": target.group(1), "metadata_tool": tool}
