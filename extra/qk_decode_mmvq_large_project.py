#!/usr/bin/env python3
"""Phase-0 inventory for the funded decode MMVQ project.

This is intentionally read-only. It maps llama.cpp's gfx1100 MMVQ object into
candidate Q4_K/Q6_K kernels, metadata, and project phases.
"""
from __future__ import annotations

import json, pathlib, re, subprocess
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-mmvq-large-project"
LLAMA = pathlib.Path("/home/ubuntu/env/llama.cpp")
OBJ = LLAMA / "build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/mmvq.cu.o.0.hipv4-amdgcn-amd-amdhsa--gfx1100"
MMVQ_SRC = LLAMA / "ggml/src/ggml-cuda/mmvq.cu"
VECDOT_SRC = LLAMA / "ggml/src/ggml-cuda/vecdotq.cuh"
READELF = pathlib.Path("/opt/rocm/llvm/bin/llvm-readelf")


def run(cmd: list[str]) -> str:
  return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60).stdout


def git_commit() -> str:
  try:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          timeout=10).stdout.strip() or "unknown"
  except Exception:
    return "unknown"


def symbol_rows(sym_text: str) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  seen: set[tuple[str, str]] = set()
  pat = re.compile(r"\b(?P<idx>\d+):\s+(?P<addr>[0-9a-f]+)\s+(?P<size>\d+)\s+(?P<kind>FUNC|OBJECT)\s+.*?\s(?P<name>_ZL(?:13|17)mul_mat_vec_q[^\s]+)")
  tpl = re.compile(r"IL9ggml_type(?P<type>\d+)ELi(?P<ncols>\d+)ELb(?P<b0>[01])ELb(?P<b1>[01])")
  for m in pat.finditer(sym_text):
    name = m.group("name")
    tm = tpl.search(name)
    type_id = int(tm.group("type")) if tm else None
    if type_id not in (12, 14):
      continue
    key = (m.group("kind"), name)
    if key in seen:
      continue
    seen.add(key)
    rows.append({
      "idx": int(m.group("idx")),
      "addr": m.group("addr"),
      "size": int(m.group("size")),
      "kind": m.group("kind"),
      "name": name,
      "type_id": type_id,
      "type_name": {12: "Q4_K", 14: "Q6_K"}[type_id],
      "ncols_dst": int(tm.group("ncols")) if tm else None,
      "template_bool_0": bool(int(tm.group("b0"))) if tm else None,
      "template_bool_1": bool(int(tm.group("b1"))) if tm else None,
      "is_descriptor": name.endswith(".kd"),
    })
  return rows


def metadata_rows(notes_text: str) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  cur: dict[str, Any] | None = None
  args: list[dict[str, int | str]] = []
  current_arg: dict[str, int | str] | None = None

  def finish_arg() -> None:
    nonlocal current_arg
    if current_arg is not None:
      args.append(current_arg)
      current_arg = None

  def finish_kernel() -> None:
    nonlocal cur, args
    finish_arg()
    if cur and cur.get("name"):
      cur["arg_count"] = len(args)
      cur["args"] = args
      rows.append(cur)
    cur = None
    args = []

  for line in notes_text.splitlines():
    s = line.strip()
    if s == "- .args:":
      finish_kernel()
      cur = {}
      args = []
    elif s.startswith("- .address_space:") or s.startswith("- .offset:"):
      finish_arg()
      current_arg = {}
      if s.startswith("- .address_space:"):
        current_arg["address_space"] = s.split(":", 1)[1].strip()
      else:
        current_arg["offset"] = int(s.split(":", 1)[1].strip())
    elif current_arg is not None and s.startswith(".offset:"):
      current_arg["offset"] = int(s.split(":", 1)[1].strip())
    elif current_arg is not None and s.startswith(".size:"):
      current_arg["size"] = int(s.split(":", 1)[1].strip())
    elif current_arg is not None and s.startswith(".value_kind:"):
      current_arg["value_kind"] = s.split(":", 1)[1].strip()
    elif cur is not None and s.startswith(".name:"):
      cur["name"] = s.split(":", 1)[1].strip()
    elif cur is not None and s.startswith(".symbol:"):
      cur["symbol"] = s.split(":", 1)[1].strip()
    elif cur is not None and s.startswith(".kernarg_segment_size:"):
      cur["kernarg_segment_size"] = int(s.split(":", 1)[1].strip())
    elif cur is not None and s.startswith(".group_segment_fixed_size:"):
      cur["group_segment_fixed_size"] = int(s.split(":", 1)[1].strip())
    elif cur is not None and s.startswith(".max_flat_workgroup_size:"):
      cur["max_flat_workgroup_size"] = int(s.split(":", 1)[1].strip())
    elif cur is not None and s.startswith(".vgpr_count:"):
      cur["vgpr_count"] = int(s.split(":", 1)[1].strip())
    elif cur is not None and s.startswith(".sgpr_count:"):
      cur["sgpr_count"] = int(s.split(":", 1)[1].strip())
    elif cur is not None and s.startswith(".vgpr_spill_count:"):
      cur["vgpr_spill_count"] = int(s.split(":", 1)[1].strip())
    elif cur is not None and s.startswith(".sgpr_spill_count:"):
      cur["sgpr_spill_count"] = int(s.split(":", 1)[1].strip())
    elif cur is not None and s.startswith(".wavefront_size:"):
      cur["wavefront_size"] = int(s.split(":", 1)[1].strip())
  finish_kernel()
  return rows


def source_refs() -> dict[str, list[str]]:
  refs: dict[str, list[str]] = {}
  patterns = {
    "rdna3_table": "MMVQ_PARAMETERS_RDNA3_0",
    "launch_calc": "calc_launch_params",
    "kernel_template": "mul_mat_vec_q<",
    "q4k_dot": "vec_dot_q4_K_q8_1",
    "q6k_dot": "vec_dot_q6_K_q8_1",
  }
  for key, pat in patterns.items():
    src = VECDOT_SRC if key.endswith("_dot") else MMVQ_SRC
    hits: list[str] = []
    if src.exists():
      for i, line in enumerate(src.read_text(errors="ignore").splitlines(), 1):
        if pat in line:
          hits.append(f"{src}:{i}:{line.strip()}")
    refs[key] = hits[:12]
  return refs


def project_plan() -> list[dict[str, Any]]:
  return [
    {
      "phase": "P0",
      "name": "contract inventory",
      "status": "done_by_this_probe",
      "gate": "Q4_K/Q6_K candidate funcs, .kd descriptors, metadata, and source launch rules identified",
    },
    {
      "phase": "P1",
      "name": "single-kernel HCQ loader smoke",
      "status": "next",
      "gate": "load a selected descriptor by name from the llama gfx1100 object; do not launch until kernarg is proven",
      "kill": "object cannot be loaded by AMDProgram/HCQ without HIP runtime or unsupported relocations",
    },
    {
      "phase": "P2",
      "name": "kernarg and launch capture",
      "status": "pending",
      "gate": "capture one llama Q4_K and one Q6_K launch's 144-byte kernarg plus grid/local through HIP-only tracing",
      "kill": "hidden runtime state or non-buffer args cannot be represented in tinygrad HCQ",
    },
    {
      "phase": "P3",
      "name": "standalone correctness",
      "status": "pending",
      "gate": "HCQ launch writes correct Q4_K/Q6_K outputs against tinygrad/llama oracle on one role shape",
      "kill": "descriptor/kernarg is runnable but not correct after VA substitution",
    },
    {
      "phase": "P4",
      "name": "standalone performance",
      "status": "pending",
      "gate": "selected llama MMVQ kernel reaches >=90% of llama standalone or >=60% HBM on role shape",
      "kill": "imported kernel lands near tinygrad in-model speed before integration",
    },
    {
      "phase": "P5",
      "name": "one-role in-model route",
      "status": "pending",
      "gate": "one high-share role improves >=10% isolated in-model with graph-safe fallback",
      "kill": "standalone win disappears for reasons not fixable by routing",
    },
    {
      "phase": "P6",
      "name": "role matrix and activation lifecycle",
      "status": "pending",
      "gate": "Q4_K roles, Q6_K roles, and activation reuse policy are mapped; projected W==D >=5%",
      "kill": "role coverage cannot reach >=5% decode projected movement",
    },
    {
      "phase": "P7",
      "name": "final W==D/dNLL verdict",
      "status": "pending",
      "gate": "ctx sweep clears >=5% sustained decode speedup; exact path byte-identical or q8 path dNLL-gated",
      "kill": "no sustained end-to-end movement under clock-controlled W==D",
    },
    {
      "phase": "P8",
      "name": "native transfer decision",
      "status": "pending",
      "gate": "decide artifact/source dependency vs tinygrad renderer/scheduler feature ownership",
    },
  ]


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  sym_text = run([str(READELF), "-s", str(OBJ)])
  notes_text = run([str(READELF), "--notes", str(OBJ)])
  syms = symbol_rows(sym_text)
  meta = metadata_rows(notes_text)
  meta_by_name = {m.get("name"): m for m in meta}
  candidates = []
  for s in syms:
    if s["kind"] != "FUNC":
      continue
    m = meta_by_name.get(s["name"])
    candidates.append({**s, "metadata": {k: v for k, v in (m or {}).items() if k != "args"},
                       "arg_offsets": (m or {}).get("args", [])})
  result = {
    "schema": "decode_mmvq_large_project_p0_v1",
    "date": "2026-06-19",
    "commit": git_commit(),
    "object": str(OBJ),
    "object_exists": OBJ.exists(),
    "candidate_func_count": len(candidates),
    "candidate_descriptor_count": len([s for s in syms if s["kind"] == "OBJECT" and s["is_descriptor"]]),
    "candidates": candidates,
    "source_refs": source_refs(),
    "project_plan": project_plan(),
    "verdict": "P0_PASS__SOURCE_IMPORT_P1_IS_LOADABLE_DESCRIPTOR_SMOKE",
    "decision": "funded large path should start with source/object import P1 before native renderer work",
  }
  (OUT / "contract_inventory.json").write_text(json.dumps(result, indent=2) + "\n")
  summary = [
    "# Decode MMVQ large project P0",
    "",
    f"- commit: `{result['commit']}`",
    f"- object exists: `{result['object_exists']}`",
    f"- Q4_K/Q6_K candidate functions: `{result['candidate_func_count']}`",
    f"- Q4_K/Q6_K descriptors: `{result['candidate_descriptor_count']}`",
    f"- verdict: `{result['verdict']}`",
    "",
    "## Candidate Snapshot",
    "",
  ]
  for c in candidates[:12]:
    md = c.get("metadata", {})
    summary.append(
      f"- `{c['type_name']}` ncols `{c['ncols_dst']}` bools `{int(c['template_bool_0'])}/{int(c['template_bool_1'])}`: "
      f"VGPR `{md.get('vgpr_count')}`, SGPR `{md.get('sgpr_count')}`, kernarg `{md.get('kernarg_segment_size')}`, "
      f"wgmax `{md.get('max_flat_workgroup_size')}`"
    )
  summary += [
    "",
    "## Decision",
    "",
    "Start with P1 source/object import. The object already has .kd descriptors and AMDGPU metadata; the next gate is",
    "whether tinygrad HCQ can load a selected descriptor by name without HIP runtime and without unsupported relocation issues.",
    "",
  ]
  (OUT / "summary.md").write_text("\n".join(summary))


if __name__ == "__main__":
  main()
