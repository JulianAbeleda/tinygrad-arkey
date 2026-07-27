#!/usr/bin/env python3
"""Join tinygrad launch sidecars to offline AMDGPU code-object resources."""
from __future__ import annotations

import argparse, collections, json, os, pathlib, re, subprocess

FIELDS = ("group_segment_fixed_size", "max_flat_workgroup_size", "sgpr_count", "sgpr_spill_count",
          "vgpr_count", "vgpr_spill_count", "wavefront_size")
DEFAULT_READOBJ = "/opt/rocm-7.2.4/lib/llvm/bin/llvm-readobj"


def _read_resources(path: pathlib.Path, readobj: str) -> dict:
  try:
    text = subprocess.run([readobj, "--notes", str(path)], check=True, capture_output=True, text=True).stdout
  except (OSError, subprocess.CalledProcessError) as e:
    return {"resource_status": "read_failed", "resource_error": str(e)}
  out = {"resource_status": "measured", "binary_path": str(path)}
  for field in FIELDS:
    match = re.search(rf"\.{re.escape(field)}:\s+([^\s]+)", text)
    if match:
      try: out[field] = int(match.group(1))
      except ValueError: out[field] = match.group(1)
  name = re.search(r"\.name:\s+([^\n]+)", text)
  if name: out["kernel_name"] = name.group(1).strip()
  if "vgpr_count" not in out or "group_segment_fixed_size" not in out:
    out["resource_status"] = "metadata_incomplete"
  return out


def _sidecar_rows(paths: list[pathlib.Path]) -> dict[str, dict]:
  joined: dict[str, dict] = {}
  for path in paths:
    data = json.loads(path.read_text())
    counts = collections.Counter(row["binary_sha256"] for row in data.get("records", []))
    geometries = collections.defaultdict(collections.Counter)
    for row in data.get("records", []):
      geometries[row["binary_sha256"]][(tuple(row["grid"]), tuple(row["workgroup"]))] += 1
    for binary, count in counts.items():
      item = joined.setdefault(binary, {"binary_sha256": binary, "observations": []})
      item["observations"].append({"sidecar": str(path), "candidate_id": data.get("candidate_id"),
                                   "dispatch_count": count,
                                   "geometries": [{"grid": list(grid), "workgroup": list(workgroup), "count": n}
                                                   for (grid, workgroup), n in geometries[binary].items()]})
  return joined


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--sidecar", action="append", required=True, type=pathlib.Path)
  ap.add_argument("--binary-dir", action="append", required=True, type=pathlib.Path)
  ap.add_argument("--llvm-readobj", default=os.environ.get("LLVM_READOBJ", DEFAULT_READOBJ))
  ap.add_argument("--out", type=pathlib.Path)
  args = ap.parse_args()

  rows = _sidecar_rows(args.sidecar)
  for directory in args.binary_dir:
    for path in directory.glob("*.hsaco"):
      binary = path.stem
      if binary not in rows: continue
      rows[binary].update(_read_resources(path, args.llvm_readobj))
  payload = {"schema": "tinygrad.decode.resource_audit.v1", "rows": sorted(rows.values(), key=lambda x: x["binary_sha256"])}
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  for row in payload["rows"]:
    print(row.get("kernel_name", "?"), row["binary_sha256"][:16], row.get("resource_status"),
          "vgpr", row.get("vgpr_count", "?"), "spill", row.get("vgpr_spill_count", "?"),
          "lds", row.get("group_segment_fixed_size", "?"),
          "observed", sum(x["dispatch_count"] for x in row["observations"]))
  return 0


if __name__ == "__main__": raise SystemExit(main())
