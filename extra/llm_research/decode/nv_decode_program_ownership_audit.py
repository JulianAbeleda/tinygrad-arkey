#!/usr/bin/env python3
"""Audit selected decode PROGRAM transport provenance from a retained census."""
from __future__ import annotations

import argparse, collections, hashlib, json, pathlib

SCHEMA = "tinygrad.nv_decode_program_ownership_audit.v1"


def _native_marker_sha(name:str) -> str:
  return hashlib.sha256(f"// precompiled native cubin: {name}".encode()).hexdigest()


def audit(census:dict) -> dict:
  capture = census.get("capture", {})
  by_jit = capture.get("program_evidence_by_jit", {})
  if not isinstance(by_jit, dict) or set(by_jit) != set(capture.get("selected_jits", ())):
    raise ValueError("program evidence must cover every selected JIT exactly")
  rows = []
  for jit_name, launches in by_jit.items():
    if len(launches) != capture.get("programs_by_jit", {}).get(jit_name):
      raise ValueError(f"PROGRAM launch census does not reconcile for {jit_name}")
    grouped:dict[str,list[dict]] = collections.defaultdict(list)
    for launch in launches:
      for field in ("program_hash", "program_name", "source_sha256", "binary_sha256", "global_size", "local_size"):
        if launch.get(field) is None: raise ValueError(f"PROGRAM launch lacks {field}: {launch.get('ordinal')}")
      grouped[launch["program_hash"]].append(launch)
    for program_hash, occurrences in grouped.items():
      first = occurrences[0]
      stable = all(tuple(row.get(key) if isinstance(row.get(key), list) else (row.get(key),)) ==
                   tuple(first.get(key) if isinstance(first.get(key), list) else (first.get(key),))
                   for row in occurrences for key in ("program_name", "source_sha256", "binary_sha256", "global_size", "local_size"))
      if not stable: raise ValueError(f"PROGRAM hash has inconsistent identity or geometry: {program_hash}")
      native_precompiled = first["source_sha256"] == _native_marker_sha(first["program_name"])
      rows.append({"jit_owner":jit_name, "program_hash":program_hash, "program_name":first["program_name"],
                   "launch_count":len(occurrences), "source_sha256":first["source_sha256"],
                   "binary_sha256":first["binary_sha256"], "global_size":first["global_size"],
                   "local_size":first["local_size"], "transport_provenance":
                   "native_precompiled_cubin" if native_precompiled else "tinygrad_rendered_source"})
  native = [row for row in rows if row["transport_provenance"] == "native_precompiled_cubin"]
  return {"schema":SCHEMA, "selected_jits":capture.get("selected_jits", []),
          "program_launches":sum(row["launch_count"] for row in rows), "unique_programs":len(rows),
          "native_precompiled_unique_programs":len(native), "native_precompiled_programs":native,
          "no_native_precompiled_cubin_in_selected_graph":not native,
          "interpretation":("The selected decode graph contains no native_nv_program transport marker. This excludes the "
                            "llama packed cubin bindings, which use native_nv_program. tinygrad-rendered-source is a transport "
                            "classification; it does not assert machine-search provenance for ordinary scheduler kernels."),
          "programs":sorted(rows, key=lambda row:(row["jit_owner"], row["program_name"], row["program_hash"]))}


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--census",type=pathlib.Path,required=True); ap.add_argument("--out",type=pathlib.Path,required=True)
  args=ap.parse_args(); result=audit(json.loads(args.census.read_text()))
  args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps({key:result[key] for key in ("program_launches","unique_programs","native_precompiled_unique_programs",
                                                "no_native_precompiled_cubin_in_selected_graph")},sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
