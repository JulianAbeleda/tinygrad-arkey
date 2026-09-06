#!/usr/bin/env python3
"""Audit selected decode PROGRAM transport provenance from a retained census."""
from __future__ import annotations

import argparse, collections, hashlib, json, pathlib

SCHEMA = "tinygrad.nv_decode_program_ownership_audit.v1"


def _native_marker_sha(name:str) -> str:
  return hashlib.sha256(f"// precompiled native cubin: {name}".encode()).hexdigest()


def audit(census:dict, compiler=None) -> dict:
  capture = census.get("capture", {})
  source_texts = capture.get("source_text_by_sha256", {})
  by_jit = capture.get("program_evidence_by_jit", {})
  if not isinstance(by_jit, dict) or set(by_jit) != set(capture.get("selected_jits", ())):
    raise ValueError("program evidence must cover every selected JIT exactly")
  rows, recompiled_by_source = [], {}
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
      recompiled_sha = None
      if compiler is not None and first["source_sha256"] in source_texts:
        if hashlib.sha256(source_texts[first["source_sha256"]].encode()).hexdigest() != first["source_sha256"]:
          raise ValueError(f"retained SOURCE text does not match its SHA-256 key: {first['source_sha256']}")
        if first["source_sha256"] not in recompiled_by_source:
          recompiled_by_source[first["source_sha256"]] = hashlib.sha256(compiler.compile(source_texts[first["source_sha256"]])).hexdigest()
        recompiled_sha = recompiled_by_source[first["source_sha256"]]
      recompile_match = recompiled_sha == first["binary_sha256"] if recompiled_sha is not None else None
      rows.append({"jit_owner":jit_name, "program_hash":program_hash, "program_name":first["program_name"],
                   "launch_count":len(occurrences), "source_sha256":first["source_sha256"],
                   "binary_sha256":first["binary_sha256"], "global_size":first["global_size"],
                   "local_size":first["local_size"], "recompiled_binary_sha256":recompiled_sha,
                   "source_recompile_match":recompile_match, "transport_provenance":
                   "native_precompiled_cubin" if native_precompiled else
                   "source_binary_recompiled_match" if recompile_match else
                   "source_recompile_mismatch" if recompile_match is False else "unknown_source_transport"})
  native = [row for row in rows if row["transport_provenance"] == "native_precompiled_cubin"]
  recompiled = [row for row in rows if row["transport_provenance"] == "source_binary_recompiled_match"]
  return {"schema":SCHEMA, "selected_jits":capture.get("selected_jits", []),
          "program_launches":sum(row["launch_count"] for row in rows), "unique_programs":len(rows),
          "native_precompiled_unique_programs":len(native), "native_precompiled_programs":native,
          "source_recompiled_unique_programs":len(recompiled),
          "all_unique_programs_source_recompiled":bool(rows) and len(recompiled) == len(rows),
          "no_known_native_precompiled_marker_in_selected_graph":not native,
          "interpretation":(("Every selected PROGRAM's retained SOURCE recompiles to its captured binary with the recorded compiler. "
                             "This positively proves SOURCE-to-binary transport, but generator/registry lineage requires a separate match. ")
                            if rows and len(recompiled) == len(rows) else
                            "An unmatched or unrecompiled source remains unknown; marker absence is not universal positive provenance. ")+
                           "The selected graph contains no recognized native_nv_program marker used by the known llama packed bindings.",
          "programs":sorted(rows, key=lambda row:(row["jit_owner"], row["program_name"], row["program_hash"]))}


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--census",type=pathlib.Path,required=True); ap.add_argument("--out",type=pathlib.Path,required=True)
  ap.add_argument("--recompile-arch", help="positively prove retained SOURCE->cubin with production NVRTC options")
  args=ap.parse_args(); compiler = None
  if args.recompile_arch:
    from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
    compiler=NVRTCCompiler(args.recompile_arch,ptx=False,cache_key="nv")
  result=audit(json.loads(args.census.read_text()),compiler)
  if compiler is not None:
    from tinygrad.runtime.support.compiler_cuda import nvrtc
    import ctypes
    major,minor=ctypes.c_int(),ctypes.c_int(); nvrtc.nvrtcVersion(ctypes.byref(major),ctypes.byref(minor))
    result["recompiler"]={"class":type(compiler).__name__,"arch":compiler.arch,"ptx":compiler.ptx,
                          "compile_options":compiler.compile_options,"nvrtc_version":[major.value,minor.value]}
  args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  print(json.dumps({key:result[key] for key in ("program_launches","unique_programs","native_precompiled_unique_programs",
                                                "no_known_native_precompiled_marker_in_selected_graph")},sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
