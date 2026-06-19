#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, subprocess, time
from types import SimpleNamespace
from typing import Any

from tinygrad.device import Device
from extra.q8_ffn_codegen_transfer_audit import inspect_blob
from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked, hip_norm_source, perf_gateup

OUT = pathlib.Path("bench/q8-ffn-amd-scheduler-project")

class FixedLaunchRunner:
  def __init__(self, prg, global_size:tuple[int, int, int], local_size:tuple[int, int, int]):
    self.prg, self.q8_global, self.q8_local = prg, global_size, local_size
  def __getattr__(self, name): return getattr(self.prg, name)
  def __call__(self, *bufs, global_size=None, local_size=None, vals=(), wait=False, timeout=None):
    return self.prg(*bufs, global_size=self.q8_global, local_size=self.q8_local, vals=vals, wait=wait, timeout=timeout)

def sha256_bytes(x:bytes) -> str: return hashlib.sha256(x).hexdigest()
def sha256_text(x:str) -> str: return hashlib.sha256(x.encode()).hexdigest()

def run(cmd:list[str]) -> dict[str, Any]:
  try:
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
    return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout.strip()}
  except Exception as e:
    return {"cmd": cmd, "error": repr(e)}

def toolchain() -> dict[str, Any]:
  return {
    "hipcc_version": run(["hipcc", "--version"]),
    "ld_lld_version": run(["/opt/rocm/llvm/bin/ld.lld", "--version"]),
    "llvm_objdump_version": run(["/opt/rocm/llvm/bin/llvm-objdump", "--version"]),
    "rocm_version_file": pathlib.Path("/opt/rocm/.info/version").read_text(errors="ignore").strip()
                         if pathlib.Path("/opt/rocm/.info/version").exists() else None,
  }

def manifest_for(name:str, source:str, blob:bytes, arch:str, global_size:tuple[int, int, int], local_size:tuple[int, int, int]) -> dict[str, Any]:
  return {
    "name": name,
    "arch": arch,
    "source_sha256": sha256_text(source),
    "hsaco_sha256": sha256_bytes(blob),
    "hsaco_bytes": len(blob),
    "build": {
      "kind": "hipcc device-only LLVM bitcode -> amdgcn object -> ld.lld -shared",
      "source": "extra.q8_ffn_fast_artifact_probe",
      "commands": [
        "hipcc -c -emit-llvm --cuda-device-only -O3 -mcumode --offload-arch=<arch> -I/opt/rocm/include/hip -o obj.bc src.cpp",
        "hipcc -target amdgcn-amd-amdhsa -mcpu=<arch> -O3 -mllvm -amdgpu-internalize-symbols -c -o rel.o obj.bc",
        "/opt/rocm/llvm/bin/ld.lld -flavor gnu -shared -o linked.so rel.o",
      ],
    },
    "launch": {"global_size": list(global_size), "local_size": list(local_size)},
    "inspect": inspect_blob(f"artifact_import_{name}", blob, f"q8_artifact_import_{name}"),
  }

def main() -> int:
  ap = argparse.ArgumentParser(description="Route B q8 artifact/import executor")
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--arch", default="gfx1100")
  ap.add_argument("--producer-threads", type=int, default=1024)
  ap.add_argument("--rows", type=int, default=12288)
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--warmups", type=int, default=8)
  ap.add_argument("--iters", type=int, default=20)
  args = ap.parse_args()

  OUT.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  prod_src = hip_norm_source(args.producer_threads)
  gateup_src = HIP_MMVQ_GATEUP_SOURCE
  prod_blob = compile_hipcc_linked(prod_src, args.arch)
  gateup_blob = compile_hipcc_linked(gateup_src, args.arch)
  compile_s = time.perf_counter() - t0

  manifest = {
    "date": "2026-06-19",
    "phase": "B1_reproducible_artifact_build",
    "policy": {
      "route": "research-only external hipcc/LLD artifact",
      "default_changed": False,
      "in_process_hip_runtime_allowed": False,
      "supported_arch": args.arch,
      "supported_shape": {"dim": 4096, "hidden": 12288, "q4": "Q4_K", "activation": "q8_1"},
    },
    "compile_s": compile_s,
    "toolchain": toolchain(),
    "artifacts": {
      "producer": manifest_for("q8_rmsnorm_side", prod_src, prod_blob, args.arch, (1, 1, 1), (args.producer_threads, 1, 1)),
      "gateup": manifest_for("q8_mmvq_gateup", gateup_src, gateup_blob, args.arch, (12288, 2, 1), (32, 4, 1)),
    },
  }
  manifest["gates"] = {
    "producer_loads_in_amdprogram": manifest["artifacts"]["producer"]["inspect"]["runtime"].get("loads_in_amdprogram") is True,
    "gateup_loads_in_amdprogram": manifest["artifacts"]["gateup"]["inspect"]["runtime"].get("loads_in_amdprogram") is True,
    "gateup_has_16_dot4": manifest["artifacts"]["gateup"]["inspect"]["disasm"].get("grouped_counts", {}).get("dot4") == 16,
    "no_unresolved_relocations": not manifest["artifacts"]["producer"]["inspect"]["readelf"].get("readelf_relocations") and
                               not manifest["artifacts"]["gateup"]["inspect"]["readelf"].get("readelf_relocations"),
  }
  manifest["verdict"] = "PASS" if all(manifest["gates"].values()) else "FAIL"
  (OUT/"artifact_build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

  dev = Device["AMD"]
  prod_prg = FixedLaunchRunner(dev.runtime("q8_artifact_import_producer", prod_blob), (1, 1, 1), (args.producer_threads, 1, 1))
  gateup_prg = FixedLaunchRunner(dev.runtime("q8_artifact_import_gateup", gateup_blob), (12288, 2, 1), (32, 4, 1))
  perf_args = SimpleNamespace(gguf=args.gguf, rows=args.rows, seed=args.seed, warmups=args.warmups, iters=args.iters,
                              producer_threads=args.producer_threads)
  perf = perf_gateup(perf_args, prod_prg, gateup_prg)
  maps = pathlib.Path("/proc/self/maps").read_text(errors="ignore")
  loader = {
    "date": "2026-06-19",
    "phase": "B2_named_artifact_loader",
    "route": "FixedLaunchRunner around AMDProgram-loaded hipcc/LLD artifacts",
    "loader": {
      "producer": {"runtime_name": prod_prg.name, "global_size": list(prod_prg.q8_global), "local_size": list(prod_prg.q8_local),
                   "kernarg_size": prod_prg.kernargs_segment_size, "group_segment_size": prod_prg.group_segment_size,
                   "private_segment_size": prod_prg.private_segment_size},
      "gateup": {"runtime_name": gateup_prg.name, "global_size": list(gateup_prg.q8_global), "local_size": list(gateup_prg.q8_local),
                 "kernarg_size": gateup_prg.kernargs_segment_size, "group_segment_size": gateup_prg.group_segment_size,
                 "private_segment_size": gateup_prg.private_segment_size},
    },
    "perf_gateup": perf,
    "no_hip_runtime_in_process": "libamdhip64.so" not in maps,
    "default_changed": False,
  }
  loader["gates"] = {
    "manifest_pass": manifest["verdict"] == "PASS",
    "producer_correct": perf["gates"].get("producer_correct") is True,
    "gate_correct": perf["gates"].get("gate_correct") is True,
    "up_correct": perf["gates"].get("up_correct") is True,
    "lifecycle_lte_129p2us": perf["gate_up_lifecycle_us"] <= 129.2,
    "no_hip_runtime_in_process": loader["no_hip_runtime_in_process"],
    "default_unchanged": True,
  }
  loader["verdict"] = "PASS" if all(loader["gates"].values()) else "FAIL"
  (OUT/"artifact_loader.json").write_text(json.dumps(loader, indent=2) + "\n")

  boundary = {
    "date": "2026-06-19",
    "phase": "B4_maintenance_boundary",
    "status": "research_only",
    "default_changed": False,
    "supported": {
      "model_family": "Qwen3-8B Q4_K_M-style dense FFN block",
      "model_path_used": str(args.gguf),
      "gpu_arch": args.arch,
      "dim": 4096,
      "hidden": 12288,
      "weight_format": "Q4_K gate/up",
      "activation_format": "q8_1 side-channel",
      "producer_threads": args.producer_threads,
      "producer_launch": [1, 1, 1],
      "producer_local": [args.producer_threads, 1, 1],
      "gateup_launch": [12288, 2, 1],
      "gateup_local": [32, 4, 1],
    },
    "requirements": {
      "rebuild_command": "PYTHONPATH=. python3 extra/q8_ffn_artifact_import_route.py",
      "source_module": "extra.q8_ffn_fast_artifact_probe",
      "no_in_process_hip_runtime": True,
      "runtime": "tinygrad AMD HCQ / AMDProgram",
      "fallback": "flag off returns to default tinygrad decode",
    },
    "non_goals": [
      "not a default route",
      "not a portable tinygrad backend feature",
      "not validated for other hidden sizes, quant formats, or GPU archs",
      "not a replacement for native scheduler/codegen ownership",
    ],
    "policy_gate": "external hipcc/LLD HSACO dependency must be accepted explicitly before using beyond research",
  }
  (OUT/"artifact_policy_boundary.json").write_text(json.dumps(boundary, indent=2) + "\n")

  result = {
    "date": "2026-06-19",
    "phase": "Route_B_B1_B2",
    "artifact_build_manifest": "bench/q8-ffn-amd-scheduler-project/artifact_build_manifest.json",
    "artifact_loader": "bench/q8-ffn-amd-scheduler-project/artifact_loader.json",
    "artifact_policy_boundary": "bench/q8-ffn-amd-scheduler-project/artifact_policy_boundary.json",
    "artifact_graph_route": "bench/q8-ffn-amd-scheduler-project/artifact_graph_route.json",
    "verdict": "PASS" if manifest["verdict"] == "PASS" and loader["verdict"] == "PASS" else "FAIL",
    "summary": {
      "compile_s": compile_s,
      "producer_hash": manifest["artifacts"]["producer"]["hsaco_sha256"],
      "gateup_hash": manifest["artifacts"]["gateup"]["hsaco_sha256"],
      "gateup_dot4": manifest["artifacts"]["gateup"]["inspect"]["disasm"].get("grouped_counts", {}).get("dot4"),
      "gateup_consumer_us": perf["gateup_consumer"]["median_ms"] * 1000.0,
      "lifecycle_us": perf["gate_up_lifecycle_us"],
      "no_hip_runtime_in_process": loader["no_hip_runtime_in_process"],
    },
    "next": "B3 graph-safe route is validated by artifact_graph_route.json when that artifact exists; W==D/dNLL authority remains q8-ffn-handwritten-a4-decode-result-20260619.md.",
  }
  (OUT/"result.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"out": str(OUT/"result.json"), "verdict": result["verdict"], "summary": result["summary"]}, indent=2))
  return 0 if result["verdict"] == "PASS" else 1

if __name__ == "__main__":
  raise SystemExit(main())
