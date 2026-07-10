#!/usr/bin/env python3
"""Capture AMD source for the failing S10 whole-prefill compile.

This diagnostic reuses the canonical whole-prefill authority path. It only wraps
the AMD compiler methods so a COMGR/LLVM failure leaves behind the exact source
that failed to compile.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, json, os, pathlib, re, sys, traceback
from typing import Any, Callable, Iterator

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.qk.prefill_harness import DEFAULT_MODEL, prefill_run_profile

DEFAULT_OUTPUT_DIR = ROOT / "bench/prefill-s10-lds2-ownership/compile-capture"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "report.json"
DEFAULT_GATE_ON_REPORT = DEFAULT_OUTPUT_DIR / "report-composed-gate-on-ab.json"
DEFAULT_GATE_OFF_REPORT = DEFAULT_OUTPUT_DIR / "report-composed-gate-off-ab.json"
S10_COMPOSED_ROUTE_ENV = {
  "PREFILL_GRAPH_GEMM": "1",
  "PREFILL_WMMA_PIPE_PRIMITIVE": "1",
  "PREFILL_WMMA_LDS_PRIMITIVE": "1",
  "PREFILL_DBUF": "1",
}
S10_LDS_ONLY_ROUTE_ENV = {
  "PREFILL_GRAPH_GEMM": "1",
  "PREFILL_WMMA_LDS_PRIMITIVE": "1",
  "PREFILL_DBUF": "1",
}
S10_ROUTE_ENVS = {
  "composed": (S10_COMPOSED_ROUTE_ENV, "prefill_wmma_pipe_lds_dbuf_primitive_generated"),
  "lds-only": (S10_LDS_ONLY_ROUTE_ENV, "prefill_wmma_lds_dbuf_primitive_mixed"),
}
S10_ROUTE_ENV = S10_COMPOSED_ROUTE_ENV

_DTYPE_BYTES = {"half": 2, "_Float16": 2, "float": 4, "int": 4, "uint": 4, "long": 8, "size_t": 8}
_PREFILL_ROLE_SHAPES = {
  (512, 4096, 4096): "attn_qo",
  (512, 1024, 4096): "attn_kv",
  (512, 4096, 12288): "ffn_down",
  (512, 12288, 4096): "ffn_gate_up",
}


def _source_suffix(src: str) -> str:
  return ".s" if src.split("\n", 1)[0].strip() == ".text" else ".cpp"


def analyze_amd_source(src: str) -> dict[str, Any]:
  """Classify a captured AMD/HIP source by kernel shape and shared-memory footprint."""
  fn = re.search(r'extern "C".*?\)\s*([A-Za-z_]\w*)\s*\((.*?)\)\s*\{', src, re.S)
  args: list[dict[str, Any]] = []
  if fn is not None:
    for typ, name, elements in re.findall(r'\b([A-Za-z_]\w*)\s*\*\s*(data\d+)_(\d+)\b', fn.group(2)):
      args.append({"type": typ, "name": name, "elements": int(elements)})

  shared_arrays: list[dict[str, Any]] = []
  for typ, name, count in re.findall(r'__attribute__\s*\(\(shared\b.*?\)\)\s*([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\[(\d+)\]', src):
    elements = int(count)
    elem_bytes = _DTYPE_BYTES.get(typ, 0)
    shared_arrays.append({"type": typ, "name": name, "elements": elements, "bytes": elements * elem_bytes})
  shared_bytes = sum(x["bytes"] for x in shared_arrays)

  m = 512
  n = k = None
  if len(args) >= 2 and args[0]["elements"] % m == 0 and args[1]["elements"] % m == 0:
    n = args[0]["elements"] // m
    k = args[1]["elements"] // m
  inferred_shape = {"m": m, "n": n, "k": k} if n is not None and k is not None else None
  role = _PREFILL_ROLE_SHAPES.get((m, n, k)) if inferred_shape is not None else None
  route_family = None
  if role in ("attn_qo", "attn_kv", "ffn_down"): route_family = "pipe"
  elif role == "ffn_gate_up": route_family = "lds_dbuf"

  lds_limit = 65536
  return {
    "kernel_name": fn.group(1) if fn is not None else None,
    "args": args,
    "inferred_prefill_shape": inferred_shape,
    "inferred_prefill_role": role,
    "inferred_route_family": route_family,
    "shared_arrays": shared_arrays,
    "shared_bytes": shared_bytes,
    "shared_limit_bytes": lds_limit,
    "shared_over_limit": shared_bytes > lds_limit,
    "contains_wmma_builtin": "__builtin_amdgcn_wmma" in src,
  }


def _record_source(out_dir: pathlib.Path, compiler: str, src: str, exc: BaseException) -> dict[str, Any]:
  out_dir.mkdir(parents=True, exist_ok=True)
  digest = hashlib.sha256(src.encode()).hexdigest()[:16]
  path = out_dir / f"failed-{len(list(out_dir.glob('failed-*'))):03d}-{compiler}-{digest}{_source_suffix(src)}"
  path.write_text(src)
  lines = src.splitlines()
  return {
    "compiler": compiler,
    "source_path": str(path),
    "sha256": hashlib.sha256(src.encode()).hexdigest(),
    "source_bytes": len(src.encode()),
    "source_lines": len(lines),
    "source_head": lines[:40],
    "source_tail": lines[-80:],
    "source_analysis": analyze_amd_source(src),
    "exception_type": type(exc).__name__,
    "exception": str(exc),
  }


@contextlib.contextmanager
def capture_amd_compile_sources(out_dir: pathlib.Path) -> Iterator[list[dict[str, Any]]]:
  from tinygrad.runtime.support import compiler_amd

  failures: list[dict[str, Any]] = []
  original_hip = compiler_amd.HIPCompiler.compile
  original_llvm = compiler_amd.AMDLLVMCompiler.compile

  def wrap(name: str, original: Callable):
    def compile_with_capture(self, src: str):
      try:
        return original(self, src)
      except BaseException as exc:
        failures.append(_record_source(out_dir, name, src, exc))
        raise
    return compile_with_capture

  compiler_amd.HIPCompiler.compile = wrap("HIPCompiler", original_hip)
  compiler_amd.AMDLLVMCompiler.compile = wrap("AMDLLVMCompiler", original_llvm)
  try:
    yield failures
  finally:
    compiler_amd.HIPCompiler.compile = original_hip
    compiler_amd.AMDLLVMCompiler.compile = original_llvm


def run_capture(*, out_dir: pathlib.Path = DEFAULT_OUTPUT_DIR, model: str = DEFAULT_MODEL,
                mode: str = "smoke", max_context: int = 4608,
                pin_clock: bool = False, scenario: str = "composed") -> dict[str, Any]:
  if scenario not in S10_ROUTE_ENVS: raise ValueError(f"unknown S10 compile-capture scenario {scenario!r}")
  route_env, required_route = S10_ROUTE_ENVS[scenario]
  route_keys = set().union(*(env.keys() for env, _ in S10_ROUTE_ENVS.values()))
  old_env = {k: os.environ.get(k) for k in route_keys}
  for key in route_keys: os.environ.pop(key, None)
  os.environ.update(route_env)
  try:
    from extra.qk.prefill_whole_synced import prefill_authority

    profile = prefill_run_profile(mode, max_context=max_context)
    report: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    with capture_amd_compile_sources(out_dir) as failures:
      try:
        report = prefill_authority(
          model_path=model,
          K=profile.K,
          warmups=profile.warmups,
          rounds=profile.rounds,
          start_positions=profile.start_positions,
          whole_lengths=profile.whole_lengths,
          chunk_n=profile.chunk_n,
          max_context=profile.max_context,
          mode=profile.mode,
          pin_clock=pin_clock,
          logits_only=False,
          require_route=required_route,
          verbose=True,
        )
      except BaseException as exc:
        error = {
          "type": type(exc).__name__,
          "message": str(exc),
          "traceback_tail": traceback.format_exc().splitlines()[-80:],
        }
    device_env = os.environ.get("DEV", "")
    pre_route_note = None
    if error is not None and device_env == "AMD:ISA" and "CAST dtypes.char -> dtypes.float unsupported" in error.get("message", ""):
      pre_route_note = (
        "This failure occurs during Q4_K -> fp16 prefill weight realization before the S10 route is entered. "
        "Use the canonical whole-prefill authority device path DEV=AMD for end-to-end S10 route smoke; "
        "reserve DEV=AMD:ISA for role-local/generated-kernel probes that do not require model-load dequant."
      )
    return {
      "schema": "prefill-s10-compile-capture.v1",
      "scenario": scenario,
      "required_route": required_route,
      "route_env": dict(route_env),
      "device_env": device_env,
      "mode": mode,
      "model": model,
      "max_context": max_context,
      "pin_clock": pin_clock,
      "status": "ok" if error is None else "compile_or_runtime_error",
      "error": error,
      "pre_route_blocker_note": pre_route_note,
      "captured_failures": failures,
      "whole_prefill_report": report,
    }
  finally:
    for key, value in old_env.items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value


def summarize_gate_ab(gate_on_report: pathlib.Path = DEFAULT_GATE_ON_REPORT,
                      gate_off_report: pathlib.Path = DEFAULT_GATE_OFF_REPORT) -> dict[str, Any]:
  gate_on = json.loads(gate_on_report.read_text())
  gate_off = json.loads(gate_off_report.read_text())
  on_whole = gate_on.get("whole_prefill_report") or {}
  off_failures = gate_off.get("captured_failures") or []
  off_analysis = off_failures[0].get("source_analysis", {}) if off_failures else {}
  on_roles = on_whole.get("prefill_role_routes") or {}
  gate_off_reproduces_overflow = (
    gate_off.get("status") == "compile_or_runtime_error" and
    off_analysis.get("inferred_prefill_role") == "attn_kv" and
    off_analysis.get("shared_over_limit") is True
  )
  gate_on_ok = (
    gate_on.get("status") == "ok" and
    len(gate_on.get("captured_failures") or []) == 0 and
    gate_off_reproduces_overflow
  )
  if gate_on_ok and on_roles.get("attn_kv") == "generated_pipe_no_local_stage":
    verdict = "S10_ATTN_KV_NO_LOCAL_STAGE_PASS"
  elif gate_on_ok and on_roles.get("attn_kv") == "pipe_resource_gated_raw_fallback":
    verdict = "S10_ATTN_KV_RESOURCE_GATE_PASS"
  else:
    verdict = "S10_ATTN_KV_GATE_AB_INCONCLUSIVE"
  return {
    "schema": "prefill-s10-attn-kv-pipe-resource-gate-ab.v1",
    "verdict": verdict,
    "gate_on": {
      "path": str(gate_on_report),
      "status": gate_on.get("status"),
      "captured_failures": len(gate_on.get("captured_failures") or []),
      "role_routes": on_roles,
      "whole_tok_s": on_whole.get("whole_tok_s"),
      "binding_gate": (on_whole.get("prefill_route_binding_gate") or {}).get("verdict"),
    },
    "gate_off": {
      "path": str(gate_off_report),
      "status": gate_off.get("status"),
      "captured_failures": len(off_failures),
      "source_analysis": off_analysis,
    },
  }


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
  ap.add_argument("--report", type=pathlib.Path, default=DEFAULT_REPORT)
  ap.add_argument("--model", default=os.environ.get("QK_MODEL", DEFAULT_MODEL))
  ap.add_argument("--mode", choices=("smoke", "authority"), default="smoke")
  ap.add_argument("--max-context", type=int, default=4608)
  ap.add_argument("--pin-clock", action="store_true")
  ap.add_argument("--scenario", choices=tuple(S10_ROUTE_ENVS), default="composed")
  ap.add_argument("--summarize-gate-ab", action="store_true")
  ap.add_argument("--gate-on-report", type=pathlib.Path, default=DEFAULT_GATE_ON_REPORT)
  ap.add_argument("--gate-off-report", type=pathlib.Path, default=DEFAULT_GATE_OFF_REPORT)
  ap.add_argument("--json", action="store_true")
  args = ap.parse_args(argv)

  payload = (summarize_gate_ab(args.gate_on_report, args.gate_off_report) if args.summarize_gate_ab else
             run_capture(out_dir=args.out_dir, model=args.model, mode=args.mode,
                         max_context=args.max_context, pin_clock=args.pin_clock, scenario=args.scenario))
  report_path = args.report if args.report.is_absolute() else ROOT / args.report
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
  if args.json:
    print(json.dumps(payload, indent=2, allow_nan=False))
  else:
    if args.summarize_gate_ab:
      print(f"{payload['verdict']} report={report_path}")
    else:
      print(f"{payload['status']} captured={len(payload['captured_failures'])} report={report_path}")
      for item in payload["captured_failures"]:
        print(f"  {item['compiler']} {item['source_lines']} lines {item['source_path']}: {item['exception']}")
  if args.summarize_gate_ab:
    return 0 if payload["verdict"] in ("S10_ATTN_KV_RESOURCE_GATE_PASS", "S10_ATTN_KV_NO_LOCAL_STAGE_PASS") else 1
  return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
  raise SystemExit(main())
