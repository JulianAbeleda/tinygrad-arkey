#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, re, subprocess, sys, time
from typing import Any

from extra.q4_k_safety import assert_q4k_native_sweep_allowed
from extra.qk_semantic_candidate import is_raw_accept_status
from extra.qk_semantic_schedule import load_json, write_json

MODEL_FILES = {
  "8b": "Qwen3-8B-Q4_K_M.gguf",
  "14b": "Qwen3-14B-Q4_K_M.gguf",
}

Q4_RESULT_RE = re.compile(
  r"^(?P<tensor>\S+) (?P<shape>\S+) q4k_primitive_gemv: .* q4_eff=(?P<gbs>[0-9.]+) GB/s .* kernels=(?P<kernels>[0-9.]+)",
  re.MULTILINE,
)
Q6_RESULT_RE = re.compile(
  r"^q6k_gemv_primitive_partial: .* device=(?P<ms>[0-9.]+) ms \((?P<gbs>[0-9.]+) quant-GB/s\)",
  re.MULTILINE,
)
Q4_GEMV_RE = re.compile(r"^primitive_gemv_correctness: PASS \S+ max_abs=(?P<max_abs>[0-9.eE+-]+)", re.MULTILINE)
Q6_GEMV_RE = re.compile(r"^correctness: max_abs=(?P<max_abs>[0-9.eE+-]+)", re.MULTILINE)


def _portable_str(value:str, base:pathlib.Path) -> str:
  resolved = str(base.resolve())
  text = str(value)
  if text == resolved: return "."
  return text.replace(resolved + "/", "")

def _portable_value(value, base:pathlib.Path):
  if isinstance(value, pathlib.Path): return _portable_str(str(value), base)
  if isinstance(value, str): return _portable_str(value, base)
  if isinstance(value, list): return [_portable_value(x, base) for x in value]
  if isinstance(value, tuple): return [_portable_value(x, base) for x in value]
  if isinstance(value, dict): return {k: _portable_value(v, base) for k, v in value.items()}
  return value


def _classify(rc:int, out:str, timeout:bool) -> str:
  if timeout: return "timeout"
  if rc == 0: return "pass"
  if "KernelOptError" in out: return "illegal-opt"
  if "CompileError" in out or "compile failed" in out: return "compile-fail"
  if "correctness failed" in out or "AssertionError" in out: return "wrong"
  return "error"


def _run(cmd:list[str], *, cwd:pathlib.Path, env:dict[str, str], timeout:float, log:pathlib.Path) -> tuple[int, str, bool, float]:
  log.parent.mkdir(parents=True, exist_ok=True)
  st = time.perf_counter()
  try:
    proc = subprocess.run(cmd, cwd=cwd, env={**os.environ, **env}, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    log.write_text(proc.stdout)
    return proc.returncode, proc.stdout, False, time.perf_counter() - st
  except subprocess.TimeoutExpired as exc:
    out = (exc.stdout or "") + "\nTIMEOUT"
    log.write_text(out)
    return 124, out, True, time.perf_counter() - st


def _row_descriptor(change:dict[str, Any]) -> dict[str, Any]:
  spec = change["schedule_spec"]
  return {"tensor": change["tensor"], "format": change["format"], "role": change.get("role"), "spec": spec}


def _q4_command(model_path:pathlib.Path, tensor:str, spec:dict[str, Any], *, device:str, iters:int, seed:int) -> list[str]:
  mode = "serial" if spec.get("codegen_mode") == "direct_out" else "grouped" if spec.get("codegen_mode") == "grouped_partial" else "partial"
  cmd = [sys.executable, "extra/q4_k_bench.py", str(model_path), "--device", device, "--tensor", tensor,
         "--iters", str(iters), "--format", "text", "--activation", "random", "--seed", str(seed),
         "--primitive", "--primitive-mode", mode, "--primitive-parts", str(spec["parts"])]
  if mode == "grouped": cmd += ["--primitive-row-group", str(spec.get("row_group", 1))]
  for opt in spec.get("opts") or []: cmd += ["--primitive-opt", str(opt)]
  return cmd


def _q6_command(model_path:pathlib.Path, tensor:str, spec:dict[str, Any], *, device:str, iters:int, seed:int) -> list[str]:
  cmd = [sys.executable, "extra/q6_k_gemv_primitive.py", str(model_path), "--device", device, "--tensor", tensor,
         "--iters", str(iters), "--parts", str(spec["parts"]), "--seed", str(seed)]
  for opt in spec.get("opts") or []: cmd += ["--opt", str(opt)]
  return cmd


def _parse_result(fmt:str, out:str) -> dict[str, Any]:
  if fmt == "Q4_K":
    row = Q4_RESULT_RE.search(out)
    corr = Q4_GEMV_RE.search(out)
    return {
      "quant_gbs": None if row is None else float(row["gbs"]),
      "kernels": None if row is None else float(row["kernels"]),
      "gemv_max_abs": None if corr is None else float(corr["max_abs"]),
    }
  if fmt == "Q6_K":
    row = Q6_RESULT_RE.search(out)
    corr = Q6_GEMV_RE.search(out)
    return {
      "quant_gbs": None if row is None else float(row["gbs"]),
      "device_ms": None if row is None else float(row["ms"]),
      "gemv_max_abs": None if corr is None else float(corr["max_abs"]),
    }
  raise ValueError(f"unsupported format {fmt!r}")


def _run_schedule(model_path:pathlib.Path, tensor:str, fmt:str, spec:dict[str, Any], *,
                  repo:pathlib.Path, device:str, iters:int, seed:int, timeout:float, log:pathlib.Path) -> dict[str, Any]:
  if fmt == "Q4_K": cmd = _q4_command(model_path, tensor, spec, device=device, iters=iters, seed=seed)
  elif fmt == "Q6_K": cmd = _q6_command(model_path, tensor, spec, device=device, iters=iters, seed=seed)
  else: raise ValueError(f"unsupported format {fmt!r}")
  rc, out, timeout_hit, elapsed_s = _run(cmd, cwd=repo, env={"PYTHONPATH": ".", "DEV": device}, timeout=timeout, log=log)
  status = _classify(rc, out, timeout_hit)
  parsed = _parse_result(fmt, out) if status == "pass" else {"quant_gbs": None, "gemv_max_abs": None}
  return {
    "status": status,
    "elapsed_s": round(elapsed_s, 3),
    "command": " ".join(_portable_value(cmd, repo)),
    "log": _portable_str(str(log), repo),
    "tail": _portable_str("\n".join(out.strip().splitlines()[-10:]), repo),
    **parsed,
  }


def _current_spec(change:dict[str, Any]) -> dict[str, Any]:
  current = change["from"]
  return {
    "name": "current",
    "format": change["format"],
    "family": current["family"],
    "parts": current["parts"],
    "opts": current["opts"],
    "codegen_mode": "partial",
  }


def _decision(current:dict[str, Any], candidate:dict[str, Any], *, min_gain:float, tie_band:float) -> tuple[str, float | None, list[str]]:
  if current["status"] != "pass": return "invalid", None, ["current schedule did not pass microbench"]
  if candidate["status"] != "pass": return "invalid", None, [f"candidate status={candidate['status']}"]
  if current.get("quant_gbs") in (None, 0) or candidate.get("quant_gbs") is None:
    return "invalid", None, ["missing quant_gbs"]
  gain = candidate["quant_gbs"] / current["quant_gbs"] - 1.0
  if gain >= min_gain: return "raw_accept", gain, ["requires confirmation before promotion"]
  if abs(gain) <= tie_band: return "tie", gain, [f"within tie_band={tie_band:.3f}"]
  return "reject", gain, [f"below min_gain={min_gain:.3f}"]


def build_microbench_report(model:str, candidate_set:dict[str, Any], gate:dict[str, Any], *,
                            model_root:pathlib.Path=pathlib.Path("~/models"), repo:pathlib.Path=pathlib.Path.cwd(),
                            out:pathlib.Path, device:str="AMD", iters:int=3, seed:int=1337,
                            timeout:float=180.0, min_gain:float=0.03, tie_band:float=0.03,
                            limit:int | None=None) -> dict[str, Any]:
  if model not in MODEL_FILES: raise ValueError(f"semantic schedule gate only supports {sorted(MODEL_FILES)} by default")
  assert_q4k_native_sweep_allowed(device, "QK semantic schedule microbench")
  surface = "codegen" if candidate_set.get("kind") == "qk_semantic_codegen_candidate_set" else "schedule"
  model_path = (model_root / MODEL_FILES[model]).expanduser().resolve()
  gate_rows = {row["id"]: row for row in gate.get("rows", [])}
  candidates = [cand for cand in candidate_set.get("candidates", []) if cand.get("id") != "current"]
  candidates = [cand for cand in candidates if (gate_rows.get(cand["id"]) or {}).get("microbench")]
  if limit is not None: candidates = candidates[:limit]
  rows = []
  out.mkdir(parents=True, exist_ok=True)
  for cand in candidates:
    change = cand["changes"][0]
    desc = _row_descriptor(change)
    current_spec, candidate_spec = _current_spec(change), desc["spec"]
    cand_out = out / cand["id"]
    current = _run_schedule(model_path, desc["tensor"], desc["format"], current_spec, repo=repo, device=device,
                            iters=iters, seed=seed, timeout=timeout, log=cand_out / "current.log")
    candidate = _run_schedule(model_path, desc["tensor"], desc["format"], candidate_spec, repo=repo, device=device,
                              iters=iters, seed=seed, timeout=timeout, log=cand_out / "candidate.log")
    status, gain, reasons = _decision(current, candidate, min_gain=min_gain, tie_band=tie_band)
    rows.append({
      "id": cand["id"],
      "tensor": desc["tensor"],
      "format": desc["format"],
      "role": desc.get("role"),
      "schedule": candidate_spec,
      "full_decode_supported": (gate_rows.get(cand["id"]) or {}).get("full_decode_supported"),
      "status": status,
      "gain": gain,
      "current": current,
      "candidate": candidate,
      "correctness_provenance": candidate_spec.get("correctness_provenance"),
      "correctness": {
        "reference_unpacked": "covered_by_qk_layout_reference_tests",
        "amd_gemv": current["status"] == "pass" and candidate["status"] == "pass",
        "full_decode_ab": False,
      },
      "storage_effect": candidate_spec.get("storage_effect"),
      "reasons": reasons,
      "policy": _portable_str(str(out / cand["id"] / "policy.json"), repo),
    })
    write_json(cand_out / "policy.json", cand["policy"])
  accepted = [row for row in rows if is_raw_accept_status(row["status"])]
  full_decode_ready = [row for row in accepted if row.get("full_decode_supported")]
  return {
    "kind": f"qk_semantic_{surface}_microbench",
    "model": model.upper(),
    "source_candidates": candidate_set.get("source_descriptor"),
    "rows": rows,
    "summary": {
      "candidates": len(rows),
      "accepted": len(accepted),
      "raw_accepts": len(accepted),
      "ties": sum(1 for row in rows if row["status"] == "tie"),
      "rejected": sum(1 for row in rows if row["status"] == "reject"),
      "invalid": sum(1 for row in rows if row["status"] == "invalid"),
      "full_decode_ready": len(full_decode_ready),
      "next_decision": "run_full_policy_benchmark" if full_decode_ready else f"semantic_{surface}_frontier_blocked",
      "acceptance_rule": "raw_accept requires full-decode confirmation before promotion",
    },
  }


def report_markdown(report:dict[str, Any]) -> str:
  surface = "Codegen" if report.get("kind") == "qk_semantic_codegen_microbench" else "Schedule"
  lines = [
    f"# QK Semantic {surface} Microbench: {report['model']}",
    "",
    "Each candidate is compared against the current schedule for the same tensor.",
    "Full decode is only justified for accepted rows that are also runtime-policy",
    "supported.",
    "",
    "## Summary",
    "",
    f"- candidates: `{report['summary']['candidates']}`",
    f"- raw accepts: `{report['summary'].get('raw_accepts', report['summary']['accepted'])}`",
    f"- ties: `{report['summary']['ties']}`",
    f"- rejected: `{report['summary']['rejected']}`",
    f"- invalid: `{report['summary']['invalid']}`",
    f"- full decode ready: `{report['summary']['full_decode_ready']}`",
    f"- next decision: `{report['summary']['next_decision']}`",
    "",
    "| id | status | gain % | current GB/s | candidate GB/s | full decode | reasons |",
    "|---|---|---:|---:|---:|---:|---|",
  ]
  for row in report["rows"]:
    gain = row.get("gain")
    reasons = "; ".join(row.get("reasons") or []) or "none"
    lines.append(
      f"| `{row['id']}` | `{row['status']}` | "
      f"{'n/a' if gain is None else f'{gain*100:.2f}'} | "
      f"{'n/a' if row['current'].get('quant_gbs') is None else f'{row['current']['quant_gbs']:.2f}'} | "
      f"{'n/a' if row['candidate'].get('quant_gbs') is None else f'{row['candidate']['quant_gbs']:.2f}'} | "
      f"`{row.get('full_decode_supported')}` | {reasons} |"
    )
  lines.append("")
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description="Microbench semantic QK schedule/codegen candidates")
  parser.add_argument("--model", choices=tuple(MODEL_FILES), required=True)
  parser.add_argument("--candidates", type=pathlib.Path, required=True)
  parser.add_argument("--static-gate", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path, required=True)
  parser.add_argument("--json", type=pathlib.Path, required=True)
  parser.add_argument("--md", type=pathlib.Path, required=True)
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--model-root", type=pathlib.Path, default=pathlib.Path("~/models"))
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--iters", type=int, default=3)
  parser.add_argument("--seed", type=int, default=1337)
  parser.add_argument("--timeout", type=float, default=180.0)
  parser.add_argument("--min-gain", type=float, default=0.03)
  parser.add_argument("--tie-band", type=float, default=0.03)
  parser.add_argument("--limit", type=int)
  args = parser.parse_args()
  report = build_microbench_report(args.model, load_json(args.candidates.expanduser()), load_json(args.static_gate.expanduser()),
                                   model_root=args.model_root, repo=args.repo.resolve(), out=args.out,
                                   device=args.device, iters=args.iters, seed=args.seed, timeout=args.timeout,
                                   min_gain=args.min_gain, tie_band=args.tie_band, limit=args.limit)
  write_json(args.json, report)
  args.md.parent.mkdir(parents=True, exist_ok=True)
  args.md.write_text(report_markdown(report))
  print(report_markdown(report))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
