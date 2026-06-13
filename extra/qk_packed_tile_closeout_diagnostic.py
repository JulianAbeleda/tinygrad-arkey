#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, re, subprocess, sys
from collections import Counter
from typing import Any

from extra.qk_load_width_report import build_report as build_load_width_report
from extra.qk_load_width_report import report_markdown as load_width_report_markdown

DEFAULT_ARTIFACT = pathlib.Path("bench/qk-packed-tile-research-closeout-20260613")
DEFAULT_MODEL = pathlib.Path("~/models/Qwen3-8B-Q4_K_M.gguf")
DEFAULT_TENSOR = "blk.0.ffn_gate.weight"
ROWS, K, PARTS = 64, 4096, 1

TARGETS = {
  "v1_partial": {
    "kernel": f"q4k_gemv_partial_{ROWS}_{K}_{PARTS}",
    "args": ["--mode", "partial", "--parts", str(PARTS), "--opt", "LOCAL:0:32"],
  },
  "tile_custom": {
    "kernel": f"q4k_gemv_tile_custom_partial_{ROWS}_{K}_{PARTS}",
    "args": ["--mode", "tile_custom", "--parts", str(PARTS)],
  },
}

PROFILE_RE = re.compile(
  r"^\*\*\* AMD\s+\d+\s+(?P<kernel>\S+)\s+arg.*?\btm\s+(?P<time>[0-9.]+)(?P<unit>us|ms|s)\b"
)
WORKGROUP_RE = re.compile(r"amdgpu_flat_work_group_size\(1,\s*(?P<size>\d+)\)")
GROUP_ID_RE = re.compile(r"int gidx(?P<axis>\d+) = .*?/\*\s*(?P<size>\d+)\s*\*/")
LOCAL_ID_RE = re.compile(r"int lidx(?P<axis>\d+) = .*?/\*\s*(?P<size>\d+)\s*\*/")
DISASM_LABEL_RE = re.compile(r"^[0-9a-fA-F]+ <(?P<kernel>[^>]+)>:")
INST_RE = re.compile(r"^\s*(?P<mnemonic>[a-zA-Z][a-zA-Z0-9_]+)\b")
MEM_INST_RE = re.compile(r"\b(?P<mnemonic>(?:global|flat|buffer)_(?:load|store)_[A-Za-z0-9_]+|s_load_[A-Za-z0-9_]+)\b")
VECTOR_SOURCE_RE = re.compile(r"\b(?:tg_uint4|uint4|unsigned_int4)\b")


def _portable(path:pathlib.Path, repo:pathlib.Path) -> str:
  try:
    return str(path.resolve().relative_to(repo.resolve()))
  except ValueError:
    return str(path)


def _time_to_us(value:float, unit:str) -> float:
  if unit == "us": return value
  if unit == "ms": return value * 1000.0
  if unit == "s": return value * 1000000.0
  raise ValueError(f"unknown DEBUG time unit {unit!r}")


def _sanitize_log(text:str, *, repo:pathlib.Path, model:pathlib.Path) -> str:
  out = text.replace(str(repo.resolve()), ".")
  if model.exists():
    home = pathlib.Path.home().resolve()
    try:
      model_label = "~/" + str(model.resolve().relative_to(home))
    except ValueError:
      model_label = model.name
    out = out.replace(str(model.resolve()), model_label)
  return "\n".join(line.rstrip() for line in out.splitlines()) + "\n"


def _extract_source_block(text:str, kernel:str) -> str:
  lines = text.splitlines()
  signature = f" {kernel}("
  start = next((idx for idx, line in enumerate(lines) if signature in line and "extern \"C\"" in line), None)
  if start is None:
    raise ValueError(f"target generated source for {kernel!r} not found")
  end = len(lines)
  for idx in range(start + 1, len(lines)):
    line = lines[idx]
    if line.startswith("system: ") or line.startswith("<stdin>:") or "Disassembly of section" in line:
      end = idx
      break
  return "\n".join(lines[start:end])


def _extract_disasm_block(text:str, kernel:str) -> str:
  lines = text.splitlines()
  start = next((idx for idx, line in enumerate(lines) if f"<{kernel}>:" in line), None)
  if start is None:
    raise ValueError(f"target disassembly for {kernel!r} not found")
  block = []
  for line in lines[start + 1:]:
    label = DISASM_LABEL_RE.match(line)
    if block and (line.startswith("*** ") or line.startswith("<stdin>:") or "Disassembly of section" in line or
                  (label is not None and label.group("kernel") != kernel)):
      break
    if line.strip(): block.append(line)
  if not block:
    raise ValueError(f"target disassembly for {kernel!r} is empty")
  return "\n".join(block)


def _profile_times(text:str, kernel:str) -> list[float]:
  times = []
  for line in text.splitlines():
    match = PROFILE_RE.match(line)
    if match and match.group("kernel") == kernel:
      times.append(_time_to_us(float(match.group("time")), match.group("unit")))
  return times


def parse_debug7_log(text:str, *, kernel:str, mode:str) -> dict[str, Any]:
  source = _extract_source_block(text, kernel)
  disasm = _extract_disasm_block(text, kernel)

  wg = WORKGROUP_RE.search(source)
  if wg is None: raise ValueError(f"{kernel}: target source has no flat work-group size")
  gidx = {int(m.group("axis")): int(m.group("size")) for m in GROUP_ID_RE.finditer(source)}
  lidx = {int(m.group("axis")): int(m.group("size")) for m in LOCAL_ID_RE.finditer(source)}

  instruction_counts: Counter[str] = Counter()
  memory_counts: Counter[str] = Counter()
  for line in disasm.splitlines():
    inst = INST_RE.match(line)
    if inst: instruction_counts[inst.group("mnemonic")] += 1
    mem = MEM_INST_RE.search(line)
    if mem: memory_counts[mem.group("mnemonic")] += 1

  times = _profile_times(text, kernel)
  return {
    "mode": mode,
    "kernel": kernel,
    "workgroup_size": int(wg.group("size")),
    "group_counts": {f"gidx{axis}": size for axis, size in sorted(gidx.items())},
    "local_counts": {f"lidx{axis}": size for axis, size in sorted(lidx.items())},
    "source_lines": len(source.splitlines()),
    "source_has_vector_type": bool(VECTOR_SOURCE_RE.search(source)),
    "source_has_tg_uint4_load": "tg_uint4 qv = *((tg_uint4*)" in source,
    "disasm_lines": len(disasm.splitlines()),
    "instruction_count": sum(instruction_counts.values()),
    "memory_instruction_count": sum(memory_counts.values()),
    "instruction_counts": dict(sorted(instruction_counts.items())),
    "memory_instruction_counts": dict(sorted(memory_counts.items())),
    "global_load_b128": memory_counts.get("global_load_b128", 0),
    "global_load_b32": memory_counts.get("global_load_b32", 0),
    "global_load_b64": memory_counts.get("global_load_b64", 0),
    "profile_times_us": times,
    "last_profile_time_us": times[-1] if times else None,
  }


def summarize(parsed:dict[str, dict[str, Any]], *, load_width_report:dict[str, Any]|None=None) -> dict[str, Any]:
  v1, tile = parsed["v1_partial"], parsed["tile_custom"]
  tile_has_wider_loads = tile["global_load_b128"] > v1["global_load_b128"]
  tile_loses_parallelism = tile["workgroup_size"] < v1["workgroup_size"]
  tile_larger_body = tile["instruction_count"] > v1["instruction_count"] * 2
  likely_cause = []
  if tile_has_wider_loads:
    likely_cause.append("tile_custom emits more target global_load_b128 instructions")
  if tile_loses_parallelism:
    likely_cause.append("tile_custom is workgroup-size 1 while v1 uses the 32-lane LOCAL schedule")
  if tile_larger_body:
    likely_cause.append("tile_custom target instruction body is more than 2x larger than v1")
  return {
    "decision": "raw_custom_tile_path_closed_not_promoted",
    "reason": (
      "The raw custom tile body proves vector packed loads can be emitted, but it is opaque to tinygrad's scheduler "
      "and gives up the 32-lane scheduled shape of the current v1 kernel. Repeated microbenchmarks already showed "
      "no general speedup, so source vectorization alone is not a sufficient optimization path."
    ),
    "next_allowed_path": (
      "Only continue this line as a first-class packed QK semantic op / renderer lowering that preserves both "
      "wide/coalesced loads and schedulable row/K parallelism. Do not broaden raw Ops.CUSTOM tg_uint4 variants."
    ),
    "tile_has_wider_loads": tile_has_wider_loads,
    "tile_loses_parallelism": tile_loses_parallelism,
    "tile_larger_body": tile_larger_body,
    "likely_cause": likely_cause,
    "load_width_summary": None if load_width_report is None else load_width_report["summary"],
  }


def build_report(logs:dict[str, pathlib.Path], *, repo:pathlib.Path) -> dict[str, Any]:
  parsed = {}
  for mode, path in logs.items():
    text = path.read_text(errors="replace")
    parsed[mode] = parse_debug7_log(text, kernel=TARGETS[mode]["kernel"], mode=mode)
  load_report = build_load_width_report(list(logs.values()), repo=repo)
  return {
    "kind": "qk_packed_tile_research_closeout",
    "schema_version": 1,
    "artifact": _portable(logs["v1_partial"].parents[1], repo),
    "shape": {"tensor": DEFAULT_TENSOR, "rows": ROWS, "k": K, "parts": PARTS},
    "modes": parsed,
    "summary": summarize(parsed, load_width_report=load_report),
  }


def report_markdown(report:dict[str, Any]) -> str:
  lines = [
    "# Packed QK Tile Research Close-Out",
    "",
    f"Decision: `{report['summary']['decision']}`",
    "",
    report["summary"]["reason"],
    "",
    "## Target Assembly Summary",
    "",
    "| mode | workgroup | group ids | local ids | source vector | disasm inst | mem inst | global_load_b128 | global_load_b32 | global_load_b64 | last DEBUG time |",
    "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for mode in ("v1_partial", "tile_custom"):
    row = report["modes"][mode]
    last = row["last_profile_time_us"]
    lines.append(
      f"| `{mode}` | `{row['workgroup_size']}` | `{row['group_counts']}` | `{row['local_counts']}` | "
      f"`{row['source_has_vector_type']}` | `{row['instruction_count']}` | `{row['memory_instruction_count']}` | "
      f"`{row['global_load_b128']}` | `{row['global_load_b32']}` | `{row['global_load_b64']}` | "
      f"`{last:.2f} us` |" if last is not None else
      f"| `{mode}` | `{row['workgroup_size']}` | `{row['group_counts']}` | `{row['local_counts']}` | "
      f"`{row['source_has_vector_type']}` | `{row['instruction_count']}` | `{row['memory_instruction_count']}` | "
      f"`{row['global_load_b128']}` | `{row['global_load_b32']}` | `{row['global_load_b64']}` | `n/a` |"
    )
  lines += [
    "",
    "## Interpretation",
    "",
  ]
  for cause in report["summary"]["likely_cause"]:
    lines.append(f"- {cause}.")
  lines += [
    "",
    "The positive evidence is real: `tile_custom` emits target `global_load_b128`",
    "instructions and the generated source contains the intended `tg_uint4` load.",
    "The negative evidence is also decisive for this raw path: the target kernel is",
    "single-work-item per row, much larger, and opaque to BEAM/tinygrad scheduling.",
    "That explains why the repeated microbench artifact did not generalize.",
    "",
    "## Next Allowed Path",
    "",
    report["summary"]["next_allowed_path"],
    "",
    "This report uses DEBUG=7 disassembly and DEBUG timing as diagnostic evidence,",
    "not as a new throughput claim. The promotion gate remains the repeated",
    "microbench/full-decode harness.",
    "",
  ]
  return "\n".join(lines)


def run_debug7_logs(*, repo:pathlib.Path, model:pathlib.Path, outdir:pathlib.Path, tensor:str, device:str, python:str) -> dict[str, pathlib.Path]:
  outdir.mkdir(parents=True, exist_ok=True)
  env = os.environ.copy()
  env["DEV"] = device
  env["DEBUG"] = "7"
  env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
  logs = {}
  for mode, spec in TARGETS.items():
    cmd = [
      python, "extra/q4_k_gemv_primitive.py", str(model), "--device", device, "--tensor", tensor,
      "--rows", str(ROWS), "--iters", "1", "--unpack-check-rows", "1", *spec["args"],
    ]
    result = subprocess.run(cmd, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    text = result.stdout.decode("utf-8", errors="replace")
    path = outdir / f"{mode}-debug7.log"
    path.write_text(_sanitize_log(text, repo=repo, model=model))
    if result.returncode != 0:
      raise RuntimeError(f"{mode} DEBUG=7 run failed with exit {result.returncode}; see {path}")
    logs[mode] = path
  return logs


def write_artifact(report:dict[str, Any], *, artifact:pathlib.Path, logs:dict[str, pathlib.Path], repo:pathlib.Path) -> None:
  artifact.mkdir(parents=True, exist_ok=True)
  (artifact / "diagnostic.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  (artifact / "diagnostic.md").write_text(report_markdown(report))
  load_report = build_load_width_report(list(logs.values()), repo=repo)
  (artifact / "source" / "load-width-report.json").write_text(json.dumps(load_report, indent=2, sort_keys=True) + "\n")
  (artifact / "source" / "load-width-report.md").write_text(load_width_report_markdown(load_report))
  (artifact / "README.md").write_text(report_markdown(report))


def main() -> int:
  parser = argparse.ArgumentParser(description="Close out raw PackedQKTile lowering with target AMD DEBUG=7 diagnostics")
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--artifact", type=pathlib.Path, default=DEFAULT_ARTIFACT)
  parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
  parser.add_argument("--tensor", default=DEFAULT_TENSOR)
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--python", default=sys.executable)
  parser.add_argument("--reuse", action="store_true", help="reuse existing source/*-debug7.log files instead of rerunning")
  args = parser.parse_args()

  repo = args.repo.resolve()
  artifact = (repo / args.artifact).resolve() if not args.artifact.is_absolute() else args.artifact.resolve()
  source_dir = artifact / "source"
  model = args.model.expanduser().resolve()

  if args.reuse:
    logs = {mode: source_dir / f"{mode}-debug7.log" for mode in TARGETS}
    missing = [str(path) for path in logs.values() if not path.exists()]
    if missing: raise FileNotFoundError(f"--reuse requested but logs are missing: {missing}")
  else:
    if not model.exists(): raise FileNotFoundError(f"model not found: {model}")
    logs = run_debug7_logs(repo=repo, model=model, outdir=source_dir, tensor=args.tensor, device=args.device, python=args.python)

  report = build_report(logs, repo=repo)
  write_artifact(report, artifact=artifact, logs=logs, repo=repo)
  print(report_markdown(report))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
