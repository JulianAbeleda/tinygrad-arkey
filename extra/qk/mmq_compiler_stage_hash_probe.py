"""CPU-only stage hash probe for native AMD lowering determinism.

This diagnostic loads one retained pre-lowering SINK and reproduces the
``do_to_program`` pipeline while retaining a content record after each major
lowering stage.  It never constructs a Device or runtime.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import difflib
import hashlib
import itertools
import json
from pathlib import Path
import pickle
from typing import Any, Mapping

from tinygrad.codegen import (
  line_rewrite, pm_linearize_cleanups, full_rewrite_to_sink,
)
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.codegen.late.regalloc import LinearScanRegallocContext, pm_regalloc_rewrite, pressure_schedule
from tinygrad.helpers import Target
from tinygrad.renderer.isa import CompilerCaptureProof, IselContext, PreRegAllocContext
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import Ops, ProgramInfo, UOp, graph_rewrite


SCHEMA = "tinygrad.mmq.compiler_stage_hash_probe.v1"
COMPARE_SCHEMA = "tinygrad.mmq.compiler_stage_hash_comparison.v1"


def _sha256(value: bytes) -> str: return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _stable(value: Any) -> Any:
  if value is None or type(value) in (bool, int, float, str): return value
  if isinstance(value, bytes): return {"bytes_sha256": _sha256(value), "nbytes": len(value)}
  if isinstance(value, UOp): return {"uop_key": value.key.hex()}
  if isinstance(value, tuple): return {"tuple": [_stable(item) for item in value]}
  if isinstance(value, list): return {"list": [_stable(item) for item in value]}
  if isinstance(value, Mapping):
    return {"mapping": [[str(key), _stable(item)] for key, item in sorted(value.items(), key=lambda row: str(row[0]))]}
  if is_dataclass(value):
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {field.name: _stable(getattr(value, field.name)) for field in fields(value)}}
  return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "text": str(value)}


def _record(uop: UOp) -> dict[str, Any]:
  arg, tag = _canonical(_stable(uop.arg)), _canonical(_stable(uop.tag))
  body = {
    "key": uop.key.hex(), "op": uop.op.name, "dtype": str(uop.dtype),
    "arg_sha256": _sha256(arg), "arg_text": str(uop.arg)[:256],
    "tag_sha256": _sha256(tag), "tag_text": str(uop.tag)[:256],
    "src_keys": [source.key.hex() for source in uop.src],
  }
  return {"token": _sha256(_canonical(body)), **body}


def _stage(name: str, uops: list[UOp], catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
  tokens = []
  for uop in uops:
    record = _record(uop)
    token = record.pop("token")
    catalog.setdefault(token, record)
    tokens.append(token)
  return {"name": name, "count": len(tokens), "sha256": _sha256(_canonical(tokens)), "tokens": tokens}


def _graph_stage(name: str, sink: UOp, catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
  return _stage(name, list(sink.toposort()), catalog)


def capture(sink_path: Path) -> dict[str, Any]:
  sink_bytes = sink_path.read_bytes()
  sink = pickle.loads(sink_bytes)
  if not isinstance(sink, UOp) or sink.op is not Ops.SINK:
    raise ValueError("stage probe requires one pickled Ops.SINK")
  renderer = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  catalog: dict[str, dict[str, Any]] = {}
  stages = [_graph_stage("retained_sink", sink, catalog)]

  selected = full_rewrite_to_sink(sink, renderer, optimize=sink.tag is None)
  program_info = ProgramInfo.from_sink(selected)
  stages.append(_graph_stage("full_rewrite_to_sink", selected, catalog))

  selected = graph_rewrite(
    selected, renderer.pre_isel_matcher, ctx=itertools.count(-1, -1),
    name="probe pre instruction selection", bottom_up=True)
  stages.append(_graph_stage("pre_instruction_selection", selected, catalog))
  isel_ctx = IselContext(selected)
  selected = graph_rewrite(
    selected, renderer.isel_matcher, ctx=isel_ctx,
    name="probe instruction selection", bottom_up=True)
  stages.append(_graph_stage("instruction_selection", selected, catalog))
  if renderer.post_isel_matcher is not None:
    selected = graph_rewrite(
      selected, renderer.post_isel_matcher, ctx=isel_ctx,
      name="probe post instruction selection", bottom_up=True)
  if (selection_proof := renderer.capture_selection_proof(isel_ctx)) is not None:
    selected = selected.replace(tag=selection_proof)
  stages.append(_graph_stage("post_instruction_selection", selected, catalog))

  lines = linearize(selected)
  stages.append(_stage("linearize", lines, catalog))
  lines = line_rewrite(lines, pm_linearize_cleanups)
  stages.append(_stage("linearize_cleanup", lines, catalog))
  lines = pressure_schedule(lines)
  stages.append(_stage("pressure_schedule", lines, catalog))
  if renderer.pre_regalloc_matcher is not None:
    lines = line_rewrite(lines, renderer.pre_regalloc_matcher, PreRegAllocContext())
  stages.append(_stage("pre_regalloc", lines, catalog))
  regalloc_ctx = LinearScanRegallocContext(lines, renderer)
  lines = line_rewrite(lines, pm_regalloc_rewrite, regalloc_ctx)
  stages.append(_stage("regalloc", lines, catalog))
  lines = line_rewrite(lines, renderer.post_regalloc_matcher, regalloc_ctx)
  stages.append(_stage("post_regalloc", lines, catalog))

  final_proof = selection_proof.finalize_zero_spill() \
    if isinstance(selection_proof, CompilerCaptureProof) and not regalloc_ctx.spills and regalloc_ctx.stack_size == 0 else None
  linear_uop = UOp(Ops.LINEAR, src=tuple(lines), arg=final_proof)
  program = UOp(Ops.PROGRAM, src=(selected, UOp(Ops.DEVICE, arg=renderer.target.device), linear_uop), arg=program_info)
  source = "\n".join(str(uop.arg) for uop in lines)
  binary = renderer.asm(program, linear_uop)
  assembly = {
    "source_sha256": _sha256(source.encode()), "source_nbytes": len(source.encode()),
    "binary_sha256": _sha256(binary), "binary_nbytes": len(binary),
    "program_key_before_payload": program.key.hex(),
  }
  body = {
    "schema": SCHEMA, "cpu_only": True, "target": "AMD:ISA:gfx1100",
    "sink_file": str(sink_path), "sink_pickle_sha256": _sha256(sink_bytes),
    "sink_key": sink.key.hex(), "stages": stages, "catalog": catalog, "assembly": assembly,
  }
  return {**body, "evidence_sha256": _sha256(_canonical(body))}


def compare(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
  if first.get("schema") != SCHEMA or second.get("schema") != SCHEMA:
    raise ValueError("comparison requires two stage probe records")
  if first.get("sink_pickle_sha256") != second.get("sink_pickle_sha256") or first.get("sink_key") != second.get("sink_key"):
    raise ValueError("stage probes do not share one retained SINK")
  left, right = first["stages"], second["stages"]
  if [row["name"] for row in left] != [row["name"] for row in right]:
    raise ValueError("stage probe inventories differ")
  rows, earliest = [], None
  for a, b in zip(left, right):
    same = a["sha256"] == b["sha256"] and a["count"] == b["count"]
    row: dict[str, Any] = {
      "name": a["name"], "identical": same,
      "first": {"count": a["count"], "sha256": a["sha256"]},
      "second": {"count": b["count"], "sha256": b["sha256"]},
    }
    if not same:
      if earliest is None: earliest = a["name"]
      atokens, btokens = a["tokens"], b["tokens"]
      changes = []
      for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=atokens, b=btokens, autojunk=False).get_opcodes():
        if tag == "equal": continue
        changes.append({
          "kind": tag, "first_range": [i1, i2], "second_range": [j1, j2],
          "first_records": [first["catalog"][token] for token in atokens[i1:i2]],
          "second_records": [second["catalog"][token] for token in btokens[j1:j2]],
        })
        if len(changes) == 8: break
      row["changes"] = changes
    rows.append(row)
  assembly = {
    key: {"first": first["assembly"][key], "second": second["assembly"][key],
          "identical": first["assembly"][key] == second["assembly"][key]}
    for key in first["assembly"]
  }
  body = {
    "schema": COMPARE_SCHEMA, "state": "PASS" if earliest is None and all(x["identical"] for x in assembly.values()) else "DIVERGED",
    "cpu_only": True, "sink_pickle_sha256": first["sink_pickle_sha256"], "sink_key": first["sink_key"],
    "earliest_divergent_stage": earliest, "stages": rows, "assembly": assembly,
  }
  return {**body, "evidence_sha256": _sha256(_canonical(body))}


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)
  cap = sub.add_parser("capture")
  cap.add_argument("--sink", type=Path, required=True)
  cap.add_argument("--output", type=Path, required=True)
  cmp = sub.add_parser("compare")
  cmp.add_argument("--first", type=Path, required=True)
  cmp.add_argument("--second", type=Path, required=True)
  cmp.add_argument("--output", type=Path, required=True)
  args = parser.parse_args(argv)
  if args.output.exists(): raise FileExistsError(args.output)
  result = capture(args.sink) if args.command == "capture" else compare(
    json.loads(args.first.read_text()), json.loads(args.second.read_text()))
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
  print(json.dumps({
    "output": str(args.output), "file_sha256": _sha256(args.output.read_bytes()),
    "evidence_sha256": result["evidence_sha256"],
    **({"earliest_divergent_stage": result["earliest_divergent_stage"]} if args.command == "compare" else
       {"stages": {row["name"]: row["sha256"] for row in result["stages"]}, "assembly": result["assembly"]}),
  }, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
