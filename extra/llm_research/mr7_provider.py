#!/usr/bin/env python3
"""Exact MR7 fused-call/outer provider over one prepared Tinygrad decode capture.

This is intentionally separate from ``search_provider``: candidate search owns
one semantic matmul, while MR7 measures identities already compiled in the
production model JIT. One process prepares the model once and serves every
JSON-lines request from that same capture/cache.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
import pathlib
import statistics
import sys
from typing import Any, Protocol, TextIO
from tinygrad.engine.metadata import PROGRAM_IDENTITY_FIELDS
from extra.llm_research.finalized_census import validate_finalized_census


PROTOCOL = "boltbeam.mr7_provider.v1"
def _canonical(value: Any) -> str: return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
def _sha(value: Any) -> str: return hashlib.sha256(value if isinstance(value, bytes) else _canonical(value).encode()).hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
  state = hashlib.sha256()
  with path.open("rb") as source:
    for block in iter(lambda:source.read(1 << 20), b""): state.update(block)
  return state.hexdigest()


def _provider_identity() -> dict[str, Any]:
  from extra.llm_research.search_provider import _revision
  return {**_revision(), "script_sha256":_file_sha256(pathlib.Path(__file__).resolve())}


def _execution_identity(row: Mapping[str, Any]) -> dict[str, Any]:
  return {key:row[key] for key in ("program_hash", "source_sha256", "binary_sha256")} | {
    "semantic_identities":sorted(({key:identity[key] for key in PROGRAM_IDENTITY_FIELDS}
      for identity in row["semantic_identities"]), key=_canonical)}


def _outer_id(row: Mapping[str, Any]) -> str:
  return f"graph:{row['batch_index']}" if row["assignment"] == "graph" else f"direct:{row['direct_call_index']}"


def validate_request(census: Mapping[str, Any], request: Mapping[str, Any]) -> tuple[str, list[Mapping[str, Any]]]:
  """Join one immutable BoltBeam request to exact finalized census records."""
  if request.get("kind") not in ("exact_isolated_execution", "complete_outer_enclosure"):
    raise ValueError("unsupported MR7 provider request kind")
  records = [row for row in census.get("records", []) if row.get("assignment") in ("graph", "direct")]
  call_indices = request.get("call_indices")
  if not isinstance(call_indices, list) or not call_indices or any(not isinstance(index, int) for index in call_indices):
    raise ValueError("MR7 provider request requires exact call indices")
  selected = [row for row in records if row.get("call_index") in call_indices]
  if len(selected) != len(call_indices) or sorted(row["call_index"] for row in selected) != sorted(call_indices):
    raise ValueError("MR7 provider request call coverage differs from census")
  if request["kind"] == "exact_isolated_execution":
    if request.get("shape_mode") != "exact_workload" or request.get("correctness_required") is not True:
      raise ValueError("isolated MR7 request requires exact workload correctness")
    if any(_execution_identity(row) != request.get("execution_identity") for row in selected):
      raise ValueError("isolated MR7 request execution identity differs from census")
    return request["kind"], selected
  if request.get("shape_mode") != "complete_outer" or request.get("no_member_duration_splitting") is not True:
    raise ValueError("outer MR7 request must forbid member-duration splitting")
  outer_id = request.get("outer_id")
  expected = sorted(row["call_index"] for row in records if _outer_id(row) == outer_id)
  if expected != sorted(call_indices): raise ValueError("outer MR7 request is not the complete census enclosure")
  return request["kind"], selected


class MR7Session(Protocol):
  def execute(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass
class TinygradDecodeMR7Session:
  model_path: pathlib.Path
  model_sha256: str
  census: Mapping[str, Any]
  fixed_depth: int
  max_context: int = 4608
  chunk_size: int = 32
  warmup_decode: int = 3
  warmups: int = 2
  samples: int = 5
  seed: int = 20260617
  _capture: Any = field(default=None, init=False, repr=False)
  _outer_calls: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

  def __post_init__(self):
    self.model_path = self.model_path.expanduser().resolve()
    if _file_sha256(self.model_path) != self.model_sha256: raise ValueError("selected GGUF SHA-256 mismatch")
    validate_finalized_census(self.census)
    if self.fixed_depth < 1 or self.max_context <= self.fixed_depth or self.chunk_size < 1 or self.warmup_decode < 2:
      raise ValueError("invalid exact decode preparation geometry")
    if self.warmups < 0 or self.samples < 5: raise ValueError("MR7 requires at least five samples")

  def _prepare(self) -> None:
    if self._capture is not None: return
    from tinygrad import Device
    if Device.DEFAULT.split(":")[0] != "METAL": raise RuntimeError("MR7 exact provider requires METAL")
    from tinygrad.llm.generate import load_model_and_tokenizer
    from extra.llm_research.decode.decode_runtime_overhead import _make_prompt, capture_decode_graph
    model, tokenizer = load_model_and_tokenizer(str(self.model_path), self.max_context, seed=self.seed)
    base_ids = (tokenizer.prefix() if hasattr(tokenizer, "prefix") else []) + tokenizer.encode("the quick brown fox jumps. " * 800)
    live = capture_decode_graph(model, _make_prompt(base_ids, self.fixed_depth), self.chunk_size, self.warmup_decode)
    expected = {key:value for key,value in self.census.items() if key != "capture"}
    if live.to_dict() != expected: raise RuntimeError("live prepared graph census differs from finalized MR5 census")
    if len(live.calls) != live.to_dict()["counts"]["logical_calls"] or live.execution_linear is None or not live.execution_inputs:
      raise RuntimeError("prepared graph lacks runtime call/input bindings")
    self._capture = live
    self._outer_calls = self._map_outer_calls()

  def _map_outer_calls(self) -> dict[str, Any]:
    from tinygrad.uop.ops import Ops
    records = [row for row in self.census["records"] if row["assignment"] in ("graph", "direct")]
    outer_ids = sorted({_outer_id(row) for row in records}, key=lambda outer:min(
      row["call_index"] for row in records if _outer_id(row) == outer))
    calls = list(self._capture.execution_linear.src)
    if len(calls) != len(outer_ids): raise RuntimeError("captured outer calls do not reconcile with census")
    result = {}
    for outer_id,call in zip(outer_ids, calls):
      expected = [row["program_hash"] for row in records if _outer_id(row) == outer_id]
      ast = call.src[0]
      programs = list(ast.src[0].src) if ast.op is Ops.CUSTOM_FUNCTION and ast.arg == "graph" else [call]
      observed = [program.src[0].key.hex() for program in programs]
      if observed != expected: raise RuntimeError("captured outer program identities differ from census")
      result[outer_id] = call
    return result

  def _time(self, call: Any) -> list[int]:
    from tinygrad.engine.realize import time_call
    for _ in range(self.warmups):
      time_call(call, self._capture.execution_var_vals, input_uops=self._capture.execution_inputs)
    values = [int(time_call(call, self._capture.execution_var_vals,
      input_uops=self._capture.execution_inputs) * 1e9) for _ in range(self.samples)]
    if any(value <= 0 or not math.isfinite(value) for value in values): raise RuntimeError("provider timing is invalid")
    return values

  def _output_sha256(self, call: Any) -> str:
    from tinygrad.engine.realize import get_call_outs_ins, resolve_params
    arguments = resolve_params(call, self._capture.execution_inputs)
    outputs, _inputs = get_call_outs_ins(call)
    if not outputs: raise RuntimeError("isolated captured call has no output buffers")
    state = hashlib.sha256()
    for index in sorted(outputs): state.update(arguments[index].buffer.ensure_allocated().numpy().tobytes())
    return state.hexdigest()

  def execute(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
    kind, rows = validate_request(self.census, request)
    self._prepare()
    if kind == "complete_outer_enclosure":
      samples = self._time(self._outer_calls[request["outer_id"]])
      evidence = {"outer_id":request["outer_id"], "call_indices":request["call_indices"],
        "samples_ns":samples, "measurement_scope":"complete_outer", "no_member_duration_splitting":True}
      return {"outer_id":request["outer_id"], "call_indices":request["call_indices"], "shape_mode":"complete_outer",
        "measurement":{"samples_ns":samples, "median_ns":statistics.median(samples)},
        "performance_evidence_hash":_sha(evidence)}

    call = self._capture.calls[rows[0]["call_index"]]
    # Refresh every intermediate through the complete captured execution, then
    # compare the isolated call's output to that full-capture reference before
    # accepting timing evidence.
    from tinygrad.engine.realize import run_linear, time_call
    run_linear(self._capture.execution_linear, self._capture.execution_var_vals,
      input_uops=self._capture.execution_inputs, jit=True, wait=True)
    reference_hash = self._output_sha256(call)
    time_call(call, self._capture.execution_var_vals, input_uops=self._capture.execution_inputs)
    observed_hash = self._output_sha256(call)
    correctness_evidence = {"oracle":"isolated exact call output compared with refreshed complete-capture output",
      "reference_output_sha256":reference_hash, "observed_output_sha256":observed_hash,
      "model_sha256":self.model_sha256, "call_indices":request["call_indices"]}
    samples = self._time(call); identity = request["execution_identity"]
    performance = {"samples_ns":samples, "execution_identity":identity, "timing_mode":"tinygrad time_call wait=True"}
    return {"execution_identity":identity, "shape_mode":"exact_workload",
      "measurement":{"samples_ns":samples, "median_ns":statistics.median(samples)},
      "correctness":{"passed":reference_hash == observed_hash, "evidence_hash":_sha(correctness_evidence),
                     "evidence":correctness_evidence},
      "performance_evidence_hash":_sha(performance),
      "generated":{"source_hash":identity["source_sha256"], "binary_hash":identity["binary_sha256"],
                   "plan_hash":identity["program_hash"]}}


def serve(source: TextIO, sink: TextIO, session: MR7Session) -> int:
  provider_identity = _provider_identity()
  for line in source:
    request_id = "unknown"
    try:
      envelope = json.loads(line)
      if not isinstance(envelope, Mapping) or set(envelope) != {"protocol", "plan_sha256", "request_id", "request"}:
        raise ValueError("invalid MR7 provider envelope")
      if envelope["protocol"] != PROTOCOL or not isinstance(envelope["request_id"], str) or not envelope["request_id"]:
        raise ValueError("MR7 provider envelope identity mismatch")
      request_id = envelope["request_id"]
      if not isinstance(envelope["plan_sha256"], str) or len(envelope["plan_sha256"]) != 64:
        raise ValueError("MR7 provider plan hash is invalid")
      if not isinstance(envelope["request"], Mapping) or envelope["request"].get("request_id") != request_id:
        raise ValueError("MR7 provider request identity mismatch")
      result = {**session.execute(envelope["request"]), "provider_identity":provider_identity}
      response = {"protocol":PROTOCOL, "request_id":request_id, "status":"ok", "result":dict(result)}
    except Exception as exc:
      response = {"protocol":PROTOCOL, "request_id":request_id, "status":"blocked",
        "error":{"code":"provider_failure", "message":str(exc), "exception_type":type(exc).__name__}}
    sink.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n"); sink.flush()
  return 0


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=pathlib.Path, required=True); parser.add_argument("--model-sha256", required=True)
  parser.add_argument("--census", type=pathlib.Path, required=True); parser.add_argument("--fixed-depth", type=int, required=True)
  parser.add_argument("--max-context", type=int, default=4608); parser.add_argument("--chunk-size", type=int, default=32)
  parser.add_argument("--warmup-decode", type=int, default=3); parser.add_argument("--warmups", type=int, default=2)
  parser.add_argument("--samples", type=int, default=5); parser.add_argument("--seed", type=int, default=20260617)
  args = parser.parse_args(argv)
  census = json.loads(args.census.read_text())
  session = TinygradDecodeMR7Session(args.model, args.model_sha256, census, args.fixed_depth, args.max_context,
    args.chunk_size, args.warmup_decode, args.warmups, args.samples, args.seed)
  return serve(sys.stdin, sys.stdout, session)


if __name__ == "__main__": raise SystemExit(main())

__all__ = ["PROTOCOL", "MR7Session", "TinygradDecodeMR7Session", "serve", "validate_request"]
