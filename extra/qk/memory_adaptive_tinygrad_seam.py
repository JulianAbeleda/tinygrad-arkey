#!/usr/bin/env python3
"""Concrete tinygrad whole-model seam for memory-adaptive machine search.

The parent is import-light and metadata-only.  Every candidate is executed in
its own process; the child binds the exact policy through a private scoped
measurement authority and uses the production ``model.generate`` path. Evidence is
fail-closed: a real run can still be incomplete, and no missing artifact is
converted into PASS.
"""
from __future__ import annotations

import argparse, contextlib, gc, hashlib, json, math, os, pathlib, subprocess, sys, time, traceback
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from extra.qk.memory_adaptive_candidate_catalog import CandidateSpec, inventory_invocation_ids
from extra.qk.memory_adaptive_transport import SelectedModelScan
from extra.qk.memory_adaptive_allocation_observer import (SCHEMA as CHECKPOINT_OBSERVER_SCHEMA,
  derive_memory_facts, validate_memory_facts)
from tinygrad.llm.prefill_memory_plan import ByteLifetime, ByteTerm, Strategy
from extra.qk.prefill_workload_plan import InvocationBytes, RemainderMapping
from tinygrad.llm.memory_semantics import MemorySemanticOwner, MemorySemanticClass

SCHEMA = "tinygrad.memory_adaptive_tinygrad_seam.v1"
WORKER_SCHEMA = "tinygrad.memory_adaptive_tinygrad_worker.v2"
ROOT = pathlib.Path(__file__).resolve().parents[2]
_RESOURCE_ZERO_FIELDS = ("scratch_bytes", "vgpr_spills", "sgpr_spills")


def _transport_encode(value: Any) -> Any:
  """Encode only known artifact values; never stringify an authority object."""
  if value is None or isinstance(value, (str, int, float, bool)): return value
  if isinstance(value, MemorySemanticOwner):
    return {"semantic_class": value.semantic_class.value, "candidate_id": value.candidate_id}
  if isinstance(value, Enum): return _transport_encode(value.value)
  if is_dataclass(value) and not isinstance(value, type):
    return {field.name: _transport_encode(getattr(value, field.name)) for field in fields(value)}
  if isinstance(value, Mapping):
    if any(not isinstance(key, str) for key in value): raise TypeError("artifact mapping keys must be strings")
    return {key: _transport_encode(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)): return [_transport_encode(item) for item in value]
  raise TypeError(f"unknown artifact transport object: {type(value).__name__}")


def _align_up(value:int, alignment:int) -> int: return ((value+alignment-1)//alignment)*alignment


def _selected_runtime_base_terms(kv:Mapping[str, Any], *, backing_bytes:int, max_context:int,
                                 alignment:int) -> tuple[ByteTerm, ...]:
  """Conservative structural residency shared by every route for the selected workload.

  The normal exact-context path may later select a narrower KV representation,
  but fp16 is the largest representation it can dispatch.  Planning that
  representation here keeps the pre-run gate safe without a model or VRAM
  tier.  Allocations are rounded exactly as separate runtime buffers.
  """
  arch = str(kv["general.architecture"])
  blocks = int(kv[f"{arch}.block_count"])-int(kv.get(f"{arch}.nextn_predict_layers", 0))
  heads = int(kv[f"{arch}.attention.head_count"])
  kv_heads = int(kv.get(f"{arch}.attention.head_count_kv", heads))
  head_dim = int(kv.get(f"{arch}.attention.key_length_mla",
                    kv.get(f"{arch}.attention.key_length", int(kv[f"{arch}.embedding_length"])//heads)))
  rope_dim = int(kv.get(f"{arch}.rope.dimension_count", head_dim))
  if min(blocks, kv_heads, head_dim, rope_dim, max_context, alignment) <= 0:
    raise ValueError("selected runtime memory geometry must be positive")
  # cache_kv is one independently allocated [K|V,B,Hkv,N,Hd] fp16 buffer per block.
  kv_per_block = _align_up(2*kv_heads*max_context*head_dim*2, alignment)
  # precompute_freqs_cis retains one [N,rope_dim] default-float table on device.
  rope = _align_up(max_context*rope_dim*4, alignment)
  return (
    ByteTerm("selected GGUF residency", backing_bytes, "selected model stat + live device scan",
             "align_up(st_size of selected GGUF, scanned allocator granularity)", ByteLifetime.PERSISTENT),
    ByteTerm("exact-context KV upper bound", blocks*kv_per_block, "selected GGUF runtime geometry + selected workload",
             "block_count * align_up(2 * batch(1) * n_kv_heads * max_context * head_dim * sizeof(fp16), scanned allocator granularity)",
             ByteLifetime.PERSISTENT),
    ByteTerm("runtime RoPE table", rope, "selected GGUF runtime geometry + selected workload",
             "align_up(max_context * rope_dim * sizeof(default_float), scanned allocator granularity)",
             ByteLifetime.PERSISTENT),
  )


def _route_graph_materialization_bound(rows:Sequence[Mapping[str, Any]], alignment:int) -> int:
  """Safe graph-wide upper bound from the exact selected route inventory.

  Summing every route input and output materialization deliberately ignores
  reuse, so the compiler's arena planner can only improve on this bound.
  """
  total = 0
  for row in rows:
    shape = row.get("shape")
    if not isinstance(shape, Mapping): raise ValueError("route inventory row has no structural shape")
    m, n, k = (int(shape[name]) for name in ("m", "n", "k"))
    if min(m, n, k) <= 0: raise ValueError("route inventory dimensions must be positive")
    total += _align_up(m*k*2, alignment) + _align_up(m*n*2, alignment)
  return total


def _validate_transport(value: Any) -> Any:
  """Validate structural semantic owners using the authoritative vocabulary."""
  if isinstance(value, list): return [_validate_transport(item) for item in value]
  if isinstance(value, dict):
    # Every semantic record is checked against the typed vocabulary. Evidence
    # rows may carry additional physical-byte fields, which remain intact.
    if "semantic_class" in value:
      if "candidate_id" not in value: raise ValueError("semantic record requires candidate_id")
      semantic_class, candidate_id = value["semantic_class"], value["candidate_id"]
      try: MemorySemanticOwner(MemorySemanticClass(semantic_class), candidate_id)
      except (TypeError, ValueError) as exc: raise ValueError(f"invalid semantic owner transport: {exc}") from exc
    out = {key: _validate_transport(item) for key, item in value.items()}
    if "semantic_owner" in value:
      owner = value["semantic_owner"]
      # ``unknown`` is the planner's explicit unclassified sentinel. Preserve
      # it so semantic reconciliation can name and reject that buffer; it is
      # never upgraded into a valid owner by transport.
      if owner == "unknown": return out
      if not isinstance(owner, dict) or set(owner) != {"semantic_class", "candidate_id"}:
        raise ValueError("semantic_owner transport requires semantic_class and candidate_id")
      _validate_transport(owner)
    return out
  return value


def _sha256(path: pathlib.Path, chunk: int = 8 << 20) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    while data := handle.read(chunk): digest.update(data)
  return "sha256:" + digest.hexdigest()


def _tensor_payload_bytes(dims: Sequence[int], ggml_type: int) -> int | None:
  from tinygrad.llm.gguf import _GGML_NATIVE, _GGML_QUANT
  count = math.prod(int(x) for x in dims)
  if ggml_type in _GGML_NATIVE: return count * _GGML_NATIVE[ggml_type].itemsize
  block = _GGML_QUANT.get(ggml_type)
  return None if block is None or count % block[0] else count // block[0] * block[1]


def _revision() -> dict[str, Any]:
  try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
  except (OSError, subprocess.SubprocessError): commit = None
  return {"git_commit": commit, "python": sys.version.split()[0], "seam_schema": SCHEMA}


def _failed_artifacts(required: Sequence[str], blockers: Sequence[str], partial: Mapping[str, Any] | None = None) -> dict[str, Any]:
  """Produce adapter-shaped FAIL records, never fabricated passing records."""
  partial = {} if partial is None else dict(partial)
  error = "; ".join(blockers) if blockers else "worker did not provide required artifacts"
  execution = partial.get("execution", {"phases": [
    {"phase": "compile", "status": "failed", "evidence": {"blocker": error}},
    {"phase": "execution", "status": "failed", "evidence": {"dispatch_state": "incomplete", "health": {}}},
    {"phase": "correctness", "status": "failed", "evidence": {"blocker": error}},
  ]})
  observed_resource = partial.get("resource") if isinstance(partial.get("resource"), Mapping) else {}
  # A partial resource record (including valid compiled-code metadata) cannot stay PASS when the guarded run has an
  # allocation, cleanup, health, census, or protocol blocker. Preserve its fields for diagnosis while making the gate
  # unambiguously fail closed.
  resource = {**dict(observed_resource), "status": "FAIL", "complete": False, "blocker": error}
  return {"execution": execution, "resource": resource,
    "route_census": partial.get("route_census", {"status": "FAIL", "complete": False,
      "covered_invocations": [], "required_invocations": list(required), "blocker": error}),
    "end_to_end_timing": partial.get("end_to_end_timing", {"scope": "end_to_end", "metric": "tok_s", "samples": []})}


def _known_term_sum(terms: Sequence[ByteTerm]) -> int:
  values = [term.bytes for term in terms]
  if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
    raise ValueError("selected candidate has an unknown or invalid planned allocation term")
  return sum(values)


class TinygradWholeModelSeam:
  def __init__(self, *, prompt_tokens: int = 544, decode_tokens: int = 8, warmup_tokens: int = 2,
               max_context: int = 2048, timeout_seconds: float = 1800.0, python: str = sys.executable):
    if min(prompt_tokens, decode_tokens, max_context) <= 0 or warmup_tokens < 0 or timeout_seconds <= 0:
      raise ValueError("invalid whole-model run bounds")
    self.prompt_tokens, self.decode_tokens, self.warmup_tokens = prompt_tokens, decode_tokens, warmup_tokens
    self.max_context, self.timeout_seconds, self.python = max_context, timeout_seconds, python
    # Deliberately process-local: controller execution is baseline-first and a
    # candidate may only be authorized against the baseline measured by this
    # seam instance for the exact selected model and workload.
    self._baselines: dict[str, Mapping[str, Any]] = {}

  @staticmethod
  def _comparison_key(model: SelectedModelScan) -> str:
    payload = {"model": model.facts.get("content_hash"), "inventory": model.facts.get("inventory_identity"),
               "workload": dict(model.workload)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

  def _authorize_correctness(self, model: SelectedModelScan, candidate: Any, row: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Retain the real baseline and compare every candidate in the parent."""
    artifacts = dict(row.get("artifacts", {})); run = row.get("run", {})
    evidence = run.get("deterministic_output_evidence") if isinstance(run, Mapping) else None
    errors: list[str] = []
    if not isinstance(evidence, Mapping): errors.append("missing required artifact: deterministic full-output evidence")
    elif evidence.get("input_digest_before") != evidence.get("input_digest_after"):
      errors.append("guarded input changed during whole-model execution")
    elif not isinstance(evidence.get("outputs"), list) or not evidence["outputs"]:
      errors.append("missing required artifact: deterministic full-output samples")
    key = self._comparison_key(model)
    is_baseline = candidate.memory.strategy is Strategy.DIRECT_PACKED_FALLBACK
    baseline = self._baselines.get(key)
    if not errors and is_baseline:
      outputs = evidence["outputs"]
      if any(value != outputs[0] for value in outputs[1:]): errors.append("baseline full outputs are not deterministic")
      else: self._baselines[key] = dict(evidence); baseline = evidence
    if not errors and baseline is None: errors.append("baseline artifact unavailable: candidates require baseline-first execution")
    atol, rtol = 0.0, 0.0
    numerical = False
    if not errors:
      # generate(temperature=0) exposes the complete greedy output sequence,
      # not logits.  Integer output comparison therefore has exact tolerance.
      numerical = all(sample == baseline["outputs"][0] for sample in evidence["outputs"])
      if not numerical: errors.append("candidate greedy output differs from retained baseline")
    comparison = {"authority": "parent retained baseline-first full-output comparison", "baseline_key": key,
      "output_kind": "complete_greedy_token_sequence", "atol": atol, "rtol": rtol,
      "greedy_equal": numerical, "sample_count": len(evidence.get("outputs", ())) if isinstance(evidence, Mapping) else 0}
    phases = [dict(x) for x in artifacts.get("execution", {}).get("phases", ())]
    phases = [({"phase": "correctness", "status": "passed" if not errors else "failed", "evidence": {
      "finite_output": not errors, "full_output_compared": not errors, "numerical_passed": not errors,
      "inputs_unchanged": not errors, "comparison": comparison,
      "input_digest": evidence.get("input_digest_before") if isinstance(evidence, Mapping) else None,
      **({"blocker": "; ".join(errors)} if errors else {})}} if x.get("phase") == "correctness" else x) for x in phases]
    artifacts["execution"] = {**dict(artifacts.get("execution", {})), "phases": phases}
    return artifacts, errors

  def scan_selected_model(self, model_path: str, device: Any) -> SelectedModelScan:
    path = pathlib.Path(model_path).expanduser().resolve(strict=True)
    if not path.is_file(): raise ValueError("selected model is not a regular file")
    from tinygrad.llm.gguf import gguf_load_metadata
    from tinygrad.llm.gguf_memory_scan import selected_gguf_backing_bytes
    from tinygrad.llm.model import PREFILL_UBATCH, derive_selected_gguf_prefill_inventory
    kv, meta = gguf_load_metadata(path)
    inventory = derive_selected_gguf_prefill_inventory(kv, meta, PREFILL_UBATCH)
    tensor_bytes = []
    for name, dims, typ, _offset in meta["tensor_infos"]:
      size = _tensor_payload_bytes(dims, typ)
      if size is None: raise ValueError(f"unknown packed byte size for selected tensor {name!r} type {typ}")
      tensor_bytes.append(size)
    semantic_kv = {k: v for k, v in kv.items() if not k.startswith("tokenizer.") and k not in ("general.name", "general.basename")}
    alignment = getattr(getattr(device, "capabilities", None), "global_allocation_granularity", None)
    backing_bytes = selected_gguf_backing_bytes(path, alignment)
    if backing_bytes is None: raise ValueError("selected GGUF backing is unknown without scanned allocator granularity")
    facts = {"content_hash": _sha256(path), "file_bytes": path.stat().st_size, "backing_bytes": backing_bytes,
      "tensor_payload_bytes": sum(tensor_bytes), "gguf": semantic_kv,
      "inventory_identity": inventory["inventory_identity"]}
    base = _selected_runtime_base_terms(kv, backing_bytes=backing_bytes, max_context=self.max_context, alignment=alignment)
    workload = {"prompt_tokens": self.prompt_tokens, "decode_tokens": self.decode_tokens,
                "warmup_tokens": self.warmup_tokens, "max_context": self.max_context,
                "prefill_ubatch": PREFILL_UBATCH,
                "timing_scope": "model.generate end_to_end"}
    return SelectedModelScan(facts, inventory, base, workload, _revision())

  def enumerate_candidate_specs(self, model: SelectedModelScan, device: Any) -> Sequence[CandidateSpec]:
    ids = inventory_invocation_ids(model.inventory)
    rows = model.inventory.get("rows", ())
    controlled_rows = tuple(row for row in rows if row.get("candidate_controlled", True) is True)
    fixed_routes = {row["invocation_id"]: row["fixed_route_id"] for row in rows
                    if row.get("candidate_controlled") is False}
    overlay = sum(int(row["shape"]["n"]) * int(row["shape"]["k"]) * 2 for row in controlled_rows)
    physical_ms = {int(row["shape"]["m"]) for row in controlled_rows}
    if len(physical_ms) != 1: return ()
    physical_m = physical_ms.pop()
    graph_bound = ByteTerm("selected prefill graph materialization upper bound",
      _route_graph_materialization_bound(rows, device.capabilities.global_allocation_granularity),
      "selected route inventory + live allocator granularity",
      "sum(align_up(M*K*sizeof(fp16)) + align_up(M*N*sizeof(fp16))) over exact selected route inventory",
      ByteLifetime.PREFILL_PEAK)
    logical_tokens = int(model.workload.get("prompt_tokens", self.prompt_tokens))
    remainder = logical_tokens % physical_m
    mappings = () if not remainder else (RemainderMapping(remainder, physical_m, physical_m+remainder),)
    # Both current bindings execute the selected inventory at its concrete M.
    # This is candidate-local capability evidence; it is not a universal 512 rule.
    activation = max((physical_m * (int(row["shape"]["k"]) + int(row["shape"]["n"])) * 2
                      for row in controlled_rows), default=0)
    kernel_facts = {"full_m_values": (physical_m,), "tail_m_values": (),
      "remainder_mappings": mappings, "correctness_m_values": (physical_m,),
      "invocation_bytes": (InvocationBytes(physical_m, activation, 0),)}
    # These are structural runtime bindings.  They contain no path, profile,
    # display name, parameter tier, or inferred size class.
    return (
      CandidateSpec("direct-packed-baseline", Strategy.DIRECT_PACKED_FALLBACK, ids, (graph_bound,),
        target_requirements={"backend": "AMD"},
        policy={"binding": {"strategy": Strategy.DIRECT_PACKED_FALLBACK.value},
                "routes": {row["invocation_id"]: fixed_routes.get(row["invocation_id"], "direct-packed-baseline")
                           for row in rows}}, **kernel_facts),
      CandidateSpec("full-resident-overlay", Strategy.FULL_RESIDENT_OVERLAY, ids,
        (ByteTerm("fp16 route-weight overlay", overlay, "selected GGUF route inventory",
                  "sum(rows * cols * sizeof(fp16))", ByteLifetime.CANDIDATE_WORKSPACE), graph_bound),
        target_requirements={"backend": "AMD"},
        policy={"binding": {"strategy": Strategy.FULL_RESIDENT_OVERLAY.value},
                "routes": {row["invocation_id"]: fixed_routes.get(row["invocation_id"], "full-resident-overlay")
                           for row in rows}}, **kernel_facts),
    )

  def collect_whole_model_artifacts(self, model_path: str, model: SelectedModelScan, candidate: Any,
                                    *, samples: int) -> Mapping[str, Any]:
    required = tuple(candidate.memory.required_invocations)
    workload_choice = dict(candidate.policy.get("workload_choice", {}))
    request = {"schema": WORKER_SCHEMA, "model_path": str(pathlib.Path(model_path).expanduser().resolve()),
      "candidate_id": candidate.candidate_id, "whole_policy_identity": candidate.whole_policy_identity,
      "strategy": candidate.memory.strategy.value,
      "routes": dict(candidate.policy.get("routes", {})), "required_invocations": list(required),
      "inventory": dict(model.inventory), "workload": dict(model.workload), "samples": samples,
      "workload_choice": workload_choice, "lifecycle_phase": "evidence",
      "planned_peak_bytes": _known_term_sum(tuple(model.base_terms)+tuple(candidate.memory.memory_terms))}
    cmd = [self.python, "-m", "extra.qk.memory_adaptive_tinygrad_seam", "--worker"]
    def launch(payload: Mapping[str, Any], profile: str):
      env = dict(os.environ)
      env.update({"PYTHONPATH": str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""), "PROFILE": profile, "DEBUG": "0"})
      return subprocess.run(cmd, cwd=ROOT, env=env, input=json.dumps(payload), text=True, capture_output=True,
                            timeout=self.timeout_seconds, check=False)
    try:
      proc = launch(request, "0")
      timing_proc = launch({**request, "lifecycle_phase": "timing"}, "0")
    except subprocess.TimeoutExpired:
      blocker = f"isolated whole-model worker timed out after {self.timeout_seconds:g}s"
      return {"actual_whole_model_run": False, "blockers": [blocker], "artifacts": _failed_artifacts(required, [blocker])}
    except OSError as exc:
      blocker = f"isolated whole-model worker could not start: {type(exc).__name__}: {exc}"
      return {"actual_whole_model_run": False, "blockers": [blocker], "artifacts": _failed_artifacts(required, [blocker])}
    try:
      row = _validate_transport(json.loads(proc.stdout.strip().splitlines()[-1]))
      timing_row = _validate_transport(json.loads(timing_proc.stdout.strip().splitlines()[-1]))
    except (IndexError, json.JSONDecodeError) as exc:
      blocker = f"isolated whole-model worker returned invalid JSON: {exc}; evidence_exit={proc.returncode}; timing_exit={timing_proc.returncode}"
      return {"actual_whole_model_run": False, "blockers": [blocker], "artifacts": _failed_artifacts(required, [blocker])}
    # Both processes start with PROFILE=0. The evidence worker enables PROFILE
    # only around post-load capture/dispatch scopes; selection timing and
    # output comparison come only from a separate, always-PROFILE=0 process.
    timing_artifacts = timing_row.get("artifacts")
    if isinstance(timing_artifacts, Mapping) and isinstance(timing_artifacts.get("end_to_end_timing"), Mapping):
      row["artifacts"]["end_to_end_timing"] = timing_artifacts["end_to_end_timing"]
    if isinstance(timing_row.get("run"), Mapping): row["run"] = timing_row["run"]
    row["actual_whole_model_run"] = row.get("actual_whole_model_run") is True and timing_row.get("actual_whole_model_run") is True
    raw_partial = row.get("artifacts") if isinstance(row.get("artifacts"), Mapping) else {}
    expected_identity = candidate.whole_policy_identity
    identity_blockers = []
    for label, output in (("evidence", row), ("PROFILE=0 timing", timing_row)):
      if output.get("whole_policy_identity") != expected_identity:
        identity_blockers.append(f"{label} worker whole_policy_identity missing or mismatched")
    if raw_partial.get("whole_policy_identity") != expected_identity:
      identity_blockers.append("evidence artifact envelope whole_policy_identity missing or mismatched")
    census_identity = raw_partial.get("route_census", {})
    if not isinstance(census_identity, Mapping) or census_identity.get("whole_policy_identity") != expected_identity:
      identity_blockers.append("route census whole_policy_identity missing or mismatched")
    if identity_blockers:
      artifacts = _failed_artifacts(required, identity_blockers)
      return {"actual_whole_model_run": False, "blockers": identity_blockers, "worker": row.get("run"),
              "physical_memory_ledger": row.get("physical_memory_ledger"),
              "schedule_manifests": row.get("schedule_manifests"), "schedule_evidence": row.get("schedule_evidence"),
              "measured_allocation": row.get("measured_allocation"),
              "memory_fact_evidence": row.get("memory_fact_evidence"), "artifacts": artifacts}
    measured_allocation = row.get("measured_allocation")
    memory_fact_evidence = row.get("memory_fact_evidence")
    partial, correctness_blockers = self._authorize_correctness(model, candidate, row)
    blockers = [x for x in row.get("blockers", ()) if "parent baseline comparison" not in x]
    blockers.extend(correctness_blockers)
    if not isinstance(measured_allocation, Mapping): blockers.append("missing required artifact: measured allocation evidence")
    elif measured_allocation.get("complete") is not True:
      blockers.extend(str(x) for x in measured_allocation.get("blockers", ()))
    if candidate.memory.strategy is not Strategy.DIRECT_PACKED_FALLBACK:
      if memory_fact_evidence is None and isinstance(row.get("memory_structure"), Mapping) and isinstance(measured_allocation, Mapping):
        try: memory_fact_evidence = derive_memory_facts(candidate.candidate_id, row["memory_structure"], measured_allocation)
        except ValueError as exc: blockers.append(f"measured policy memory facts rejected: {exc}")
      memory_fact_evidence = validate_memory_facts(memory_fact_evidence, candidate_id=candidate.candidate_id)
      if memory_fact_evidence is None: blockers.append("missing required artifact: complete measured policy memory facts")
    census = raw_partial.get("route_census")
    expected_routes = dict(request["routes"])
    if not isinstance(census, Mapping) or census.get("status") != "PASS" or census.get("complete") is not True:
      blockers.append("missing required artifact: genuine complete runtime route census")
    else:
      census_rows = census.get("rows")
      if not isinstance(census_rows, list): blockers.append("runtime route census rows are unavailable")
      else:
        observed = {x.get("invocation_id"): x for x in census_rows if isinstance(x, Mapping)}
        if len(observed) != len(census_rows) or set(observed) != set(required):
          blockers.append("runtime route census does not exactly equal required selected inventory")
        elif any(row.get("route_id") != expected_routes.get(key) or row.get("call_count") != row.get("expected_call_count")
                 for key, row in observed.items()):
          blockers.append("runtime route census route IDs or expected counts differ from selected policy")
    if proc.returncode != 0: blockers.append(f"isolated evidence worker exited {proc.returncode}")
    if timing_proc.returncode != 0: blockers.append(f"isolated PROFILE=0 timing worker exited {timing_proc.returncode}")
    required_keys = ("execution", "resource", "route_census", "end_to_end_timing")
    blockers.extend(f"missing required artifact: {key}" for key in required_keys if key not in raw_partial)
    artifacts = _failed_artifacts(required, blockers, partial) if blockers else dict(partial)
    actual = row.get("actual_whole_model_run") is True
    return {"actual_whole_model_run": actual, "blockers": list(dict.fromkeys(blockers)), "worker": row.get("run"),
            "physical_memory_ledger": row.get("physical_memory_ledger"),
            "schedule_manifests": row.get("schedule_manifests"), "schedule_evidence": row.get("schedule_evidence"),
            "measured_allocation": measured_allocation, "memory_fact_evidence": memory_fact_evidence, "artifacts": artifacts}


def _health() -> dict[str, Any]:
  cmd = ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp"]
  try: proc = subprocess.run(cmd, text=True, capture_output=True, timeout=15, check=False)
  except (OSError, subprocess.TimeoutExpired) as exc: return {"healthy": False, "error": f"{type(exc).__name__}: {exc}"}
  output = (proc.stdout + "\n" + proc.stderr).strip()
  bad = any(x in output.lower() for x in ("gpu reset", "unresponsive", "xgmi error"))
  return {"healthy": proc.returncode == 0 and not bad, "returncode": proc.returncode, "output": output[-4000:]}


def _compiled_resource_artifact(profile_start: int, selected_device: str) -> tuple[dict[str, Any], list[str]]:
  """Read the actual code objects retained by tinygrad's PROFILE surface."""
  from tinygrad.device import Compiled, ProfileProgramEvent
  from extra.qk.amdgpu_metadata import parse_amdgpu_metadata
  programs, failures = [], []
  seen: set[str] = set()
  events = Compiled.profile_events[profile_start:]
  for event in events:
    if not isinstance(event, ProfileProgramEvent) or event.device != selected_device or not event.lib: continue
    identity = "sha256:" + hashlib.sha256(event.lib).hexdigest()
    if identity in seen: continue
    seen.add(identity)
    try: metadata = parse_amdgpu_metadata(event.lib)
    except Exception as exc:
      failures.append(f"{event.name}: {type(exc).__name__}: {exc}"); continue
    programs.append({"program": event.name, "device": event.device, "binary_sha256": identity,
                     "binary_bytes": len(event.lib), "resources": metadata})
  if not programs: failures.append("no selected-device compiled-program code objects were exposed by tinygrad PROFILE")
  violations = []
  for program in programs:
    resources = program["resources"]
    for field in _RESOURCE_ZERO_FIELDS:
      if resources.get(field) != 0: violations.append(f"{program['program']} {field}={resources.get(field)!r}")
    if resources.get("dynamic_stack") is not False: violations.append(f"{program['program']} uses dynamic stack")
  failures.extend("resource violation: " + value for value in violations)
  maxima = {field: max((int(x["resources"][field]) for x in programs), default=0)
            for field in ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "vgpr_spills", "sgpr_spills")}
  artifact = {"status": "PASS" if not failures else "FAIL", "authority": "tinygrad PROFILE code objects + AMDGPU metadata notes",
    "aggregation": "all unique selected-device compiled programs; maxima reported; any parse failure/scratch/spill/dynamic-stack fails",
    "program_count": len(programs), "programs": programs, "resources": maxima,
    "violations": violations, "complete": not failures}
  if failures: artifact["blocker"] = "; ".join(failures)
  return artifact, failures


def _schedule_evidence_json(evidence: Any) -> dict[str, Any]:
  return {"complete": evidence.complete, "blockers": list(evidence.blockers),
    "peak_physical_bytes": evidence.peak_physical_bytes,
    "peak_by_semantic_class": [{"semantic_class": x.semantic_class, "candidate_id": x.candidate_id,
                                 "physical_bytes": x.physical_bytes} for x in evidence.peak_by_semantic_class],
    "indices": [{"index": x.index, "physical_bytes": x.physical_bytes,
                 "by_semantic_class": [{"semantic_class": y.semantic_class, "candidate_id": y.candidate_id,
                                         "physical_bytes": y.physical_bytes} for y in x.by_semantic_class]}
                for x in evidence.indices]}


def _manifest_arena_owner_id(manifest_index:int, arena:Any) -> str|None:
  if not arena.identity.startswith("arena:") or arena.backing_uop is None: return None
  return f"manifest:{manifest_index}:{arena.identity}:backing:{arena.backing_uop.key.hex()}"


def _manifest_json(manifest: Any, manifest_index:int) -> dict[str, Any]:
  return {"peak_physical_bytes": manifest.peak_physical_bytes,
    "arenas": [{"identity": x.identity, "device": x.device, "lane": x.lane, "size": x.size,
                "shared_rewritten_backing": x.identity.startswith("arena:") and x.backing_uop is not None,
                "physical_owner_id": _manifest_arena_owner_id(manifest_index, x)}
               for x in manifest.arenas],
    "buffers": [{"identity": x.identity, "device": x.device, "arena_identity": x.arena_identity,
                 "byte_range": list(x.byte_range), "first_index": x.first_index, "last_index": x.last_index,
                 "semantic_owner": _transport_encode(x.semantic_owner)} for x in manifest.buffers]}


def _drain_manifest_rows(manifests:list[Any]) -> list[dict[str, Any]]:
  """Serialize manifest evidence and release backing UOps before ledger cleanup."""
  try: return [_manifest_json(manifest, n) for n, manifest in enumerate(manifests)]
  finally: manifests.clear()

def _bind_manifest_physical_owners(ledger, manifest, manifest_index:int) -> None:
  """Bind exact rewritten/dedicated backing UOps before their allocations dispatch."""
  from tinygrad.llm.memory_semantics import MemorySemanticClass, MemorySemanticOwner
  from tinygrad.llm.physical_memory_ledger import AllocationOwner, allocation_owner_from_semantic, bind_allocation_owner
  rows_by_arena: dict[str, list[Any]] = {}
  for row in manifest.buffers: rows_by_arena.setdefault(row.arena_identity, []).append(row)
  for arena in manifest.arenas:
    if arena.backing_uop is None: continue
    if arena.identity.startswith("arena:"):
      owner = AllocationOwner("schedule_arena", "schedule", semantic_owner_id=_manifest_arena_owner_id(manifest_index, arena))
    else:
      owners = {row.semantic_owner for row in rows_by_arena.get(arena.identity, ())
                if isinstance(row.semantic_owner, MemorySemanticOwner)}
      if len(owners) != 1: continue
      semantic_owner = owners.pop()
      # A dedicated schedule allocation can be retained and reused by cached
      # programs across prefill and decode. Its physical lease is one schedule
      # transient even while per-manifest logical roles change over time; the
      # manifest/evidence rows retain those exact phase-specific classes.
      if semantic_owner.semantic_class in {
        MemorySemanticClass.PREFILL_ACTIVATION, MemorySemanticClass.PREFILL_OUTPUT, MemorySemanticClass.PREFILL_SCRATCH,
        MemorySemanticClass.RUNTIME_ACTIVATION, MemorySemanticClass.RUNTIME_OUTPUT, MemorySemanticClass.RUNTIME_SCRATCH,
      }:
        owner = AllocationOwner("schedule_transient", "schedule")
      else: owner = allocation_owner_from_semantic(semantic_owner)
    # The same normalized structural UOp can back separate prefill and decode
    # allocations in different invocations. Bind the exact Buffer base selected
    # by this manifest, not a process-wide UOp key, so independently allocated
    # lifetimes may carry truthful phase-specific owners.
    bind_allocation_owner(arena.backing_uop.buffer, owner)


def _reconcile_memory_authorities(physical: Mapping[str, Any], manifests: Sequence[Mapping[str, Any]],
                                  schedule_evidence: Sequence[Mapping[str, Any]], observer: Mapping[str, Any],
                                  *, selected_device: str, granularity: int | None, free_vram_bytes: int | None,
                                  planned_peak_bytes: int) -> dict[str, Any]:
  """Join physical lifetimes and schedule semantics without phase/size-based classification."""
  blockers = list(str(x) for x in physical.get("blockers", ()))
  if physical.get("complete") is not True: blockers.append("physical allocation ledger is incomplete")
  if observer.get("schema") != CHECKPOINT_OBSERVER_SCHEMA:
    blockers.append("external checkpoint evidence has an unsupported schema")
  elif observer.get("complete") is not True:
    blockers.extend(str(x) for x in observer.get("blockers", ()))
  if not isinstance(granularity, int) or isinstance(granularity, bool) or granularity <= 0:
    blockers.append("selected device allocator granularity is unavailable")
  if not isinstance(free_vram_bytes, int) or isinstance(free_vram_bytes, bool) or free_vram_bytes < 0:
    blockers.append("selected device free memory is unavailable")
  if not manifests: blockers.append("no schedule memory manifests were collected")
  if len(manifests) != len(schedule_evidence): blockers.append("schedule manifest/evidence count mismatch")
  for n, evidence in enumerate(schedule_evidence):
    if evidence.get("complete") is not True:
      blockers.extend(f"schedule manifest {n}: {x}" for x in evidence.get("blockers", ()))

  expected: dict[str, set[int]] = {}
  for n, manifest in enumerate(manifests):
    for arena in manifest.get("arenas", ()):
      if arena.get("identity", "").startswith("arena:"):
        if arena.get("shared_rewritten_backing") is not True:
          blockers.append(f"schedule manifest {n} shared arena {arena.get('identity')!r} has no rewritten backing UOp")
        if arena.get("device") != selected_device:
          blockers.append(f"schedule manifest {n} arena device differs from scanned selected device")
        size = arena.get("size")
        owner_id = arena.get("physical_owner_id")
        if not isinstance(owner_id, str) or not owner_id:
          blockers.append(f"schedule manifest {n} shared arena {arena.get('identity')!r} has no unique physical owner ID")
        if isinstance(size, int) and not isinstance(size, bool) and isinstance(granularity, int) and granularity > 0:
          expected.setdefault(str(owner_id), set()).add(((size+granularity-1)//granularity)*granularity)

  observed: dict[str, set[int]] = {}
  requested_events: list[tuple[int, int]] = []
  for row in physical.get("lifetimes", ()):
    owner = row.get("owner") if isinstance(row, Mapping) else None
    if not isinstance(owner, Mapping): continue
    if owner.get("kind") == "schedule_arena":
      aid = owner.get("semantic_owner_id")
      if not isinstance(aid, str) or aid not in expected:
        blockers.append(f"physical schedule arena has no matching manifest owner: {aid!r}")
      elif isinstance(row.get("physical_nbytes"), int): observed.setdefault(aid, set()).add(row["physical_nbytes"])
    if row.get("device") == selected_device and isinstance(row.get("requested_nbytes"), int) and isinstance(row.get("free_sequence"), int):
      requested_events.extend(((row["alloc_sequence"], row["requested_nbytes"]), (row["free_sequence"], -row["requested_nbytes"])))
  for aid, sizes in expected.items():
    if observed.get(aid) != sizes: blockers.append(f"physical/manifest mismatch for shared arena {aid!r}: expected {sorted(sizes)}, observed {sorted(observed.get(aid, set()))}")

  physical_peak = physical.get("peak_physical_bytes")
  if not isinstance(physical_peak, int): blockers.append("physical ledger peak is unavailable")
  elif physical_peak > planned_peak_bytes: blockers.append(f"physical peak {physical_peak} exceeds planned peak {planned_peak_bytes}")
  current = requested_peak = 0
  for _, delta in sorted(requested_events, key=lambda x: (x[0], x[1])):
    current += delta; requested_peak = max(requested_peak, current)
  external_growth = observer.get("peak_growth_bytes")
  if not isinstance(external_growth, int): blockers.append("external checkpoint peak is unavailable")
  elif isinstance(physical_peak, int) and not requested_peak <= external_growth <= physical_peak:
    blockers.append(f"external peak inconsistency: growth {external_growth} is outside requested/physical peak bounds {requested_peak}..{physical_peak}")
  if observer.get("post_run_retained_bytes") != 0: blockers.append("external checkpoint cleanup is incomplete")
  blockers = list(dict.fromkeys(blockers))
  return {"schema": "tinygrad.reconciled_measured_allocation.v1", "complete": not blockers, "blockers": blockers,
    "peak_bytes": physical_peak, "planned_peak_bytes": planned_peak_bytes,
    "allocations": list(physical.get("lifetimes", ())), "physical_ledger": dict(physical),
    "schedule_manifests": list(manifests), "schedule_evidence": list(schedule_evidence),
    "external_peak_corroboration": dict(observer),
    "authority": "physical allocation ledger + explicit schedule semantic ownership; observer checkpoints corroborate peaks only"}


def _worker(request: Mapping[str, Any]) -> dict[str, Any]:
  """Execute the real generate path; report unavailable authorities precisely."""
  blockers: list[str] = []
  before = _health()
  if not before["healthy"]:
    blocker = "GPU health preflight failed"
    artifacts = _failed_artifacts(request.get("required_invocations", ()), [blocker])
    artifacts["whole_policy_identity"] = request["whole_policy_identity"]
    artifacts["route_census"]["whole_policy_identity"] = request["whole_policy_identity"]
    return {"schema": WORKER_SCHEMA, "whole_policy_identity": request["whole_policy_identity"],
            "actual_whole_model_run": False, "blockers": [blocker],
            "artifacts": artifacts}
  required, routes = tuple(request["required_invocations"]), dict(request["routes"])
  policy = {"strategy": request["strategy"], "candidate_id": request["candidate_id"],
            "whole_policy_identity": request["whole_policy_identity"], "routes": routes,
            "provenance": SCHEMA, "measured": True}
  def collector(runtime_request):
    if runtime_request.get("inventory") != request["inventory"]: raise ValueError("runtime inventory differs from parent scan")
    return {"validated_request": runtime_request, "decision": "SELECTED", "validation": "measurement_trial", "policy": policy}
  started = time.time(); outputs: list[list[int]] = []; speeds: list[float] = []; route_census = None
  input_before = input_after = None
  failure_traceback = None
  profile_start = 0; observer = None; allocation = None
  ledger = device_facts = None; manifests: list[Any] = []; manifest_rows: list[dict[str, Any]] = []
  schedule_rows: list[dict[str, Any]] = []
  clear_model_caches = None
  stack = contextlib.ExitStack()
  model = _kv = census_gen = gen = None
  try:
    from extra.qk.memory_adaptive_runtime_collector import install_model_adapters
    install_model_adapters()
    from tinygrad import Context, Device, Tensor, TinyJit
    from tinygrad.device import Compiled
    from tinygrad.llm.device_facts import scan_device_facts
    from tinygrad.llm.model import Transformer, _memory_adaptive_measurement_authority, precompute_freqs_cis
    clear_model_caches = precompute_freqs_cis.cache_clear
    from extra.qk.physical_memory_ledger import PhysicalMemoryLedger, allocation_phase
    from tinygrad.llm.physical_memory_ledger import AllocationOwner
    from extra.qk.prefill_route_census import collect_prefill_route_census
    from extra.qk.schedule_memory_evidence import schedule_memory_evidence
    from tinygrad.schedule.memory import collect_memory_plan_manifests
    from extra.qk.memory_adaptive_allocation_observer import AllocationObserver
    profile_start = len(Compiled.profile_events)
    workload = request["workload"]
    device_facts = scan_device_facts()
    selected = device_facts.selected_device
    granularity = device_facts.capabilities.global_allocation_granularity
    ledger = PhysicalMemoryLedger(devices=(selected,))
    def on_manifest(manifest):
      # The callback is synchronous and receives the exact post-rewrite arena UOps. Bind only shared arena bases;
      # logical buffers retain their semantic tags for schedule evidence and the shared physical arena gets no class.
      manifest_index = len(schedule_rows)
      ledger.record_manifest()
      evidence = schedule_memory_evidence(manifest)
      schedule_rows.append(_schedule_evidence_json(evidence))
      _bind_manifest_physical_owners(ledger, manifest, manifest_index)
    stack.enter_context(ledger.active())
    manifests = stack.enter_context(collect_memory_plan_manifests(on_manifest=on_manifest))
    observer = AllocationObserver((selected,), planned_peak_bytes=request["planned_peak_bytes"]).start()
    measurement_workload = {"prefill_ubatch": int(workload["prefill_ubatch"])}
    with _memory_adaptive_measurement_authority(device_facts=device_facts, inventory=request["inventory"],
                                                workload=measurement_workload, collector=collector):
      with allocation_phase("model_load"):
        model, _kv = Transformer.from_gguf(request["model_path"], int(workload["max_context"]))
    observer.post_load()
    prompt = [((i * 17) % 1000) + 1 for i in range(int(workload["prompt_tokens"]))]
    input_before = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
    # Route hooks execute at graph capture.  Capture one genuine isolated prefill under the context-local census;
    # decode from that generate is explicitly outside the model's prefill-forward scope.
    model.prefill_jit, model.prefill_v2_jit, model.prefill_v2_jits = TinyJit(model.forward), TinyJit(model.forward), {}
    expected_calls = int(request.get("workload_choice", {}).get("total_call_count", 1))
    with Context(PROFILE=1), allocation_phase("prefill_capture_dispatch"):
      with collect_prefill_route_census(required, {key: expected_calls for key in required}) as census:
        census_gen = model.generate(list(prompt)); next(census_gen)
    route_census = census.artifact()
    route_census["whole_policy_identity"] = request["whole_policy_identity"]
    if not route_census["complete"]: blockers.append("runtime route census incomplete: " + route_census.get("blocker", "unknown"))
    # Exercise at least the full measured decode span before collecting samples. This is workload-derived (not a
    # model/GPU tier) and prevents a later token position from turning the first measured sample into a compile trial.
    warmup = max(int(workload["warmup_tokens"]), int(workload["decode_tokens"]))
    if warmup:
      with Context(PROFILE=1), allocation_phase("warmup_dispatch"): list(zip(range(warmup), model.generate(list(prompt))))
    for _ in range(int(request["samples"])):
      with Context(PROFILE=1), allocation_phase("measured_dispatch"):
        gen = model.generate(list(prompt)); Device[selected].synchronize(); t0 = time.perf_counter()
        out = [int(token) for token in list(zip(range(int(workload["decode_tokens"])), gen)) for token in [token[1]]]
        Device[selected].synchronize(); elapsed = time.perf_counter() - t0
      outputs.append(out); speeds.append(len(out) / elapsed)
    input_after = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
  except Exception as exc:
    blockers.append(f"whole-model generate failed: {type(exc).__name__}: {exc}")
    failure_traceback = traceback.format_exc()
  finally:
    # Cleanup is inside both exact authorities, after every live model/generator reference has been released.
    for active_gen in (census_gen, gen):
      if active_gen is not None:
        try: active_gen.close()
        except Exception as exc: blockers.append(f"generator cleanup failed: {type(exc).__name__}: {exc}")
    if model is not None:
      # Captured JIT return graphs can retain their first invocation's tiny
      # request-input buffers. Reset before dropping the cyclic model/JIT graph
      # so every physical allocation receives a free event inside the ledger.
      jits = [getattr(model, name, None) for name in ("prefill_jit", "rollout_jit", "rollout_jit_flash",
                                                      "rollout_jit_ring", "rollout_jit_ring_full", "prefill_v2_jit")]
      jits.extend(getattr(model, "prefill_v2_jits", {}).values())
      for jit in jits:
        if hasattr(jit, "reset"):
          try: jit.reset()
          except Exception as exc: blockers.append(f"JIT cleanup failed: {type(exc).__name__}: {exc}")
      getattr(model, "prefill_v2_jits", {}).clear()
      jit = None
      jits.clear()
    active_gen = None
    try: manifest_rows = _drain_manifest_rows(manifests)
    except Exception as exc: blockers.append(f"schedule manifest serialization failed: {type(exc).__name__}: {exc}")
    if ledger is not None:
      # Polling can miss very short-lived state (the RNG counter is eight
      # bytes). Record the fully retained post-dispatch state synchronously
      # before any JIT/model/RNG lifetime is released.
      if observer is not None: observer.checkpoint("pre_cleanup")
      with allocation_phase("cleanup"):
        census_gen = gen = model = _kv = None
        if clear_model_caches is not None: clear_model_caches()
        # Sampling owns process-persistent device seed/counter tensors. This is
        # an isolated worker, so reset the process RNG registry at the run
        # boundary to end those physical lifetimes inside the ledger.
        Tensor.manual_seed(Tensor._seed)
        gc.collect()
    else:
      census_gen = gen = model = _kv = None
      if clear_model_caches is not None: clear_model_caches()
      gc.collect()
    if observer is not None:
      observer_row = observer.stop()
    else: observer_row = {}
    try: stack.close()
    except Exception as exc: blockers.append(f"memory authority cleanup failed: {type(exc).__name__}: {exc}")
    if ledger is not None and device_facts is not None:
      physical = ledger.export_evidence(scanned_granularities={device_facts.selected_device:
                                 device_facts.capabilities.global_allocation_granularity}).to_json()
      allocation = _reconcile_memory_authorities(physical, manifest_rows, schedule_rows, observer_row,
        selected_device=device_facts.selected_device,
        granularity=device_facts.capabilities.global_allocation_granularity,
        free_vram_bytes=device_facts.free_vram_bytes, planned_peak_bytes=request["planned_peak_bytes"])
      blockers.extend(allocation["blockers"])
    else:
      allocation = {"schema": "tinygrad.reconciled_measured_allocation.v1", "complete": False,
                    "blockers": ["live internal device scan or physical ledger unavailable"]}
      blockers.extend(allocation["blockers"])
  try: resource, resource_failures = _compiled_resource_artifact(profile_start, device_facts.selected_device)
  except Exception as exc:
    resource_failures = [f"compiled-program resource capture failed: {type(exc).__name__}: {exc}"]
    resource = {"status": "FAIL", "blocker": resource_failures[0]}
  blockers.extend("missing required artifact: final compiled-program resource metadata: " + x for x in resource_failures)
  after = _health()
  if not after["healthy"]: blockers.append("GPU health postflight failed")
  actual = bool(outputs)
  if actual: blockers.append("awaiting parent baseline comparison")
  execution = {"phases": [
    {"phase": "compile", "status": "passed" if resource.get("status") == "PASS" else "failed",
     "evidence": {"program_count": resource.get("program_count", 0), "resource_authority": resource.get("authority"),
                  **({"blocker": resource.get("blocker")} if resource.get("status") != "PASS" else {})}},
    {"phase": "execution", "status": "passed" if actual and after["healthy"] else "failed",
     "evidence": {"dispatch_state": "completed" if actual else "failed", "health": {
       "preflight": before["healthy"], "postflight": after["healthy"], "device_fault": actual and not after["healthy"]}}},
    {"phase": "correctness", "status": "failed", "evidence": {"finite_output": actual,
      "full_output_compared": False, "numerical_passed": False, "inputs_unchanged": input_before == input_after,
      "blocker": "awaiting parent baseline comparison"}},
  ]}
  artifacts = {"whole_policy_identity": request["whole_policy_identity"], "execution": execution,
    "resource": {**resource, "measured_allocation": allocation},
    "route_census": route_census if route_census is not None else {"status": "FAIL", "complete": False,
      "whole_policy_identity": request["whole_policy_identity"],
      "covered_invocations": [], "required_invocations": list(required), "blocker": "runtime census forward failed"},
    "end_to_end_timing": {"scope": "end_to_end", "metric": "tok_s", "samples": speeds,
      "authority": "isolated tinygrad Transformer.from_gguf + model.generate"}}
  execution["phases"][1]["evidence"]["measured_allocation"] = allocation
  return {"schema": WORKER_SCHEMA, "whole_policy_identity": request["whole_policy_identity"],
          "actual_whole_model_run": actual, "blockers": list(dict.fromkeys(blockers)),
          "physical_memory_ledger": allocation.get("physical_ledger"),
          "schedule_manifests": allocation.get("schedule_manifests"),
          "schedule_evidence": allocation.get("schedule_evidence"),
          "measured_allocation": allocation, "artifacts": artifacts,
          "run": {"candidate_id": request["candidate_id"], "whole_policy_identity": request["whole_policy_identity"],
                  "strategy": request["strategy"],
                  "workload_choice": dict(request.get("workload_choice", {})),
                  "output_token_digests": [hashlib.sha256(json.dumps(x).encode()).hexdigest() for x in outputs],
                  "deterministic_output_evidence": {"schema": "tinygrad.deterministic_full_output.v1",
                    "output_kind": "complete_greedy_token_sequence", "outputs": outputs,
                    "input_digest_before": input_before, "input_digest_after": input_after},
                  "failure_traceback": failure_traceback,
                  "elapsed_s": time.time()-started, "health_pre": before, "health_post": after}}


def _timing_worker(request: Mapping[str, Any]) -> dict[str, Any]:
  """Run selection timing without profiling, ledgers, manifests, or evidence hooks."""
  if os.environ.get("PROFILE", "0") != "0": raise RuntimeError("timing worker must start with PROFILE=0")
  before = _health()
  if not before["healthy"]: return {"schema": WORKER_SCHEMA, "whole_policy_identity": request["whole_policy_identity"],
    "actual_whole_model_run": False,
    "blockers": ["GPU health preflight failed"], "artifacts": {
      "whole_policy_identity": request["whole_policy_identity"], "end_to_end_timing": {"samples": []}}}
  policy = {"strategy": request["strategy"], "candidate_id": request["candidate_id"],
            "whole_policy_identity": request["whole_policy_identity"],
            "routes": dict(request["routes"]), "provenance": SCHEMA, "measured": True}
  def collector(runtime_request):
    if runtime_request.get("inventory") != request["inventory"]: raise ValueError("runtime inventory differs from parent scan")
    return {"validated_request": runtime_request, "decision": "SELECTED", "validation": "measurement_trial", "policy": policy}
  outputs: list[list[int]] = []; speeds: list[float] = []
  input_before = input_after = None; model = gen = None
  clear_model_caches = tensor_type = None
  try:
    from extra.qk.memory_adaptive_runtime_collector import install_model_adapters
    install_model_adapters()
    from tinygrad import Device, Tensor, TinyJit
    from tinygrad.llm.device_facts import scan_device_facts
    from tinygrad.llm.model import Transformer, _memory_adaptive_measurement_authority, precompute_freqs_cis
    clear_model_caches, tensor_type = precompute_freqs_cis.cache_clear, Tensor
    workload = request["workload"]
    device_facts = scan_device_facts(); selected = device_facts.selected_device
    with _memory_adaptive_measurement_authority(device_facts=device_facts, inventory=request["inventory"],
        workload={"prefill_ubatch": int(workload["prefill_ubatch"])}, collector=collector):
      model, _ = Transformer.from_gguf(request["model_path"], int(workload["max_context"]))
    prompt = [((i * 17) % 1000) + 1 for i in range(int(workload["prompt_tokens"]))]
    input_before = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
    model.prefill_jit, model.prefill_v2_jit, model.prefill_v2_jits = TinyJit(model.forward), TinyJit(model.forward), {}
    warmup = max(int(workload["warmup_tokens"]), int(workload["decode_tokens"]))
    if warmup: list(zip(range(warmup), model.generate(list(prompt))))
    for _ in range(int(request["samples"])):
      gen = model.generate(list(prompt)); Device[selected].synchronize(); t0 = time.perf_counter()
      out = [int(pair[1]) for pair in zip(range(int(workload["decode_tokens"])), gen)]
      Device[selected].synchronize(); elapsed = time.perf_counter() - t0
      outputs.append(out); speeds.append(len(out) / elapsed)
    input_after = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
  finally:
    if gen is not None: gen.close()
    model = gen = None
    if clear_model_caches is not None: clear_model_caches()
    if tensor_type is not None: tensor_type.manual_seed(tensor_type._seed)
    gc.collect()
  after = _health(); actual = len(outputs) == int(request["samples"]) and after["healthy"]
  return {"schema": WORKER_SCHEMA, "whole_policy_identity": request["whole_policy_identity"],
    "actual_whole_model_run": actual,
    "blockers": [] if actual else ["clean timing run incomplete"],
    "artifacts": {"whole_policy_identity": request["whole_policy_identity"],
      "end_to_end_timing": {"scope": "end_to_end", "metric": "tok_s", "samples": speeds,
      "authority": "isolated PROFILE=0 tinygrad Transformer.from_gguf + model.generate"}},
    "run": {"candidate_id": request["candidate_id"], "whole_policy_identity": request["whole_policy_identity"],
      "strategy": request["strategy"],
      "deterministic_output_evidence": {"schema": "tinygrad.deterministic_full_output.v1",
        "output_kind": "complete_greedy_token_sequence", "outputs": outputs,
        "input_digest_before": input_before, "input_digest_after": input_after},
      "health_pre": before, "health_post": after}}


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--worker", action="store_true")
  args = parser.parse_args(argv)
  if not args.worker: parser.error("this module is loaded as a controller seam; --worker is internal")
  try:
    request = json.loads(sys.stdin.read())
    row = _timing_worker(request) if request.get("lifecycle_phase") == "timing" else _worker(request)
  except Exception as exc: row = {"schema": WORKER_SCHEMA, "actual_whole_model_run": False,
                                  "blockers": [f"worker protocol error: {type(exc).__name__}: {exc}"]}
  print(json.dumps(_transport_encode(row), sort_keys=True, separators=(",", ":")))
  return 0


SEAM = TinygradWholeModelSeam()
if __name__ == "__main__": raise SystemExit(main())

__all__ = ["SCHEMA", "WORKER_SCHEMA", "TinygradWholeModelSeam", "SEAM"]
