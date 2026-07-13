"""Fail-closed gates for the future pure register-resident prefill path.

This module is an evidence join, not a kernel builder.  It consumes compile,
resource, correctness, timing, and route-census artifacts produced by the
existing authorities.  A missing register artifact, an LDS artifact, or an
incomplete hash/provenance join is a hard block.
"""
from __future__ import annotations

from typing import Any, Iterable

SCHEMA = "prefill-pure-register-evaluation-gate.v1"
COMPILE_SCHEMA = "prefill-pure-register-compile.v1"
REGISTER_STORAGE = "global_register_resident"
TARGET = {"backend": "AMD", "arch": "gfx1100", "wave_size": 32}
ROLES = ("attn_qo", "ffn_down", "attn_kv", "ffn_gate_up")
REQUIRED_SEARCH_FIELDS = ("storage_kind", "wait_kind", "buffer_count", "slot_addressing", "consumer_identity")


def _consumer_identity(row: dict[str, Any] | None) -> str | None:
  """Extract the generic GEMM consumer adapter identity from evidence.

  The identity is deliberately opaque to this gate: WMMA, MFMA, and dot2
  adapters can all participate, but a route must name the adapter that
  consumed its logical register tile.  Keeping this as an evidence join
  avoids coupling the pure-route authority to one instruction family.
  """
  if not isinstance(row, dict): return None
  pipeline = row.get("pipeline") if isinstance(row.get("pipeline"), dict) else {}
  value = row.get("consumer_identity", pipeline.get("consumer_identity"))
  return value if isinstance(value, str) and value.strip() else None


def _candidate_fields(row: dict[str, Any] | None) -> tuple[str, ...] | None:
  """Return the explicitly typed policy fields exposed to machine search."""
  if not isinstance(row, dict): return None
  fields = row.get("candidate_fields")
  if not isinstance(fields, list) or any(not isinstance(field, str) or not field.strip() for field in fields): return None
  result = tuple(fields)
  if len(set(result)) != len(result) or any(field not in result for field in REQUIRED_SEARCH_FIELDS): return None
  return result


def _identity(row: dict[str, Any] | None) -> str | None:
  if not isinstance(row, dict): return None
  value = row.get("canonical_identity", row.get("candidate_hash"))
  return value if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value) else None


def _binary_hash(row: dict[str, Any] | None) -> str | None:
  if not isinstance(row, dict): return None
  program = row.get("program") if isinstance(row.get("program"), dict) else {}
  value = row.get("binary_sha256", program.get("binary_sha256"))
  return value if isinstance(value, str) and len(value) == 64 else None


def _result(stage: str, passed: bool, errors: Iterable[str], **evidence: Any) -> dict[str, Any]:
  errors = tuple(str(x) for x in errors)
  return {"schema": f"{SCHEMA}.{stage}", "stage": stage, "passed": bool(passed and not errors),
          "errors": list(errors), **evidence}


def compile_only(candidate: dict[str, Any] | None, artifact: dict[str, Any] | None) -> dict[str, Any]:
  """Validate a compiler-owned register artifact without launching it."""
  errors: list[str] = []
  if not isinstance(candidate, dict): errors.append("candidate payload is unavailable")
  if not isinstance(artifact, dict):
    errors.append("register compile artifact is unavailable")
    return _result("compile", False, errors)
  if artifact.get("schema") not in (COMPILE_SCHEMA, "prefill-pure-anchor-compiled-resource-authority.v1"):
    errors.append("unsupported register compile artifact schema")
  identity = _identity(artifact)
  if identity is None: errors.append("compile artifact has no canonical candidate identity")
  candidate_identity = _identity(candidate)
  if candidate_identity is not None and identity != candidate_identity:
    errors.append("compile artifact identity does not match candidate payload")
  binary = _binary_hash(artifact)
  if binary is None: errors.append("compile artifact has no binary hash")
  resource_artifact = artifact.get("resource_artifact")
  resource_obj = None
  if not isinstance(resource_artifact, dict):
    errors.append("final AMD resource artifact is unavailable")
  else:
    try:
      from tinygrad.codegen.opt.amd_resource_artifact import AMDResourceArtifact, validate_amd_resource_artifact
      resource_obj = AMDResourceArtifact.from_json(resource_artifact)
      if resource_obj.target not in ("gfx1100", "AMD:gfx1100"):
        errors.append("final AMD resource artifact target is not gfx1100")
      validate_amd_resource_artifact(resource_obj, expected_candidate_identity=identity)
      if binary is not None and resource_obj.binary_sha256 != binary:
        errors.append("final resource artifact binary identity differs from compile binary")
    except (TypeError, ValueError) as exc:
      errors.append(f"final AMD resource artifact is invalid: {exc}")
  surface = artifact.get("surface") if isinstance(artifact.get("surface"), dict) else {}
  if artifact.get("passed") is not True: errors.append("compile authority did not pass")
  if surface.get("strict_pure") is not True or artifact.get("strict_pure") is False:
    errors.append("compiled surface is not strict pure")
  if surface.get("ops_ins_count", artifact.get("ops_ins_count", 0)) != 0:
    errors.append("register compile artifact contains Ops.INS")
  if surface.get("source_kind") == "native_isa": errors.append("register compile artifact uses native ISA source")
  pipeline = artifact.get("pipeline") if isinstance(artifact.get("pipeline"), dict) else {}
  if pipeline.get("storage_kind") != REGISTER_STORAGE:
    errors.append("compile artifact does not prove global_register_resident storage")
  if _consumer_identity(artifact) is None:
    errors.append("compile artifact lacks generic GEMM consumer identity")
  if pipeline.get("lds_bytes", pipeline.get("active_lds_bytes", 0)) != 0:
    errors.append("register compile artifact claims LDS storage")
  mapping = pipeline.get("register_mapping")
  required_roles: tuple[str, ...] = ()
  if not isinstance(mapping, dict):
    errors.append("register compile artifact lacks physical VGPR mapping evidence")
  else:
    if mapping.get("backend") != "amd_vgpr":
      errors.append("register compile artifact mapping backend is not amd_vgpr")
    if mapping.get("addressing") not in ("static", "sequential"):
      errors.append("register compile artifact mapping does not prove static VGPR addressing")
    raw_roles = mapping.get("required_roles")
    if not isinstance(raw_roles, list) or not raw_roles or any(not isinstance(role, str) or not role for role in raw_roles):
      errors.append("register compile artifact mapping has no required logical register roles")
    elif len(set(raw_roles)) != len(raw_roles):
      errors.append("register compile artifact mapping has duplicate logical register roles")
    else:
      required_roles = tuple(raw_roles)
      try:
        validate_amd_resource_artifact(resource_obj, expected_candidate_identity=identity,
                                       required_roles=required_roles)
      except (TypeError, ValueError) as exc:
        errors.append(f"register physical mapping is invalid: {exc}")
  wait = artifact.get("wait") if isinstance(artifact.get("wait"), dict) else {}
  if wait.get("typed") is not True or wait.get("kind") != "targeted_vmcnt":
    errors.append("compile artifact lacks typed targeted wait evidence")
  coverage = wait.get("coverage")
  if not isinstance(coverage, dict):
    errors.append("compile artifact lacks serialized wait dependency coverage")
  else:
    try:
      from tinygrad.codegen.opt.compiler_policies import WaitDependencyCoverage
      coverage_obj = WaitDependencyCoverage.from_json(coverage)
      if not coverage_obj.passed:
        errors.append("compile artifact wait dependency coverage did not pass")
      raw_required = pipeline.get("wait_required_edges")
      if not isinstance(raw_required, list) or not raw_required:
        errors.append("compile artifact lacks required wait stage edges")
      else:
        required_edges = set()
        for edge in raw_required:
          if not isinstance(edge, list) or len(edge) != 3 or not isinstance(edge[0], str) or not edge[0] or \
             any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in edge[1:]):
            errors.append("compile artifact has malformed required wait stage edge")
            continue
          required_edges.add((edge[0], edge[1], edge[2]))
        if required_edges - set(coverage_obj.covered):
          errors.append("compile artifact wait coverage is missing required stage edges")
    except (TypeError, ValueError) as exc:
      errors.append(f"compile artifact wait dependency coverage is invalid: {exc}")
  abi = artifact.get("abi") if isinstance(artifact.get("abi"), dict) else {}
  for name, expected in (("wave_size", 32), ("fragment_carrier", "half.vec(16)"),
                         ("accumulator_carrier", "float.vec(8)")):
    if abi.get(name) != expected: errors.append(f"compile artifact ABI field {name!r} is unproven")
  artifact_resources = {}
  if resource_obj is not None:
    facts = resource_obj.resources
    wave_count = (facts.workgroup_threads // facts.wavefront_size
                  if facts.workgroup_threads is not None and facts.wavefront_size and
                  facts.workgroup_threads % facts.wavefront_size == 0 else None)
    artifact_resources = {**facts.to_json(), "stage": resource_obj.resource_stage, "wave_count": wave_count}
  return _result("compile", not errors, errors, canonical_identity=identity, binary_sha256=binary,
                 storage_kind=pipeline.get("storage_kind"), consumer_identity=_consumer_identity(artifact),
                 pipeline=pipeline, wait=wait, abi=abi,
                 resources=artifact_resources or (artifact.get("resources") if isinstance(artifact.get("resources"), dict) else {}),
                 resource_artifact=resource_artifact)


def final_resources(compile: dict[str, Any] | None) -> dict[str, Any]:
  """Require final-program resource facts joined to the compile artifact."""
  errors: list[str] = []
  if not isinstance(compile, dict) or compile.get("passed") is not True:
    errors.append("compile gate must pass before final-resource evaluation")
    return _result("resources", False, errors)
  resources = compile.get("resources") if isinstance(compile.get("resources"), dict) else {}
  if resources.get("stage") != "final_program": errors.append("resource plan is not final_program")
  for name in ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "workgroup_threads", "wave_count"):
    value = resources.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
      errors.append(f"final resource {name!r} is unknown")
  if resources.get("lds_bytes") != 0: errors.append("register resource plan must report zero LDS bytes")
  if resources.get("scratch_bytes") != 0: errors.append("register candidate uses scratch")
  for name in ("vgpr_spills", "sgpr_spills"):
    if resources.get(name) != 0: errors.append(f"register candidate has {name}")
  if _identity(compile) is None or _binary_hash(compile) is None:
    errors.append("final resources are not joined to candidate and binary hashes")
  return _result("resources", not errors, errors, canonical_identity=_identity(compile),
                 binary_sha256=_binary_hash(compile), resources=resources)


def runtime_compile_resource_eligibility(candidate: dict[str, Any] | None, artifact: dict[str, Any] | None, *,
                                         profile: str, role: str, shape: tuple[int, int, int],
                                         target: dict[str, Any]) -> dict[str, Any]:
  """Default-closed runtime join for register warmstart installation; never launches a kernel."""
  compiled = compile_only(candidate, artifact)
  resources = final_resources(compiled)
  errors = [*compiled["errors"], *resources["errors"]]
  binding = artifact.get("runtime_binding") if isinstance(artifact, dict) else None
  expected = {"profile": profile, "role": role, "shape": {"m": shape[0], "n": shape[1], "k": shape[2]}, "target": target}
  if not isinstance(binding, dict): errors.append("compile artifact lacks exact runtime binding")
  elif binding != expected: errors.append("compile artifact runtime binding is not an exact workload/target match")
  candidate_identity = _identity(candidate)
  if candidate_identity is None or _identity(compiled) != candidate_identity:
    errors.append("runtime candidate/compile identity join failed")
  return _result("runtime_compile_resource_eligibility", not errors, errors,
                 canonical_identity=_identity(compiled), binary_sha256=_binary_hash(compiled), binding=binding,
                 compile=compiled, resources=resources)


def correctness_timing(resources: dict[str, Any] | None, correctness: dict[str, Any] | None,
                       timing: dict[str, Any] | None, *, baseline_tok_s: float | None = None) -> dict[str, Any]:
  """Require nonconstant correctness and an isolated pinned timing sample."""
  errors: list[str] = []
  if not isinstance(resources, dict) or resources.get("passed") is not True:
    errors.append("final-resource gate must pass before correctness/timing")
  if not isinstance(correctness, dict):
    errors.append("correctness artifact is unavailable")
  else:
    if correctness.get("passed") is not True: errors.append("correctness authority did not pass")
    if correctness.get("nonconstant_cases") is not True: errors.append("nonconstant correctness cases are unproven")
    if correctness.get("all_output_parity") is not True: errors.append("full output parity is unproven")
    if _identity(correctness) != _identity(resources): errors.append("correctness candidate identity join failed")
    if _binary_hash(correctness) != _binary_hash(resources): errors.append("correctness binary identity join failed")
  if not isinstance(timing, dict):
    errors.append("timing artifact is unavailable")
  else:
    if timing.get("passed") is not True: errors.append("timing authority did not pass")
    protocol = timing.get("protocol") if isinstance(timing.get("protocol"), dict) else {}
    if protocol.get("scope") != "kernel_only" or protocol.get("compile_excluded") is not True:
      errors.append("timing is not compile-excluded kernel-only")
    if timing.get("clock_pin") is not True: errors.append("timing clocks are not pinned")
    if _identity(timing) != _identity(resources): errors.append("timing candidate identity join failed")
    if _binary_hash(timing) != _binary_hash(resources): errors.append("timing binary identity join failed")
    measured = timing.get("tok_s", timing.get("median_tok_s"))
    if not isinstance(measured, (int, float)) or measured <= 0: errors.append("timing throughput is unavailable")
    elif baseline_tok_s is not None and measured <= baseline_tok_s:
      errors.append("register candidate does not improve the pure baseline")
  return _result("correctness_timing", not errors, errors, canonical_identity=_identity(resources),
                 binary_sha256=_binary_hash(resources), correctness=correctness, timing=timing)


def validate_role_attribution(report: dict[str, Any] | None, *, expected_roles: tuple[str, ...] = ROLES,
                              require_pure: bool = False) -> dict[str, Any]:
  """Require an explicit route row for every role in a whole-prefill artifact.

  A top-level route flag never substitutes for the role map: candidate runs can
  bind one role while the remaining roles use a fallback route.
  """
  errors: list[str] = []
  if not isinstance(report, dict):
    return _result("role_attribution", False, ("whole-prefill route artifact is unavailable",))
  role_routes = report.get("prefill_role_routes")
  if not isinstance(role_routes, dict):
    errors.append("prefill_role_routes is unavailable")
    role_routes = {}
  for role in expected_roles:
    route = role_routes.get(role)
    if not isinstance(route, str) or not route:
      errors.append(f"{role}: route attribution is missing")
    elif require_pure and route not in ("register", "pure_register", "global_register_resident"):
      errors.append(f"{role}: route {route!r} is not an admitted pure register route")
  census = report.get("candidate_set_route_census")
  if census is not None and (not isinstance(census, dict) or census.get("passed") is not True):
    errors.append("candidate-set route census is missing or failed")
  return _result("role_attribution", not errors, errors, roles={role: role_routes.get(role) for role in expected_roles},
                 candidate_set_route_census=census)


def machine_search(role_evidence: dict[str, dict[str, Any]] | None, *, selected_roles: tuple[str, ...] = ROLES,
                   route_report: dict[str, Any] | None = None) -> dict[str, Any]:
  """Admit search only when every selected role has complete pure evidence."""
  errors: list[str] = []
  if not isinstance(role_evidence, dict):
    errors.append("role evidence census is unavailable")
    return _result("machine_search", False, errors, selected_roles=list(selected_roles))
  if not selected_roles or any(role not in ROLES for role in selected_roles): errors.append("selected roles are unsupported")
  rows = {}
  for role in selected_roles:
    row = role_evidence.get(role)
    if not isinstance(row, dict):
      errors.append(f"{role}: evidence is unavailable")
      continue
    rows[role] = row
    if row.get("passed") is not True: errors.append(f"{role}: pure evidence gates did not pass")
    if row.get("strict_pure") is not True or row.get("fallback_used") is not False:
      errors.append(f"{role}: strict pure fallback-free route is unproven")
    if row.get("route_family") != "pure": errors.append(f"{role}: route family is not pure")
    if not _identity(row) or not _binary_hash(row): errors.append(f"{role}: candidate/binary identity join is incomplete")
    if _consumer_identity(row) is None:
      errors.append(f"{role}: generic GEMM consumer identity is unavailable")
    if _candidate_fields(row) is None:
      errors.append(f"{role}: candidate fields are missing or not the required typed policy set")
    policy = row.get("policy") if isinstance(row.get("policy"), dict) else {}
    if policy.get("storage_kind") != REGISTER_STORAGE or policy.get("wait_kind") != "targeted_vmcnt":
      errors.append(f"{role}: typed register policy fields are unavailable")
    if row.get("search_space") != "typed_policy_fields":
      errors.append(f"{role}: search space is not typed policy fields")
  if route_report is not None:
    attribution = validate_role_attribution(route_report, expected_roles=selected_roles, require_pure=True)
    errors.extend(f"role attribution: {error}" for error in attribution["errors"])
    consumers = route_report.get("prefill_role_consumers") if isinstance(route_report, dict) else None
    if consumers is not None:
      if not isinstance(consumers, dict):
        errors.append("role attribution: prefill_role_consumers is malformed")
      else:
        for role in selected_roles:
          expected = consumers.get(role)
          actual = _consumer_identity(rows.get(role))
          if not isinstance(expected, str) or not expected.strip():
            errors.append(f"role attribution: {role}: consumer identity is missing")
          elif actual != expected:
            errors.append(f"role attribution: {role}: consumer identity does not match evidence")
  return _result("machine_search", not errors, errors, selected_roles=list(selected_roles), roles=rows,
                 search_space="typed_policy_fields" if not errors else None)


def evaluate(candidate: dict[str, Any] | None, *, compile_artifact: dict[str, Any] | None = None,
             correctness: dict[str, Any] | None = None, timing: dict[str, Any] | None = None,
             role_evidence: dict[str, dict[str, Any]] | None = None,
             selected_roles: tuple[str, ...] = ROLES, baseline_tok_s: float | None = None,
             route_report: dict[str, Any] | None = None) -> dict[str, Any]:
  """Run R6-R10 in order and stop at the first unavailable or failed gate."""
  report: dict[str, Any] = {"schema": SCHEMA, "passed": False, "blocked_at": None, "blockers": {}, "stages": {}}
  compile = compile_only(candidate, compile_artifact)
  report["stages"]["compile"] = compile
  if not compile["passed"]:
    report.update(blocked_at="compile", blockers={"compile": compile["errors"]})
    return report
  resources = final_resources(compile)
  report["stages"]["resources"] = resources
  if not resources["passed"]:
    report.update(blocked_at="resources", blockers={"resources": resources["errors"]})
    return report
  correctness_stage = correctness_timing(resources, correctness, timing, baseline_tok_s=baseline_tok_s)
  report["stages"]["correctness_timing"] = correctness_stage
  if not correctness_stage["passed"]:
    report.update(blocked_at="correctness_timing", blockers={"correctness_timing": correctness_stage["errors"]})
    return report
  search = machine_search(role_evidence, selected_roles=selected_roles, route_report=route_report)
  report["stages"]["machine_search"] = search
  if not search["passed"]:
    report.update(blocked_at="machine_search", blockers={"machine_search": search["errors"]})
    return report
  report["passed"] = True
  return report
