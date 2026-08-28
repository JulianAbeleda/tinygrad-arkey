"""Production runtime for exact, search-selected fp16 prefill graph GEMMs.

Research tooling builds and admits candidate registries.  This module only
consumes an exact model-owned attachment, installs the compiler warmstart
context for the selected workload, and otherwise declines to the ordinary
tinygrad linear path.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Mapping
from typing import Any

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.llm.prefill_candidate_runtime import canonical_candidate_set_identity
from tinygrad.uop.ops import Ops


_CANDIDATE_ROUTE_CENSUS: ContextVar[dict[str, Any] | None] = ContextVar("candidate_route_census", default=None)

# Measured per-target warmstart schedule for admitted fp16 overlay GEMMs, keyed by the same declared
# (backend, arch, wave_size) triple the compact artifacts use.  The candidate
# context owns its complete output tile.  Applying generic output UPCASTs on
# top of that tile makes several logical rows alias the same LDS slots; the
# old apparent 46 ms result was therefore not correctness-qualified.  The
# exact sm_120 gate is TC-only, which matches the generic reference bitwise.
_CANDIDATE_WARMSTART_OPTS: tuple[tuple[tuple[str, str, int], tuple[Opt, ...]], ...] = (
  (("NV", "sm_120", 32), (Opt(OptOps.TC, 0, (-1, 2, 1)),)),
)


def _candidate_warmstart_opts(backend: str, arch: str, wave_size: int) -> tuple[Opt, ...]:
  for target, opts in _CANDIDATE_WARMSTART_OPTS:
    if target == (backend, arch, wave_size): return opts
  return (Opt(OptOps.TC, 0, (-1, 2, 1)),)


@contextmanager
def candidate_route_census():
  collector = {"selected":{}, "model_forward":{}}
  token = _CANDIDATE_ROUTE_CENSUS.set(collector)
  try: yield collector
  finally: _CANDIDATE_ROUTE_CENSUS.reset(token)


def _candidate_route_row(admission) -> dict[str, Any]:
  workload = admission.normalized_payload["workload"]
  shape, target = workload["shape"], workload["target"]
  return {"profile":workload["profile"], "role":workload["role"],
          "shape":{"m":shape["m"], "n":shape["n"], "k":shape["k"]},
          "target":{"backend":target["backend"], "arch":target["arch"], "wave_size":target["wave_size"]},
          "canonical_identity":admission.canonical_identity}


def _structural_route_key(row: dict[str, Any]) -> tuple[Any, ...]:
  shape, target = row["shape"], row["target"]
  return (row["role"], shape["m"], shape["n"], shape["k"], target["backend"], target["arch"], target["wave_size"])


def _record_candidate_route(admission) -> None:
  collector = _CANDIDATE_ROUTE_CENSUS.get()
  if collector is None: return
  row, key = _candidate_route_row(admission), _structural_route_key(_candidate_route_row(admission))
  prior = collector["selected"].get(key)
  if prior is not None and prior["canonical_identity"] != row["canonical_identity"]:
    raise RuntimeError(f"candidate route census identity drift for {key!r}")
  collector["selected"][key] = {**row, "bindings":1 if prior is None else prior["bindings"] + 1}


def record_model_forward_candidate(*, role: str, shape: tuple[int, int, int], canonical_identity: str, one_buffer: bool) -> None:
  """Record a separately admitted forward binding without relaxing registry identity checks."""
  collector = _CANDIDATE_ROUTE_CENSUS.get()
  if collector is None or not one_buffer: return
  if not isinstance(canonical_identity, str) or not canonical_identity or not all(isinstance(x, int) for x in shape): return
  key = (role, *shape, canonical_identity)
  prior = collector["model_forward"].get(key)
  collector["model_forward"][key] = {"role":role, "shape":{"m":shape[0], "n":shape[1], "k":shape[2]},
    "canonical_identity":canonical_identity, "one_buffer":True, "bindings":1 if prior is None else prior["bindings"] + 1}


def finalize_candidate_route_census(collector: dict[str, Any], registry) -> dict[str, Any]:
  enabled_roles = {admission.normalized_payload["workload"]["role"] for admission in registry.admissions}
  expected = {_structural_route_key(_candidate_route_row(admission)):{**_candidate_route_row(admission), "bindings":0}
              for _,admission in zip(registry.candidate_set.entries, registry.admissions)
              if admission.normalized_payload["workload"]["role"] in enabled_roles}
  selected = dict(collector["selected"])
  missing = [expected[key] for key in sorted(expected.keys() - selected.keys())]
  unexpected = [selected[key] for key in sorted(selected.keys() - expected.keys())]
  mismatched = [selected[key] for key in sorted(expected.keys() & selected.keys())
                if selected[key]["canonical_identity"] != expected[key]["canonical_identity"]]
  return {"schema":"prefill-candidate-set-route-census.v1", "passed":not (missing or unexpected or mismatched),
          "policy_roles":sorted(enabled_roles), "expected_entry_count":len(expected), "selected_entry_count":len(selected),
          "model_forward":[collector["model_forward"][key] for key in sorted(collector["model_forward"])],
          "selected":[selected[key] for key in sorted(selected)], "missing":missing, "unexpected":unexpected,
          "identity_mismatches":mismatched}


def _contiguous_candidate_operand(value: Tensor) -> Tensor:
  """Keep semantic metadata around an already concrete allocation without copying it."""
  return value if value.uop.op is Ops.MEMORY_SEMANTIC and value.uop.src[0].has_buffer_identity() else value.contiguous()


def _candidate_storage_kind(payload: Mapping[str, Any]) -> str:
  residency = payload.get("schedule", {}).get("residency", {}) if isinstance(payload, Mapping) else {}
  resident = residency.get("resident", ()) if isinstance(residency, Mapping) else ()
  return "global_register_resident" if isinstance(resident, (list, tuple)) and "stage_ab_register" in resident else "lds"


def _install_candidate_matmul(x, w, out_f, in_f, admission, compile_artifact: Mapping[str, Any] | None = None):
  # Register-resident candidates remain experimental until their compile/resource authority is promoted too.
  # Fail closed instead of importing that research-only evidence stack into production.
  if _candidate_storage_kind(admission.normalized_payload) == "global_register_resident": return None
  import tinygrad.codegen.opt.postrange as pr
  m = int(x.shape[-2])
  packed_dtype = admission.context.packed_weight.storage_dtype if admission.context.packed_weight is not None else None
  key = pr.warmstart_key({m, out_f}, in_f, packed_dtype)
  existing = (pr._WARMSTART_CANDIDATE_CONTEXTS or {}).get(key)
  if existing is not None and existing.canonical_identity != admission.canonical_identity:
    raise ValueError(f"candidate warmstart key collision for {key!r}")
  target = admission.normalized_payload["workload"]["target"]
  opts = _candidate_warmstart_opts(target["backend"], target["arch"], target["wave_size"])
  pr._WARMSTART_OPTS = {**(pr._WARMSTART_OPTS or {}), key:opts}
  pr._WARMSTART_CANDIDATE_CONTEXTS = {**(pr._WARMSTART_CANDIDATE_CONTEXTS or {}), key:admission.context}
  a = x.reshape(m, in_f).cast(dtypes.float16).contiguous()
  bt = _contiguous_candidate_operand(w.cast(dtypes.float16))
  return (a @ bt.transpose()).reshape(*x.shape[:-1], out_f)


def _attached_candidate_admission(lin, role: str, shape: tuple[int, int, int]):
  """Resolve an exact policy attachment; provenance labels never select runtime work."""
  binding = getattr(lin, "_prefill_graph_gemm_binding", None)
  if not isinstance(binding, dict): return None
  registry, policy, facts = binding.get("candidate_registry"), binding.get("selected_policy"), binding.get("scanned_target_facts")
  inventory_identity, candidate_set_identity = binding.get("inventory_identity"), binding.get("candidate_set_identity")
  if registry is None or not isinstance(policy, dict) or not isinstance(facts, dict): return None
  if not all(isinstance(value, str) and value for value in (inventory_identity, candidate_set_identity)): return None
  target = facts.get("target", facts)
  if not isinstance(target, dict) or not all(key in target for key in ("backend", "arch", "wave_size")): return None
  target = {key:target[key] for key in ("backend", "arch", "wave_size")}
  expected_shape = {"m":shape[0], "n":shape[1], "k":shape[2]}
  if policy.get("role") != role or policy.get("shape") != expected_shape or policy.get("target") != target: return None
  if policy.get("inventory_identity") != inventory_identity or policy.get("candidate_set_identity") != candidate_set_identity: return None
  identity = policy.get("candidate_identity")
  if not isinstance(identity, str) or not identity: return None
  try:
    if canonical_candidate_set_identity(registry.candidate_set.to_json()) != candidate_set_identity: return None
  except (AttributeError, TypeError, ValueError): return None
  matches = []
  for admission in registry.admissions:
    row = _candidate_route_row(admission)
    if (_structural_route_key(row) == (role, *shape, target["backend"], target["arch"], target["wave_size"]) and
        admission.canonical_identity == identity): matches.append(admission)
  return matches[0] if len(matches) == 1 else None


def route_pf16_graph_gemm(lin, x: Tensor, w: Tensor | None = None) -> Tensor | None:
  """Run an exactly attached selected candidate, or decline to generic tinygrad."""
  if w is None: w = getattr(lin, "_pf16_w", None)
  if w is None or getattr(lin, "bias", None) is not None or x.ndim < 2: return None
  if not isinstance(x.shape[-2], int) or not isinstance(x.shape[-1], int): return None
  m = x.shape[-2]
  out_f, in_f = w.shape
  if in_f != x.shape[-1]: return None
  role = getattr(lin, "_prefill_graph_role", None)
  admission = None if role is None else _attached_candidate_admission(lin, role, (m, out_f, in_f))
  if admission is None: return None
  binding = getattr(lin, "_prefill_graph_gemm_binding", {})
  result = _install_candidate_matmul(x, w, out_f, in_f, admission,
                                     binding.get("compile_artifact") if isinstance(binding, dict) else None)
  if result is None: return None
  setattr(lin, "_prefill_full_kernel_candidate_identity", admission.canonical_identity)
  _record_candidate_route(admission)
  return result


__all__ = ["candidate_route_census", "finalize_candidate_route_census", "record_model_forward_candidate",
           "route_pf16_graph_gemm"]
