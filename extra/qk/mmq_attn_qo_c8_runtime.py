"""Exact attn_qo bootstrap and runner factory for frozen staged C8.

This module deliberately composes existing execution/attestation primitives.
It does not create a second launcher.  Qualification is untimed and queue
specific; timing refuses to start until two distinct immutable qualification
files already exist.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from extra.qk.direct_packed_executable_attestor import (
  DirectPackedAttestationBindings, make_production_direct_packed_attested_runner,
  qualify_and_freeze_production_direct_packed,
)
from extra.qk.mmq_attn_qo_c6_binding import (
  COMPOSITION_SCHEMA, read_json, rebuild_attn_qo_exact_fixture,
)
from extra.qk.mmq_exact_role_spec import exact_role_spec
from extra.qk.mmq_frozen_staged_c8_timing import (
  QueueTimingRunners, make_frozen_staged_candidate_runner,
)
from extra.qk.mmq_frozen_staged_family import (
  QUEUE_MODES, FrozenStagedFamily, load_frozen_staged_family_manifest,
)
from extra.qk.mmq_staged_c7_authority import validate_staged_c7_authority_snapshot


REQUIRED_PROGRAM_PREFIX = "q4k_gen_prefill_"


@dataclass(frozen=True)
class DirectPackedObjects:
  """Exact production objects shared by role-specific C8 bootstraps."""

  linear: Any
  activation: Any
  route_spec: Any


DirectPackedObjectBuilder = Callable[[Any, Any, Any, str], DirectPackedObjects]


def production_direct_packed_object_builder(
    role: Any, words: Any, activation: Any, activation_dtype: str,
    ) -> DirectPackedObjects:
  """Lazily build real tinygrad objects without import-time Device access."""
  from tinygrad import Tensor, dtypes
  from tinygrad.llm.prefill_routes import PrefillLinearRouteSpec

  if activation_dtype not in ("float16", "float32"):
    raise ValueError("production activation dtype must be float16 or float32")
  dtype = dtypes.float16 if activation_dtype == "float16" else dtypes.float32
  packed = Tensor(words, dtype=dtypes.uint32, device="AMD")
  activation_tensor = Tensor(activation, dtype=dtype, device="AMD")
  linear = SimpleNamespace(
    bias=None, out_features=role.n, in_features=role.k,
    q4k_storage=object(), route_role=role.role, _prefill_graph_role=role.role,
    prefill_packed_weight=lambda: packed)
  spec = PrefillLinearRouteSpec(
    "direct_packed", "q4k", role.role, role.m, role.n, role.k)
  return DirectPackedObjects(linear, activation_tensor, spec)


def build_direct_packed_objects(
    *, role: Any, words: Any, activation: Any, activation_dtype: str,
    object_builder: DirectPackedObjectBuilder | None = None,
    ) -> DirectPackedObjects:
  """Build one exact fallback object set through an injectable lazy builder."""
  if getattr(role, "role", None) not in ("attn_qo", "ffn_gate_up") or \
     any(not isinstance(getattr(role, name, None), int) or
         getattr(role, name) <= 0 for name in ("m", "n", "k")):
    raise ValueError("direct-packed objects require an admitted exact role")
  builder = production_direct_packed_object_builder if object_builder is None else object_builder
  if not callable(builder):
    raise TypeError("direct-packed object builder must be callable")
  result = builder(role, words, activation, activation_dtype)
  if not isinstance(result, DirectPackedObjects):
    raise TypeError("direct-packed object builder must return DirectPackedObjects")
  if result.linear is None or result.activation is None or result.route_spec is None:
    raise ValueError("direct-packed object builder returned incomplete objects")
  return result


def _composition(value: Any, family: FrozenStagedFamily) -> dict[str, Any]:
  if not isinstance(value, Mapping) or value.get("schema") != COMPOSITION_SCHEMA or \
     value.get("status") != "PASS" or \
     value.get("family_identity") != family.family_identity or \
     value.get("promotion_eligible_on_candidate_win") is not False:
    raise ValueError("attn_qo C8 composition is not an exact non-promotable PASS")
  c6 = value.get("c6_correctness_evidence")
  if not isinstance(c6, Mapping) or c6.get("status") != "PASS" or \
     c6.get("family_identity") != family.family_identity:
    raise ValueError("attn_qo C8 composition lacks exact C6 evidence")
  if not isinstance(value.get("runtime_canary_by_queue"), Mapping) or \
     set(value["runtime_canary_by_queue"]) != set(QUEUE_MODES):
    raise ValueError("attn_qo C8 composition lacks both runtime canaries")
  return dict(value)


def validate_live_software(
    authority_snapshot: Mapping[str, Any], composition: Mapping[str, Any],
    ) -> dict[str, Any]:
  """Prove the current clean commit/tree is the C6/C7 software authority."""
  authority = validate_staged_c7_authority_snapshot(
    authority_snapshot, verify_current_software=True)
  if composition.get("c7_authority_snapshot_identity") != authority["snapshot_identity"] or \
     composition.get("c7_memory_authority") != authority["memory_authority"] or \
     composition["c6_correctness_evidence"].get("software_identity") != \
       authority["software_identity"]:
    raise ValueError("live software or C6 composition differs from C7 authority")
  return authority


def build_attn_qo_direct_packed_objects(
    family: FrozenStagedFamily, composition: Mapping[str, Any], *,
    object_builder: DirectPackedObjectBuilder | None = None,
    ) -> tuple[Any, Any, Any]:
  """Build the production direct-packed objects from exact retained bytes."""
  role = family.binding.role_spec
  if role.role != "attn_qo":
    raise ValueError("C8 runtime is exact to attn_qo")
  fixture = rebuild_attn_qo_exact_fixture(role, composition["execution_fixture"])
  objects = build_direct_packed_objects(
    role=role, words=fixture.words,
    activation=fixture.source.reshape(1, role.m, role.k),
    activation_dtype="float32", object_builder=object_builder)
  return objects.linear, objects.activation, objects.route_spec


def queue_attestation_bindings(
    c6_by_queue: Mapping[str, Mapping[str, Any]], *, clock_identity: str,
    ) -> dict[str, DirectPackedAttestationBindings]:
  """Build the exact queue bindings used by both Qo and ffn bootstraps."""
  if not isinstance(c6_by_queue, Mapping) or set(c6_by_queue) != set(QUEUE_MODES):
    raise ValueError(f"C6 queue bindings must contain exactly {QUEUE_MODES!r}")
  return {
    queue: DirectPackedAttestationBindings(
      queue_mode=queue, workload_identity=c6_by_queue[queue]["workload_identity"],
      input_identity=c6_by_queue[queue]["input_identity"],
      device_identity=c6_by_queue[queue]["device_identity"],
      software_identity=c6_by_queue[queue]["software_identity"],
      comparator_identity=c6_by_queue[queue]["comparator_identity"],
      clock_identity=clock_identity, required_program_prefix=REQUIRED_PROGRAM_PREFIX,
    ).validate()
    for queue in QUEUE_MODES
  }


def attestation_bindings(
    composition: Mapping[str, Any], *, clock_identity: str,
    ) -> dict[str, DirectPackedAttestationBindings]:
  c6 = composition["c6_correctness_evidence"]
  return queue_attestation_bindings({
    queue: {
      "workload_identity": c6["workload_identity"],
      "input_identity": c6["input_identity"],
      "device_identity": c6["device_identity"],
      "software_identity": c6["software_identity"],
      "comparator_identity": c6["queue_comparators"][queue],
    } for queue in QUEUE_MODES
  }, clock_identity=clock_identity)


def qualification_paths(
    config: Mapping[str, Any], *, pm4_key: str = "qualification_pm4",
    aql_key: str = "qualification_aql",
    ) -> dict[str, Path]:
  """Require two distinct immutable queue qualification files."""
  paths = {"PM4": Path(config[pm4_key]).resolve(), "AQL": Path(config[aql_key]).resolve()}
  if paths["PM4"] == paths["AQL"] or not all(path.is_file() for path in paths.values()):
    raise ValueError("both distinct frozen qualification files must preexist")
  return paths


def compose_queue_timing_runners(
    *, queue_mode: str, family: FrozenStagedFamily,
    frozen_bundle: str | Path, staged_family_manifest: str | Path,
    runtime_canary_by_queue: Mapping[str, Any],
    direct_objects: DirectPackedObjects,
    qualification_paths_by_queue: Mapping[str, str | Path],
    bindings_by_queue: Mapping[str, DirectPackedAttestationBindings],
    clock_ns: Any,
    candidate_runner_builder: Callable[..., Any] =
      make_frozen_staged_candidate_runner,
    fallback_runner_builder: Callable[..., Any] =
      make_production_direct_packed_attested_runner,
    ) -> QueueTimingRunners:
  """Role-neutral composition of the existing candidate and fallback launchers."""
  if not isinstance(direct_objects, DirectPackedObjects):
    raise TypeError("direct_objects must be DirectPackedObjects")
  if not callable(candidate_runner_builder) or not callable(fallback_runner_builder):
    raise TypeError("C8 runner builders must be callable")
  candidate = candidate_runner_builder(
    role_spec=family.binding.role_spec, frozen_bundle=frozen_bundle,
    staged_family_manifest=staged_family_manifest,
    runtime_canary_by_queue=runtime_canary_by_queue)
  fallback_result = fallback_runner_builder(
    linear=direct_objects.linear, input_tensor=direct_objects.activation,
    route_spec=direct_objects.route_spec,
    qualification_paths_by_queue=qualification_paths_by_queue,
    bindings_by_queue=bindings_by_queue, clock_ns=clock_ns)
  if not isinstance(fallback_result, tuple) or len(fallback_result) != 2:
    raise TypeError("fallback runner builder must return (runner, attestor)")
  fallback, _attestor = fallback_result
  return QueueTimingRunners(candidate, fallback).validate(queue_mode)


def qualify_one_queue(
    *, family: FrozenStagedFamily, composition: Mapping[str, Any],
    authority_snapshot: Mapping[str, Any], queue_mode: str, output: str | Path,
    clock_identity: str,
    ) -> dict[str, Any]:
  """Freeze one untimed production fallback qualification."""
  composition = _composition(composition, family)
  validate_live_software(authority_snapshot, composition)
  linear, activation, spec = build_attn_qo_direct_packed_objects(family, composition)
  return qualify_and_freeze_production_direct_packed(
    linear=linear, input_tensor=activation, route_spec=spec,
    queue_mode=queue_mode,
    bindings_by_queue=attestation_bindings(
      composition, clock_identity=clock_identity),
    output=output)


def attn_qo_c8_runner_factory(
    *, queue_mode: str, family: FrozenStagedFamily,
    c6_correctness_evidence: Mapping[str, Any], clock_identity: str,
    clock_ns: Any, config: Mapping[str, Any],
    ) -> QueueTimingRunners:
  """Factory loaded by ``mmq_frozen_staged_c8_sessions`` inside each child."""
  required = {
    "composition", "authority_snapshot", "frozen_bundle",
    "staged_family_manifest", "qualification_pm4", "qualification_aql",
  }
  if not isinstance(config, Mapping) or set(config) != required:
    raise ValueError(f"attn_qo C8 runner config must contain exactly {sorted(required)!r}")
  composition = _composition(read_json(config["composition"], "C8 composition"), family)
  if dict(c6_correctness_evidence) != composition["c6_correctness_evidence"]:
    raise ValueError("session C6 evidence differs from attn_qo composition")
  validate_live_software(
    read_json(config["authority_snapshot"], "C7 authority"), composition)
  frozen_qualifications = qualification_paths(config)
  linear, activation, spec = build_attn_qo_direct_packed_objects(family, composition)
  bindings = attestation_bindings(composition, clock_identity=clock_identity)
  return compose_queue_timing_runners(
    queue_mode=queue_mode, family=family,
    frozen_bundle=config["frozen_bundle"],
    staged_family_manifest=config["staged_family_manifest"],
    runtime_canary_by_queue=composition["runtime_canary_by_queue"],
    direct_objects=DirectPackedObjects(linear, activation, spec),
    qualification_paths_by_queue=frozen_qualifications,
    bindings_by_queue=bindings, clock_ns=clock_ns)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--role", default="attn_qo")
  parser.add_argument("--frozen-bundle", type=Path, required=True)
  parser.add_argument("--staged-family-manifest", type=Path, required=True)
  parser.add_argument("--composition", type=Path, required=True)
  parser.add_argument("--authority-snapshot", type=Path, required=True)
  parser.add_argument("--queue-mode", choices=QUEUE_MODES, required=True)
  parser.add_argument("--clock-identity", default="clock-policy-0")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args(argv)
  role = exact_role_spec(args.role)
  family = load_frozen_staged_family_manifest(
    args.staged_family_manifest, role_spec=role, frozen_bundle=args.frozen_bundle)
  artifact = qualify_one_queue(
    family=family, composition=read_json(args.composition, "C8 composition"),
    authority_snapshot=read_json(args.authority_snapshot, "C7 authority"),
    queue_mode=args.queue_mode, output=args.output,
    clock_identity=args.clock_identity)
  print(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False))
  return 0


if __name__ == "__main__": raise SystemExit(main())


__all__ = [
  "DirectPackedObjectBuilder", "DirectPackedObjects",
  "REQUIRED_PROGRAM_PREFIX", "attestation_bindings",
  "attn_qo_c8_runner_factory", "build_attn_qo_direct_packed_objects",
  "build_direct_packed_objects", "compose_queue_timing_runners",
  "production_direct_packed_object_builder", "qualification_paths",
  "qualify_one_queue", "queue_attestation_bindings", "validate_live_software",
]
