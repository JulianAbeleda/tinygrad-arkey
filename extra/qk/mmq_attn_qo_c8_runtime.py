"""Exact attn_qo bootstrap and runner factory for frozen staged C8.

This module deliberately composes existing execution/attestation primitives.
It does not create a second launcher.  Qualification is untimed and queue
specific; timing refuses to start until two distinct immutable qualification
files already exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

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
    family: FrozenStagedFamily, composition: Mapping[str, Any],
    ) -> tuple[Any, Any, Any]:
  """Build the production direct-packed objects from exact retained bytes."""
  from tinygrad import Tensor, dtypes
  from tinygrad.llm.prefill_routes import PrefillLinearRouteSpec

  role = family.binding.role_spec
  if role.role != "attn_qo":
    raise ValueError("C8 runtime is exact to attn_qo")
  fixture = rebuild_attn_qo_exact_fixture(role, composition["execution_fixture"])
  packed = Tensor(fixture.words, dtype=dtypes.uint32, device="AMD")
  activation = Tensor(
    fixture.source.reshape(1, role.m, role.k), dtype=dtypes.float32, device="AMD")
  linear = SimpleNamespace(
    bias=None, out_features=role.n, in_features=role.k,
    q4k_storage=object(), route_role=role.role, _prefill_graph_role=role.role,
    prefill_packed_weight=lambda: packed)
  spec = PrefillLinearRouteSpec(
    "direct_packed", "q4k", role.role, role.m, role.n, role.k)
  return linear, activation, spec


def attestation_bindings(
    composition: Mapping[str, Any], *, clock_identity: str,
    ) -> dict[str, DirectPackedAttestationBindings]:
  c6 = composition["c6_correctness_evidence"]
  return {
    queue: DirectPackedAttestationBindings(
      queue_mode=queue, workload_identity=c6["workload_identity"],
      input_identity=c6["input_identity"], device_identity=c6["device_identity"],
      software_identity=c6["software_identity"],
      comparator_identity=c6["queue_comparators"][queue],
      clock_identity=clock_identity, required_program_prefix=REQUIRED_PROGRAM_PREFIX,
    ).validate()
    for queue in QUEUE_MODES
  }


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
  qualification_paths = {
    "PM4": Path(config["qualification_pm4"]).resolve(),
    "AQL": Path(config["qualification_aql"]).resolve(),
  }
  if qualification_paths["PM4"] == qualification_paths["AQL"] or \
     not all(path.is_file() for path in qualification_paths.values()):
    raise ValueError("both distinct frozen qualification files must preexist")
  linear, activation, spec = build_attn_qo_direct_packed_objects(family, composition)
  bindings = attestation_bindings(composition, clock_identity=clock_identity)
  candidate = make_frozen_staged_candidate_runner(
    role_spec=family.binding.role_spec, frozen_bundle=config["frozen_bundle"],
    staged_family_manifest=config["staged_family_manifest"],
    runtime_canary_by_queue=composition["runtime_canary_by_queue"])
  fallback, _attestor = make_production_direct_packed_attested_runner(
    linear=linear, input_tensor=activation, route_spec=spec,
    qualification_paths_by_queue=qualification_paths,
    bindings_by_queue=bindings, clock_ns=clock_ns)
  return QueueTimingRunners(candidate, fallback).validate(queue_mode)


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
  "REQUIRED_PROGRAM_PREFIX", "attestation_bindings",
  "attn_qo_c8_runner_factory", "build_attn_qo_direct_packed_objects",
  "qualify_one_queue", "validate_live_software",
]
