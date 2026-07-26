"""LR-060: `KernelSpec` -- a common descriptor contract for model-specific custom kernels.

This module is INERT: nothing in the default path imports it, it changes no route selection and no
admission logic (Phase 6 non-goal). It exists to answer one question for the refactor: for each of the
ten fields the scope asks a `KernelSpec` to own or reference, where does that information already live,
and can a descriptor emit the manifest row that `extra/qk/route_manifest.json` already carries for that
route, byte-for-byte on the fields it claims to own?

Six contracts already exist on this branch. A `KernelSpec` that copies their fields instead of holding a
reference to them would recreate exactly the drift LR-060 is supposed to close, so it does not:

  * `tinygrad.codegen.plan.TargetCapabilities` -- the target field.
  * `tinygrad.codegen.opt.Opt` tuple -- the lowering-plan field (`OptimizationPlan`'s own compatibility
    encoding, per that module's docstring; the two prefill/decode route specs already carry this).
  * `tinygrad.codegen.opt.compiler_policies.ResourcePlan` / `PipelinePolicy` -- the resource contract.
  * The route's own dataclass (`Q4KPrefillRouteSpec`, `FlashDecodeAttentionSpec`, ...) -- geometry. These
    are ALREADY typed per-route descriptors; `KernelSpec.geometry` holds one, it does not restate its
    fields.
  * `extra.qk.route_manifest.route(route_id)` -- route identity, shape guards, evidence paths, selector,
    and provenance. This is DATA (`extra/qk/route_manifest.json`), loaded once by the sibling
    `route_manifest.py` loader described in that module's own docstring.

Where the manifest already carries a fact (roles, quant, expected kernel-name globs, evidence paths,
authority gate), `KernelSpec.to_manifest_row()` derives it from the typed fields above and
`check_against_manifest()` proves that derivation agrees with the real `route_manifest.json` entry --
which is the whole point of "the manifest should be generated from these descriptors, not manually
duplicated." Fields the descriptor cannot honestly compute (performance evidence is a bench-artifact
path, not a typed number) are carried as artifact references, not invented values; see
`PerformanceEvidence` below.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Callable

from tinygrad.codegen.opt import Opt
from tinygrad.codegen.opt.compiler_policies import PipelinePolicy, ResourcePlan
from tinygrad.codegen.plan import TargetCapabilities

from extra.qk import route_manifest


# ---- 7/8. correctness / performance evidence --------------------------------------------------------
# Both are reference-only: the manifest already stores paths (`authority_gate`, `promotion_artifacts`),
# never re-derived numbers. A KernelSpec that invented a pass/fail or a tok/s figure here would be
# lying about having re-run evidence it did not re-run. So these hold paths, exactly as the manifest
# does, and `check_against_manifest` proves the paths are the same paths.

@dataclass(frozen=True)
class CorrectnessEvidence:
  """Reference to the gate that proves token/output correctness for this route. Not re-executed here."""
  gate: str                             # extra/qk/route_manifest.json "authority_gate"
  artifacts: tuple[str, ...] = ()        # the subset of "promotion_artifacts" that are correctness evidence


@dataclass(frozen=True)
class PerformanceEvidence:
  """Reference to the benchmark artifact(s)/doc that carry the speed verdict. No number is restated
  here on purpose: a KernelSpec cannot honestly claim a tok/s figure it did not just measure, and the
  scope explicitly warns against claiming final resource/perf facts from host estimates."""
  artifacts: tuple[str, ...] = ()        # bench/*.json or docs/*.md paths ("promotion_artifacts")
  note: str = ""                         # free-text pointer when no structured artifact exists yet


# ---- 9. admission predicate --------------------------------------------------------------------------
# The manifest's shape_guards are heterogeneous data (exact ints, "*", live boolean conditions as
# strings). A fully generic evaluator over all of that is route-unification work (LR-061), which this
# refactor slice must not do (non-goal: do not change route admission). So AdmissionPredicate stays a
# thin, explicit, per-route callable -- but it is built to CHECK ITSELF against the manifest's own
# shape_guards row rather than asserting facts nothing verifies.

@dataclass(frozen=True)
class AdmissionPredicate:
  """A callable geometry check plus the manifest shape_guards row it claims to implement."""
  check: Callable[[], bool]
  shape_guard: dict[str, Any]

  def admits(self) -> bool:
    return bool(self.check())


@dataclass(frozen=True)
class KernelSpec:
  """The ten-field descriptor. Every field either IS a reference to an existing typed contract, or is
  the minimum new glue needed to point at one. See the module docstring for the mapping."""

  # 1. stable route/variant identity -- the exact key into extra/qk/route_manifest.json.
  route_id: str

  # 2. target -- typed target capability, not a bare string.
  target: TargetCapabilities

  # 3. geometry -- the route's OWN typed spec object (Q4KPrefillRouteSpec, FlashDecodeAttentionSpec,
  #    ...). KernelSpec does not restate rows/k/tokens/Hq/Hd/... fields; it holds the object.
  geometry: Any

  # 4. input/output ABI -- role names as the manifest and the runtime call site name them
  #    (tinygrad/llm/{prefill,decode}_routes.py). No repo-wide typed buffer-slot ABI object exists yet
  #    (extra/qk/prefill/execution_bridge_contracts.py types a *different*, research-only dispatch
  #    contract); until one lands, this stays a tuple of role strings, cross-checked against the
  #    manifest's own "roles" list rather than asserted independently.
  abi_roles: tuple[str, ...]

  # 5. lowering plan -- the Opt tuple, i.e. OptimizationPlan's own compatibility encoding (see
  #    tinygrad/codegen/plan.py docstring: "The Opt list is retained verbatim as the compatibility
  #    encoding the scope allows"). Both example routes below already carry this field on their own
  #    geometry dataclass; KernelSpec reads it from there rather than duplicating it.
  lowering_plan: tuple[Opt, ...]

  # 6. resource contract -- PipelinePolicy/ResourcePlan reference. Optional: not every route has a
  #    compiler_policies.py-shaped contract populated yet (see report), so this is allowed to be None
  #    with the omission surfaced by `to_manifest_row`, never silently assumed.
  resource_contract: PipelinePolicy | ResourcePlan | None

  # 7/8. evidence.
  correctness_evidence: CorrectnessEvidence
  performance_evidence: PerformanceEvidence

  # 9. admission predicate.
  admission: AdmissionPredicate

  # 10. emitter/builder -- the actual callable that produces the kernel (e.g.
  #     emit_q4k_packed_prefill_kernel, FlashDecodeAttentionSpec.emit_tile). Stored as a zero-arg
  #     thunk so decode (two kernels) and prefill (one kernel) share the same slot shape.
  emitter: Callable[[], Any]

  # ---- manifest cross-reference -----------------------------------------------------------------

  def manifest_entry(self) -> dict:
    """The real, current route_manifest.json row for this spec's route_id. Raises KeyError if the
    route_id is not (or no longer) in the manifest -- a spec must never invent a manifest identity."""
    return route_manifest.route(self.route_id)

  def to_manifest_row(self) -> dict[str, Any]:
    """Emit the subset of the manifest row this descriptor can honestly derive from its own typed
    fields, rather than the caller hand-copying route_manifest.json. Fields this descriptor does not
    own (status, selector, provenance, env/rollback -- these are route-lifecycle facts recorded by
    promotion, not computed from geometry) are deliberately absent, not guessed at."""
    return {
      "route_id": self.route_id,
      "roles": sorted(self.abi_roles),
      "authority_gate": self.correctness_evidence.gate,
      "promotion_artifacts": tuple(sorted({*self.correctness_evidence.artifacts, *self.performance_evidence.artifacts})),
      "expected_kernel_names": self.emitted_kernel_names(),
    }

  def emitted_kernel_names(self) -> tuple[str, ...]:
    """Call the real emitter/geometry naming, not a guess. Every geometry object in this repo already
    exposes a `kernel_name` (single-kernel routes) or `emitted_kernel_names` (multi-kernel routes)
    property; this reads whichever the geometry actually has."""
    if hasattr(self.geometry, "emitted_kernel_names"):
      return tuple(self.geometry.emitted_kernel_names)
    return (self.geometry.kernel_name,)

  def check_against_manifest(self) -> list[str]:
    """Prove `to_manifest_row()` is consistent with the real manifest entry. Returns an empty list iff
    consistent; a non-empty list of human-readable mismatches otherwise. This is the acceptance test
    the scope asks for in place of "we wrote a dataclass": the manifest is generated DATA, so the
    proof obligation is agreement with that data, not merely well-typed Python."""
    errors: list[str] = []
    try:
      entry = self.manifest_entry()
    except KeyError as exc:
      return [f"route_id {self.route_id!r} is not in route_manifest.json: {exc}"]

    row = self.to_manifest_row()

    manifest_roles = set(entry.get("roles", ()))
    if not set(row["roles"]).issubset(manifest_roles):
      errors.append(f"abi_roles {row['roles']} not a subset of manifest roles {sorted(manifest_roles)}")

    manifest_gate = entry.get("authority_gate", "")
    if row["authority_gate"] != manifest_gate:
      errors.append(f"authority_gate {row['authority_gate']!r} != manifest authority_gate {manifest_gate!r}")

    manifest_artifacts = set(entry.get("promotion_artifacts", ()))
    spec_artifacts = set(row["promotion_artifacts"])
    if not spec_artifacts.issubset(manifest_artifacts):
      errors.append(f"promotion_artifacts {sorted(spec_artifacts)} not a subset of manifest "
                    f"promotion_artifacts {sorted(manifest_artifacts)}")

    manifest_expected = entry.get("expected_kernels", ())
    for name in row["expected_kernel_names"]:
      # Manifest globs are inconsistently terminated (see extra/qk/route_manifest.json: some rows end
      # in "_*", some are bare prefixes with no trailing wildcard at all, e.g. decode_flash_live_split_
      # g4_kvboth's "flash_fused_gmax_combine" vs the emitted "flash_fused_gmax_combine_32_128"). This
      # checker treats every manifest entry as a glob, trying it first verbatim and then with an
      # appended "*", so both conventions are honoured without rewriting the manifest's own dialect.
      if not any(fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(name, pat + "*") for pat in manifest_expected):
        errors.append(f"emitted kernel name {name!r} matches none of manifest expected_kernels {manifest_expected!r}")

    if self.admission.shape_guard not in entry.get("shape_guards", ()):
      errors.append(f"admission shape_guard {self.admission.shape_guard!r} is not one of the manifest's own "
                    f"shape_guards {entry.get('shape_guards', ())!r}")

    return errors
