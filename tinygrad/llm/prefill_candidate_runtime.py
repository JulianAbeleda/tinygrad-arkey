"""Exact runtime decoder for the promoted four-role fp16 prefill candidate set.

The compact checked-in artifact factors the identical schedule shared by all
four searched workloads.  This module expands it to the historical candidate
set shape, verifies every entry and aggregate identity, and constructs only the
typed compiler context required by the production graph-GEMM executor.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import hashlib, json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from tinygrad.codegen.opt.kernel_pipeline import KernelStage1PipelinePlan


COMPACT_SCHEMA = "tinygrad.prefill_wmma_lds_compact.v1"
CANDIDATE_SET_SCHEMA = "boltbeam.full_kernel_candidate_set.v1"
CANDIDATE_SCHEMA = "boltbeam.full_kernel_candidate.v1"
ROUTE_ID = "prefill_wmma_lds_dbuf_generated"
ARTIFACT = Path(__file__).with_name("generated") / "prefill_wmma_lds_dbuf_candidate_set.json"
_PROVENANCE_KEYS = frozenset(("profile", "profile_id", "profiles", "model", "model_id", "model_name",
                              "model_path", "filename", "size_label", "model_size"))


def _semantic_json(value: Any) -> str:
  def clean(item: Any) -> Any:
    if isinstance(item, Mapping):
      return {str(key):clean(child) for key,child in sorted(item.items(), key=lambda pair: str(pair[0]))
              if str(key).lower().replace("-", "_") not in _PROVENANCE_KEYS}
    if isinstance(item, (list, tuple)): return [clean(child) for child in item]
    if item is None or isinstance(item, (str, bool, int, float)): return item
    raise TypeError(f"semantic identity requires JSON values, got {type(item).__name__}")
  return json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _candidate_identity(payload: Mapping[str, Any]) -> str:
  return hashlib.sha256(_semantic_json(payload).encode("ascii")).hexdigest()


def _legacy_candidate_identity(payload: Mapping[str, Any]) -> str:
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()


def canonical_candidate_set_identity(candidate_set: Mapping[str, Any]) -> str:
  if set(candidate_set) != {"schema", "entries"} or candidate_set.get("schema") != CANDIDATE_SET_SCHEMA:
    raise ValueError("candidate set has an unsupported schema or fields")
  entries = candidate_set.get("entries")
  if not isinstance(entries, list) or not entries: raise ValueError("candidate set requires entries")
  canonical = []
  for entry in entries:
    if not isinstance(entry, Mapping) or set(entry) != {"canonical_identity", "payload"} or not isinstance(entry["payload"], Mapping):
      raise ValueError("candidate set contains a malformed entry")
    canonical.append({"canonical_identity":entry["canonical_identity"], "payload":entry["payload"]})
  digest = hashlib.sha256(_semantic_json(sorted(canonical, key=_semantic_json)).encode("ascii")).hexdigest()
  return f"candidate_set:sha256:{digest}"


@dataclass(frozen=True)
class KernelLDSWindow:
  role: str
  base: int
  end: int
  stride_bytes: int

  def __post_init__(self):
    if self.role not in ("A", "B") or any(not isinstance(value, int) or isinstance(value, bool)
                                           for value in (self.base, self.end, self.stride_bytes)):
      raise ValueError("invalid candidate LDS window")
    if self.base < 0 or self.end <= self.base or self.stride_bytes <= 0 or \
       self.base % 16 or self.end % 16 or self.stride_bytes % 16:
      raise ValueError("candidate LDS windows must be non-empty and b128 aligned")


@dataclass(frozen=True)
class KernelTileGeometry:
  tile: tuple[int, int, int]
  waves: tuple[int, int]
  threads: int
  wave_size: int
  lds_windows: tuple[KernelLDSWindow, ...]

  def __post_init__(self):
    if (len(self.tile) != 3 or len(self.waves) != 2 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0
                                                           for x in (*self.tile, *self.waves, self.threads, self.wave_size))):
      raise ValueError("invalid candidate tile geometry")
    if self.threads != self.waves[0] * self.waves[1] * self.wave_size:
      raise ValueError("candidate waves do not account for all threads")
    if len(self.lds_windows) != 2 or tuple(window.role for window in self.lds_windows) != ("A", "B") or \
       self.lds_windows[0].base != 0 or self.lds_windows[0].end != self.lds_windows[1].base:
      raise ValueError("candidate requires contiguous A/B LDS windows")

  @property
  def lds_bytes(self) -> int: return self.lds_windows[-1].end


@dataclass(frozen=True)
class KernelCandidateContext:
  schema_version: str
  canonical_identity: str
  geometry: KernelTileGeometry
  pipeline: KernelStage1PipelinePlan
  packed_weight: None = None

  def __post_init__(self):
    if self.schema_version != CANDIDATE_SCHEMA or len(self.canonical_identity) != 64 or \
       any(char not in "0123456789abcdef" for char in self.canonical_identity):
      raise ValueError("invalid candidate compiler context identity")


@dataclass(frozen=True)
class CandidateSetEntry:
  canonical_identity: str
  payload: Mapping[str, Any]
  def to_json(self) -> dict[str, Any]: return {"canonical_identity":self.canonical_identity, "payload":json.loads(json.dumps(dict(self.payload)))}


@dataclass(frozen=True)
class CandidateSet:
  entries: tuple[CandidateSetEntry, ...]
  schema: str = CANDIDATE_SET_SCHEMA
  def to_json(self) -> dict[str, Any]: return {"schema":self.schema, "entries":[entry.to_json() for entry in self.entries]}


@dataclass(frozen=True)
class CandidateAdmission:
  canonical_identity: str
  normalized_payload: Mapping[str, Any]
  geometry: KernelTileGeometry
  pipeline_plan: KernelStage1PipelinePlan
  active_lds_bytes: int
  context: KernelCandidateContext


@dataclass(frozen=True)
class CandidateRegistry:
  candidate_set: CandidateSet
  admissions: tuple[CandidateAdmission, ...]
  exact_index: Mapping[tuple[Any, ...], CandidateAdmission]

  def get(self, role: str, shape: tuple[int, int, int], target: Mapping[str, Any]) -> CandidateAdmission | None:
    try: key = (role, *shape, target["backend"], target["arch"], target["wave_size"])
    except (KeyError, TypeError): return None
    return self.exact_index.get(key)


def _strict_dict(value: Any, keys: set[str], label: str) -> dict[str, Any]:
  if not isinstance(value, dict) or set(value) != keys: raise ValueError(f"{label} fields drifted")
  return value


@cache
def promoted_candidate_set() -> CandidateSet:
  raw = json.loads(ARTIFACT.read_text())
  _strict_dict(raw, {"schema", "route_id", "candidate_set_identity", "profile", "target", "template", "entries"}, "compact artifact")
  if raw["schema"] != COMPACT_SCHEMA or raw["route_id"] != ROUTE_ID: raise ValueError("compact artifact identity drifted")
  target = _strict_dict(raw["target"], {"backend", "arch", "wave_size"}, "compact target")
  if target != {"backend":"AMD", "arch":"gfx1100", "wave_size":32}: raise ValueError("compact target is unsupported")
  template = _strict_dict(raw["template"], {"schema_version", "dtypes", "layout", "schedule", "static_constraints"}, "candidate template")
  if template["schema_version"] != CANDIDATE_SCHEMA: raise ValueError("candidate schema drifted")
  rows = raw["entries"]
  if not isinstance(rows, list) or len(rows) != 4: raise ValueError("promoted candidate set must contain four rows")
  entries = []
  for row in rows:
    _strict_dict(row, {"role", "shape", "canonical_identity", "legacy_identity"}, "candidate row")
    shape = _strict_dict(row["shape"], {"m", "n", "k"}, "candidate shape")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape.values()):
      raise ValueError("candidate shape must contain positive integers")
    role = row["role"]
    payload = {"schema_version":template["schema_version"],
      "workload":{"profile":raw["profile"], "role":role, "shape":dict(shape),
                  "dtypes":dict(template["dtypes"]), "layout":dict(template["layout"]), "target":dict(target)},
      "schedule":json.loads(json.dumps(template["schedule"])),
      "static_constraints":dict(template["static_constraints"]),
      "applicability":{"exact_shape":True, "profiles":[raw["profile"]], "roles":[role],
                       "targets":[f"{target['backend']}:{target['arch']}:wave{target['wave_size']}"]}}
    if (_candidate_identity(payload), _legacy_candidate_identity(payload)) != (row["canonical_identity"], row["legacy_identity"]):
      raise ValueError(f"candidate identity drifted for {role}")
    entries.append(CandidateSetEntry(row["canonical_identity"], MappingProxyType(payload)))
  result = CandidateSet(tuple(entries))
  if canonical_candidate_set_identity(result.to_json()) != raw["candidate_set_identity"]:
    raise ValueError("promoted candidate-set identity drifted")
  return result


@cache
def promoted_candidate_registry() -> CandidateRegistry:
  candidate_set = promoted_candidate_set()
  admissions, exact = [], {}
  for entry in candidate_set.entries:
    workload, schedule = entry.payload["workload"], entry.payload["schedule"]
    shape, target = workload["shape"], workload["target"]
    tile = tuple(schedule["tile"][axis] for axis in ("m", "n", "k"))
    waves = tuple(schedule["waves"][axis] for axis in ("m", "n"))
    windows = tuple(KernelLDSWindow(role.upper(), *schedule["lds"]["windows"][role], schedule["lds"]["strides"][role])
                    for role in ("a", "b"))
    geometry = KernelTileGeometry(tile, waves, schedule["threads"], target["wave_size"], windows)
    if any(shape[axis] % tile[index] for index,axis in enumerate(("m", "n", "k"))):
      raise ValueError("candidate workload is not exactly tile divisible")
    pipeline = KernelStage1PipelinePlan(schedule["pipeline"]["buffer_count"], geometry.lds_bytes,
                                        schedule["pipeline"]["stage_count"])
    if pipeline.active_lds_bytes > entry.payload["static_constraints"]["max_lds_bytes"]:
      raise ValueError("candidate active LDS exceeds its declared limit")
    context = KernelCandidateContext(entry.payload["schema_version"], entry.canonical_identity, geometry, pipeline)
    admission = CandidateAdmission(entry.canonical_identity, entry.payload, geometry, pipeline,
                                   pipeline.active_lds_bytes, context)
    key = (workload["role"], shape["m"], shape["n"], shape["k"], target["backend"], target["arch"], target["wave_size"])
    if key in exact: raise ValueError("duplicate promoted candidate key")
    exact[key] = admission
    admissions.append(admission)
  return CandidateRegistry(candidate_set, tuple(admissions), MappingProxyType(exact))


def decode_prefill_graph_candidate_set(value: Mapping[str, Any]) -> CandidateRegistry:
  """Decode only the exact promoted set; altered, partial, or foreign sets fail closed."""
  canonical = promoted_candidate_set().to_json()
  if not isinstance(value, Mapping) or value != canonical: raise ValueError("candidate set is not the exact promoted production set")
  if canonical_candidate_set_identity(value) != canonical_candidate_set_identity(canonical):
    raise ValueError("candidate-set identity mismatch")
  return promoted_candidate_registry()


__all__ = ["ARTIFACT", "ROUTE_ID", "CandidateRegistry", "canonical_candidate_set_identity",
           "decode_prefill_graph_candidate_set", "promoted_candidate_registry", "promoted_candidate_set"]
