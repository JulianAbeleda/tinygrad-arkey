"""Fail-closed dual-queue device and memory authority for staged C7.

The live collector runs PM4 and AQL DeviceFacts scans in separate spawned
children.  This module joins those queue-specific observations only after
proving that their queue-neutral hardware, allocator implementation, and
allocation granularity agree.  The shared admitted budget is the conservative
minimum of the two production scanned-memory budgets.

Importing this module is CPU-only.  Device construction occurs only in the
explicit ``collect`` command's spawned workers.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

from tinygrad.llm.admission import scanned_device_memory_budget
from tinygrad.llm.device_facts import DeviceFacts

from extra.qk.mmq_frozen_staged_family import QUEUE_MODES
from extra.qk.mmq_staged_c7_c8_contract import staged_c7_budget_identity


SCHEMA = "tinygrad.mmq_q4k_q8_1.staged_c7_dual_queue_authority.v1"
SOFTWARE_SEMANTICS = (
  "clean tinygrad repository commit and tree executing both DeviceFacts scans "
  "and the subsequent staged C7 collector"
)
BUDGET_POLICY = "minimum_of_pm4_aql_scanned_device_memory_budget"
BUDGET_PROVENANCE = (
  "minimum of independent PM4/AQL live free VRAM minus each scan's live "
  "occupied-byte disturbance reserve, rounded to their equal scanned allocator granularity"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
IsolatedRunner = Callable[..., Any]


def _canonical(value: Any) -> bytes:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value: Any) -> str:
  return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _json_normalized(value: Any) -> Any:
  """Return the exact JSON transport shape used by persisted evidence."""
  return json.loads(_canonical(value))


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    while data := handle.read(8 << 20): digest.update(data)
  return "sha256:" + digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
  if set(value) != expected:
    raise ValueError(
      f"{label} fields differ: expected {sorted(expected)!r}, got {sorted(value)!r}")


def _nonempty(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value:
    raise ValueError(f"{label} must be a non-empty string")
  return value


def _positive(value: Any, label: str) -> int:
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise ValueError(f"{label} must be a positive integer")
  return value


def _git(*args: str) -> str:
  try:
    return subprocess.run(
      ("git", *args), cwd=REPO_ROOT, check=True, text=True,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
  except (OSError, subprocess.SubprocessError) as exc:
    raise ValueError(f"cannot establish C7 software revision: {exc}") from exc


def collect_clean_repository_provenance() -> dict[str, Any]:
  """Return the exact clean repository revision used by scans and collection."""
  if _git("status", "--porcelain", "--untracked-files=all"):
    raise ValueError("C7 authority collection requires a clean repository")
  return {
    "vcs": "git", "commit": _git("rev-parse", "HEAD"),
    "tree": _git("rev-parse", "HEAD^{tree}"), "clean": True,
  }


def _validate_repository(value: Any, label: str) -> dict[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} must be a mapping")
  _exact_keys(value, {"vcs", "commit", "tree", "clean"}, label)
  if value.get("vcs") != "git" or value.get("clean") is not True:
    raise ValueError(f"{label} must identify one clean git revision")
  commit, tree = value.get("commit"), value.get("tree")
  for name, digest in (("commit", commit), ("tree", tree)):
    if not isinstance(digest, str) or len(digest) not in (40, 64) or \
       any(char not in "0123456789abcdef" for char in digest):
      raise ValueError(f"{label}.{name} must be a lowercase git digest")
  return {"vcs": "git", "commit": commit, "tree": tree, "clean": True}


def _class_source(value: type) -> dict[str, str]:
  source = inspect.getsourcefile(value)
  if source is None:
    raise ValueError(f"allocator implementation class {value!r} has no source file")
  resolved = Path(source).resolve()
  try: relative = resolved.relative_to(REPO_ROOT)
  except ValueError as exc:
    raise ValueError("allocator implementation source is outside the tinygrad repository") from exc
  if not resolved.is_file():
    raise ValueError("allocator implementation source file is missing")
  return {
    "class": f"{value.__module__}.{value.__qualname__}",
    "source": relative.as_posix(), "source_sha256": _sha256(resolved),
  }


def _allocator_implementation(opened: Any) -> dict[str, Any]:
  allocator, interface = getattr(opened, "allocator", None), getattr(opened, "iface", None)
  if allocator is None or interface is None:
    raise ValueError("selected AMD device lacks allocator or interface implementation")
  return {
    "allocator": _class_source(type(allocator)),
    "interface": _class_source(type(interface)),
  }


def _validate_implementation(value: Any, label: str, *, verify_sources: bool) -> dict[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{label} must be a mapping")
  _exact_keys(value, {"allocator", "interface"}, label)
  normalized: dict[str, Any] = {}
  for kind in ("allocator", "interface"):
    row = value.get(kind)
    if not isinstance(row, Mapping):
      raise ValueError(f"{label}.{kind} must be a mapping")
    _exact_keys(row, {"class", "source", "source_sha256"}, f"{label}.{kind}")
    class_name = _nonempty(row.get("class"), f"{label}.{kind}.class")
    source = _nonempty(row.get("source"), f"{label}.{kind}.source")
    source_hash = _nonempty(row.get("source_sha256"), f"{label}.{kind}.source_sha256")
    if not source_hash.startswith("sha256:") or len(source_hash) != 71:
      raise ValueError(f"{label}.{kind}.source_sha256 is malformed")
    path = (REPO_ROOT / source).resolve()
    try: path.relative_to(REPO_ROOT)
    except ValueError as exc:
      raise ValueError(f"{label}.{kind}.source escapes the repository") from exc
    if verify_sources and (not path.is_file() or _sha256(path) != source_hash):
      raise ValueError(f"{label}.{kind} source differs from the authority snapshot")
    normalized[kind] = {
      "class": class_name, "source": source, "source_sha256": source_hash,
    }
  return normalized


def _queue_neutral_hardware(facts: DeviceFacts) -> dict[str, Any]:
  hardware = facts.canonical_hardware()
  if set(hardware) != {
      "schema_version", "selected_device", "backend", "architecture",
      "queue_mode", "total_vram_bytes", "capabilities"}:
    raise ValueError("DeviceFacts canonical hardware fields differ from the C7 authority contract")
  return _json_normalized({
    key: value for key, value in hardware.items() if key != "queue_mode"})


def _scan_worker(selected_device: str, queue_mode: str) -> dict[str, Any]:
  """Spawn-only live scan entry.  The parent never opens a Device."""
  os.environ.update({
    "AMD_AQL": "1" if queue_mode == "AQL" else "0",
    "DEV": selected_device,
  })
  from tinygrad.device import Device
  from tinygrad.llm.device_facts import scan_device_facts
  facts = scan_device_facts(selected_device)
  opened = Device[facts.selected_device]
  return {
    "queue_mode": queue_mode, "facts": facts.to_json(),
    "allocator_implementation": _allocator_implementation(opened),
  }


def _normalize_observation(value: Any, queue_mode: str, *,
                           verify_sources: bool) -> dict[str, Any]:
  if not isinstance(value, Mapping):
    raise ValueError(f"{queue_mode} scan result must be a mapping")
  _exact_keys(value, {"queue_mode", "facts", "allocator_implementation"},
              f"{queue_mode} scan result")
  if value.get("queue_mode") != queue_mode:
    raise ValueError(f"{queue_mode} scan request identity differs")
  facts_raw = value.get("facts")
  if not isinstance(facts_raw, Mapping):
    raise ValueError(f"{queue_mode} DeviceFacts snapshot must be a mapping")
  facts = DeviceFacts.from_json(facts_raw)
  if facts.state != "ok" or facts.backend != "AMD" or facts.queue_mode != queue_mode:
    raise ValueError(f"{queue_mode} DeviceFacts do not attest a healthy effective AMD queue")
  granularity = _positive(
    facts.capabilities.global_allocation_granularity,
    f"{queue_mode} global allocation granularity")
  budget = scanned_device_memory_budget(facts)
  budget_row = budget.to_dict()
  admitted = _positive(budget.admitted_bytes, f"{queue_mode} admitted budget")
  if budget_row["admitted_bytes"] != admitted:
    raise ValueError(f"{queue_mode} scanned budget is internally inconsistent")
  implementation = _validate_implementation(
    value.get("allocator_implementation"),
    f"{queue_mode}.allocator_implementation", verify_sources=verify_sources)
  return {
    "queue_mode": queue_mode, "facts": _json_normalized(facts.to_json()),
    "queue_neutral_hardware": _queue_neutral_hardware(facts),
    "allocation_granularity_bytes": granularity,
    "budget": budget_row, "allocator_implementation": implementation,
  }


def build_staged_c7_authority_snapshot(
    observations: Mapping[str, Any], *, repository: Mapping[str, Any],
    verify_sources: bool = True,
    ) -> dict[str, Any]:
  """Join already isolated queue observations into one C7 authority snapshot."""
  if not isinstance(observations, Mapping) or set(observations) != set(QUEUE_MODES):
    raise ValueError(f"C7 authority observations must contain exactly {QUEUE_MODES!r}")
  repo = _validate_repository(repository, "repository")
  queues = {
    queue: _normalize_observation(observations[queue], queue, verify_sources=verify_sources)
    for queue in QUEUE_MODES
  }
  granularity = queues["PM4"]["allocation_granularity_bytes"]
  if queues["AQL"]["allocation_granularity_bytes"] != granularity:
    raise ValueError("PM4/AQL global allocation granularity differs")
  neutral = queues["PM4"]["queue_neutral_hardware"]
  if queues["AQL"]["queue_neutral_hardware"] != neutral:
    raise ValueError("PM4/AQL queue-neutral hardware differs")
  implementation = queues["PM4"]["allocator_implementation"]
  if queues["AQL"]["allocator_implementation"] != implementation:
    raise ValueError("PM4/AQL allocator implementation differs")

  software = {
    "semantics": SOFTWARE_SEMANTICS, "repository": repo,
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
  }
  software_identity = _identity(software)
  device_identity = _identity({
    "schema": "tinygrad.staged_c7.queue_neutral_device.v1",
    "hardware": neutral,
  })
  allocator_identity = _identity({
    "schema": "tinygrad.staged_c7.allocator_implementation.v1",
    "repository": repo, "implementation": implementation,
  })
  admitted = min(_positive(
    queues[queue]["budget"]["admitted_bytes"], f"{queue} admitted budget")
    for queue in QUEUE_MODES)
  authority = {
    "device_identity": device_identity,
    "software_identity": software_identity,
    "allocator_identity": allocator_identity,
    "allocation_granularity_bytes": granularity,
  }
  authority["budget_identity"] = staged_c7_budget_identity(
    **authority, admitted_budget_bytes=admitted,
    budget_provenance=BUDGET_PROVENANCE)
  payload = {
    "schema": SCHEMA, "status": "PASS",
    "selected_device": neutral["selected_device"],
    "queue_neutral_hardware": neutral, "device_identity": device_identity,
    "software": software, "software_identity": software_identity,
    "allocator_implementation": implementation,
    "allocator_identity": allocator_identity,
    "allocation_granularity_bytes": granularity,
    "queue_observations": queues,
    "budget": {
      "policy": BUDGET_POLICY, "provenance": BUDGET_PROVENANCE,
      "admitted_bytes": admitted,
      "queue_budgets": {
        queue: queues[queue]["budget"] for queue in QUEUE_MODES
      },
    },
    "memory_authority": authority,
  }
  return {**payload, "snapshot_identity": _identity(payload)}


def validate_staged_c7_authority_snapshot(
    value: Any, *, verify_current_software: bool = True,
    verify_sources: bool = True,
    repository_probe: Callable[[], Mapping[str, Any]] = collect_clean_repository_provenance,
    ) -> dict[str, Any]:
  """Deeply recompute identities, budgets, and the dual-queue join."""
  if not isinstance(value, Mapping):
    raise ValueError("C7 authority snapshot must be a mapping")
  _exact_keys(value, {
    "schema", "status", "selected_device", "queue_neutral_hardware",
    "device_identity", "software", "software_identity",
    "allocator_implementation", "allocator_identity",
    "allocation_granularity_bytes", "queue_observations", "budget",
    "memory_authority", "snapshot_identity",
  }, "C7 authority snapshot")
  if value.get("schema") != SCHEMA or value.get("status") != "PASS":
    raise ValueError("C7 authority snapshot schema or PASS state differs")
  software = value.get("software")
  if not isinstance(software, Mapping):
    raise ValueError("C7 authority software payload must be a mapping")
  _exact_keys(software, {
    "semantics", "repository", "python_implementation", "python_version",
  }, "software")
  if software.get("semantics") != SOFTWARE_SEMANTICS:
    raise ValueError("C7 authority software identity semantics differ")
  repository = _validate_repository(software.get("repository"), "software.repository")
  if verify_current_software:
    current = _validate_repository(repository_probe(), "current repository")
    if current != repository:
      raise ValueError("current clean repository differs from the C7 authority snapshot")
    if software.get("python_implementation") != platform.python_implementation() or \
       software.get("python_version") != platform.python_version():
      raise ValueError("current Python runtime differs from the C7 authority snapshot")
  rebuilt = build_staged_c7_authority_snapshot(
    {
      queue: {
        "queue_mode": value["queue_observations"][queue]["queue_mode"],
        "facts": value["queue_observations"][queue]["facts"],
        "allocator_implementation":
          value["queue_observations"][queue]["allocator_implementation"],
      }
      for queue in QUEUE_MODES
    },
    repository=repository, verify_sources=verify_sources)
  if rebuilt != dict(value):
    raise ValueError("C7 authority snapshot content or derived identity differs")
  return rebuilt


def collect_staged_c7_authority_snapshot(
    *, selected_device: str = "AMD", timeout_seconds: float = 60.0,
    isolated_runner: IsolatedRunner | None = None,
    repository_probe: Callable[[], Mapping[str, Any]] = collect_clean_repository_provenance,
    ) -> dict[str, Any]:
  """Run independent spawned PM4/AQL scans and build their shared authority."""
  selected_device = _nonempty(selected_device, "selected_device")
  if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or \
     timeout_seconds <= 0:
    raise ValueError("timeout_seconds must be positive")
  repository = dict(repository_probe())
  _validate_repository(repository, "repository")
  if isolated_runner is None:
    from tinygrad.runtime.process_isolated import run_isolated
    isolated_runner = run_isolated
  observations = {}
  for queue in QUEUE_MODES:
    isolated = isolated_runner(
      _scan_worker, args=(selected_device, queue),
      timeout_seconds=float(timeout_seconds), start_method="spawn")
    if getattr(isolated, "status", None) != "passed" or \
       not isinstance(getattr(isolated, "result", None), Mapping):
      error = getattr(isolated, "error", None) or "isolated scan returned no structured result"
      raise ValueError(f"{queue} C7 authority scan failed: {error}")
    observations[queue] = isolated.result
  repository_after = _validate_repository(
    repository_probe(), "post-scan repository")
  if repository_after != repository:
    raise ValueError("clean repository changed during dual-queue C7 authority scans")
  snapshot = build_staged_c7_authority_snapshot(
    observations, repository=repository, verify_sources=True)
  if snapshot["selected_device"] != selected_device:
    raise ValueError("selected device differs from dual-queue DeviceFacts scans")
  return snapshot


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
  encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(
    prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
  try:
    with os.fdopen(fd, "w") as handle:
      handle.write(encoded)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, path)
  except BaseException:
    try: os.unlink(temporary)
    except FileNotFoundError: pass
    raise


def load_staged_c7_authority_snapshot(
    path: str | Path, *, verify_current_software: bool = True,
    ) -> dict[str, Any]:
  return validate_staged_c7_authority_snapshot(
    json.loads(Path(path).read_text()),
    verify_current_software=verify_current_software)


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)
  collect = sub.add_parser("collect")
  collect.add_argument("--selected-device", default="AMD")
  collect.add_argument("--timeout-seconds", type=float, default=60.0)
  collect.add_argument("--output", type=Path, required=True)
  validate = sub.add_parser("validate")
  validate.add_argument("--input", type=Path, required=True)
  args = parser.parse_args(argv)
  if args.command == "collect":
    result = collect_staged_c7_authority_snapshot(
      selected_device=args.selected_device, timeout_seconds=args.timeout_seconds)
    write_json_atomic(args.output, result)
  else:
    result = load_staged_c7_authority_snapshot(args.input)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
  return 0


if __name__ == "__main__": raise SystemExit(main())


__all__ = [
  "BUDGET_POLICY", "BUDGET_PROVENANCE", "SCHEMA", "SOFTWARE_SEMANTICS",
  "build_staged_c7_authority_snapshot", "collect_clean_repository_provenance",
  "collect_staged_c7_authority_snapshot", "load_staged_c7_authority_snapshot",
  "validate_staged_c7_authority_snapshot", "write_json_atomic",
]
