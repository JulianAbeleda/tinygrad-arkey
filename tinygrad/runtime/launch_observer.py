"""Explicit child-process launch observation seam for HCQ programs.

The observer is inert unless ``TINYGRAD_LAUNCH_SIDECAR`` is set.  It is a
producer-side module with no device imports; the HCQ runtime supplies launch
geometry and code-object bytes, while this module owns atomic serialization.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import pathlib
import tempfile
import time
from dataclasses import dataclass
from typing import Any

SCHEMA = "tinygrad.kfd_launch_sidecar.v1"


def _validate_sha256(value: str, field: str) -> str:
  if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
    raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")
  return value


def _sha256(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def _dim3(value: tuple[int, int, int], field: str) -> list[int]:
  if len(value) != 3 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in value):
    raise ValueError(f"{field} must contain three positive integers")
  return [int(x) for x in value]


@dataclass(frozen=True)
class _Pending:
  program_id: str | int
  source_sha256: str
  binary_sha256: str
  grid: list[int]
  workgroup: list[int]
  submit_ns: int
  dispatch_id: str | int
  queue_id: str | int | None
  launch_order: int


class LaunchObserver:
  """Atomic sidecar writer used only when explicitly configured."""

  def __init__(self, path: str, *, candidate_id: str, source_sha256: str, source_commit: str,
               source_tree_sha256: str, target_id: str, runtime_id: str, run_id: str,
               attempt_id: str, sync: bool, flush_every: int = 64, binary_dir: str | None = None,
               clock_ns=time.monotonic_ns):
    if not path or not candidate_id or not source_sha256 or not source_commit or not source_tree_sha256:
      raise ValueError("sidecar requires path, candidate, and source identity")
    self.path = pathlib.Path(path).expanduser().resolve()
    self.candidate_id, self.source_sha256 = candidate_id, _validate_sha256(source_sha256, "source_sha256")
    self.source_commit, self.source_tree_sha256 = source_commit, source_tree_sha256
    self.target_id, self.runtime_id = target_id, runtime_id
    self.run_id, self.attempt_id, self.sync, self.clock_ns = run_id, attempt_id, sync, clock_ns
    if flush_every < 1: raise ValueError("flush_every must be positive")
    self.flush_every, self._since_flush = flush_every, 0
    self.binary_dir = pathlib.Path(binary_dir).expanduser().resolve() if binary_dir else None
    if self.binary_dir is not None: self.binary_dir.mkdir(parents=True, exist_ok=True)
    self.records: list[dict[str, Any]] = []
    self.pending: dict[str, _Pending] = {}
    self._launch_order = 0
    self.path.parent.mkdir(parents=True, exist_ok=True)
    atexit.register(self.flush)

  @classmethod
  def from_env(cls) -> "LaunchObserver | None":
    path = os.environ.get("TINYGRAD_LAUNCH_SIDECAR")
    if not path:
      return None
    required = {
      "candidate_id": os.environ.get("TINYGRAD_OBSERVATION_CANDIDATE_ID"),
      "source_sha256": os.environ.get("TINYGRAD_OBSERVATION_SOURCE_SHA256"),
      "source_commit": os.environ.get("TINYGRAD_OBSERVATION_SOURCE_COMMIT"),
      "source_tree_sha256": os.environ.get("TINYGRAD_OBSERVATION_SOURCE_TREE_SHA256"),
      "target_id": os.environ.get("TINYGRAD_OBSERVATION_TARGET_ID", "unknown"),
      "runtime_id": os.environ.get("TINYGRAD_OBSERVATION_RUNTIME_ID", "tinygrad"),
      "run_id": os.environ.get("TINYGRAD_OBSERVATION_RUN_ID", "run-unknown"),
      "attempt_id": os.environ.get("TINYGRAD_OBSERVATION_ATTEMPT_ID", "attempt-unknown"),
    }
    missing = [name for name in ("candidate_id", "source_sha256", "source_commit", "source_tree_sha256") if not required[name]]
    if missing:
      raise ValueError(f"TINYGRAD_LAUNCH_SIDECAR missing identity environment: {', '.join(missing)}")
    try: flush_every = int(os.environ.get("TINYGRAD_OBSERVATION_FLUSH_EVERY", "64"))
    except ValueError as e: raise ValueError("TINYGRAD_OBSERVATION_FLUSH_EVERY must be an integer") from e
    return cls(path, **required, sync=os.environ.get("TINYGRAD_OBSERVATION_SYNC", "0") == "1", flush_every=flush_every,
               binary_dir=os.environ.get("TINYGRAD_OBSERVATION_BINARY_DIR"))

  @property
  def enabled(self) -> bool: return True

  def submit(self, *, program_id: str | int, source_sha256: str, binary_sha256: str,
             grid: tuple[int, int, int], workgroup: tuple[int, int, int],
             dispatch_id: str | int, queue_id: str | int | None = None, binary: bytes | None = None) -> str:
    if self.binary_dir is not None and binary:
      destination = self.binary_dir / f"{binary_sha256}.hsaco"
      if not destination.exists():
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=self.binary_dir)
        try:
          with os.fdopen(fd, "wb") as handle:
            handle.write(binary); handle.flush(); os.fsync(handle.fileno())
          os.replace(temporary, destination)
        except BaseException:
          try: os.unlink(temporary)
          except FileNotFoundError: pass
          raise
    self._launch_order += 1
    token = f"{self.run_id}:{self.attempt_id}:{self._launch_order}"
    self.pending[token] = _Pending(program_id, source_sha256, binary_sha256, _dim3(grid, "grid"),
                                   _dim3(workgroup, "workgroup"), self.clock_ns(), dispatch_id, queue_id,
                                   self._launch_order)
    return token

  def complete(self, token: str, *, counters: dict[str, int | float] | None = None) -> None:
    pending = self.pending.pop(token)
    complete_ns = self.clock_ns()
    if complete_ns < pending.submit_ns:
      raise ValueError("completion clock moved backwards")
    row: dict[str, Any] = {
      "candidate_id": self.candidate_id, "program_id": pending.program_id,
      "source_sha256": pending.source_sha256, "binary_sha256": pending.binary_sha256,
      "grid": pending.grid, "workgroup": pending.workgroup, "submit_ns": pending.submit_ns,
      "complete_ns": complete_ns, "dispatch_id": pending.dispatch_id, "queue_id": pending.queue_id,
      "launch_order": pending.launch_order,
    }
    if counters is not None: row["counters"] = dict(counters)
    self.records.append(row)
    self._since_flush += 1
    if self._since_flush >= self.flush_every: self.flush()

  def abort(self, token: str) -> None:
    self.pending.pop(token, None)

  def flush(self) -> None:
    payload = {
      "schema": SCHEMA, "candidate_id": self.candidate_id,
      "source_commit": self.source_commit, "source_tree_sha256": self.source_tree_sha256,
      "target_id": self.target_id, "runtime_id": self.runtime_id,
      "run_id": self.run_id, "attempt_id": self.attempt_id, "records": list(self.records),
    }
    fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
    try:
      with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
      os.replace(temporary, self.path)
      self._since_flush = 0
    except BaseException:
      try: os.unlink(temporary)
      except FileNotFoundError: pass
      raise


_OBSERVER: LaunchObserver | None | bool = False


def get_launch_observer() -> LaunchObserver | None:
  global _OBSERVER
  if _OBSERVER is False:
    _OBSERVER = LaunchObserver.from_env()
  return _OBSERVER if isinstance(_OBSERVER, LaunchObserver) else None


def reset_launch_observer_for_tests() -> None:
  global _OBSERVER
  _OBSERVER = False


__all__ = ["LaunchObserver", "SCHEMA", "get_launch_observer", "reset_launch_observer_for_tests"]
