"""Run one bounded experiment callback in an isolated process.

This module is hardware-agnostic.  It provides the hard timeout and process
group cleanup that a GPU-facing adapter must have; it does not allocate or
dispatch anything itself.
"""
from __future__ import annotations

import contextlib, io, multiprocessing, os, signal, time
from dataclasses import dataclass
from typing import Any, Callable


class _BoundedWriter(io.TextIOBase):
  def __init__(self, limit: int): self.limit, self.parts, self.size = limit, [], 0
  def write(self, value: str) -> int:
    if not isinstance(value, str): value = str(value)
    remaining = max(0, self.limit - self.size)
    if remaining:
      piece = value[:remaining]
      self.parts.append(piece); self.size += len(piece)
    return len(value)
  def flush(self) -> None: pass
  def text(self) -> str: return "".join(self.parts)


@dataclass(frozen=True)
class IsolatedResult:
  status: str
  result: Any = None
  error: str | None = None
  stdout: str = ""
  stderr: str = ""
  elapsed_seconds: float = 0.0
  timed_out: bool = False
  # Typed diagnostics carried by an exception raised after a child has
  # already captured useful execution evidence (for example a PM4 census
  # followed by a delayed queue synchronize failure).  This is intentionally
  # limited to plain mappings so the parent never receives live runtime
  # objects.
  evidence: dict[str, Any] | None = None


def _child(queue: Any, callback: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], limit: int) -> None:
  # The child is its own process group so descendants can be terminated too.
  with contextlib.suppress(OSError): os.setsid()
  out, err = _BoundedWriter(limit), _BoundedWriter(limit)
  started = time.monotonic()
  try:
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
      value = callback(*args, **kwargs)
    row = {"status": "passed", "result": value, "error": None}
  except BaseException as exc:  # serialize failure; never let the child hang on an exception
    evidence = {}
    for name in ("pm4_dispatch_census", "aql_packet_census", "runtime_preconstruction"):
      value = getattr(exc, name, None)
      if isinstance(value, dict): evidence[name] = value
    row = {"status": "failed", "result": None, "error": f"{type(exc).__name__}: {exc}",
           "evidence": evidence or None}
  row.update(stdout=out.text(), stderr=err.text(), elapsed_seconds=time.monotonic() - started)
  with contextlib.suppress(Exception): queue.put(row)


def _stop_group(proc: multiprocessing.Process, grace_seconds: float) -> None:
  if proc.pid is None: return
  with contextlib.suppress(ProcessLookupError, PermissionError): os.killpg(proc.pid, signal.SIGTERM)
  proc.join(max(0.0, grace_seconds))
  if proc.is_alive():
    with contextlib.suppress(ProcessLookupError, PermissionError): os.killpg(proc.pid, signal.SIGKILL)
    proc.join(max(0.0, grace_seconds))
  if proc.is_alive(): proc.kill(); proc.join(max(0.0, grace_seconds))


def run_isolated(callback: Callable[..., Any], *, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None,
                 timeout_seconds: float = 10.0, terminate_grace_seconds: float = 0.25,
                 output_limit: int = 65536, start_method: str | None = None) -> IsolatedResult:
  """Run ``callback`` with a hard deadline and fail closed on timeout/no result.

  ``start_method`` selects the multiprocessing context.  The default keeps the
  historic behaviour (``fork`` on posix).  Callers that dispatch to a GPU MUST
  pass ``"spawn"`` so the child initializes the device FRESH -- a fork()ed child
  inherits the parent's (unusable) GPU/driver fds and the AMD/ROCm runtime fails
  to initialize there (``OSError: [Errno 9] Bad file descriptor``).  Under spawn
  the callback and every arg are pickled, so they must be picklable (module-level
  callables + picklable args, never closures/lambdas or live runtime objects).
  """
  if not callable(callback): raise TypeError("callback must be callable")
  if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0: raise ValueError("timeout must be positive")
  if not isinstance(output_limit, int) or output_limit <= 0: raise ValueError("output_limit must be positive")
  kwargs = {} if kwargs is None else dict(kwargs)
  ctx = multiprocessing.get_context(start_method or ("fork" if os.name == "posix" else "spawn"))
  queue = ctx.Queue(maxsize=1)
  proc = ctx.Process(target=_child, args=(queue, callback, tuple(args), kwargs, output_limit), daemon=True)
  started = time.monotonic(); proc.start(); proc.join(float(timeout_seconds))
  elapsed = time.monotonic() - started
  if proc.is_alive():
    _stop_group(proc, terminate_grace_seconds)
    return IsolatedResult("timed_out", error="isolated callback exceeded hard timeout", elapsed_seconds=elapsed, timed_out=True)
  try: row = queue.get_nowait()
  except Exception:
    return IsolatedResult("failed", error="isolated callback exited without a result", elapsed_seconds=elapsed)
  return IsolatedResult(row.get("status", "failed"), row.get("result"), row.get("error"),
                        row.get("stdout", ""), row.get("stderr", ""), row.get("elapsed_seconds", elapsed),
                        evidence=row.get("evidence"))


__all__ = ["IsolatedResult", "run_isolated"]
