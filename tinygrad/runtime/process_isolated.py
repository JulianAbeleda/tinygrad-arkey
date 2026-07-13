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
    row = {"status": "failed", "result": None, "error": f"{type(exc).__name__}: {exc}"}
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
                 output_limit: int = 65536) -> IsolatedResult:
  """Run ``callback`` with a hard deadline and fail closed on timeout/no result."""
  if not callable(callback): raise TypeError("callback must be callable")
  if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0: raise ValueError("timeout must be positive")
  if not isinstance(output_limit, int) or output_limit <= 0: raise ValueError("output_limit must be positive")
  kwargs = {} if kwargs is None else dict(kwargs)
  ctx = multiprocessing.get_context("fork" if os.name == "posix" else "spawn")
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
                        row.get("stdout", ""), row.get("stderr", ""), row.get("elapsed_seconds", elapsed))


__all__ = ["IsolatedResult", "run_isolated"]
