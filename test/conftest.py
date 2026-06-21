"""pytest test-infra: ensure a host C compiler is available for tinygrad's CPU backend.

tinygrad compiles CPU kernels via `getenv("CC", "clang")` (tinygrad/runtime/support/compiler_cpu.py). On a machine
where the unversioned `clang` is not on PATH but a versioned clang / gcc / cc is, a COLD-cache test run fails with
`FileNotFoundError: 'clang'` (warm-cache runs pass via the kernel cache, which masks it). This conftest points `CC`
at an available compiler so the suite passes from a cold cache on any such machine.

Scope/safety: only acts when `CC` is unset AND `clang` is not found (never overrides an explicit `CC`, never changes
anything when `clang` exists). Affects only the pytest process env; no production-runtime effect.
"""
import os
import shutil

if "CC" not in os.environ and shutil.which("clang") is None:
  for _cc in ("clang-18", "clang-17", "clang-16", "clang-15", "gcc", "cc"):
    if shutil.which(_cc):
      os.environ["CC"] = _cc
      break
