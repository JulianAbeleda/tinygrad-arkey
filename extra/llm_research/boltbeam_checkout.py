"""Find the BoltBeam checkout that research code imports BoltBeam modules from.

Research only. Nothing under tinygrad/ imports this module, and a test enforces
that. Production reads BoltBeam's exported files and never needs a checkout.

The first place that has a boltbeam package wins:
  1. BOLTBEAM_ROOT, when it is set. Nothing else is tried, so a wrong value
     fails instead of silently using some other BoltBeam.
  2. A boltbeam package Python can already import (installed, or on PYTHONPATH).
  3. The sibling checkout: the directory next to this repository named BoltBeam.
When no place has it, or the checkout found lacks a module, ImportError names
every place tried.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

SIBLING_ROOT = Path(__file__).resolve().parents[3] / "BoltBeam"


def _has_package(root: Path) -> bool:
  return (root / "boltbeam" / "__init__.py").is_file()


def _message(modules: tuple[str, ...], tried: list[str], problem: str) -> str:
  return "\n".join([f"BoltBeam module(s) {', '.join(modules)} unavailable: {problem}",
                    "Research code under extra/llm_research needs a BoltBeam checkout; production tinygrad/ does not.",
                    "Places tried, in order:", *(f"  - {place}" for place in tried),
                    f"Fix: set BOLTBEAM_ROOT to a BoltBeam checkout, or clone BoltBeam at {SIBLING_ROOT}."])


def require_boltbeam(*modules: str) -> Path:
  """Make each BoltBeam module importable and return the checkout root, or raise ImportError."""
  tried: list[str] = []
  env_root = os.environ.get("BOLTBEAM_ROOT")
  if env_root:
    root = Path(env_root).expanduser().resolve()
    tried.append(f"BOLTBEAM_ROOT={env_root}")
    if not _has_package(root): raise ImportError(_message(modules, tried, f"{root} has no boltbeam/__init__.py"))
    loaded = sys.modules.get("boltbeam")
    loaded_file = getattr(loaded, "__file__", None)
    if loaded_file is not None and Path(loaded_file).resolve().parents[1] != root:
      raise ImportError(_message(modules, tried, f"boltbeam is already imported from {Path(loaded_file).resolve().parents[1]}"))
    if str(root) not in sys.path: sys.path.insert(0, str(root))
  else:
    tried.append("BOLTBEAM_ROOT (unset)")
    spec = importlib.util.find_spec("boltbeam")
    if spec is not None and spec.origin is not None:
      root = Path(spec.origin).resolve().parents[1]
      tried.append(f"importable boltbeam at {root}")
    else:
      tried.append("importable boltbeam (none)")
      tried.append(f"sibling checkout {SIBLING_ROOT}")
      if not _has_package(SIBLING_ROOT): raise ImportError(_message(modules, tried, "no place has a boltbeam package"))
      root = SIBLING_ROOT
      # Appended, so this repository's own top-level packages keep priority.
      if str(root) not in sys.path: sys.path.append(str(root))
  for module in modules:
    try: found = importlib.util.find_spec(module) is not None
    except ImportError as exc: raise ImportError(_message(modules, tried, f"importing {module} from {root} failed: {exc}")) from exc
    if not found: raise ImportError(_message(modules, tried, f"the checkout at {root} has no {module}; update it"))
  return root


__all__ = ["SIBLING_ROOT", "require_boltbeam"]
