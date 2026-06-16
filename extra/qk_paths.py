from __future__ import annotations

import pathlib

def portable_path(path:pathlib.Path, repo:pathlib.Path) -> str:
  try:
    return str(path.resolve().relative_to(repo.resolve()))
  except ValueError:
    return str(path)
