from __future__ import annotations

import pathlib

# Single source of truth for the default Qwen3-8B-Q4_K_M weights path used by the fork's decode/eval
# scripts (the machine-local test fixture; the real CLI takes -m). Centralized here so the literal is not
# spread across scripts. qk_paths imports only pathlib, so it is safe to import before the env-ordering
# barrier (`set DEV/JIT/QK flags before `from tinygrad import ...``).
DEFAULT_MODEL_GGUF = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
# 14B fixture used by the large-shape / role-attribution probes (same rationale: one source, not spread).
DEFAULT_MODEL_14B_GGUF = "/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf"

def portable_path(path:pathlib.Path, repo:pathlib.Path) -> str:
  try:
    return str(path.resolve().relative_to(repo.resolve()))
  except ValueError:
    return str(path)
