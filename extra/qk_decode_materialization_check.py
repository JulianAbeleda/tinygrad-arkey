"""Materialization / buffer-identity ABI checker: does E_49152 (full-MAXC K/V materialization) return, and are the
tile inputs buffer-identity (not sliced views)? Standalone CLI + importable check_materialization(captured).

  DEV=AMD JIT=1 [DECODE_ATTN_KV_IDENTITY=1] PYTHONPATH=. .venv/bin/python extra/qk_decode_materialization_check.py
See docs/decode-machine-search-readiness-package-scope-20260623.md (P3). Enforces principle #12 (buffer-identity ABI)."""
import json
from extra.qk_decode_search_gate import check_materialization, _setup_model, capture_decode  # noqa

if __name__ == "__main__":
  m, tok = _setup_model()
  _, captured, *_ = capture_decode(m, tok)
  print("MATERIALIZATION " + json.dumps(check_materialization(captured)))
