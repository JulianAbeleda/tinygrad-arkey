"""Route-fire checker (machine-readable): is the candidate decode kernel present in the captured TinyJit graph,
and is the slice route absent? Standalone CLI + importable check_route_fire(captured).

  DEV=AMD JIT=1 [DECODE_ATTN_KV_IDENTITY=1] PYTHONPATH=. .venv/bin/python extra/qk_decode_route_fire_check.py
See docs/decode-machine-search-readiness-package-scope-20260623.md (P2)."""
import json
from extra.qk_decode_search_gate import check_route_fire, _setup_model, capture_decode  # noqa

if __name__ == "__main__":
  m, tok = _setup_model()
  _, captured, *_ = capture_decode(m, tok)
  r = check_route_fire(captured); r["program_node_names"] = f"<{len(r['program_node_names'])} kernels>"
  print("ROUTE_FIRE " + json.dumps(r))
