#!/usr/bin/env python3
"""Phase R0 boundary audit (see docs/tinygrad-runtime-client-separation-roadmap-20260630.md).

Documents the tinygrad runtime server's boundary surface and checks for runtime/client leakage. It inspects the
server implementation in tinygrad/llm/cli.py (route literals + RuntimeState capabilities) and, if a server is
running, probes it live. Writes:

  bench/tinygrad-runtime-boundary/latest.json
  bench/tinygrad-runtime-boundary/summary.md

Verdicts: R0_PASS_BOUNDARY_PINNED / R0_BLOCKED_SERVER_ENTRYPOINT_AMBIGUOUS
"""
from __future__ import annotations
import os, sys, json, inspect, pathlib, argparse, urllib.request, urllib.error
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import tinygrad.llm.cli as cli

OUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "bench" / "tinygrad-runtime-boundary"

def _handler_source() -> str:
  return inspect.getsource(cli.Handler)

def static_audit() -> dict:
  src = _handler_source()
  state_methods = {m for m in dir(cli.RuntimeState) if not m.startswith("__")}
  # each check: (id, description, predicate, owner) — owner documents which side of the boundary owns it
  checks = [
    ("v1_models",        "GET /v1/models route exists",                 '"/v1/models"' in src,                 "runtime"),
    ("v1_chat",          "POST /v1/chat/completions route exists",      '"/v1/chat/completions"' in src,       "runtime"),
    ("v1_completions",   "POST /v1/completions route exists",           '"/v1/completions"' in src,            "runtime"),
    ("model_load_path",  "model load path (RuntimeState.load+from_gguf)", "load" in state_methods,             "runtime"),
    ("max_context",      "max_context handling (from_gguf caps by ctx)", "max_context" in inspect.getsource(cli.RuntimeState), "runtime"),
    ("prefix_cache",     "prompt/KV prefix reuse (get_start_pos)",       "get_start_pos" in inspect.getsource(cli.Handler), "runtime"),
    ("streaming",        "SSE streaming behavior (stream_json)",         "stream_json" in src,                  "runtime"),
    ("prompt_guard",     "oversized-prompt guard before generate",      "_guard_context" in src and "context_length_exceeded" in src, "runtime"),
    ("status_ctrl",      "GET /runtime/status control",                 '"/runtime/status"' in src,            "runtime"),
    ("models_ctrl",      "GET /runtime/models control",                 '"/runtime/models"' in src,            "runtime"),
    ("metrics_ctrl",     "GET /runtime/metrics control",                '"/runtime/metrics"' in src,           "runtime"),
    ("cache_ctrl",       "GET /runtime/cache control",                  '"/runtime/cache"' in src,             "runtime"),
    ("compile_counters", "compile-cache hit/miss + kernel-count observability", "compile_cache" in inspect.getsource(cli.RuntimeState.cache_dict), "runtime"),
    ("load_ctrl",        "POST /runtime/load control",                  '"/runtime/load"' in src,              "runtime"),
    ("unload_ctrl",      "POST /runtime/unload control",                '"/runtime/unload"' in src,            "runtime"),
    ("warmup_ctrl",      "POST /runtime/warmup control",                '"/runtime/warmup"' in src,            "runtime"),
    ("cancel_ctrl",      "POST /runtime/cancel control",                '"/runtime/cancel"' in src,            "runtime"),
    ("busy_policy",      "one-generation-at-a-time busy policy (gen_lock)", "gen_lock" in inspect.getsource(cli.RuntimeState), "runtime"),
    ("error_contract",   "structured JSON error contract",              hasattr(cli, "RUNTIME_ERROR_STATUS"), "runtime"),
    ("registry",         "model registry (build_registry)",             hasattr(cli, "build_registry"),       "runtime"),
  ]
  # leakage check: the runtime must NOT own these client concerns
  leakage_terms = ["index_repo", "search_repo", "embeddings_index", "summarize_session", "tool_execution"]
  leakage = [t for t in leakage_terms if t in inspect.getsource(cli)]
  return {
    "checks": [{"id": cid, "description": desc, "present": bool(ok), "owner": owner} for cid, desc, ok, owner in checks],
    "client_concern_leakage": leakage,
  }

def live_probe(base_url:str) -> dict|None:
  def get(path):
    try:
      with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=5) as r:
        return r.status, json.loads(r.read().decode())
    except Exception as e:
      return None, str(e)
  st, status = get("/runtime/status")
  if st is None: return None
  _, models = get("/v1/models")
  return {"base_url": base_url, "status": status,
          "v1_models_count": len(models.get("data", [])) if isinstance(models, dict) else None}

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--base-url", default=os.environ.get("TINYGRAD_RUNTIME_URL", "http://127.0.0.1:8000"))
  ap.add_argument("--no-live", action="store_true", help="skip the live server probe")
  args = ap.parse_args()

  audit = static_audit()
  present = [c for c in audit["checks"] if c["present"]]
  missing = [c for c in audit["checks"] if not c["present"]]
  live = None if args.no_live else live_probe(args.base_url)

  verdict = "R0_PASS_BOUNDARY_PINNED" if not missing and not audit["client_concern_leakage"] \
            else "R0_BLOCKED_SERVER_ENTRYPOINT_AMBIGUOUS"
  result = {
    "verdict": verdict,
    "entrypoint": "tinygrad/llm/cli.py :: main() --serve (class LLMServer / Handler)",
    "present_count": len(present), "total_count": len(audit["checks"]),
    "missing": [c["id"] for c in missing],
    "client_concern_leakage": audit["client_concern_leakage"],
    "checks": audit["checks"],
    "live_probe": live,
  }
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  (OUT_DIR / "latest.json").write_text(json.dumps(result, indent=2))

  lines = [f"# Tinygrad Runtime Boundary Audit (R0)", "",
           f"Verdict: **{verdict}**", "",
           f"Entrypoint: `{result['entrypoint']}`", "",
           f"Surface present: {len(present)}/{len(audit['checks'])} checks", ""]
  lines.append("| check | present | owner |")
  lines.append("|---|---|---|")
  for c in audit["checks"]:
    lines.append(f"| {c['description']} | {'yes' if c['present'] else 'NO'} | {c['owner']} |")
  lines += ["", f"Client-concern leakage into runtime: {audit['client_concern_leakage'] or 'none'}", ""]
  if live: lines += [f"Live probe `{args.base_url}`: loaded={live['status'].get('loaded')} "
                     f"max_context={live['status'].get('max_context')} v1_models={live['v1_models_count']}", ""]
  else: lines += ["Live probe: skipped or no server running.", ""]
  (OUT_DIR / "summary.md").write_text("\n".join(lines))

  print(f"wrote {OUT_DIR/'latest.json'} and {OUT_DIR/'summary.md'}")
  print(f"{len(present)}/{len(audit['checks'])} surface checks present, leakage={audit['client_concern_leakage'] or 'none'}")
  print(f"== {verdict} ==")
  sys.exit(0 if verdict == "R0_PASS_BOUNDARY_PINNED" else 1)

if __name__ == "__main__":
  main()
