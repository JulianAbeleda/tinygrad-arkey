#!/usr/bin/env python3
"""Phase 2 probe: server/long-prompt PREFILL_CONCRETE_KV policy (PREFILL_CONCRETE_KV=auto + PREFILL_SERVER_PROFILE).

Verifies the DECISION wiring (perf is already measured in docs/prefill-default-policy-evaluation-result-20260620.md):
concrete-KV auto = ON iff PREFILL_V2 active AND server profile; OFF for one-shot; explicit 1 forces; server profile
implies V2=auto + concrete-KV on (on a 24GB card). NO new kernels. Run:
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_concrete_kv_policy_probe.py
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys

MODEL = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
# (env_PREFILL_V2, env_PREFILL_CONCRETE_KV, env_PREFILL_SERVER_PROFILE)
CASES = {
  "v2off_default":       (None, None, None),
  "v2on_oneshot":        ("1",  None, None),
  "v2on_ckv_auto":       ("1",  "auto", None),
  "v2on_explicit_ckv":   ("1",  "1",  None),
  "server_profile":      (None, None, "1"),
  "server_but_v2_forced_off": ("0", None, "1"),
}


def _load(case):
  v2, ckv, srv = CASES[case]
  for k, val in (("PREFILL_V2", v2), ("PREFILL_CONCRETE_KV", ckv), ("PREFILL_SERVER_PROFILE", srv)):
    os.environ.pop(k, None)
    if val is not None: os.environ[k] = val
  import io, contextlib
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    from extra.llm_generate import load_model_and_tokenizer
    import tinygrad.llm.model as M
    m, _ = load_model_and_tokenizer(MODEL, 1024, seed=0)
    njits = len(getattr(m, "prefill_v2_jits", {}))     # precompiled concrete jits (concrete-KV side effect)
  reasons = [l for l in buf.getvalue().splitlines() if l.startswith("PREFILL_")]
  print("@@C@@" + json.dumps({"case": case, "resolved_V2": M.PREFILL_V2, "resolved_CKV": M.PREFILL_CONCRETE_KV,
    "precompiled_concrete_jits": njits, "reasons": reasons}))


def main() -> int:
  if len(sys.argv) >= 3 and sys.argv[1] == "--load": _load(sys.argv[2]); return 0
  import tinygrad.llm.model as M
  units = {
    "v2on+server->ON": M.prefill_concrete_kv_auto_decision(True, True)[0] is True,
    "v2on+noserver->OFF": M.prefill_concrete_kv_auto_decision(False, True)[0] is False,
    "v2off->OFF": M.prefill_concrete_kv_auto_decision(True, False)[0] is False,
  }
  loads = {}
  for case in CASES:
    p = subprocess.run([sys.executable, __file__, "--load", case], env={**os.environ, "DEV": "AMD", "PYTHONPATH": "."},
                       capture_output=True, text=True, timeout=900)
    line = next((l for l in p.stdout.splitlines() if l.startswith("@@C@@")), None)
    loads[case] = json.loads(line[5:]) if line else {"case": case, "ERROR": p.stderr[-300:]}
  g = {
    "unit_v2on_server_ON": units["v2on+server->ON"], "unit_v2on_noserver_OFF": units["v2on+noserver->OFF"],
    "unit_v2off_OFF": units["v2off->OFF"],
    "v2on_oneshot_ckv_off": loads["v2on_oneshot"].get("resolved_CKV") is False,
    "v2on_ckv_auto_off (no server)": loads["v2on_ckv_auto"].get("resolved_CKV") is False,
    "explicit_ckv_on": loads["v2on_explicit_ckv"].get("resolved_CKV") is True,
    "server_profile_v2_on": loads["server_profile"].get("resolved_V2") is True,
    "server_profile_ckv_on": loads["server_profile"].get("resolved_CKV") is True,
    "server_profile_precompiled": loads["server_profile"].get("precompiled_concrete_jits", 0) >= 1,
    "server_but_v2_forced_off->ckv_off": loads["server_but_v2_forced_off"].get("resolved_CKV") is False,
  }
  result = {"date": "2026-06-20", "model": pathlib.Path(MODEL).name, "units": units, "loads": loads, "gates": g,
            "all_gates_pass": all(g.values()),
            "perf_reference": "docs/prefill-default-policy-evaluation-result-20260620.md (warm prefill 0.17-1.6s ckv vs 7.5-73.5s default)"}
  out = pathlib.Path("bench/qk-prefill-policy-integration"); out.mkdir(parents=True, exist_ok=True)
  (out / "concrete_kv_policy.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"loads": {k: {kk: v.get(kk) for kk in ("resolved_V2", "resolved_CKV", "precompiled_concrete_jits")} for k, v in loads.items()},
                    "gates": g, "all_gates_pass": result["all_gates_pass"]}, indent=2))
  return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
