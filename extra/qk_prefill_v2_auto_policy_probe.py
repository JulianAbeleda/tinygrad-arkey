#!/usr/bin/env python3
"""Phase 1 probe: VRAM-aware PREFILL_V2 auto policy (prefill_v2_auto_decision + PREFILL_V2=auto).

Verifies: (a) the decision function picks ON for a 24GB card / OFF for 16GB / OFF when VRAM unknown; (b) a real
`PREFILL_V2=auto` load resolves + logs a reason + loads successfully with sane peak VRAM; (c) explicit
PREFILL_V2=0/1 still force the module global. NO new kernels. Run:
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_v2_auto_policy_probe.py
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys

MODEL = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")


def _real_load(env_val: str):
  """child: set PREFILL_V2=<env_val>, load, report resolved PREFILL_V2 + peak VRAM + the auto reason line."""
  if env_val == "<unset>": os.environ.pop("PREFILL_V2", None)
  else: os.environ["PREFILL_V2"] = env_val
  import io, contextlib
  buf = io.StringIO()
  from tinygrad import Device, GlobalCounters
  with contextlib.redirect_stdout(buf):
    from extra.llm_generate import load_model_and_tokenizer
    import tinygrad.llm.model as M
    try:
      m, _ = load_model_and_tokenizer(MODEL, 2048, seed=0)
      ok = True
    except Exception as e:
      print("@@C@@" + json.dumps({"env": env_val, "load_ok": False, "err": str(e)[:200]})); return
  reason = next((l for l in buf.getvalue().splitlines() if l.startswith("PREFILL_V2=auto")), None)
  print("@@C@@" + json.dumps({"env": env_val, "load_ok": ok, "resolved_PREFILL_V2": M.PREFILL_V2,
    "auto_reason": reason, "peak_vram_gb": round(GlobalCounters.mem_used_per_device["AMD"] / 1e9, 2)}))


def main() -> int:
  if len(sys.argv) >= 3 and sys.argv[1] == "--load":
    _real_load(sys.argv[2]); return 0
  import tinygrad.llm.model as M
  # (a) unit: decision for synthetic cards (8B-ish: q4 5GB, fp16 14GB, kv 1.2GB)
  q4, fp16, kv = int(5e9), int(14e9), int(1.2e9)
  scenarios = {"24GB_card": int(25.7e9), "16GB_card": int(17.2e9), "vram_unknown": None, "exactly_22GB": int(22e9)}
  units = {name: dict(zip(("enabled", "reason"), M.prefill_v2_auto_decision(v, fp16, q4, kv))) for name, v in scenarios.items()}
  detected = M._detect_total_vram_bytes()
  # (b,c) real loads in subprocesses (PREFILL_V2 read at import)
  loads = {}
  for env_val in ("<unset>", "0", "1", "auto"):
    p = subprocess.run([sys.executable, __file__, "--load", env_val],
                       env={**os.environ, "DEV": "AMD", "PYTHONPATH": "."}, capture_output=True, text=True, timeout=600)
    line = next((l for l in p.stdout.splitlines() if l.startswith("@@C@@")), None)
    loads[env_val] = json.loads(line[5:]) if line else {"env": env_val, "ERROR": p.stderr[-300:]}
  result = {"date": "2026-06-20", "model": pathlib.Path(MODEL).name, "detected_total_vram_gb": (detected/1e9 if detected else None),
            "decision_unit_tests": units, "real_loads": loads,
            "gates": {
              "24GB->ON": units["24GB_card"]["enabled"] is True,
              "16GB->OFF": units["16GB_card"]["enabled"] is False,
              "unknown->OFF": units["vram_unknown"]["enabled"] is False,
              "explicit_0_off": loads["0"].get("resolved_PREFILL_V2") is False,
              "explicit_1_on": loads["1"].get("resolved_PREFILL_V2") is True,
              "unset_off": loads["<unset>"].get("resolved_PREFILL_V2") is False,
              "auto_resolves": loads["auto"].get("auto_reason") is not None}}
  result["all_gates_pass"] = all(result["gates"].values())
  out = pathlib.Path("bench/qk-prefill-policy-integration"); out.mkdir(parents=True, exist_ok=True)
  (out / "prefill_v2_auto_policy.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"detected_total_vram_gb": result["detected_total_vram_gb"], "units": units,
                    "real_loads": {k: {kk: v.get(kk) for kk in ("resolved_PREFILL_V2", "peak_vram_gb", "auto_reason")} for k, v in loads.items()},
                    "gates": result["gates"], "all_gates_pass": result["all_gates_pass"]}, indent=2))
  return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
