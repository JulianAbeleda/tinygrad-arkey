#!/usr/bin/env python3
"""FMI-4 B1 knob probe for decode fused-MMVQ integration.

Runs a single loaded Qwen3-8B model through warm eager decode measurements while
varying only existing launch-shape knobs. This is the bounded Track-B surface:
if it cannot move role bandwidth, the remaining occupancy route is not an env
knob and must move to runtime/cache or renderer work.
"""
from __future__ import annotations

import contextlib, io, json, os, pathlib, re, sys
from collections import defaultdict
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "qk-decode-fused-mmvq-integration"
HBM_PEAK_GBS = 960.0
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem\s+[\d.]+\s+GB\s+tm\s+([\d.]+)us.*?(\d+)\|(\d+)\s+GB/s")
_GEMV = re.compile(r"q([46])k_gemv\w*_(\d+)_(\d+)_")
ROLE_BYTES = {"ffn_down": (1252.8e6, 6), "ffn_gate/up": (2 * 1019.2e6, 4), "lm_head": (510.5e6, 6),
              "attn_q/o": (2 * 339.7e6, 4), "attn_k/v": (84.9e6 + 104.4e6, 4)}
SHAPE_ROLE = {(151936, 4096): "lm_head", (4096, 12288): "ffn_down", (12288, 4096): "ffn_gate/up",
              (4096, 4096): "attn_q/o", (1024, 4096): "attn_k/v"}


def _set_env(cfg: dict[str, str]) -> dict[str, str | None]:
  keys = ["Q4K_COOP_RT", "Q6K_COOP_RT", "Q4K_ATTN_QO_COOP", "Q6K_FFN_DOWN_COOP", "Q6K_LM_HEAD_COOP"]
  old = {k: os.environ.get(k) for k in keys}
  for k in keys:
    if k in cfg: os.environ[k] = cfg[k]
    else: os.environ.pop(k, None)
  return old


def _restore_env(old: dict[str, str | None]) -> None:
  for k, v in old.items():
    if v is None: os.environ.pop(k, None)
    else: os.environ[k] = v


def _measure_once(m: Any, sp: int, tokid: int) -> list[dict[str, Any]]:
  from tinygrad import Tensor, Context, GlobalCounters
  with Context(DEBUG=0):
    for _ in range(2):
      m.logits(Tensor([[tokid]], dtype="int32").contiguous(), sp).realize()
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    GlobalCounters.reset()
    m.logits(Tensor([[tokid]], dtype="int32").contiguous(), sp).realize()
  roles = defaultdict(lambda: [0, 0.0, 0.0])
  for line in buf.getvalue().splitlines():
    mt = _LINE.search(_ANSI.sub("", line))
    if not mt: continue
    g = _GEMV.search(mt.group(1).lower())
    if not g: continue
    role = SHAPE_ROLE.get((int(g.group(2)), int(g.group(3))))
    if role is None: continue
    roles[role][0] += 1
    roles[role][1] += float(mt.group(2))
    roles[role][2] += max(int(mt.group(3)), int(mt.group(4)))
  rows = []
  for role, (cnt, tm_us, read_gbs) in sorted(roles.items(), key=lambda kv: -kv[1][1]):
    by, qb = ROLE_BYTES.get(role, (0, 4))
    eff_bw = by / (tm_us * 1e-6) / 1e9 if tm_us else 0
    rows.append({"role": role, "kernels": cnt, "total_tm_ms": round(tm_us / 1000, 3), "weight_MB": round(by / 1e6, 1),
                 "qbits": qb, "effective_read_GBs": round(eff_bw, 1), "pct_hbm_peak": round(100 * eff_bw / HBM_PEAK_GBS, 1),
                 "debug_max_GBs": round(read_gbs / cnt, 1) if cnt else 0})
  return rows


def _by_role(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  return {r["role"]: r for r in rows}


def main() -> None:
  model_path = next((a for a in sys.argv[1:] if a.endswith(".gguf")), os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  from tinygrad import Tensor, Context
  from extra.llm_generate import load_model_and_tokenizer

  m, tok = load_model_and_tokenizer(model_path, 2048, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox. " * 40)
  sp, tokid = 64, int(ids[64])
  with Context(DEBUG=0):
    m.logits(Tensor([ids[:sp]], dtype="int32").contiguous(), 0).realize()

  configs = [
    ("default", {}),
    ("q6_rt1", {"Q6K_COOP_RT": "1"}),
    ("q6_rt2", {"Q6K_COOP_RT": "2"}),
    ("q6_rt8", {"Q6K_COOP_RT": "8"}),
    ("q6_rt16", {"Q6K_COOP_RT": "16"}),
    ("q6_coop_off", {"Q6K_FFN_DOWN_COOP": "0", "Q6K_LM_HEAD_COOP": "0"}),
    ("q4_rt4", {"Q4K_COOP_RT": "4"}),
    ("q4_rt8", {"Q4K_COOP_RT": "8"}),
    ("q4_rt32", {"Q4K_COOP_RT": "32"}),
    ("q4_attn_coop_off", {"Q4K_ATTN_QO_COOP": "0"}),
  ]

  rows = []
  for name, cfg in configs:
    old = _set_env(cfg)
    try:
      measured = _measure_once(m, sp, tokid)
    finally:
      _restore_env(old)
    rows.append({"config": name, "env": cfg, "roles": measured})
    print(f"{name}: " + ", ".join(f"{r['role']}={r['pct_hbm_peak']}%" for r in measured))

  base = _by_role(next(r for r in rows if r["config"] == "default")["roles"])
  comparisons = []
  best_by_role: dict[str, dict[str, Any]] = {}
  for row in rows:
    by = _by_role(row["roles"])
    for role, rr in by.items():
      b = base.get(role)
      if not b or b["pct_hbm_peak"] == 0: continue
      rel = rr["pct_hbm_peak"] / b["pct_hbm_peak"]
      comp = {"config": row["config"], "role": role, "pct_hbm_peak": rr["pct_hbm_peak"],
              "baseline_pct_hbm_peak": b["pct_hbm_peak"], "relative": round(rel, 3)}
      comparisons.append(comp)
      cur = best_by_role.get(role)
      if cur is None or comp["relative"] > cur["relative"]: best_by_role[role] = comp

  pass_rows = [r for r in comparisons if r["config"] != "default" and r["relative"] >= 1.10 and r["role"] in ("ffn_gate/up", "ffn_down", "lm_head", "attn_q/o")]
  result = {
    "schema": "decode_fused_mmvq_fmi4_b1_knob_probe_v1",
    "phase": "FMI-4-B1",
    "status": "PASS_BOUNDED_KNOB" if pass_rows else "FAIL_B1_NO_ENV_KNOB_CLEARS_GATE",
    "model": pathlib.Path(model_path).name,
    "gate": "one high-share role group moves >=10% relative isolated in-model",
    "rows": rows,
    "comparisons": comparisons,
    "best_by_role": best_by_role,
    "passing_rows": pass_rows,
    "decision": ("Implement/validate the winning knob under W==D before deeper Track B work"
                 if pass_rows else
                 "Existing env launch-shape knobs do not close Track B; next Track B surface is runtime/cache identity or renderer project."),
  }
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "fmi4_b1_knob_probe.json").write_text(json.dumps(result, indent=2) + "\n")
  lines = ["# FMI-4 B1 knob probe", "", f"Verdict: `{result['status']}`.", ""]
  for role, row in sorted(best_by_role.items()):
    lines.append(f"- `{role}` best `{row['config']}`: `{row['relative']}x` baseline (`{row['pct_hbm_peak']}%` HBM)")
  lines += ["", result["decision"], ""]
  (OUT / "fmi4_b1_summary.md").write_text("\n".join(lines))
  print(json.dumps({"status": result["status"], "passing": pass_rows, "out": str(OUT / "fmi4_b1_knob_probe.json")}, indent=2))


if __name__ == "__main__":
  main()
