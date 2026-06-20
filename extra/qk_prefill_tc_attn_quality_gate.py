#!/usr/bin/env python3
"""Quality gate for Branch B (PREFILL_TC_ATTN explicit attention) — dNLL + greedy(argmax)-exact.

Reuses the VRAM-safe sampled-NLL method (model.logits(window, start_pos=0) -> per-window NLL + argmax) from
qk_prefill_graph_gemm_quality_sampled. Both arms keep PREFILL_GRAPH_GEMM=1 (the promoted baseline); the ONLY
difference is PREFILL_TC_ATTN 0 vs 1, so the gate isolates the attention-path change. start_pos=0 is concrete
so the TC-attn branch fires. Gate: max_abs_dNLL <= 0.01 AND argmax matches every window (greedy-exact).

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_tc_attn_quality_gate.py [model.gguf]
"""
from __future__ import annotations
import json, math, os, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
WINDOWS, STRIDE, UBATCH = 4, 64, 512   # CALIB_TEXT ~783 toks -> need = STRIDE*(W-1)+512 = 704 <= 783


def _lse(xs):
  import numpy as np
  m = float(np.max(xs)); return m + math.log(float(np.exp(xs - m).sum()))


def _child(model_path, tc):
  import numpy as np
  from tinygrad import Tensor
  from extra.llm_generate import load_model_and_tokenizer
  from extra.qk_prefill_v2_nll_eval import CALIB_TEXT
  Tensor.manual_seed(20260620)
  model, tok = load_model_and_tokenizer(model_path, 2048, seed=20260620)
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(CALIB_TEXT)
  need = STRIDE * (WINDOWS - 1) + UBATCH
  if len(ids) < need: raise ValueError(f"calib too short: {len(ids)} < {need}")
  rows = []
  for w in range(WINDOWS):
    win = ids[w * STRIDE: w * STRIDE + UBATCH]
    logits = model.logits(Tensor([win], dtype="int32").contiguous(), 0)[:, -2, :].realize()[0].numpy()
    target = int(win[-1]); nll = _lse(logits) - float(logits[target])
    rows.append({"window": w, "nll": round(float(nll), 6), "argmax": int(np.argmax(logits)),
                 "target": target, "finite": bool(np.isfinite(logits).all())})
  print("@@R@@" + json.dumps({"tc": tc, "rows": rows}))


def _run(model_path, tc):
  env = {**os.environ, "DEV": os.environ.get("DEV", "AMD"), "PREFILL_V2": "1",
         "PREFILL_GRAPH_GEMM": "1", "PREFILL_TC_ATTN": str(tc), "PYTHONPATH": "."}
  p = subprocess.run([sys.executable, __file__, "--child", str(tc), model_path], cwd=ROOT, env=env,
                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
  line = next((l for l in p.stdout.splitlines() if l.startswith("@@R@@")), None)
  if line is None: raise RuntimeError(f"tc={tc} child failed:\n{p.stdout[-800:]}")
  return json.loads(line[5:])


def main() -> int:
  model_path = sys.argv[-1] if (len(sys.argv) > 1 and not sys.argv[-1].isdigit()) else \
    os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  if len(sys.argv) >= 3 and sys.argv[1] == "--child":
    _child(model_path, int(sys.argv[2])); return 0
  off = _run(model_path, 0); on = _run(model_path, 1)
  rows = []
  for b, g in zip(off["rows"], on["rows"]):
    rows.append({"window": b["window"], "off_nll": b["nll"], "on_nll": g["nll"],
                 "dNLL": round(g["nll"] - b["nll"], 6), "argmax_match": b["argmax"] == g["argmax"],
                 "finite": b["finite"] and g["finite"]})
  max_abs_dnll = max(abs(r["dNLL"]) for r in rows)
  greedy_exact = all(r["argmax_match"] for r in rows)
  finite = all(r["finite"] for r in rows)
  gate = max_abs_dnll <= 0.01 and greedy_exact and finite
  result = {"date": "2026-06-20", "phase": "BRANCH_B_TC_ATTN_QUALITY_GATE",
            "regime": "concrete start_pos=0, graph route ON both arms, toggle PREFILL_TC_ATTN",
            "windows": WINDOWS, "max_abs_dNLL": max_abs_dnll, "greedy_exact": greedy_exact, "finite": finite,
            "gate_pass": gate, "rows": rows,
            "verdict": "PASS_TC_ATTN_QUALITY" if gate else "BLOCKED_TC_ATTN_QUALITY"}
  out = pathlib.Path("bench/qk-prefill-tc-attention"); out.mkdir(parents=True, exist_ok=True)
  (out / "quality_gate_result.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({k: result[k] for k in ("max_abs_dNLL", "greedy_exact", "finite", "gate_pass", "verdict")}, indent=2))
  return 0 if gate else 1


if __name__ == "__main__":
  raise SystemExit(main())
