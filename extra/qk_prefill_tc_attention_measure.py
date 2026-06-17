#!/usr/bin/env python3
"""Gated integration probe: does the Option-B TC-attention win survive INSIDE the full prefill-v2 forward?

Standalone (concrete KV) the explicit TC attention beat SDPA ~2.5x. In-model the prefill-v2 chunk uses a
SYMBOLIC start_pos, so KV=start_pos+T is symbolic -- the same obstacle that slowed SDPA. This probe measures
the FULL prefill-v2 forward GPU time (GlobalCounters.time_sum_s under DEBUG>=2) at start_pos in {0,512,1536,3072}
with PREFILL_TC_ATTENTION on vs off (SDPA), each in its OWN subprocess (avoid compile-accumulation faults).

Acceptance: long-context full forward improves >=1.25x at sp=3072 (expected ~1.43x if attention is ~50% and
speeds up 2.5x). Else bank Option B as standalone-only and don't keep the model path wired. dNLL + decode
checks are separate. Run: DEV=AMD PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_tc_attention_measure.py
"""
from __future__ import annotations

import json, os, pathlib, subprocess, sys

MODEL = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
SPS = [0, 512, 1536, 3072]
MAXC = 4096

def _child(tc:int):
  os.environ["PREFILL_V2"] = "1"; os.environ["PREFILL_TC_ATTENTION"] = str(tc); os.environ.setdefault("JIT", "1")
  from tinygrad import Tensor, UOp, GlobalCounters, Context
  from tinygrad.llm.model import Transformer
  Tensor.manual_seed(0)
  m, _ = Transformer.from_gguf(pathlib.Path(MODEL).expanduser(), MAXC)
  vsp = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0])
  t = Tensor([5, 6, 7, 8, 9, 10] * (MAXC // 6) + [0] * (MAXC - 6 * (MAXC // 6)), dtype="int32").reshape(1, MAXC)
  res = {}
  for sp_val in SPS:
    sp = vsp.bind(sp_val); chunk = t[:, sp:sp + 512]
    m(chunk, sp, temp).realize()  # warmup / capture (compile)
    best = 1e9
    with Context(DEBUG=2):
      for _ in range(5):
        GlobalCounters.reset(); m(chunk, sp, temp).realize(); best = min(best, GlobalCounters.time_sum_s)
    res[sp_val] = round(best * 1e3, 3)
  print("@@R@@" + json.dumps(res))

def main():
  if len(sys.argv) >= 2 and sys.argv[1] == "--child":
    _child(int(sys.argv[2])); return
  def run(tc):
    p = subprocess.run([sys.executable, __file__, "--child", str(tc)], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": "."}, timeout=400)
    line = next((l for l in p.stdout.splitlines() if l.startswith("@@R@@")), None)
    if line is None: raise RuntimeError(f"child tc={tc} failed:\n{p.stderr[-500:]}")
    return {int(k): v for k, v in json.loads(line[5:]).items()}
  sdpa = run(0); tc = run(1)
  rows = []
  for sp in SPS:
    su = round(sdpa[sp] / tc[sp], 3) if tc[sp] else None
    rows.append({"start_pos": sp, "KV": sp + 512, "sdpa_ms": sdpa[sp], "tc_ms": tc[sp], "speedup": su})
    print(f"sp={sp:5d} (KV={sp+512:5d}): sdpa {sdpa[sp]:7.2f}ms | tc {tc[sp]:7.2f}ms -> {su}x", file=sys.__stdout__)
  long = next(r for r in rows if r["start_pos"] == max(SPS))
  out = {"model": pathlib.Path(MODEL).name, "shape_note": "full prefill-v2 forward, T=512, symbolic start_pos",
         "rows": rows, "score_materialization_MB_at_3584": round(32 * 512 * 3584 * 2 / 1e6, 1),
         "passes": bool(long["speedup"] and long["speedup"] >= 1.25)}
  out["verdict"] = (f"PASS: TC attention survives in-model, {long['speedup']}x full forward at sp={max(SPS)} "
                    f">=1.25x -> keep wired (pending dNLL)" if out["passes"] else
                    f"REFUTED in-model: {long['speedup']}x at sp={max(SPS)} <1.25x (symbolic KV likely blocks "
                    f"TC) -> bank Option B as standalone-only, unwire")
  print(out["verdict"], file=sys.__stdout__)
  art = pathlib.Path("bench/qk-prefill-tc-attention/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}", file=sys.__stdout__)

if __name__ == "__main__":
  main()
