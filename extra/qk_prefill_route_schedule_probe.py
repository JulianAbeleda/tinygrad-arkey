#!/usr/bin/env python3
"""Phase 3 probe: the 32-token symbolic fallback trap + the remainder-fix A/B.

For PREFILL_V2 prompts whose length is NOT a multiple of PREFILL_UBATCH, the sub-512 remainder falls to many
slow 32-token symbolic calls. PREFILL_REMAINDER_FIX routes that remainder through ONE shifted prefill-v2 chunk.
This probe logs the call schedule + prefill wall + tok0 for fix OFF vs ON, per prompt length. GATE: tok0 must
MATCH (byte-identical greedy) and the 32-token calls must disappear. NO new kernels. Run:
  DEV=AMD PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_route_schedule_probe.py
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, time
from collections import Counter

MODEL = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
LENS = [600, 1024, 1500, 2100]


def _child(fix: int):
  os.environ["PREFILL_V2"] = "1"; os.environ["PREFILL_REMAINDER_FIX"] = str(fix)
  from tinygrad import Device
  dev = Device["AMD"]
  from extra.llm_generate import load_model_and_tokenizer
  m, tok = load_model_and_tokenizer(MODEL, 2304, seed=0)
  pfx = (tok.prefix() if hasattr(tok, "prefix") else [])
  filler = "the quick brown fox jumps over a lazy dog near rivers and bright hills under wheeling stars "
  T = type(m); orig = T.__call__; sched = []
  def tr(self, tokens, start_pos, *a, **k):
    sh = tokens.shape[1]; sched.append(("int" if isinstance(start_pos, int) else "sym") + str(sh if isinstance(sh, int) else "sym"))
    return orig(self, tokens, start_pos, *a, **k)
  T.__call__ = tr
  rows = []
  for L in LENS:
    ids = (pfx + tok.encode("Probe " + filler * 400))[:L]
    sched.clear(); dev.synchronize(); t0 = time.perf_counter()
    tok0 = next(m.generate(list(ids), chunk_size=32, temperature=0.0)); dev.synchronize()
    prefill_s = time.perf_counter() - t0
    c = Counter(sched)
    rows.append({"prompt_len": L, "tok0": tok0, "prefill_s": round(prefill_s, 2),
                 "ncalls": len(sched), "n_32tok": sum(v for k, v in c.items() if k.endswith("32") or k == "symsym"),
                 "schedule": dict(c)})
  print("@@R@@" + json.dumps({"fix": fix, "rows": rows}))


def main() -> int:
  if len(sys.argv) >= 3 and sys.argv[1] == "--child": _child(int(sys.argv[2])); return 0
  res = {}
  for fix in (0, 1):
    p = subprocess.run([sys.executable, __file__, "--child", str(fix)],
                       env={**os.environ, "DEV": "AMD", "PYTHONPATH": "."}, capture_output=True, text=True, timeout=1200)
    line = next((l for l in p.stdout.splitlines() if l.startswith("@@R@@")), None)
    if line is None: print(f"fix={fix} FAILED:\n{p.stdout[-300:]}\n{p.stderr[-400:]}"); return 1
    res[fix] = {r["prompt_len"]: r for r in json.loads(line[5:])["rows"]}
  rows = []
  for L in LENS:
    off, on = res[0][L], res[1][L]
    rows.append({"prompt_len": L, "tok0_off": off["tok0"], "tok0_on": on["tok0"], "tok0_match": off["tok0"] == on["tok0"],
                 "prefill_s_off": off["prefill_s"], "prefill_s_on": on["prefill_s"],
                 "speedup": round(off["prefill_s"] / on["prefill_s"], 2) if on["prefill_s"] else None,
                 "ncalls_off": off["ncalls"], "ncalls_on": on["ncalls"],
                 "n_32tok_off": off["n_32tok"], "n_32tok_on": on["n_32tok"],
                 "sched_off": off["schedule"], "sched_on": on["schedule"]})
    print(f"  len{L:5d}: tok0 {off['tok0']}=={on['tok0']}? {off['tok0']==on['tok0']} | "
          f"prefill {off['prefill_s']:.2f}->{on['prefill_s']:.2f}s ({round(off['prefill_s']/on['prefill_s'],2) if on['prefill_s'] else '-'}x) | "
          f"32tok {off['n_32tok']}->{on['n_32tok']} | sched_on {on['schedule']}")
  gate = {"all_tok0_match": all(r["tok0_match"] for r in rows),
          "remainder_32tok_eliminated": all(r["n_32tok_on"] == 0 for r in rows if r["prompt_len"] >= 512),
          "no_slower": all((r["prefill_s_on"] <= r["prefill_s_off"] + 0.05) for r in rows)}
  result = {"date": "2026-06-20", "model": pathlib.Path(MODEL).name, "rows": rows, "gates": gate,
            "all_gates_pass": all(gate.values())}
  out = pathlib.Path("bench/qk-prefill-policy-integration"); out.mkdir(parents=True, exist_ok=True)
  (out / "route_schedule_probe.json").write_text(json.dumps(result, indent=2) + "\n")
  print(f"\ngates: {gate} all_pass={result['all_gates_pass']}")
  return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
