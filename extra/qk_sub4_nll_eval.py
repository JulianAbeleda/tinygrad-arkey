#!/usr/bin/env python3
"""Phase 3 of the sub-4-bit decode arc: dNLL fake-dequant quality gate -- the decision point, NO GPU kernel.

For each candidate (role -> Q3/Q2), quantize that role's weights offline (the Phase-2 proxy), dequantize back to
fp, SUBSTITUTE them into the model's dense (fallback) path, and measure teacher-forced NLL vs the unmodified
model. dNLL = candidate - baseline. Accept <= 0.01, borderline <= 0.02, reject > 0.01. A candidate justifies a
kernel only if dNLL <= 0.01 AND it saves a meaningful share of decode bandwidth (>=5%).

Reuses _window_nll (dense fp path) from qk_prefill_v2_nll_eval and roundtrip() from qk_sub4_quant_probe. Slow
(offline re-quant of all 36 blocks per candidate); that's fine. Run in background:
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_sub4_nll_eval.py [model.gguf]
"""
from __future__ import annotations

import gc, json, os, pathlib, sys, time

# decode-bandwidth share per role (from the Phase-1 census) -> projected byte saving of a demotion
_BW_SHARE = {"ffn_down": 26.8, "ffn_gate": 21.8, "ffn_up": 21.8, "lm_head": 10.9,
             "attn_output": 7.3, "attn_q": 7.3, "attn_v": 2.2, "attn_k": 1.8}
# Q4_K 4.5bpw, Q6_K 6.5625bpw -> Q3_K 3.4375, Q2_K 2.625. saved fraction of that role's bytes:
_CUR_BPW = {"ffn_down": 6.5625, "attn_v": 6.5625, "lm_head": 6.5625, "ffn_gate": 4.5, "ffn_up": 4.5,
            "attn_q": 4.5, "attn_output": 4.5, "attn_k": 4.5}
_TGT_BPW = {"Q3": 3.4375, "Q2": 2.625}

def _proj_bw_saving(roles, qt):  # % of decode bandwidth reclaimed by demoting `roles` to qt
  return round(sum(_BW_SHARE[r] * (1 - _TGT_BPW[qt] / _CUR_BPW[r]) for r in roles), 1)

# candidate ladder: high-byte single roles first, then the FFN trio; one Q2 to confirm Phase-2's reject.
CANDIDATES = [("ffn_down", [("ffn_down", "Q3")]), ("ffn_gate", [("ffn_gate", "Q3")]),
              ("ffn_up", [("ffn_up", "Q3")]), ("attn_q", [("attn_q", "Q3")]),
              ("attn_output", [("attn_output", "Q3")]),
              ("ffn_trio_Q3", [("ffn_down", "Q3"), ("ffn_gate", "Q3"), ("ffn_up", "Q3")]),
              ("ffn_down_Q2", [("ffn_down", "Q2")])]

def _match(name, role): return (f".{role}.weight" in name) or (role == "lm_head" and name == "output.weight")

def main():
  if len(sys.argv) > 1 and sys.argv[1].endswith(".gguf"): model_path = sys.argv[1]
  else: model_path = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
  from tinygrad import Tensor
  from extra.llm_generate import load_model_and_tokenizer
  from extra.qk_prefill_v2_nll_eval import _window_nll, CALIB_TEXT
  from extra.qk_sub4_quant_probe import roundtrip
  UBATCH = 384
  WINDOWS = int(os.environ.get("QK_SUB4_WINDOWS", 1))   # mean dNLL over W windows (re-quant once; cheap forwards)
  STRIDE = 128
  only = set(os.environ.get("QK_SUB4_ONLY", "").split(",")) - {""}   # restrict candidates (avoid VRAM accumulation)
  cands = [(l, s) for l, s in CANDIDATES if not only or l in only]
  model, tok = load_model_and_tokenizer(model_path, 2048, seed=20260617)
  lins = model._q4k_linears.linears
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(CALIB_TEXT)
  starts = [w * STRIDE for w in range(WINDOWS)]
  def mean_nll(): return sum(_window_nll(model, ids, s, UBATCH, prefill_v2=False) for s in starts) / len(starts)
  base = mean_nll()
  print(f"baseline NLL: {base:.5f}  (UBATCH={UBATCH}, {WINDOWS} window(s), dense fp path)", file=sys.__stdout__)

  rows = []
  for label, spec in cands:
    t0 = time.time(); saved = {}
    for lin in lins:
      for role, qt in spec:
        if _match(lin.name, role):
          saved[lin.name] = lin.weight
          lin.weight = Tensor(roundtrip(lin.weight.numpy(), qt)).contiguous().realize()
          break
    nll = mean_nll(); nt = len(saved)
    for lin in lins:
      if lin.name in saved: lin.weight = saved.pop(lin.name)
    del saved; gc.collect()   # drop the substituted Q3 tensors' refs (VRAM) before the next candidate
    roles = [r for r, _ in spec]; qt = spec[0][1]
    dnll = round(nll - base, 5); bw = _proj_bw_saving(roles, qt)
    verdict = "accept" if dnll <= 0.01 else ("maybe" if dnll <= 0.02 else "reject")
    row = {"label": label, "spec": [list(s) for s in spec], "nll": round(nll, 5), "dNLL": dnll,
           "proj_bw_saving_pct": bw, "n_tensors": nt, "verdict": verdict,
           "worth_kernel": bool(dnll <= 0.01 and bw >= 5.0)}
    rows.append(row)
    print(f"{label:16} dNLL {dnll:+.5f}  bw_saving {bw:4.1f}%  -> {verdict}"
          f"{'  [WORTH KERNEL]' if row['worth_kernel'] else ''}  ({time.time()-t0:.0f}s)", file=sys.__stdout__)

  accepted = [r for r in rows if r["worth_kernel"]]
  out = {"model": pathlib.Path(model_path).name, "ubatch": UBATCH, "baseline_nll": round(base, 5), "rows": rows,
         "accepted_worth_kernel": [r["label"] for r in accepted],
         "verdict": (f"PROCEED: {[r['label'] for r in accepted]} pass dNLL<=0.01 with >=5% bw saving -> Q3 kernel "
                     f"arc (Phase 4) earned" if accepted else
                     "REFUTED: no candidate passes dNLL<=0.01 with >=5% bw saving -> bank sub4 as quality-refuted")}
  print(out["verdict"], file=sys.__stdout__)
  art = pathlib.Path("bench/qk-sub4-nll/search.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}", file=sys.__stdout__)

if __name__ == "__main__":
  main()
