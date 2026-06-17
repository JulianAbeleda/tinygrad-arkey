#!/usr/bin/env python3
"""Teacher-forced PREFILL-path NLL gate for PREFILL_V2 (the fp16-prefill quality gate).

Why prefill-path (not decode, cf. qk_nll_eval.py): PREFILL_V2 changes only the PREFILL forward -- it runs
the FFN/attn matmuls in fp16 with realized fp16 weights (lossy vs the baseline fp32-activation path). The
quality question is therefore "does the fp16 prefill degrade the logits it produces?". We measure that
directly: prefill a real >=512-token window once and accumulate -log p(true_next | logits_at_pos) over the
whole window, with PREFILL_V2 OFF (reference: fp32 activations) vs ON (fp16 + realized weights + warmstart).

  dNLL = nll(prefill_v2) - nll(baseline).  Accept iff dNLL <= EPS (default 0.01, the decode-arc convention).

Greedy byte-identical (the cheap smoke) is necessary but not sufficient for a lossy path; this is the real
gate. Run with PREFILL_V2=1 so the model realizes fp16 weights + builds the warmstart table at load:
  DEV=AMD PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_v2_nll_eval.py <model.gguf>
"""
from __future__ import annotations

import argparse, json, os, pathlib, platform, subprocess, sys

# Fixed real prose (content irrelevant, only that it is constant + long enough for a 512-token window). Varied
# registers so the NLL isn't dominated by one style. ~800+ tokens.
CALIB_TEXT = (
  "The history of computation is a history of moving work closer to the data it acts on. Early machines "
  "shuttled numbers between memory and a single arithmetic unit, and the cost of that shuttling came to "
  "dominate everything else. Caches, pipelines, and vector units were all attempts to amortize the same "
  "stubborn expense: reading a value from far away is slow, and a processor that cannot keep its arithmetic "
  "units fed will idle no matter how fast they are. Modern accelerators push this idea to an extreme, with "
  "thousands of lanes that must all be supplied from a shared pool of bandwidth. When the supply runs short, "
  "the lanes wait, and the advertised peak becomes a number that no real program ever reaches.\n\n"
  "Far to the north the river bent through a valley of black pines, and in winter the water ran clear and "
  "shallow over stones the color of iron. The people who lived there measured the season not by the calendar "
  "but by the ice: when it thickened enough to bear a sled, the markets opened on the frozen channel, and "
  "traders came down from the hills with furs and dried fish and small bitter apples wrapped in straw. An old "
  "woman kept the only clock in the village, a brass instrument her grandfather had carried from the coast, "
  "and each morning she wound it and announced the hour to no one in particular, as if the act of counting "
  "the minutes were itself a kind of prayer against the long dark.\n\n"
  "Consider a function that takes a list of integers and returns the running maximum. The naive approach "
  "scans the list once, holding the largest value seen so far, and appends it at every step; this is linear "
  "in the length of the input and uses constant extra space beyond the output. A subtler version must answer "
  "queries about arbitrary windows, and there the simple scan no longer suffices, because a value that was "
  "the maximum may fall out of the window and leave no record of the next-largest behind it. The standard "
  "remedy is a double-ended queue of indices kept in decreasing order of their values, which amortizes each "
  "element to constant work despite the occasional burst of evictions at the front.\n\n"
  "Economists have long argued about whether prices carry all the information a market needs. In the strong "
  "form of the claim, the current price already reflects every fact that anyone knows, so no amount of study "
  "can reliably beat the average; in weaker forms, some patterns persist long enough to be exploited before "
  "they are competed away. The debate is hard to settle because the act of trading on a belief tends to "
  "erase the very signal that justified it, and because the rare events that matter most are, almost by "
  "definition, too infrequent to measure with confidence. What survives is less a law than a discipline: be "
  "skeptical of stories that explain the past too neatly, and humble about the future.\n\n"
  "Botanists distinguish between plants that invest in many cheap seeds and those that produce a few "
  "expensive ones, and the contrast runs deeper than mere arithmetic. A dandelion scatters hundreds of "
  "feather-light parachutes that will mostly land on pavement or in shade and die, betting that sheer number "
  "will find the rare hospitable crack. An oak, by comparison, drops heavy acorns near its own roots, each "
  "packed with enough starch to push a seedling through its first hard season, and accepts that most will be "
  "eaten by squirrels and jays. Neither strategy is correct in isolation; each is a wager tuned to a "
  "particular kind of uncertainty, and a forest holds both because the weather itself cannot decide which "
  "year will be kind. The same tension reappears wherever something must be spread against an unknown future: "
  "in how a language keeps redundant words against the noise of a crowded room, in how a body keeps more "
  "immune cells than any single infection requires, in how a careful engineer keeps a margin that looks like "
  "waste right up until the morning it is the only thing that saves the bridge."
)

def _git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  except Exception: return "unknown"

def _window_nll(model, ids:list[int], start:int, n:int, prefill_v2:bool) -> float:
  # prefill window ids[start:start+n] at position 0; NLL over predictions 1..n-1 from the prefill logits.
  import numpy as np
  from tinygrad import Tensor
  import tinygrad.codegen.opt.postrange as pr
  for lin in (getattr(model, "_q4k_linears", None).linears if getattr(model, "_q4k_linears", None) else []):
    lin.decode_enabled = False                     # prefill (fallback / fp16 path), not the decode GEMV
  for b in model.blk: b._prefill_v2 = prefill_v2
  win = ids[start:start + n]
  t = Tensor([win], dtype="int32").contiguous()
  saved = pr._WARMSTART_OPTS
  if prefill_v2: pr._WARMSTART_OPTS = model._pf16_warmstart   # contained: only around this prefill forward
  try:
    arr = model.logits(t, 0).realize()[0].numpy()  # (n, vocab) fp32
  finally:
    pr._WARMSTART_OPTS = saved
  logits_pred = arr[:n - 1]                         # positions 0..n-2 predict tokens 1..n-1
  mx = logits_pred.max(axis=1, keepdims=True)
  lse = mx[:, 0] + np.log(np.exp(logits_pred - mx).sum(axis=1))
  tgt = np.array(win[1:n])
  return float((lse - logits_pred[np.arange(n - 1), tgt]).mean())

def eval_prefill_nll(model_path:str, max_context:int, ubatch:int, windows:int, stride:int, seed:int) -> dict:
  from extra.llm_generate import load_model_and_tokenizer
  model, tok = load_model_and_tokenizer(model_path, max_context, seed=seed)
  if not getattr(model, "_pf16_warmstart", None):
    raise RuntimeError("model has no prefill-v2 warmstart table -- run with PREFILL_V2=1 (and a validated UBATCH).")
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(CALIB_TEXT)
  need = stride * (windows - 1) + ubatch
  if len(ids) < need:
    raise ValueError(f"calibration too short: have {len(ids)} tokens, need {need} for {windows}x{ubatch} (stride {stride}). "
                     f"Lengthen CALIB_TEXT or reduce --windows.")
  rows = []
  for w in range(windows):
    s = w * stride
    nref = _window_nll(model, ids, s, ubatch, prefill_v2=False)
    nv2  = _window_nll(model, ids, s, ubatch, prefill_v2=True)
    rows.append({"window": w, "start": s, "n": ubatch, "nll_baseline": round(nref, 6),
                 "nll_prefill_v2": round(nv2, 6), "dNLL": round(nv2 - nref, 6)})
    print(f"window {w} [{s}:{s+ubatch}]: baseline {nref:.5f} | v2 {nv2:.5f} | dNLL {nv2-nref:+.5f}", file=sys.__stdout__)
  mean_d = sum(r["dNLL"] for r in rows) / len(rows)
  max_d = max(r["dNLL"] for r in rows)
  return {"model_id": pathlib.Path(model_path).name, "hardware": platform.node(), "commit": _git_sha(),
          "ubatch": ubatch, "windows": windows, "stride": stride, "tokens_scored": (ubatch - 1) * windows,
          "rows": rows, "mean_dNLL": round(mean_d, 6), "max_dNLL": round(max_d, 6)}

def main():
  ap = argparse.ArgumentParser(description="teacher-forced prefill-path NLL gate for PREFILL_V2")
  ap.add_argument("model")
  ap.add_argument("--max-context", type=int, default=2048)
  ap.add_argument("--ubatch", type=int, default=512)
  ap.add_argument("--windows", type=int, default=2)
  ap.add_argument("--stride", type=int, default=256)
  ap.add_argument("--eps", type=float, default=0.01)
  ap.add_argument("--seed", type=int, default=20260617)
  ap.add_argument("--artifact", default="bench/qk-prefill-v2-nll/result.json")
  args = ap.parse_args()
  if not os.environ.get("PREFILL_V2"):
    print("ERROR: run with PREFILL_V2=1 (realizes fp16 weights + builds the warmstart table at load).", file=sys.__stdout__)
    sys.exit(2)
  res = eval_prefill_nll(args.model, args.max_context, args.ubatch, args.windows, args.stride, args.seed)
  res["eps"] = args.eps
  res["verdict"] = "accept" if res["max_dNLL"] <= args.eps else "reject"
  print(f"mean dNLL {res['mean_dNLL']:+.5f} | max dNLL {res['max_dNLL']:+.5f} | eps {args.eps} -> {res['verdict'].upper()}",
        file=sys.__stdout__)
  out = pathlib.Path(args.artifact); out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(res, indent=2))
  print(f"artifact: {out}", file=sys.__stdout__)
  print(json.dumps(res))

if __name__ == "__main__":
  main()
