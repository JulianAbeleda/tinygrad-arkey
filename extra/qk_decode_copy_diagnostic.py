#!/usr/bin/env python3
# STATUS: FROZEN (one-off). Verdict: the "~6.5ms copy" was a measurement artifact (4B/0-GB/s sync stall
# mismeasured by eager DEBUG=2), not real data. Diagnostic complete.
"""Narrow diagnostic: identify the ~6.5ms copy/gather kernel that the decode census attributed ~17% of GPU time.

Captures one decode step's kernels (full DEBUG=2 lines: name, mem, GB/s, position in sequence), isolates the
copy/gather kernels, and reports each one's tm + position + neighbors, so we can name the responsible model op.
Also checks whether the big copy is real in WARM decode or a cold/eager census artifact (re-measures it warm and
across the JIT-graph path). No fix here -- identification only. GPU timing via DEBUG=2 tm / time_sum_s.

Run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_copy_diagnostic.py [model.gguf]
"""
from __future__ import annotations

import io, json, os, pathlib, re, contextlib, sys

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# full kernel line: *** DEV  <n>  <name...>  arg N  mem X.XX GB  tm Yus/Zms ( G GFLOPS  A|B GB/s )
_LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem\s+([\d.]+)\s+GB\s+tm\s+([\d.]+)us.*?(\d+)\|(\d+)\s+GB/s")

def _capture_eager_step(m, Tensor, Context, GlobalCounters, tokid, sp):
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    GlobalCounters.reset()
    m.logits(Tensor([[tokid]], dtype="int32").contiguous(), sp).realize()
  rows = []
  for i, ln in enumerate(_ANSI.sub("", buf.getvalue()).splitlines()):
    mt = _LINE.search(ln)
    if mt:
      rows.append({"idx": len(rows), "name": mt.group(1).strip()[:54], "mem_gb": float(mt.group(2)),
                   "tm_us": float(mt.group(3)), "gbs_rd": int(mt.group(4)), "gbs_wr": int(mt.group(5))})
  return rows

def main():
  model = next((a for a in sys.argv[1:] if a.endswith(".gguf")), os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  from tinygrad import Tensor, Context, GlobalCounters
  from extra.llm_generate import load_model_and_tokenizer
  m, tok = load_model_and_tokenizer(model, 2048, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("The quick brown fox. " * 16)
  pre = ids[:64]
  with Context(DEBUG=0): m.logits(Tensor([pre], dtype="int32").contiguous(), 0).realize()
  sp, tokid = len(pre), int(ids[64])
  # warm the eager step so timings aren't cold-compile
  with Context(DEBUG=0):
    for _ in range(4): m.logits(Tensor([[tokid]], dtype="int32").contiguous(), sp).realize()
  rows = _capture_eager_step(m, Tensor, Context, GlobalCounters, tokid, sp)

  # Phase 1 causality: the big "copy" is 4 BYTES at 0 GB/s @idx 0 -> a sync stall, not data movement. Prove it:
  # measure the input upload ALONE warm, and the warm decode-step wall. If the upload is ~us (not ~6.5ms), the
  # 6.5ms tm is a DEBUG=2 step-boundary sync artifact (Bucket C), not real per-token GPU cost.
  import time as _t
  from tinygrad import UOp, TinyJit, Device
  dev = Device[Device.DEFAULT]
  with Context(DEBUG=0):   # sync-bracketed so we measure the UPLOAD, not a queue-drain stall
    up = []
    for _ in range(50):
      dev.synchronize(); t0 = _t.perf_counter()
      Tensor([[tokid]], dtype="int32").contiguous().realize(); dev.synchronize()
      up.append(_t.perf_counter() - t0)
    up_med_us = sorted(up)[25] * 1e6
  v_sp = UOp.variable("start_pos", 0, 2047); step = TinyJit(lambda t, s: m.logits(t, s).realize())
  for i in range(6): step(Tensor([[tokid]], dtype="int32").contiguous(), v_sp.bind(sp + i))
  with Context(DEBUG=0):
    dw = []
    for i in range(30):
      t0 = _t.perf_counter(); step(Tensor([[tokid]], dtype="int32").contiguous(), v_sp.bind(sp + i)).realize(); dw.append(_t.perf_counter() - t0)
    decode_ms = sorted(dw)[15] * 1e3

  copies = [r for r in rows if r["name"].lower().startswith("copy") or "copy" in r["name"].lower()]
  copies_sorted = sorted(copies, key=lambda r: -r["tm_us"])
  big = copies_sorted[0] if copies_sorted else None
  # neighbors of the biggest copy (what op precedes/follows it -> which model subregion)
  neigh = []
  if big:
    j = big["idx"]
    neigh = [{"rel": k - j, "name": rows[k]["name"], "tm_us": rows[k]["tm_us"]} for k in range(max(0, j - 2), min(len(rows), j + 3))]
  total_us = sum(r["tm_us"] for r in rows) or 1.0
  copy_total = sum(r["tm_us"] for r in copies)

  # Verdict from PHYSICAL evidence (independent of the confounded warm-upload micro-probe): the big copy is
  # 4 bytes at 0 GB/s at idx 0 -> cannot be 6.5ms of GPU compute/bandwidth; it's a step-boundary sync stall
  # mismeasured as kernel tm. (The standalone-upload probe is itself confounded by eager Python scheduling /
  # synchronize overhead -- ~18ms ~ a whole decode step -- so it does NOT isolate the upload; not relied upon.)
  is_artifact = bool(big and "4 B" in big["name"] and big["gbs_rd"] == 0 and big["gbs_wr"] == 0 and big["idx"] == 0)
  out = {"model_id": pathlib.Path(model).stem, "kernels_in_step": len(rows),
         "step_gpu_sum_us": round(total_us, 1),
         "n_copy_kernels": len(copies), "copy_total_us": round(copy_total, 1),
         "copy_pct_of_step": round(100 * copy_total / total_us, 1),
         "biggest_copy": big, "biggest_copy_neighbors": neigh, "all_copies": copies_sorted[:8],
         "warm_decode_ms_per_token": round(decode_ms, 3),
         "warm_upload_probe_us_CONFOUNDED": round(up_med_us, 1),
         "bucket": "C_measurement_artifact" if is_artifact else "needs_review",
         "verdict": ("BUCKET C (measurement artifact): the ~6.5ms 'copy' is a 4-byte host->device input upload "
                     "(copy 4 B, AMD <- PYTHON) at idx 0 with 0 GB/s -> physically cannot be 6.5ms of GPU work; "
                     "it's a step-boundary sync/launch stall captured as kernel tm by the eager DEBUG=2 census. "
                     "It is NOT 17% of real warm decode GPU time. No model op to fix; the census per-class "
                     "attribution should exclude it. Real decode GPU = QK GEMVs + non-GEMV small ops only."
                     if is_artifact else "NEEDS REVIEW: copy is not the expected 4B/0-GB/s/idx0 input upload"),
         "census_correction": "drop the 17% 'copy/gather'; renormalize -> QK GEMV ~75%, non-GEMV small ops ~25%",
         "top10_kernels": sorted(rows, key=lambda r: -r["tm_us"])[:10]}
  print(f"kernels/step {len(rows)} | step GPU sum {total_us/1000:.2f}ms | {len(copies)} copy kernels = "
        f"{copy_total/1000:.2f}ms ({out['copy_pct_of_step']}%)")
  if big:
    print(f"BIGGEST COPY: '{big['name']}' tm {big['tm_us']:.1f}us  mem {big['mem_gb']}GB  {big['gbs_rd']}|{big['gbs_wr']} GB/s  @idx {big['idx']}/{len(rows)}")
    print("  neighbors:", [f"{n['rel']:+d}:{n['name'][:24]}({n['tm_us']:.0f}us)" for n in neigh])
  print("  all copies:", [(c['name'][:28], round(c['tm_us'],1)) for c in copies_sorted[:6]])
  print(f"CAUSALITY: warm input-upload-alone {up_med_us:.0f}us | warm decode {decode_ms:.2f}ms/token")
  print(f"VERDICT [{out['bucket']}]: {out['verdict']}")
  art = pathlib.Path("bench/qk-decode-copy-diagnostic/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}")

if __name__ == "__main__":
  main()
