#!/usr/bin/env python3
"""Gate 1 — PREFILL_GRAPH_GEMM repeated performance (default-on readiness).

Proves the graph-route speedup is stable and NOT a sync/clock/session artifact. Each measurement runs in a
fresh subprocess (clean VRAM); the gate metric is the SYNCED arbiter throughput (K forwards back-to-back, ONE
dev.synchronize(), total/K) -- the trustworthy GPU-throughput method. We also report the nosync number (the
qk_prefill_v2_measure style that produced the prior 1.89x) so the sync-artifact question is answered explicitly.

Run: DEV=AMD PYTHONPATH=. python3 extra/qk_prefill_graph_gemm_default_perf.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
     (orchestrates baseline vs graph subprocesses; pass --worker to run one measurement)
"""
from __future__ import annotations
import json, os, statistics, subprocess, sys, time, pathlib

MODEL = None
K_ARBITER = 8


def perflevel(x): subprocess.run(["rocm-smi", "--setperflevel", x], capture_output=True, text=True)


def worker(model_path: str):
  import tinygrad.codegen.opt.postrange as pr  # noqa
  from tinygrad import Tensor, UOp, Device
  from tinygrad.llm.model import Transformer, PREFILL_UBATCH
  dev = Device["AMD"]; Tensor.manual_seed(0)
  model, _ = Transformer.from_gguf(pathlib.Path(model_path).expanduser(), 2048)
  N = PREFILL_UBATCH; maxc = model.max_context
  vsp = UOp.variable("start_pos", 0, maxc - 1); temp = Tensor([0.0])
  t = Tensor([5, 6, 7, 8, 9, 10] * 200 + [0] * (maxc - 1200), dtype="int32").reshape(1, maxc)
  sp = vsp.bind(0); chunk = t[:, sp:sp + N]
  fwd = lambda: model(chunk, sp, temp)
  perflevel("high")
  try:
    for _ in range(5): fwd().realize(); dev.synchronize()                 # warm + JIT-capture
    # SYNCED arbiter: K forwards, ONE sync, total/K = true GPU throughput
    best = 1e9
    for _ in range(3):
      dev.synchronize(); t0 = time.perf_counter()
      for _ in range(K_ARBITER): fwd().realize()
      dev.synchronize(); best = min(best, (time.perf_counter() - t0) / K_ARBITER)
    ms_synced = best * 1e3
    # nosync (qk_prefill_v2_measure style) for the sync-artifact comparison
    nos = []
    for _ in range(5): GC0 = time.perf_counter(); fwd().realize(); nos.append(time.perf_counter() - GC0)
    ms_nosync = min(nos) * 1e3
  finally:
    perflevel("auto")
  print(json.dumps({"ms512_synced": round(ms_synced, 1), "toks_synced": round(N / (ms_synced / 1e3), 1),
                    "ms512_nosync": round(ms_nosync, 1), "toks_nosync": round(N / (ms_nosync / 1e3), 1)}))


def run_one(model_path, graph: bool):
  env = dict(os.environ); env["DEV"] = "AMD"; env["PREFILL_V2"] = "1"; env["PYTHONPATH"] = "."
  if graph: env["PREFILL_GRAPH_GEMM"] = "1"
  else: env.pop("PREFILL_GRAPH_GEMM", None)
  p = subprocess.run([sys.executable, __file__, "--worker", model_path], env=env, capture_output=True, text=True, timeout=1200)
  if p.returncode != 0: return {"rc": p.returncode, "err": p.stderr[-300:]}
  line = [l for l in p.stdout.strip().splitlines() if l.startswith("{")]
  if not line: return {"rc": -1, "err": "no json: " + p.stdout[-200:]}
  d = json.loads(line[-1]); d["rc"] = 0; return d


def main() -> int:
  if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
    worker(sys.argv[2]); return 0
  model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
  SESSIONS = int(os.environ.get("SESSIONS", "3"))
  pairs = []
  for s in range(SESSIONS):
    # alternate which side runs first to mix clock/session state
    if s % 2 == 0: base = run_one(model_path, False); graph = run_one(model_path, True)
    else: graph = run_one(model_path, True); base = run_one(model_path, False)
    pair = {"session": s, "baseline": base, "graph": graph,
            "speedup_synced": (round(base["ms512_synced"] / graph["ms512_synced"], 3)
                               if base.get("rc") == 0 and graph.get("rc") == 0 else None),
            "speedup_nosync": (round(base["ms512_nosync"] / graph["ms512_nosync"], 3)
                               if base.get("rc") == 0 and graph.get("rc") == 0 else None)}
    pairs.append(pair)
    print(f"  session {s}: baseline {base.get('ms512_synced','ERR')}ms graph {graph.get('ms512_synced','ERR')}ms "
          f"synced_speedup={pair['speedup_synced']} (nosync {pair['speedup_nosync']})")

  failures = sum(1 for p in pairs for side in ("baseline", "graph") if p[side].get("rc") != 0)
  ok = [p for p in pairs if p["speedup_synced"] is not None]
  speedups = [p["speedup_synced"] for p in ok]
  base_p50 = statistics.median([p["baseline"]["ms512_synced"] for p in ok]) if ok else None
  graph_p50 = statistics.median([p["graph"]["ms512_synced"] for p in ok]) if ok else None
  median_speedup = statistics.median(speedups) if speedups else None
  worst_speedup = min(speedups) if speedups else None
  passed = (len(ok) >= 3 and failures == 0 and median_speedup is not None and median_speedup >= 1.5
            and worst_speedup >= 1.25 and graph_p50 < base_p50)
  result = {"date": "2026-06-20", "gate": 1, "schema": "prefill_graph_gemm_default_perf_v1",
            "sessions": SESSIONS, "metric": "SYNCED arbiter (K=%d forwards, one sync, total/K)" % K_ARBITER,
            "pairs": pairs, "run_failures": failures, "median_speedup_synced": median_speedup,
            "worst_speedup_synced": worst_speedup, "baseline_p50_ms512": base_p50, "graph_p50_ms512": graph_p50,
            "median_speedup_nosync": (statistics.median([p["speedup_nosync"] for p in ok]) if ok else None),
            "thresholds": {"paired>=3": len(ok) >= 3, "median>=1.5": median_speedup is not None and median_speedup >= 1.5,
                           "worst>=1.25": worst_speedup is not None and worst_speedup >= 1.25,
                           "graph_p50<base_p50": (graph_p50 < base_p50) if ok else False, "failures==0": failures == 0},
            "verdict": "PASS_PREFILL_GRAPH_GEMM_REPEATED_PERF" if passed else "BLOCKED_PREFILL_GRAPH_GEMM_REPEATED_PERF"}
  out = pathlib.Path("bench/amd-broad-backend-roadmap"); out.mkdir(parents=True, exist_ok=True)
  (out / "prefill_graph_gemm_default_perf_result.json").write_text(json.dumps(result, indent=2) + "\n")
  print(f"\n  SYNCED: median speedup {median_speedup}x worst {worst_speedup}x | baseline p50 {base_p50}ms graph p50 {graph_p50}ms")
  print(f"  (nosync median speedup {result['median_speedup_nosync']}x -- prior promotion used nosync ~1.89x)")
  print(f"  failures: {failures}")
  print(f"\n{result['verdict']}")
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
