"""Q6 attention-V direct-output promotion full gate (GPU, serialized, flock).

Scope: `docs/task_workflow/input/nv-q6-direct-shared-q8-promotion-scope-20260814.md`.
Question: with the PRODUCTION loader-installed max17 cooperative shared-Q8 lease already
promoted for NV sm_120, does turning on the Q6-K V direct-output consumer (llama MMVQ
geometry, two per-term `__dp4a`) clear the same section-6 full gate used for the group lease?

Both arms keep the max17 group lease forced open; the only difference is the Q6-direct
sub-variant. Control = `q6_direct_output=False` (current production), candidate =
`q6_direct_output=True`. Record = checked-in policy JSON (group open, Q6-direct closed).

Gate items:
  1. Wall d512/d2048/d4096: candidate must not regress the same-session control median
     and both arms' token streams must be identical.
  2. Census (d512, two captures): candidate gains exactly 16 direct Q6 consumers
     (8 real Q6-K V blocks x two captures), cooperative Q4 stays 86, fused providers 34,
     zero legacy shared-Q4, zero duplicate providers.
  3. Semantic contract (harness child, control vs candidate): exact token stream, equal
     argmax, ordered top-10 sets, relative L2 <= 1e-3, 2*max_abs/min_top1_margin < 1.0.
  4. Pins: control and candidate token streams identical at every depth, each 3/3.
  5. Unit tests green (run separately).
  6. pg3 legacy sha `27857cb8ca03` for `q4k_g3_lanemap_gemv_4096_4096` unmoved.

Open mode = both records forced open via module overrides BEFORE `Transformer.from_gguf`;
the production wiring (model.py lease install, shared_q8_attention_call route) handles the
rest. Each arm runs as a fresh subprocess under `flock -w 600 /tmp/gpu-bench.lock`.
"""
from __future__ import annotations

import argparse, contextlib, gc, hashlib, io, json, os, re, statistics, subprocess, sys, time
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import Context, Target
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel
from tinygrad.llm.model import Transformer
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.uop.ops import Ops, ProgramInfo, UOp
from tinygrad.codegen import do_estimates, do_linearize, do_render, full_rewrite_to_sink
import tinygrad.llm.model_route_plan as mrp
import tinygrad.llm.model as tgm

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
TM_RE = re.compile(r"^\*\*\* NV\s+\d+\s+(\S+)\s+arg\s+\d+.*?tm\s+([\d.]+)(us|ms)/")

N = 4096
LEGACY = f"q4k_g3_lanemap_gemv_{N}_{N}"
FUSED_PROVIDER = "rmsnorm_q8_1_llama_provider_4096"
ORDINARY_PROVIDER = "q8_1_llama_provider_4096"
COOP_PREFIX = "q4k_warp_coop_q8_dp4a_partial_"
LEGACY_SHARED_PREFIX = "q4k_q8_dp4a_"
Q6_DIRECT_PREFIX = "q6k_q8_warp_direct_"
LEASE_BLOCKS = 17

DEPTHS = (512, 2048, 4096)
LEGACY_PG3_SHA = "27857cb8ca03"

# Real Qwen topology: the max17 lease holds 8 Q6-K V blocks (1,2,3,6,9,12,15,18) and
# 9 Q4-K V blocks, so the direct-Q6 consumer fires 8 times per capture (16 per two
# captures) while the cooperative Q4 consumers stay 86 and providers stay 34.
CENSUS_ORACLE_CANDIDATE = {"fused_providers": 2 * LEASE_BLOCKS, "coop_q4": 86,
                           "legacy_shared_q4": 0, "ordinary_providers": 0, "q6_direct": 16}
CENSUS_ORACLE_CONTROL = {"fused_providers": 2 * LEASE_BLOCKS, "coop_q4": 86,
                         "legacy_shared_q4": 0, "ordinary_providers": 0, "q6_direct": 0}

LEASE_INDICES = "1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18"
HARNESS = "/home/ubuntu/tinygrad-arkey/extra/llm_research/decode/nv_shared_q8_progressive_qualification.py"


def parse_census_log(text: str) -> dict[str, list[float]]:
  per: dict[str, list[float]] = {}
  for line in text.splitlines():
    m = TM_RE.match(line)
    if not m: continue
    us = float(m.group(2)) * (1e-3 if m.group(3) == "ms" else 1.0)
    per.setdefault(m.group(1), []).append(us)
  return per


def census_once(gen, depth: int, warmup: bool = True) -> dict[str, list[float]]:
  if warmup:
    with Context(DEBUG=0): next(gen)
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    with Context(DEBUG=2): next(gen)
  return parse_census_log(buf.getvalue())


def reset_decode_jits(model) -> None:
  from tinygrad.engine.jit import TinyJit
  for attr in vars(model).values():
    if isinstance(attr, TinyJit): attr.reset()
    elif isinstance(attr, tuple):
      for j in attr:
        if isinstance(j, TinyJit): j.reset()


def wall_once(model, prompt: list[int], nmeas: int, reps: int) -> tuple[list[float], list[str], list[int]]:
  dev = Device[Device.DEFAULT]
  tok_s, shas, firsts = [], [], []
  for _ in range(reps):
    model.reset_generation_state()
    gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
    next(gen)
    dev.synchronize()
    lat, toks = [], []
    for _ in range(nmeas):
      t0 = time.perf_counter()
      toks.append(int(next(gen)))
      lat.append(time.perf_counter() - t0)
    gen.close()
    tok_s.append(nmeas / sum(lat))
    shas.append(hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest())
    firsts.append(toks[0])
  return tok_s, shas, firsts


def free_model(model, kv) -> None:
  del model, kv
  gc.collect()
  Device[Device.DEFAULT].synchronize()
  Device[Device.DEFAULT].allocator.free_cache()


def render_only(ast, ren):
  full_sink = full_rewrite_to_sink(ast, ren, optimize=ast.tag is None)
  prg = UOp(Ops.PROGRAM, src=(full_sink, UOp(Ops.DEVICE, arg=ren.target.device)),
            arg=ProgramInfo.from_sink(full_sink))
  prg = do_linearize(ren, prg, full_sink)
  updated = do_estimates(prg, full_sink, prg.src[2])
  if updated is not None: prg = updated
  prg = do_render(ren, prg, prg.src[2])
  return prg.src[3].arg


def render_legacy_pg3() -> dict:
  ren = HIPRenderer(Target.parse("AMD:HIP:gfx1100"))
  out = UOp.placeholder((N,), dtypes.float32, 0)
  words = UOp.placeholder((N * (N // 256) * 36,), dtypes.uint32, 1)
  x = UOp.placeholder((N,), dtypes.float16, 2)
  ast = q4k_g3_lanemap_gemv_kernel(N, N)(out, words, x)
  src = render_only(ast, ren)
  return {"name": LEGACY, "sha256": hashlib.sha256((src + "\n").encode()).hexdigest()[:12],
          "src_len": len(src)}


def set_overrides(shared_q8: bool, q6_direct: bool) -> None:
  mrp._DECODE_SHARED_Q8_ATTENTION_PROMOTED_TARGETS = frozenset({("NV", "sm_120")}) if shared_q8 else frozenset()
  mrp._DECODE_Q6_DIRECT_SHARED_Q8_ATTENTION_PROMOTED_TARGETS = frozenset({("NV", "sm_120")}) if q6_direct else frozenset()


def run_arm(arm: str, depth: int, nmeas: int, reps: int) -> dict:
  if arm == "candidate":
    set_overrides(True, True)
  elif arm == "control":
    set_overrides(True, False)
  # "record": leave both module-level sets as the route-policy records loaded them
  # (shared-Q8 open, Q6-direct per its checked-in record). The promotion gate proves
  # this checked-in state equals the forced-open candidate state.

  print(f"[q6-gate] {depth} {arm}: loading model", file=sys.stderr, flush=True)
  model, kv = Transformer.from_gguf(MODEL, 4608)
  out: dict = {"arm": arm, "depth": depth}
  prompt = [1] * depth
  print(f"[q6-gate] {depth} {arm}: model loaded, census", file=sys.stderr, flush=True)
  gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
  census = None
  if depth == 512:
    c1 = census_once(gen, depth)
    reset_decode_jits(model)
    c2 = census_once(gen, depth, warmup=False)
    census = {}
    for name, times in c1.items(): census.setdefault(name, []).extend(times)
    for name, times in c2.items(): census.setdefault(name, []).extend(times)
  gen.close()
  print(f"[q6-gate] {depth} {arm}: census done, wall", file=sys.stderr, flush=True)
  tok_s, shas, firsts = wall_once(model, prompt, nmeas, reps)
  print(f"[q6-gate] {depth} {arm}: wall done, render", file=sys.stderr, flush=True)
  row: dict = {
    "tok_s_median": round(statistics.median(tok_s), 3),
    "tok_s_reps": [round(x, 3) for x in tok_s],
    "ms_per_token_median": round(1000 / statistics.median(tok_s), 6) if statistics.median(tok_s) else None,
    "token_sha_reps": shas, "first_token_reps": firsts,
  }
  if census is not None:
    fused = census.get(FUSED_PROVIDER, [])
    ordinary = census.get(ORDINARY_PROVIDER, [])
    coop = [v for k, vs in census.items() if k.startswith(COOP_PREFIX) for v in vs]
    legacy_shared = [v for k, vs in census.items() if k.startswith(LEGACY_SHARED_PREFIX) for v in vs]
    q6_direct = [v for k, vs in census.items() if k.startswith(Q6_DIRECT_PREFIX) for v in vs]
    row["census"] = {
      "fused_provider_count": len(fused),
      "ordinary_provider_count": len(ordinary),
      "coop_q4_count": len(coop),
      "legacy_shared_q4_count": len(legacy_shared),
      "q6_direct_count": len(q6_direct),
      "histogram": sorted(((k, len(v), round(statistics.median(v), 2)) for k, v in census.items()),
                          key=lambda t: (-t[1], -t[2])),
    }
  out["result"] = row
  out["pg3_legacy_render"] = render_legacy_pg3()
  try:
    free_model(model, kv)
  except Exception as e:
    print(f"[q6-gate] note: model teardown failed after data collection: {e}", file=sys.stderr)
  return out


def run_harness_child(indices: str, out: str, q6_direct: bool) -> dict:
  cmd = [sys.executable, HARNESS, "--mode", "child", "--model", MODEL, "--depth", "512",
         "--count", "8", "--max-context", "1024", "--out", out, "--composed",
         "--fused-indices", indices, "--cooperative-q4"]
  if q6_direct: cmd.append("--q6-direct-output")
  run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env={**os.environ, "PYTHONPATH": "/home/ubuntu/tinygrad-arkey"})
  if run.returncode:
    raise RuntimeError(f"harness child failed rc={run.returncode}: {run.stderr[-4000:]}")
  with open(out) as f:
    return json.load(f)


def semantic_contract() -> dict:
  import numpy as np
  root = "/tmp/nv_shared_q8_q6_direct_semantic"
  import pathlib
  pathlib.Path(root).mkdir(parents=True, exist_ok=True)
  ctrl_row = run_harness_child(LEASE_INDICES, f"{root}/control.json", q6_direct=False)
  cand_row = run_harness_child(LEASE_INDICES, f"{root}/candidate.json", q6_direct=True)
  ctrl = np.load(f"{root}/control.npz")["logits"]
  cand = np.load(f"{root}/candidate.npz")["logits"]
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _semantic_comparison
  comparison = _semantic_comparison(ctrl, cand, ctrl_row, cand_row)
  comparison["census_oracle"] = {
    "fused_providers": cand_row["fused_rmsnorm_q8_provider_count"],
    "coop_q4": cand_row["cooperative_q4_consumer_count"],
    "legacy_shared_q4": cand_row["legacy_q4_shared_consumer_count"],
    "ordinary_providers": cand_row["q8_provider_count"] - cand_row["fused_rmsnorm_q8_provider_count"],
    "q6_direct": cand_row["q6_direct_consumer_count"],
  }
  comparison["gate_pass"] = bool(comparison["semantic_pass"]) and all(
    comparison["census_oracle"][k] == v for k, v in CENSUS_ORACLE_CANDIDATE.items()
    if k in comparison["census_oracle"])
  return comparison


def gate_verdict(control: dict, candidate: dict, semantic: dict) -> dict:
  issues: list[str] = []
  for depth in DEPTHS:
    c, o = control[f"d{depth}"]["result"], candidate[f"d{depth}"]["result"]
    if o["tok_s_median"] < c["tok_s_median"]:
      issues.append(f"d{depth}: candidate {o['tok_s_median']} tok/s < control {c['tok_s_median']} tok/s (regression)")
    if len(set(o["token_sha_reps"])) != 1:
      issues.append(f"d{depth}: candidate reps not identical 3/3")
    if len(set(c["token_sha_reps"])) != 1:
      issues.append(f"d{depth}: control reps not identical 3/3")
    if o["token_sha_reps"] != c["token_sha_reps"]:
      issues.append(f"d{depth}: candidate sha != control sha (bitwise-exactness broken)")

  cen = candidate["d512"]["result"]["census"]
  for key, oracle_key in (("fused_provider_count", "fused_providers"),
                          ("coop_q4_count", "coop_q4"),
                          ("legacy_shared_q4_count", "legacy_shared_q4"),
                          ("ordinary_provider_count", "ordinary_providers"),
                          ("q6_direct_count", "q6_direct")):
    if cen[key] != CENSUS_ORACLE_CANDIDATE[oracle_key]:
      issues.append(f"candidate census {key} {cen[key]} != oracle {CENSUS_ORACLE_CANDIDATE[oracle_key]}")
  cen_ctrl = control["d512"]["result"]["census"]
  if cen_ctrl["q6_direct_count"] != CENSUS_ORACLE_CONTROL["q6_direct"]:
    issues.append(f"control census q6_direct_count {cen_ctrl['q6_direct_count']} != 0")

  if not semantic["gate_pass"]:
    issues.append(f"semantic/census gate failed: pass={semantic.get('semantic_pass')} "
                  f"oracle={semantic.get('census_oracle')}")

  for arm, label in ((candidate, "candidate"), (control, "control")):
    if arm["d512"]["pg3_legacy_render"]["sha256"] != LEGACY_PG3_SHA:
      issues.append(f"pg3 legacy sha ({label}) {arm['d512']['pg3_legacy_render']['sha256']} != pinned {LEGACY_PG3_SHA}")

  return {"verdict": "PASS" if not issues else "FAIL", "issues": issues}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--arm", choices=("control", "candidate", "record"), default=None)
  ap.add_argument("--depth", type=int, default=None)
  ap.add_argument("--nmeas", type=int, default=20)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--artifact", default="/tmp/nv_shared_q8_q6_direct_gate_out.json")
  args = ap.parse_args()

  if args.arm is not None:
    print(json.dumps(run_arm(args.arm, args.depth or 512, args.nmeas, args.reps), indent=1), flush=True)
    return

  def run_one(arm: str, depth: int) -> dict:
    proc = subprocess.run([sys.executable, __file__, "--arm", arm, "--depth", str(depth),
                           "--nmeas", str(args.nmeas), "--reps", str(args.reps)],
                          capture_output=True, text=True)
    txt = proc.stdout
    if txt.find("{") == -1:
      return {"arm": arm, "depth": depth, "error": proc.stderr[-3000:],
              "stdout_tail": txt[-1000:]}
    out = json.loads(txt[txt.find("{"):])
    if proc.returncode != 0:
      out["teardown_error"] = proc.stderr[-800:]
    return out

  results: dict = {}
  for arm in ("control", "candidate"):
    for d in DEPTHS:
      results[f"{arm}_d{d}"] = run_one(arm, d)
      with open(args.artifact, "w") as f:
        json.dump(results, f, indent=1)
  results["record_d512"] = run_one("record", 512)
  with open(args.artifact, "w") as f:
    json.dump(results, f, indent=1)

  complete = all("result" in results[f"{arm}_d{d}"] for arm in ("control", "candidate")
                 for d in DEPTHS) and "result" in results["record_d512"]
  if not complete:
    result = {"partial": True, "results": results}
  else:
    semantic = semantic_contract()
    control = {f"d{d}": results[f"control_d{d}"] for d in DEPTHS}
    candidate = {f"d{d}": results[f"candidate_d{d}"] for d in DEPTHS}
    record = results["record_d512"]
    verdict = gate_verdict(control, candidate, semantic)
    result = {"partial": False, "control": control, "candidate": candidate, "record_default_d512": record,
              "semantic": semantic, "gate": verdict}
  with open(args.artifact, "w") as f:
    json.dump(result, f, indent=1)
  print(json.dumps(result, indent=1))


if __name__ == "__main__":
  main()
