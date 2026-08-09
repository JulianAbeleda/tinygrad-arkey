"""Shared-Q8 attention landing section-6 full gate (GPU, serialized, flock).

Scope: `docs/task_workflow/input/nv-gemv-substrate-landing-scope-20260808.md` section 4.
Question: with the PRODUCTION loader-installed max17 cooperative lease (blocks 1-12 and
14-18, `SharedQ8AttentionAdmission(cooperative_q4=True)`), does the open mode (record forced
open for NV sm_120) clear the full gate?

Gate items:
  1. Wall d512/d2048/d4096, open mode: must not regress the M2-on baseline
     (172.80 / 161.50 / 149.00 tok/s) and must show a positive delta vs the same-session
     closed mode (the cooperative Q8_1+DP4A lease is the only difference).
  2. Census (d512, open mode, two captures): fused RMSNorm/Q8 provider 34 (17 blocks x two
     captures), cooperative Q4 consumers 86, legacy shared-Q4 consumers 0, ordinary
     providers 0 (no duplicates), per the max17 record oracle.
  3. Semantic contract (harness child, control vs candidate): exact token stream, equal
     argmax, ordered top-10 sets, aggregate relative L2 <= 1e-3,
     2*max_abs/min_top1_margin < 1.0, all finite.
  4. Pins 3/3 at every depth, both modes: d512 sha `227ad3ce...` first `271`, d2048 sha
     `aca13ac6...` (also first `271`), d4096 sha `d9f1700a...` (first `374`); open and
     closed streams identical at every depth.
  5. Unit tests green (run separately).
  6. pg3 legacy sha `27857cb8ca03` for `q4k_g3_lanemap_gemv_4096_4096` unmoved.

Open mode = gate forced open via the module override
`mrp._DECODE_SHARED_Q8_ATTENTION_PROMOTED_TARGETS = frozenset({("NV","sm_120")})` BEFORE
`Transformer.from_gguf`; the production wiring (model.py lease install, shared_q8_attention_call
route) handles everything else. Closed mode = default records (lease dormant). Record mode =
checked-in policy JSON. Each arm runs as a fresh subprocess. Run under
`flock -w 600 /tmp/gpu-bench.lock`.
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
LAYERS = 36
LEASE_BLOCKS = 17

DEPTHS = (512, 2048, 4096)
BASELINE = {512: 172.80, 2048: 161.50, 4096: 149.00}
PIN_SHA = {512: "227ad3ce9621f2c382cc722a3c2f1677637d3e3f2bfbf37d6ca652f98880eb4e",
           2048: "aca13ac6d085808f43111945d9353a7491ecb45b261beb55acf11aaeaec8ea1d",
           4096: "d9f1700aac269e5b5f9667c280ba0e744b6516566d7a2ba666712aef3f4dd9e1"}
PIN_FIRST = 271
# The first-token pin was validated for the shallow depths only; d4096 deterministically
# samples `374` (identical across closed/open arms and every rep, pre- and post-fix).
PIN_FIRST_DEPTHS = frozenset({512, 2048})
LEGACY_PG3_SHA = "27857cb8ca03"

# Max17 record oracle, per two captures (composed harness protocol).
CENSUS_ORACLE = {"fused_providers": 2 * LEASE_BLOCKS, "coop_q4": 86, "legacy_shared_q4": 0,
                 "ordinary_providers": 0, "q6_direct": 0}

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


def census_once(gen, depth: int) -> dict[str, list[float]]:
  """One token capture (DEBUG=2). Two captures are summed by the caller for the oracle."""
  with Context(DEBUG=0): next(gen)
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    with Context(DEBUG=2): next(gen)
  return parse_census_log(buf.getvalue())


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


def run_arm(arm: str, depth: int, nmeas: int, reps: int) -> dict:
  if arm == "open":
    mrp._DECODE_SHARED_Q8_ATTENTION_PROMOTED_TARGETS = frozenset({("NV", "sm_120")})
  elif arm == "closed":
    mrp._DECODE_SHARED_Q8_ATTENTION_PROMOTED_TARGETS = frozenset()
  # "record": leave the module-level set as the route-policy record loaded it.
  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()

  print(f"[gate] {depth} {arm}: loading model", file=sys.stderr, flush=True)
  model, kv = Transformer.from_gguf(MODEL, 4608)
  out: dict = {"arm": arm, "depth": depth}
  prompt = [1] * depth
  print(f"[gate] {depth} {arm}: model loaded, census", file=sys.stderr, flush=True)
  gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
  census = None
  if depth == 512:
    # Two captures (the max17 oracle is per two captures), summed below.
    c1, c2 = census_once(gen, depth), census_once(gen, depth)
    census = {}
    for name, times in c1.items(): census.setdefault(name, []).extend(times)
    for name, times in c2.items(): census.setdefault(name, []).extend(times)
  gen.close()
  print(f"[gate] {depth} {arm}: census done, wall", file=sys.stderr, flush=True)
  tok_s, shas, firsts = wall_once(model, prompt, nmeas, reps)
  print(f"[gate] {depth} {arm}: wall done, render", file=sys.stderr, flush=True)
  row: dict = {
    "tok_s_median": round(statistics.median(tok_s), 3),
    "tok_s_reps": [round(x, 3) for x in tok_s],
    "token_sha_reps": shas, "first_token_reps": firsts,
  }
  if census is not None:
    total_kernels = sum(len(v) for v in census.values())
    total_us = sum(statistics.median(v) * len(v) for v in census.values())
    fused = census.get(FUSED_PROVIDER, [])
    ordinary = census.get(ORDINARY_PROVIDER, [])
    coop = [v for k, vs in census.items() if k.startswith(COOP_PREFIX) for v in vs]
    legacy_shared = [v for k, vs in census.items() if k.startswith(LEGACY_SHARED_PREFIX) for v in vs]
    q6_direct = [v for k, vs in census.items() if k.startswith(Q6_DIRECT_PREFIX) for v in vs]
    row["census"] = {
      "kernels_per_token": total_kernels, "kernel_us_total": round(total_us, 1),
      "fused_provider_count": len(fused), "fused_provider_median_us": round(statistics.median(fused), 3) if fused else None,
      "ordinary_provider_count": len(ordinary),
      "coop_q4_count": len(coop), "coop_q4_median_us": round(statistics.median(coop), 3) if coop else None,
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
    print(f"[gate] note: model teardown failed after data collection: {e}", file=sys.stderr)
  return out


def run_arm_record(depth: int, nmeas: int, reps: int) -> dict:
  return run_arm("record", depth, nmeas, reps)


def run_harness_child(indices: str, out: str, cooperative: bool) -> dict:
  cmd = [sys.executable, HARNESS, "--mode", "child", "--model", MODEL, "--depth", "512",
         "--count", "8", "--max-context", "1024", "--out", out, "--composed"]
  if indices: cmd += ["--fused-indices", indices]
  if cooperative: cmd.append("--cooperative-q4")
  run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env={**os.environ, "PYTHONPATH": "/home/ubuntu/tinygrad-arkey"})
  if run.returncode:
    raise RuntimeError(f"harness child failed rc={run.returncode}: {run.stderr[-4000:]}")
  with open(out) as f:
    return json.load(f)


def semantic_contract() -> dict:
  """Control vs candidate full-logit comparison via the qualification harness."""
  import numpy as np
  root = "/tmp/nv_shared_q8_semantic"
  import pathlib
  pathlib.Path(root).mkdir(parents=True, exist_ok=True)
  ctrl_row = run_harness_child("", f"{root}/control.json", cooperative=False)
  cand_row = run_harness_child(LEASE_INDICES, f"{root}/candidate.json", cooperative=True)
  ctrl = np.load(f"{root}/control.npz")["logits"]
  cand = np.load(f"{root}/candidate.npz")["logits"]
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _semantic_comparison
  comparison = _semantic_comparison(ctrl, cand, ctrl_row, cand_row)
  comparison["census_oracle"] = {
    "fused_providers": cand_row["fused_rmsnorm_q8_provider_count"],
    "coop_q4": cand_row["cooperative_q4_consumer_count"],
    "legacy_shared_q4": cand_row["legacy_q4_shared_consumer_count"],
    "ordinary_providers": cand_row["q8_provider_count"] - cand_row["fused_rmsnorm_q8_provider_count"],
  }
  comparison["gate_pass"] = bool(comparison["semantic_pass"]) and all(
    comparison["census_oracle"][k] == v for k, v in CENSUS_ORACLE.items() if k in comparison["census_oracle"])
  return comparison


def gate_verdict(closed: dict, opened: dict, semantic: dict) -> dict:
  issues: list[str] = []
  for depth in DEPTHS:
    c, o = closed[f"d{depth}"]["result"], opened[f"d{depth}"]["result"]
    if o["tok_s_median"] < BASELINE[depth]:
      issues.append(f"d{depth}: open {o['tok_s_median']} < M2-on baseline {BASELINE[depth]}")
    if o["tok_s_median"] <= c["tok_s_median"]:
      issues.append(f"d{depth}: open {o['tok_s_median']} <= closed {c['tok_s_median']} (no positive delta)")
    if len(set(o["token_sha_reps"])) != 1:
      issues.append(f"d{depth}: open reps not identical 3/3")
    if len(set(c["token_sha_reps"])) != 1:
      issues.append(f"d{depth}: closed reps not identical 3/3")
    if o["token_sha_reps"] != c["token_sha_reps"]:
      issues.append(f"d{depth}: open sha != closed sha (bitwise-exactness broken)")
    if depth in PIN_SHA and c["token_sha_reps"][0] != PIN_SHA[depth]:
      issues.append(f"d{depth}: closed sha != pin {PIN_SHA[depth][:12]}...")
    if depth in PIN_FIRST_DEPTHS and set(c["first_token_reps"]) != {PIN_FIRST}:
      issues.append(f"d{depth}: closed first token != pin {PIN_FIRST}")

  cen = opened["d512"]["result"]["census"]
  for key, oracle_key in (("fused_provider_count", "fused_providers"),
                          ("coop_q4_count", "coop_q4"),
                          ("legacy_shared_q4_count", "legacy_shared_q4"),
                          ("ordinary_provider_count", "ordinary_providers"),
                          ("q6_direct_count", "q6_direct")):
    if cen[key] != CENSUS_ORACLE[oracle_key]:
      issues.append(f"census {key} {cen[key]} != oracle {CENSUS_ORACLE[oracle_key]}")

  if not semantic["gate_pass"]:
    issues.append(f"semantic/census gate failed: pass={semantic.get('semantic_pass')} "
                  f"oracle={semantic.get('census_oracle')}")

  # render_legacy_pg3 is arm/depth-independent, so the d512 slot stands for the run.
  for arm, label in ((opened, "open"), (closed, "closed")):
    if arm["d512"]["pg3_legacy_render"]["sha256"] != LEGACY_PG3_SHA:
      issues.append(f"pg3 legacy sha ({label}) {arm['d512']['pg3_legacy_render']['sha256']} != pinned {LEGACY_PG3_SHA}")

  return {"verdict": "PASS" if not issues else "FAIL", "issues": issues}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--arm", choices=("closed", "open", "record"), default=None)
  ap.add_argument("--depth", type=int, default=None)
  ap.add_argument("--nmeas", type=int, default=20)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--artifact", default="/tmp/nv_shared_q8_section6_gate_out.json")
  args = ap.parse_args()

  if args.arm is not None:
    if args.arm == "record":
      print(json.dumps(run_arm_record(args.depth or 512, args.nmeas, args.reps), indent=1), flush=True)
    else:
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
  for arm in ("closed", "open", "record"):
    for d in DEPTHS:
      results[f"{arm}_d{d}"] = run_one(arm, d)
      with open(args.artifact, "w") as f:
        json.dump(results, f, indent=1)
  complete = all("result" in results[f"{arm}_d{d}"] for arm in ("closed", "open", "record")
                 for d in DEPTHS)
  if not complete:
    result = {"partial": True, "results": results}
  else:
    semantic = semantic_contract()
    closed = {f"d{d}": results[f"closed_d{d}"] for d in DEPTHS}
    opened = {f"d{d}": results[f"open_d{d}"] for d in DEPTHS}
    record = {f"d{d}": results[f"record_d{d}"] for d in DEPTHS}
    verdict = gate_verdict(closed, opened, semantic)
    result = {"partial": False, "closed": closed, "open": opened, "record_default": record,
              "semantic": semantic, "gate": verdict}
  with open(args.artifact, "w") as f:
    json.dump(result, f, indent=1)
  print(json.dumps(result, indent=1))


if __name__ == "__main__":
  main()
