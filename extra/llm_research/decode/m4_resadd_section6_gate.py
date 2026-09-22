"""M4 residual_add landing section-6 full gate (GPU, serialized, flock).

Scope: `docs/task_workflow/input/m4-resadd-landing-scope-20260806.md` section 3.
Question: with the PRODUCTION residual-slot fold active (no admission surgery), does the
open mode (per-variant record forced open for NV sm_120) clear the full gate?

Gate items:
  1. d512/d2048/d4096 wall, open mode: must not regress the M2-on baseline
     (172.80 / 161.50 / 149.00 tok/s) and must show a positive delta vs the same-session
     closed mode (copy-free residual_add fusion is the only difference).
  2. d512 census, open mode: epi_resadd 36, legacy attn_qo GEMV 0 (legacy 4096_4096 total
     36 = ffn_down only), E_32_32_4_86a2 copy class back to the control baseline 1
     (residual-slot copies 0), E_32_32_4_02a residual-add class 36 (ffn_down only),
     kernels/token 912 (984 variant row minus 72 residual-slot copies).
  3. Pins 3/3 at every depth, both modes: d512 closed sha `227ad3ce...` first 271; open
     and closed streams identical at every depth (bitwise-exactness).
  4. Legacy pg3 render sha `27857cb8ca03` for `q4k_g3_lanemap_gemv_4096_4096` unmoved.

Open mode = gate forced open via the module override
`mrp._DECODE_Q4K_EPILOGUE_RESADD_PROMOTED_TARGETS = frozenset({("NV","sm_120")})` BEFORE
`Transformer.from_gguf`; the production wiring (model.py per-block flag, qk_primitives
install site, decode_routes ResidualViewRequest, kernel_program fold) handles everything
else. Closed mode = default records (fold dormant). No admission surgery.

Each arm runs as a fresh subprocess (a fresh interpreter releases the model's GPU memory
completely). Run the whole gate under `flock -w 600 /tmp/gpu-bench.lock`.
"""
from __future__ import annotations

import argparse, contextlib, gc, hashlib, io, json, re, statistics, subprocess, sys, time
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
EPI_RESADD = f"q4k_g3_lanemap_gemv_epi_resadd_{N}_{N}"
LEGACY = f"q4k_g3_lanemap_gemv_{N}_{N}"
COPY_PREFIX = "E_32_32_4_86a2"
RESADD_PREFIX = "E_32_32_4_02a"
LAYERS = 36

DEPTHS = (512, 2048, 4096)
BASELINE = {512: 172.80, 2048: 161.50, 4096: 149.00}
PIN_SHA = "227ad3ce9621f2c382cc722a3c2f1677637d3e3f2bfbf37d6ca652f98880eb4e"
PIN_FIRST = 271
LEGACY_PG3_SHA = "27857cb8ca03"


def parse_census_log(text: str) -> dict[str, list[float]]:
  per: dict[str, list[float]] = {}
  for line in text.splitlines():
    m = TM_RE.match(line)
    if not m: continue
    us = float(m.group(2)) * (1e-3 if m.group(3) == "ms" else 1.0)
    per.setdefault(m.group(1), []).append(us)
  return per


def census_once(gen, depth: int) -> dict[str, list[float]]:
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
  """pg3_proto render path (AMD:HIP:gfx1100 pins the pg3 table; house convention)."""
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
    mrp._DECODE_Q4K_EPILOGUE_RESADD_PROMOTED_TARGETS = frozenset({("NV", "sm_120")})
  elif arm == "closed":
    mrp._DECODE_Q4K_EPILOGUE_RESADD_PROMOTED_TARGETS = frozenset()
  # "record": leave the module-level set as the route-policy record loaded it.
  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()

  print(f"[gate] {depth} {arm}: loading model", file=sys.stderr, flush=True)
  model, kv = Transformer.from_gguf(MODEL, 4608)
  out: dict = {"arm": arm, "depth": depth}
  prompt = [1] * depth
  print(f"[gate] {depth} {arm}: model loaded, generating", file=sys.stderr, flush=True)
  gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
  census = census_once(gen, depth) if depth == 512 else None
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
    epi = census.get(EPI_RESADD, [])
    legacy = census.get(LEGACY, [])
    copies = [v for k, vs in census.items() if k.startswith(COPY_PREFIX) for v in vs]
    resadd = [v for k, vs in census.items() if k.startswith(RESADD_PREFIX) for v in vs]
    row["census"] = {
      "kernels_per_token": total_kernels, "kernel_us_total": round(total_us, 1),
      "epi_resadd_count": len(epi), "epi_resadd_median_us": round(statistics.median(epi), 3) if epi else None,
      "legacy_gemv_count": len(legacy), "legacy_gemv_median_us": round(statistics.median(legacy), 3) if legacy else None,
      "copy_class_count": len(copies), "copy_class_median_us": round(statistics.median(copies), 3) if copies else None,
      "resadd_class_count": len(resadd), "resadd_class_median_us": round(statistics.median(resadd), 3) if resadd else None,
      "histogram": sorted(((k, len(v), round(statistics.median(v), 2)) for k, v in census.items()),
                          key=lambda t: (-t[1], -t[2])),
    }
  out["result"] = row
  out["pg3_legacy_render"] = render_legacy_pg3()
  print(json.dumps(out, indent=1), flush=True)
  # Data first: the HCQ device can stall during teardown (observed 30s timeline wait timeout),
  # and losing a completed arm to a free-side error wastes a full model load. Free is best-effort.
  try:
    free_model(model, kv)
  except Exception as e:
    print(f"[gate] note: model teardown failed after data collection: {e}", file=sys.stderr)
  return out


def run_arm_record(depth: int, nmeas: int, reps: int) -> dict:
  """record mode: no module override, the checked-in route-policy JSON decides."""
  return run_arm("record", depth, nmeas, reps)


def gate_verdict(closed: dict, opened: dict) -> dict:
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
  c512 = closed["d512"]["result"]
  if c512["token_sha_reps"][0] != PIN_SHA:
    issues.append(f"d512 closed sha != pin {PIN_SHA[:12]}...")
  if set(c512["first_token_reps"]) != {PIN_FIRST}:
    issues.append(f"d512 closed first token != pin {PIN_FIRST}")

  cen = opened["d512"]["result"]["census"]
  if cen["epi_resadd_count"] != LAYERS:
    issues.append(f"census epi_resadd {cen['epi_resadd_count']} != {LAYERS}")
  if cen["legacy_gemv_count"] != LAYERS:
    issues.append(f"census legacy 4096_4096 {cen['legacy_gemv_count']} != {LAYERS} (attn_qo legacy should be 0)")
  if cen["copy_class_count"] != 1:
    issues.append(f"census copy class {cen['copy_class_count']} != 1 (residual-slot copies should be 0)")
  if cen["resadd_class_count"] != LAYERS:
    issues.append(f"census resadd class {cen['resadd_class_count']} != {LAYERS} (attn_qo residual-add should be 0)")
  if cen["kernels_per_token"] != 912:
    issues.append(f"census kernels/token {cen['kernels_per_token']} != 912")

  if opened["pg3_legacy_render"]["sha256"] != LEGACY_PG3_SHA:
    issues.append(f"pg3 legacy sha {opened['pg3_legacy_render']['sha256']} != pinned {LEGACY_PG3_SHA}")
  if closed["pg3_legacy_render"]["sha256"] != LEGACY_PG3_SHA:
    issues.append(f"pg3 legacy sha (closed) {closed['pg3_legacy_render']['sha256']} != pinned {LEGACY_PG3_SHA}")

  return {"verdict": "PASS" if not issues else "FAIL", "issues": issues}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--arm", choices=("closed", "open", "record"), default=None,
                  help="run a single arm/depth; omit to run all arms and depths as fresh subprocesses")
  ap.add_argument("--depth", type=int, default=None,
                  help="depth for --arm mode (default 512)")
  ap.add_argument("--nmeas", type=int, default=20)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--artifact", default="/tmp/m4_resadd_section6_gate_out.json")
  args = ap.parse_args()

  if args.arm is not None:
    # record mode: leave the module-level target set untouched so the checked-in
    # route-policy record itself decides (post-promotion equality check).
    if args.arm == "record":
      print(json.dumps(run_arm_record(args.depth or 512, args.nmeas, args.reps), indent=1), flush=True)
    else:
      run_arm(args.arm, args.depth or 512, args.nmeas, args.reps)
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
    closed = {f"d{d}": results[f"closed_d{d}"] for d in DEPTHS}
    opened = {f"d{d}": results[f"open_d{d}"] for d in DEPTHS}
    record = {f"d{d}": results[f"record_d{d}"] for d in DEPTHS}
    verdict = gate_verdict(closed, opened)
    result = {"partial": False, "closed": closed, "open": opened, "record_default": record,
              "gate": verdict}
  with open(args.artifact, "w") as f:
    json.dump(result, f, indent=1)
  print(json.dumps(result, indent=1))


if __name__ == "__main__":
  main()
