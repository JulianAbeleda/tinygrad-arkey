"""M4 variant-reopen P0 probe 2 - fused epi_resadd per-kernel economics (GPU, serialized).

Scope: `docs/task_workflow/input/m4-variant-reopen-boundary-p0-scope-20260806.md` section 4,
probe 2. Question: is `q4k_g3_lanemap_gemv_epi_resadd_4096_4096` economical per kernel vs the
legacy `q4k_g3_lanemap_gemv_4096_4096`, in isolation (one variant open)?

This probe runs BOTH arms in one session (control all-closed, then residual_add one-open) so
the per-kernel medians come from the same GPU state. It extracts per-kernel medians for the
legacy attn_qo GEMV (control census), the fused epi_resadd GEMV (variant census), and the
E_32_32_4_86a2 opaque-boundary copy class, then computes the copy-free ceiling arithmetic:
copy_mass - fused_penalty must be positive (or at parity), and the fused kernel median must
not regress the legacy per-kernel time materially (>10% is material here).

Protocol (M4 decomposition record): Qwen3-8B-Q4_K_M, d512, nmeas 20, reps 3, tokens pinned
(token sha 9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9, first token
151936). Must run under `flock -w 600 /tmp/gpu-bench.lock`.

The probe is an evaluation, not an implementation: it never edits kernel_program.py /
decode_routes.py / decode_kernels.py, and it touches no promotion record.
"""
from __future__ import annotations

import argparse, contextlib, dataclasses, gc, hashlib, io, json, re, statistics, sys, time
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")
from tinygrad import Device, Tensor
from tinygrad.uop.ops import UOp
from tinygrad.helpers import Context
from tinygrad.llm.model import Transformer
from tinygrad.llm.qk_primitives import Q4KPrimitiveLinear
import tinygrad.llm.model_route_plan as mrp
import tinygrad.llm.model as tgm

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
TM_RE = re.compile(r"^\*\*\* NV\s+\d+\s+(\S+)\s+arg\s+\d+.*?tm\s+([\d.]+)(us|ms)/")

N = 4096
FUSED = f"q4k_g3_lanemap_gemv_epi_resadd_{N}_{N}"
LEGACY = f"q4k_g3_lanemap_gemv_{N}_{N}"
COPY_PREFIX = "E_32_32_4_86a2"
LAYERS = 36


def parse_census_log(text: str) -> dict[str, list[float]]:
  """Split a DEBUG=2 census log into per-kernel median sources (us per launch)."""
  per: dict[str, list[float]] = {}
  for line in text.splitlines():
    m = TM_RE.match(line)
    if not m: continue
    us = float(m.group(2)) * (1e-3 if m.group(3) == "ms" else 1.0)
    per.setdefault(m.group(1), []).append(us)
  return per


def extract_rows(per: dict[str, list[float]]) -> dict:
  """Pull the three load-bearing rows from a census: fused, legacy, and the copy class."""
  fused = per.get(FUSED, [])
  legacy = per.get(LEGACY, [])
  copies = [v for k, vs in per.items() if k.startswith(COPY_PREFIX) for v in vs]
  return {
    "fused_count": len(fused), "fused_median_us": round(statistics.median(fused), 3) if fused else None,
    "legacy_count": len(legacy), "legacy_median_us": round(statistics.median(legacy), 3) if legacy else None,
    "copy_count": len(copies), "copy_median_us": round(statistics.median(copies), 3) if copies else None,
  }


def microgate_verdict(rows: dict, book_copy_count: int = LAYERS, book_copy_us: float = 1.47) -> dict:
  """Copy-free ceiling arithmetic + verdict. The ceiling is the copy mass we would remove
  (measured, plus the scope's book value 36 x 1.47us) minus the fused-kernel penalty over the
  legacy baseline. Positive ceiling and a <=10% per-kernel regression is the PASS gate."""
  fused, legacy = rows["fused_median_us"], rows["legacy_median_us"]
  if fused is None or legacy is None:
    return {"verdict": "NO-CENSUS", "reason": "fused or legacy median missing"}
  penalty = LAYERS * (fused - legacy)
  measured_mass = (rows["copy_count"] or 0) * (rows["copy_median_us"] or 0.0)
  book_mass = book_copy_count * book_copy_us
  ceiling_measured = measured_mass - penalty
  ceiling_book = book_mass - penalty
  pct = (fused - legacy) / legacy * 100.0
  regress_ok = pct <= 10.0
  verdict = "PASS" if (regress_ok and ceiling_measured > 0 and ceiling_book > 0) else "FAIL"
  return {
    "verdict": verdict,
    "fused_vs_legacy_pct": round(pct, 2),
    "per_kernel_penalty_us": round(fused - legacy, 3),
    "fused_penalty_us_total": round(penalty, 2),
    "copy_mass_measured_us": round(measured_mass, 2),
    "copy_mass_book_us": round(book_mass, 2),
    "copy_free_ceiling_measured_us": round(ceiling_measured, 2),
    "copy_free_ceiling_book_us": round(ceiling_book, 2),
    "regress_ok": regress_ok,
  }


def gate_admissions(model, target_role: str | None):
  """One-variant-open admission surgery (m4_decomp protocol): True only for target_role."""
  seen, n_target, n_closed = set(), 0, 0
  def walk(obj):
    nonlocal n_target, n_closed
    if isinstance(obj, Q4KPrimitiveLinear):
      adm = getattr(obj, "route_admission", None)
      if adm is None: return
      role = getattr(obj, "route_role", "")
      want = (role == target_role) if target_role else False
      if want:
        n_target += 1
        obj.route_admission = dataclasses.replace(adm, q4k_epilogue_fusion_promoted=True)
      else:
        n_closed += 1
        obj.route_admission = dataclasses.replace(adm, q4k_epilogue_fusion_promoted=False)
      return
    if id(obj) in seen: return
    seen.add(id(obj))
    if isinstance(obj, dict):
      for v in obj.values(): walk(v)
    elif isinstance(obj, (list, tuple, set, frozenset)):
      for v in obj: walk(v)
    elif not isinstance(obj, (Tensor, UOp)) and hasattr(obj, "__dict__"):
      for v in vars(obj).values(): walk(v)
  walk(model)
  return n_target, n_closed


def census_once(gen, depth):
  """Prime with DEBUG=0, then capture exactly one decode token's DEBUG=2 log."""
  with Context(DEBUG=0): next(gen)
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    with Context(DEBUG=2): next(gen)
  return parse_census_log(buf.getvalue())


def wall_once(model, prompt, nmeas: int, reps: int):
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
  """Release the model's GPU buffers so the second arm can load in the same session."""
  del model, kv
  gc.collect()
  Device[Device.DEFAULT].synchronize()
  Device[Device.DEFAULT].allocator.free_cache()


def run_arm(open_variant: bool, depth: int, nmeas: int, reps: int) -> dict:
  if open_variant:
    mrp._DECODE_Q4K_EPILOGUE_FUSION_PROMOTED_TARGETS = frozenset({("NV", "sm_120")})
  else:
    mrp._DECODE_Q4K_EPILOGUE_FUSION_PROMOTED_TARGETS = frozenset()
  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()

  model, kv = Transformer.from_gguf(MODEL, 4608)
  n_target = n_closed = 0
  if open_variant:
    n_target, n_closed = gate_admissions(model, "attn_qo")

  prompt = [1] * depth
  gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
  per = census_once(gen, depth)
  gen.close()
  rows = extract_rows(per)
  tok_s, shas, firsts = wall_once(model, prompt, nmeas, reps)
  verdict = microgate_verdict(rows)

  total_kernels = sum(len(v) for v in per.values())
  total_us = sum(statistics.median(v) * len(v) for v in per.values())
  out = {
    "arm": "variant" if open_variant else "control", "depth": depth,
    "gate_admitted_target": n_target, "gate_closed_other": n_closed,
    "kernels_per_token": total_kernels,
    "kernel_us_total": round(total_us, 1),
    "rows": rows, "verdict": verdict,
    "tok_s_median": round(statistics.median(tok_s), 3),
    "tok_s_reps": [round(x, 3) for x in tok_s],
    "token_sha_reps": shas, "first_token_reps": firsts,
    "histogram": sorted(((k, len(v), round(statistics.median(v), 2)) for k, v in per.items()),
                        key=lambda t: (-t[1], -t[2])),
  }
  free_model(model, kv)
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--nmeas", type=int, default=20)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--arm", choices=("control", "variant"), default=None,
                  help="run a single arm; omit to run both arms as fresh subprocesses "
                       "(a fresh interpreter per arm releases the model's GPU memory completely)")
  args = ap.parse_args()

  if args.arm is not None:
    arm = run_arm(args.arm == "variant", args.depth, args.nmeas, args.reps)
    print(json.dumps({args.arm: arm}, indent=1))
    return

  import subprocess
  def run_one(arm: str) -> dict:
    proc = subprocess.run(
      [sys.executable, __file__, "--arm", arm, "--depth", str(args.depth),
       "--nmeas", str(args.nmeas), "--reps", str(args.reps)],
      capture_output=True, text=True)
    if proc.returncode != 0:
      raise RuntimeError(f"arm {arm} failed:\n{proc.stderr[-2000:]}")
    txt = proc.stdout
    return json.loads(txt[txt.find("{"):])

  control = run_one("control")
  variant = run_one("variant")
  print(json.dumps({"control": control, "variant": variant}, indent=1))


if __name__ == "__main__":
  main()
