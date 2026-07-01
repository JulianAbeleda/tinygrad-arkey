#!/usr/bin/env python3
"""TG-P11.1: minimal reduce/upcast accumulator-widening invariant microgate (no attention).

Four tiny generated-UOp kernels isolate the codegen invariant the TG-P10 combine tripped:
  scalar_no_upcast     out[h]   = sum_s x[h,s]                          (baseline scalar reduce)
  invariant_upcast     out[h,d] = sum_s x[h,s]                          (result invariant along the upcast d -> scalar acc)
  varies_upcast        out[h,d] = sum_s x[h,s]*y[h,s,d]                 (result varies along upcast d -> one acc slot/lane)
  mixed_var_inv        out[h,d] = (sum_s x[h,s]*y[h,s,d]) / (sum_s x[h,s])  (num varies, den invariant)

Each case checks numeric correctness vs a numpy reference AND inspects the generated code (DEBUG=4) for the two
failure fingerprints: an invalid `make_float4(...) = ...` REG-accumulator store, and slot-0 lane aliasing. Under the
default (no fix) the varies/mixed cases fail to compile (invalid store); under REDUCE_ACC_UPCAST_FIX=1 they must
compile and be numeric_ok with a widened (non-aliased) accumulator.

Run: DEV=AMD PYTHONPATH=. python3 extra/qk_tg_p11_reduce_upcast_microgate.py   (add REDUCE_ACC_UPCAST_FIX=1 for the fixed arm)
"""
from __future__ import annotations
import contextlib, io, json, os, pathlib, re

os.environ.setdefault("DEV", "AMD")
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/tg-p11-reduce-upcast-accumulator"
Hq, Hd, S = 32, 128, 36
_INVALID = re.compile(r"make_float4\([^)]*\)\s*=")   # store to a vector rvalue (non-assignable)


def _kernels():
  from tinygrad.uop.ops import AxisType, KernelInfo, UOp
  from tinygrad.dtype import AddrSpace, dtypes
  F32 = dtypes.float32

  def scalar_no_upcast(out, x):
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    acc = UOp.placeholder((1,), F32, 300, addrspace=AddrSpace.REG)
    s = UOp.range(S, 1, axis_type=AxisType.REDUCE)
    acc = acc.after(h)[0].set(0.0)
    upd = acc[0].store(acc.after(s)[0] + x[h * S + s]).end(s)
    return out[h].store(acc.after(upd)[0]).end(h).sink(arg=KernelInfo(name="p11_scalar_no_upcast"))

  def invariant_upcast(out, x):
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    d = UOp.range(Hd, 1, AxisType.GLOBAL)               # output axis; result is invariant along it
    acc = UOp.placeholder((1,), F32, 301, addrspace=AddrSpace.REG)
    s = UOp.range(S, 2, axis_type=AxisType.REDUCE)
    acc = acc.after(h, d)[0].set(0.0)
    upd = acc[0].store(acc.after(s)[0] + x[h * S + s]).end(s)
    return out[h * Hd + d].store(acc.after(upd)[0]).end(h, d).sink(arg=KernelInfo(name="p11_invariant_upcast"))

  def varies_upcast(out, x, y):
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    d = UOp.range(Hd, 1, AxisType.GLOBAL)               # result varies along the upcast d
    acc = UOp.placeholder((1,), F32, 302, addrspace=AddrSpace.REG)
    s = UOp.range(S, 2, axis_type=AxisType.REDUCE)
    acc = acc.after(h, d)[0].set(0.0)
    upd = acc[0].store(acc.after(s)[0] + x[h * S + s] * y[(h * S + s) * Hd + d]).end(s)
    return out[h * Hd + d].store(acc.after(upd)[0]).end(h, d).sink(arg=KernelInfo(name="p11_varies_upcast"))

  def mixed_var_inv(out, x, y):
    h = UOp.range(Hq, 0, AxisType.GLOBAL)
    d = UOp.range(Hd, 1, AxisType.GLOBAL)
    num = UOp.placeholder((1,), F32, 303, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), F32, 304, addrspace=AddrSpace.REG)
    s = UOp.range(S, 2, axis_type=AxisType.REDUCE)
    num = num.after(h, d)[0].set(0.0)
    den = den.after(h, d)[0].set(0.0)
    upd = num[0].store(num.after(s)[0] + x[h * S + s] * y[(h * S + s) * Hd + d])
    upd = den.after(upd)[0].store(den.after(s)[0] + x[h * S + s]).end(s)
    return out[h * Hd + d].store(num.after(upd)[0] / den.after(upd)[0]).end(h, d).sink(arg=KernelInfo(name="p11_mixed_var_inv"))

  return scalar_no_upcast, invariant_upcast, varies_upcast, mixed_var_inv


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  from tinygrad import Tensor, dtypes, Device, Context
  rng = np.random.RandomState(3)
  x = rng.randn(Hq, S).astype(np.float32)
  y = rng.randn(Hq, S, Hd).astype(np.float32)
  xt = Tensor(x.reshape(-1), device="AMD").realize()
  yt = Tensor(y.reshape(-1), device="AMD").realize()
  refs = {
    "scalar_no_upcast": x.sum(1),
    "invariant_upcast": np.repeat(x.sum(1)[:, None], Hd, axis=1),
    "varies_upcast": np.einsum("hs,hsd->hd", x, y),
    "mixed_var_inv": np.einsum("hs,hsd->hd", x, y) / x.sum(1)[:, None],
  }
  scalar_no_upcast, invariant_upcast, varies_upcast, mixed_var_inv = _kernels()
  outshape = {"scalar_no_upcast": Hq, "invariant_upcast": Hq * Hd, "varies_upcast": Hq * Hd, "mixed_var_inv": Hq * Hd}

  def run(name, fn, args):
    src_holder = {}
    def build():
      out = Tensor.empty(outshape[name], dtype=dtypes.float32, device="AMD")
      return out.custom_kernel(*args, fxn=fn)[0]
    # capture generated source (DEBUG=4) while realizing
    buf = io.StringIO()
    try:
      with contextlib.redirect_stdout(buf), Context(DEBUG=4):
        res = build().realize().numpy()
      src = buf.getvalue()
      invalid = bool(_INVALID.search(src))
      exp = refs[name].reshape(-1)
      rel = float(np.abs(res - exp).max() / (np.abs(exp).max() + 1e-6))
      return {"case": name, "compile_ok": True, "numeric_ok": rel < 1e-3, "rel_err": rel,
              "invalid_reg_vector_store": invalid}
    except Exception as e:
      src = buf.getvalue()
      return {"case": name, "compile_ok": False, "numeric_ok": False,
              "invalid_reg_vector_store": bool(_INVALID.search(src)) or "not assignable" in str(e),
              "error": f"{type(e).__name__}: {str(e)[:80]}"}

  cases = [run("scalar_no_upcast", scalar_no_upcast, (xt,)),
           run("invariant_upcast", invariant_upcast, (xt,)),
           run("varies_upcast", varies_upcast, (xt, yt)),
           run("mixed_var_inv", mixed_var_inv, (xt, yt))]
  fixed = os.environ.get("REDUCE_ACC_UPCAST_FIX") == "1"
  all_ok = all(c["compile_ok"] and c["numeric_ok"] for c in cases)
  # baseline (no fix): the varies/mixed cases are expected to fail; with the fix all must pass.
  if fixed:
    verdict = "TG_P11_1_PASS_INVARIANT_TEST_READY" if all_ok else "TG_P11_2_BLOCKED_LOWERING_STILL_WRONG"
  else:
    reproduced = any(not c["compile_ok"] and c["case"] in ("varies_upcast", "mixed_var_inv") for c in cases)
    verdict = "TG_P11_1_PASS_INVARIANT_TEST_READY" if reproduced else "TG_P11_1_BLOCKED_TEST_NOT_MINIMAL"
  latest = {"scope": "TG-P11.1 reduce/upcast accumulator invariant microgate", "verdict": verdict,
            "reduce_acc_upcast_fix": fixed, "all_ok": all_ok, "cases": cases}
  json.dump(latest, open(OUT / ("invariant_microgate_fixed.json" if fixed else "invariant_microgate.json"), "w"), indent=2)
  print(verdict, "fix=", fixed, "| " + " ".join(f"{c['case']}:{'ok' if c['numeric_ok'] else ('cfail' if not c['compile_ok'] else 'nnum')}" for c in cases))
  return 0 if verdict.startswith("TG_P11_1_PASS") else 1


if __name__ == "__main__":
  raise SystemExit(main())
