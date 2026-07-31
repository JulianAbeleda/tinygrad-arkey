#!/usr/bin/env python3
"""MB1 follow-up: characterize the residual max_abs_error after the LDS-coverage fix -- is it a
few outlier cells, or a broad per-cell shuffle? Reuses the same lane, keeps the npz dump.
"""
import sys, tempfile
sys.path.insert(0, ".")
import numpy as np
import extra.llm_research.prefill.metal_precontract_lane as lane_mod
from extra.llm_research.prefill.metal_precontract_lane import ProbeConfig, run_precontract_probe

captured_paths = []
_orig_mkstemp = tempfile.mkstemp
def _mkstemp(*a, **kw):
  fd, path = _orig_mkstemp(*a, **kw)
  if kw.get("prefix", "").startswith("m1e_precontract_RA_") or (a and a[0] == "m1e_precontract_RA_"):
    captured_paths.append(path)
  return fd, path
tempfile.mkstemp = _mkstemp
lane_mod.tempfile = tempfile

def main():
  cfg = ProbeConfig(quant="Q4_K", role="ffn_gate_up", shape=(512, 12288, 4096),
                     geometry=(256, 64, 32, 8, 1, 1), device="METAL", rounds=1, warmups=0)
  res = run_precontract_probe(cfg, keep_npz=True)
  d = res.to_json()
  print("max_abs_error:", d["max_abs_error"], "coverage:", d["coverage"]["written_fraction"])
  npz_path = captured_paths[-1]
  with np.load(npz_path) as npz:
    ref = npz["reference"].astype(np.float64)
    out = npz["output_round0"].astype(np.float64)
  diff = np.abs(ref - out)
  print("shape", ref.shape, "total", ref.size)
  for thresh in (0.02, 0.1, 1.0, 5.0, 20.0, 40.0):
    print(f"  cells with |diff| > {thresh}: {int((diff > thresh).sum())} ({(diff > thresh).mean()*100:.4f}%)")
  print("  ref nonzero frac:", float((ref != 0).mean()))
  print("  out nonzero frac:", float((out != 0).mean()))
  both_zero = (ref == 0) & (out == 0)
  ref_only_zero = (ref == 0) & (out != 0)
  out_only_zero = (ref != 0) & (out == 0)
  both_nonzero = (ref != 0) & (out != 0)
  print("  both_zero:", int(both_zero.sum()), "ref_zero_out_nonzero:", int(ref_only_zero.sum()),
        "ref_nonzero_out_zero:", int(out_only_zero.sum()), "both_nonzero:", int(both_nonzero.sum()))
  if both_nonzero.sum():
    d2 = diff[both_nonzero]
    print("  among both-nonzero cells: max diff", float(d2.max()), "mean diff", float(d2.mean()),
          "median diff", float(np.median(d2)), "frac >0.02:", float((d2>0.02).mean()))

if __name__ == "__main__":
  main()
