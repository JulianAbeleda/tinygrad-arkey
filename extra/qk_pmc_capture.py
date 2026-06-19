#!/usr/bin/env python3
"""Native-PMC hardware-counter capture for tinygrad AMD (HCQ) kernels — the PMU-4 fallback.

rocprofv3 cannot trace tinygrad's HCQ/KFD dispatches (rocprof_hcq_visibility_gap, confirmed
2026-06-19 + by docs/primitive-pmu-observability-scope). But tinygrad's AMD backend has NATIVE
PMC (PMC=1): it programs the perf counters around each kernel and stores ProfilePMCEvent. This
tool drives PMC=1 and DECODES the raw blob (sum each counter's u64 across instances at off+i*8),
giving per-kernel L2 hit-rate / VALU / occupancy / LDS-bank-conflict.

Run:  DEV=AMD PMC=1 PROFILE=1 PYTHONPATH=. .venv/bin/python extra/qk_pmc_capture.py
Default counters (ops_amd PMC_COUNTERS): SQ_BUSY_CYCLES,SQ_INSTS_VALU,SQ_INSTS_SALU,
  SQC_LDS_IDX_ACTIVE,SQC_LDS_BANK_CONFLICT,GRBM_GUI_ACTIVE,GL2C_HIT,GL2C_MISS.
Override with PMC_COUNTERS=... (e.g. add GL2C_MC_RDREQ for HBM read traffic).
"""
from __future__ import annotations
import struct
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.device import Compiled

def decode_pmc(ev) -> dict[str, int]:
  """Sum each PMC counter over all (xcc,inst,se,sa,wgp) instances; each instance is a u64 at off+i*8."""
  out: dict[str, int] = {}
  for s in ev.sched:
    n = s.xcc * s.inst * s.se * s.sa * s.wgp
    tot = sum(struct.unpack_from("<Q", ev.blob, s.off + i*8)[0] for i in range(n) if s.off + i*8 + 8 <= len(ev.blob))
    out[s.name] = out.get(s.name, 0) + tot
  return out

def capture(fn, warmup=2, label=""):
  """Run fn() (warmup+1 times), return decoded PMC counters for the LAST kernel-set as a list of dicts."""
  for _ in range(warmup): fn()
  base = len([e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"])
  fn()
  Device['AMD'].synchronize(); Device['AMD']._at_profile_finalize()
  evs = [e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"][base:]
  rows = []
  for ev in evs:
    c = decode_pmc(ev); hit, miss = c.get("GL2C_HIT", 0), c.get("GL2C_MISS", 0)
    busy, active, valu = c.get("SQ_BUSY_CYCLES", 0), c.get("GRBM_GUI_ACTIVE", 0), c.get("SQ_INSTS_VALU", 0)
    rows.append({"label": label, "kern": ev.kern, "L2_hit%": round(100*hit/(hit+miss+1e-9), 1),
                 "l2_hit": hit, "l2_miss": miss, "VALU": valu, "SQ_busy": busy, "GRBM_active": active,
                 "VALU_per_active": round(valu/(active+1e-9), 2), "bankconf": c.get("SQC_LDS_BANK_CONFLICT", 0),
                 "mc_rdreq": c.get("GL2C_MC_RDREQ", 0)})
  return rows

if __name__ == "__main__":
  rng = np.random.default_rng(0)
  # Reused square matmul (data fits caches, high reuse) vs decode-shaped streaming GEMV (weight read once, no reuse).
  big = 8192
  Asq = Tensor((rng.standard_normal((2048, 2048))*0.1).astype(np.float16)); Bsq = Tensor((rng.standard_normal((2048, 2048))*0.1).astype(np.float16))
  xg = Tensor((rng.standard_normal((1, big))*0.1).astype(np.float16)); Wg = Tensor((rng.standard_normal((big, big))*0.1).astype(np.float16))
  print("=== reused square matmul 2048^3 (expect HIGH L2 hit) ===")
  for r in capture(lambda: (Asq@Bsq).realize(), label="matmul2048"): print(r)
  print("=== decode-shaped GEMV 1x8192 @ 8192x8192 (weight streamed once, no reuse — expect LOW L2 hit / HBM-bound) ===")
  for r in capture(lambda: (xg@Wg).realize(), label="gemv8192"): print(r)
