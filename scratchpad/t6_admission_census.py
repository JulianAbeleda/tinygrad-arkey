"""T6 -- pure-Python admission census (no GPU) for the device-aware admission fix.

Checks, in the caller's own process (no Device[...] touched, mirrors precontract_probe_lane.py's own
admit_probe_config no-GPU guarantee):
  1. M1e's wave_count group at (256,64,32,wm,1,1) for wm in (8,4,2) -- sanity, these already admitted
     before the fix and must still admit after it.
  2. All 23 of M1a's Metal-legal (tm,tn,tk,wm,wn,bc) tuples (docs/task_workflow/output/
     m1a-readiness-and-geometry-population-result-20260730.md), which M1e found were ALL rejected under
     the old AMD-tc-hardcoded admission gate ("capability_geometry: tile must divide into whole per-wave
     tensor-core subtiles and K steps").

Shape held fixed at (512, 12288, 4096) -- the qwen3_8b_q4k_m_gfx1100 ffn_gate_up natural shape M1e's own
sweep used, and the shape M1a's population's LDS/thread legality was itself checked against.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from extra.llm_research.prefill.precontract_probe_lane import ProbeConfig, admit_probe_config
from extra.llm_research.runtime_specs import FullKernelAdmissionError

SHAPE = (512, 12288, 4096)

WAVE_COUNT_GROUP = [(256, 64, 32, wm, 1, 1) for wm in (8, 4, 2)]

# All 23 of M1a's Metal-legal tuples, tk=32 throughout, expanding the 3 "both splits collide" rows to
# their two (wm,wn) alternatives each (20 table rows + 3 extra alternatives = 23 tuples).
M1A_23_TUPLES = [
  (64, 32, 32, 8, 1, 1), (64, 32, 32, 8, 4, 1),
  (64, 64, 32, 8, 1, 1), (64, 64, 32, 4, 8, 1),
  (64, 128, 32, 8, 1, 1), (64, 128, 32, 2, 16, 1),
  (128, 32, 32, 16, 1, 1), (128, 32, 32, 8, 4, 1),
  (128, 64, 32, 16, 1, 1), (128, 64, 32, 4, 8, 1),
  (128, 128, 32, 16, 1, 1), (128, 128, 32, 2, 16, 1),
  (256, 32, 32, 32, 1, 1), (256, 32, 32, 8, 4, 1),
  (256, 64, 32, 32, 1, 1), (256, 64, 32, 4, 8, 1),
  (256, 128, 32, 32, 1, 1), (256, 128, 32, 2, 16, 1),
  (64, 32, 32, 8, 1, 2),
  (64, 64, 32, 8, 1, 2),
  (64, 128, 32, 8, 1, 2),
  (128, 32, 32, 16, 1, 2),
  (128, 64, 32, 16, 1, 2),
]
assert len(M1A_23_TUPLES) == 23, len(M1A_23_TUPLES)


def census(label: str, geometries: list[tuple[int, int, int, int, int, int]]) -> None:
  print(f"\n=== {label} ({len(geometries)} geometries) ===")
  admitted, rejected = 0, 0
  for geom in geometries:
    config = ProbeConfig(quant="Q4_K", role="ffn_gate_up", shape=SHAPE, geometry=geom, device="METAL",
                         rounds=3, warmups=1)
    try:
      entry, admission = admit_probe_config(config)
      admitted += 1
      print(f"  ADMIT  geom={geom} active_lds_bytes={admission.active_lds_bytes}")
    except (FullKernelAdmissionError, ValueError) as exc:
      rejected += 1
      print(f"  REJECT geom={geom} reason={exc}")
  print(f"  -- {admitted} admitted / {rejected} rejected of {len(geometries)}")


if __name__ == "__main__":
  census("M1e wave_count group (sanity: must still admit)", WAVE_COUNT_GROUP)
  census("M1a's 23 Metal-legal tuples", M1A_23_TUPLES)
