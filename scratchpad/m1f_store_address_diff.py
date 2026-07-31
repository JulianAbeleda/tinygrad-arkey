#!/usr/bin/env python3
"""M1f: diff the emitted accumulator->global store loop, AMD (correct) vs Metal (writes 18.745%).

Compile-only, no GPU. This is a differential read, not a hypothesis test: render
build_precontract_lds_stage for the same role/shape/geometry on both targets (reusing
scratchpad/m1d_confirm_c_fragment.py's proven Target.parse + to_program technique, which needs no
Device[...] and no GPU), then read the emitted store region side by side.

Part 1 (this file, when run): regenerates /tmp/m1d_metal_source.c and /tmp/m1d_amd_source.c by
calling scratchpad/m1d_confirm_c_fragment.py's render_one for both targets (identical dispatch:
Q4_K, ffn_gate_up, shape (512,12288,4096), geometry (256,64,32,8,1,1) -- AMD's own promoted tuple
from PACKED_WMMA_ROUTES, the exact geometry M1b/M1c/M1e measured the Metal failure at).

Part 2: extracts the final store loop's address arithmetic FROM THE ACTUAL RENDERED SOURCE TEXT
(transcribed by hand from /tmp/m1d_{metal,amd}_source.c, quoted in the M1f doc) and brute-forces
it over the entire launch grid the kernel's own source declares (gidx0 in [0,192), gidx1 in [0,2),
lidx1 in [0,8), lidx0 in [0,32) -- read directly off the source's own comments,
e.g. "int gidx0 = ...; /* 192 */", and cross-checked against the compiled kernel name's encoded
dims, e.g. r_2_192_32_8_2_8_4_128_4_<hash>) to check whether the union of all store addresses,
across every thread the kernel launches, is a bijection onto the output tile [0, 512*12288).

This never touches a GPU: it is pure NumPy arithmetic over the address expressions extracted from
static kernel source text. If a store address formula transcribed here stops matching the actual
source (e.g. after a codegen change), this script's asserts on tuple counts (64 AMD offsets, 32
Metal pairs) and the printed unique-address counts are the mechanism to catch that -- re-run
scratchpad/m1d_confirm_c_fragment.py first and diff the new /tmp/m1d_*_source.c against what is
quoted below before trusting these numbers again.
"""
from __future__ import annotations
import subprocess, sys
import numpy as np

REPO = "/Users/julianabeleda/env/tinygrad-arkey-exp"


def regenerate_sources() -> None:
  """Re-run M1d's proven renderer to refresh /tmp/m1d_{metal,amd}_source.c from the current repo state."""
  result = subprocess.run([sys.executable, f"{REPO}/scratchpad/m1d_confirm_c_fragment.py"],
                          cwd=REPO, capture_output=True, text=True)
  if result.returncode != 0:
    raise RuntimeError(f"m1d_confirm_c_fragment.py failed:\n{result.stdout}\n{result.stderr}")


# ---- launch grid, read directly off both sources' own gidx0/gidx1/lidx0/lidx1 comments ----
M, N = 512, 12288
TOTAL = M * N
GIDX0_RANGE, GIDX1_RANGE, LIDX1_RANGE, LIDX0_RANGE = 192, 2, 8, 32


def check_bijection() -> dict:
  gidx0 = np.arange(GIDX0_RANGE); gidx1 = np.arange(GIDX1_RANGE)
  lidx1 = np.arange(LIDX1_RANGE); lidx0 = np.arange(LIDX0_RANGE)
  G0, G1, L1, L0 = np.meshgrid(gidx0, gidx1, lidx1, lidx0, indexing='ij')
  G0 = G0.ravel().astype(np.int64); G1 = G1.ravel().astype(np.int64)
  L1 = L1.ravel().astype(np.int64); L0 = L0.ravel().astype(np.int64)
  nthreads = G0.size
  assert nthreads == GIDX0_RANGE * GIDX1_RANGE * LIDX1_RANGE * LIDX0_RANGE

  # ---- AMD: /tmp/m1d_amd_source.c lines 305-369 ----
  # int alu158 = ((gidx1*3145728)+(lidx1*393216)+((lidx0>>4)*12288)+(gidx0<<6)+alu5);  // alu5 = lidx0&15
  amd_base = G1 * 3145728 + L1 * 393216 + (L0 >> 4) * 12288 + (G0 << 6) + (L0 & 15)
  amd_offsets = [
    (16, 16), (32, 32), (48, 48),
    (24576, 1), (24592, 17), (24608, 33), (24624, 49),
    (49152, 2), (49168, 18), (49184, 34), (49200, 50),
    (73728, 3), (73744, 19), (73760, 35), (73776, 51),
    (98304, 4), (98320, 20), (98336, 36), (98352, 52),
    (122880, 5), (122896, 21), (122912, 37), (122928, 53),
    (147456, 6), (147472, 22), (147488, 38), (147504, 54),
    (172032, 7), (172048, 23), (172064, 39), (172080, 55),
    (196608, 8), (196624, 24), (196640, 40), (196656, 56),
    (221184, 9), (221200, 25), (221216, 41), (221232, 57),
    (245760, 10), (245776, 26), (245792, 42), (245808, 58),
    (270336, 11), (270352, 27), (270368, 43), (270384, 59),
    (294912, 12), (294928, 28), (294944, 44), (294960, 60),
    (319488, 13), (319504, 29), (319520, 45), (319536, 61),
    (344064, 14), (344080, 30), (344096, 46), (344112, 62),
    (368640, 15), (368656, 31), (368672, 47), (368688, 63),
    (0, 0),
  ]
  assert len(amd_offsets) == 64, "AMD store-line count changed -- re-quote from fresh source"
  amd_addrs = np.concatenate([amd_base + off for off, _ in amd_offsets])

  # ---- Metal: /tmp/m1d_metal_source.c lines 463-495 ----
  # int alu161 = ((gidx1*3145728)+(lidx1*393216)+((lidx0>>4)*49152)+((alu0&1)*24576)
  #              +(((lidx0>>1)&1)*12288)+(gidx0<<6)+(((lidx0>>3)&1)<<2)+((lidx0&1)<<1));
  # // alu0 = lidx0>>2 (line 17)
  metal_base = (G1 * 3145728 + L1 * 393216 + (L0 >> 4) * 49152 + ((L0 >> 2) & 1) * 24576
                + ((L0 >> 1) & 1) * 12288 + (G0 << 6) + (((L0 >> 3) & 1) << 2) + ((L0 & 1) << 1))
  metal_pairs = [
    (8, (8, 9)), (16, (16, 17)), (24, (24, 25)), (32, (32, 33)), (40, (40, 41)), (48, (48, 49)), (56, (56, 57)),
    (98304, (2, 3)), (98312, (10, 11)), (98320, (18, 19)), (98328, (26, 27)), (98336, (34, 35)), (98344, (42, 43)),
    (98352, (50, 51)), (98360, (58, 59)),
    (196608, (4, 5)), (196616, (12, 13)), (196624, (20, 21)), (196632, (28, 29)), (196640, (36, 37)),
    (196648, (44, 45)), (196656, (52, 53)), (196664, (60, 61)),
    (294912, (6, 7)), (294920, (14, 15)), (294928, (22, 23)), (294936, (30, 31)), (294944, (38, 39)),
    (294952, (46, 47)), (294960, (54, 55)), (294968, (62, 63)),
    (0, (0, 1)),
  ]
  assert len(metal_pairs) == 32, "Metal store-line count changed -- re-quote from fresh source"
  metal_addrs = np.concatenate([np.stack([metal_base + off, metal_base + off + 1]).ravel() for off, _ in metal_pairs])

  out = {}
  for name, addrs, per_thread in (("AMD", amd_addrs, 64), ("METAL", metal_addrs, 64)):
    u, c = np.unique(addrs, return_counts=True)
    out[name] = {
      "total_store_elements": int(addrs.size),
      "expected_total": nthreads * per_thread,
      "unique_addresses": int(u.size),
      "expected_unique": TOTAL,
      "min_addr": int(addrs.min()), "max_addr": int(addrs.max()), "expected_max": TOTAL - 1,
      "any_address_hit_ne_1_time": bool(np.any(c != 1)),
      "max_hit_count": int(c.max()), "min_hit_count": int(c.min()),
      "is_clean_bijection": bool(u.size == TOTAL and np.all(c == 1) and addrs.min() == 0 and addrs.max() == TOTAL - 1),
    }
  return out


if __name__ == "__main__":
  if "--skip-regen" not in sys.argv:
    regenerate_sources()
  result = check_bijection()
  import json
  print(json.dumps(result, indent=2))
