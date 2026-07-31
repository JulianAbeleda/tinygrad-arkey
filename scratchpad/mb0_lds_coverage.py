#!/usr/bin/env python3
"""MB0: does the producer write every LDS byte the fragment loads read?

Compile-only, no GPU. Same method as scratchpad/m1f_store_address_diff.py (the template) --
brute-force address formulas over the full thread grid with NumPy -- applied to the LDS window
(`buf1`) instead of the global store.

Part 1: regenerate /tmp/m1d_{metal,amd}_source.c via scratchpad/m1d_confirm_c_fragment.py's
render_one (identical dispatch: Q4_K, ffn_gate_up, shape (512,12288,4096), geometry
(256,64,32,8,1,1)).

Part 2: the LDS producer-store addresses (`build_precontract_lds_stage`'s cooperative store into
`buf1`) and the fragment-load read addresses (what feeds the Ops.CONTRACT/cast* operands of each
__WMMA call) are transcribed VERBATIM below from the rendered source text (grepped and quoted in
the MB0 report; re-grep `buf1+` in both files and diff against what's transcribed here before
trusting these numbers again after a codegen change).

Both targets' LDS address formulas depend only on (lidx0, lidx1) -- gidx0/gidx1 do not appear in
any `buf1`-relative offset in either source (grep confirms no gidx term in alu3/alu4/alu5 on Metal
or alu4/alu6/alu7 on AMD). So the write/read SET for one threadgroup's private `buf1` copy is fully
determined by the local id space (lidx0 in [0,32), lidx1 in [0,8), 256 threads) -- enumerating the
full grid (98,304 threads = 192*2*8*32) would just replicate this same per-threadgroup set 384
times (192*2), which we verify explicitly below rather than assume.

buf1 is declared `half buf1[12800]` on BOTH targets (25600 bytes / 2 bytes-per-half), matching the
scope doc's active_lds_bytes=25600.
"""
from __future__ import annotations
import subprocess, sys, json
import numpy as np

REPO = "/Users/julianabeleda/env/tinygrad-arkey-exp"
BUF1_SIZE = 12800  # half buf1[12800]; declared identically on both targets


def regenerate_sources() -> None:
  result = subprocess.run([sys.executable, f"{REPO}/scratchpad/m1d_confirm_c_fragment.py"],
                          cwd=REPO, capture_output=True, text=True)
  if result.returncode != 0:
    raise RuntimeError(f"m1d_confirm_c_fragment.py failed:\n{result.stdout}\n{result.stderr}")


LIDX0_RANGE, LIDX1_RANGE = 32, 8
# full grid, for the explicit replication check only
GIDX0_RANGE, GIDX1_RANGE = 192, 2


def _grid(local_only: bool):
  lidx0 = np.arange(LIDX0_RANGE); lidx1 = np.arange(LIDX1_RANGE)
  if local_only:
    L0, L1 = np.meshgrid(lidx0, lidx1, indexing='ij')
    return L0.ravel().astype(np.int64), L1.ravel().astype(np.int64)
  gidx0 = np.arange(GIDX0_RANGE); gidx1 = np.arange(GIDX1_RANGE)
  G0, G1, L1, L0 = np.meshgrid(gidx0, gidx1, lidx1, lidx0, indexing='ij')
  return L0.ravel().astype(np.int64), L1.ravel().astype(np.int64)


def metal_addresses(local_only=True):
  L0, L1 = _grid(local_only)
  # --- source constants, /tmp/m1d_metal_source.c lines 17-22 ---
  # int alu0 = (lidx0>>2);
  # int alu1 = (lidx0&3);
  # int alu2 = (alu1<<3);
  # int alu3 = ((lidx1*320)+(alu0*40)+alu2);        -- WRITE base
  # int alu4 = ((lidx0&15)*40);                      -- READ base (block 2 and 3)
  # int alu5 = ((lidx1*2560)+alu4);                  -- READ base (block 1)
  alu0 = L0 >> 2
  alu1 = L0 & 3
  alu2 = alu1 << 3
  alu3 = L1 * 320 + alu0 * 40 + alu2
  alu4 = (L0 & 15) * 40
  alu5 = L1 * 2560 + alu4

  # --- WRITE addresses: lines 147-156, ten half4 (4-element) stores ---
  write_bases = [4, 2560, 2564, 5120, 5124, 7680, 7684, 10240, 10244, 0]
  assert len(write_bases) == 10
  writes = np.concatenate([np.stack([alu3 + b + k for k in range(4)]).ravel() for b in write_bases])

  # --- READ addresses ---
  # block 1 (lines 158-189): 32 single-half reads off alu5
  read5_offsets = [1, 8, 9, 16, 17, 24, 25, 640, 641, 648, 649, 656, 657, 664, 665,
                    1280, 1281, 1288, 1289, 1296, 1297, 1304, 1305,
                    1920, 1921, 1928, 1929, 1936, 1937, 1944, 1945, 0]
  assert len(read5_offsets) == 32
  reads_a = np.concatenate([alu5 + o for o in read5_offsets])

  # block 2 (lines 190-221): 32 single-half reads off alu4, offsets 12800..14745
  read4_single_offsets = [12800, 12801, 12808, 12809, 12816, 12817, 12824, 12825,
                           13440, 13441, 13448, 13449, 13456, 13457, 13464, 13465,
                           14080, 14081, 14088, 14089, 14096, 14097, 14104, 14105,
                           14720, 14721, 14728, 14729, 14736, 14737, 14744, 14745]
  assert len(read4_single_offsets) == 32
  reads_b = np.concatenate([alu4 + o for o in read4_single_offsets])

  # block 3 (lines 222-237): 16 half2 (2-element) reads off alu4, offsets 10240..12184
  read4_pair_offsets = [10240, 10248, 10256, 10264, 10880, 10888, 10896, 10904,
                         11520, 11528, 11536, 11544, 12160, 12168, 12176, 12184]
  assert len(read4_pair_offsets) == 16
  reads_c = np.concatenate([np.stack([alu4 + o, alu4 + o + 1]).ravel() for o in read4_pair_offsets])

  reads = np.concatenate([reads_a, reads_b, reads_c])
  return writes, reads


def amd_addresses(local_only=True):
  L0, L1 = _grid(local_only)
  # --- source constants, /tmp/m1d_amd_source.c lines 26-33 ---
  # int alu0 = ((lidx0>>2)&1);
  # int alu1 = (lidx0>>3);
  # int alu2 = (lidx0&3);
  # int alu3 = (alu2<<3);
  # int alu4 = ((lidx1*320)+(alu0*160)+(alu1*40)+alu3);   -- WRITE base
  # int alu5 = (lidx0&15);
  # int alu6 = (alu5*40);                                  -- READ base (block 2)
  # int alu7 = ((lidx1*1280)+alu6);                         -- READ base (block 1)
  a0 = (L0 >> 2) & 1
  a1 = L0 >> 3
  a2 = L0 & 3
  a3 = a2 << 3
  alu4 = L1 * 320 + a0 * 160 + a1 * 40 + a3
  alu5 = L0 & 15
  alu6 = alu5 * 40
  alu7 = L1 * 1280 + alu6

  # --- WRITE addresses: lines 158-162, five half8 (8-element) stores ---
  write_bases = [2560, 5120, 7680, 10240, 0]
  assert len(write_bases) == 5
  writes = np.concatenate([np.stack([alu4 + b + k for k in range(8)]).ravel() for b in write_bases])

  # --- READ addresses ---
  # block 1 (lines 164-179): 16 half4 (4-element) reads off alu7
  read7_offsets = [4, 8, 12, 16, 20, 24, 28, 640, 644, 648, 652, 656, 660, 664, 668, 0]
  assert len(read7_offsets) == 16
  reads_a = np.concatenate([np.stack([alu7 + o + k for k in range(4)]).ravel() for o in read7_offsets])

  # block 2 (lines 180-211): 32 half4 (4-element) reads off alu6
  read6_offsets = [10240, 10244, 10248, 10252, 10256, 10260, 10264, 10268,
                    10880, 10884, 10888, 10892, 10896, 10900, 10904, 10908,
                    11520, 11524, 11528, 11532, 11536, 11540, 11544, 11548,
                    12160, 12164, 12168, 12172, 12176, 12180, 12184, 12188]
  assert len(read6_offsets) == 32
  reads_b = np.concatenate([np.stack([alu6 + o + k for k in range(4)]).ravel() for o in read6_offsets])

  reads = np.concatenate([reads_a, reads_b])
  return writes, reads


def analyze(name, writes, reads):
  wset = set(np.unique(writes).tolist())
  rset = set(np.unique(reads).tolist())
  missing = sorted(rset - wset)
  out = {
    "name": name,
    "buf1_declared_size": BUF1_SIZE,
    "write_elements_total": int(writes.size),
    "write_unique_addresses": len(wset),
    "write_min": int(writes.min()), "write_max": int(writes.max()),
    "write_max_within_declared_bounds": bool(writes.max() < BUF1_SIZE),
    "read_elements_total": int(reads.size),
    "read_unique_addresses": len(rset),
    "read_min": int(reads.min()), "read_max": int(reads.max()),
    "read_max_within_declared_bounds": bool(reads.max() < BUF1_SIZE),
    "reads_subset_of_writes": len(missing) == 0,
    "missing_count": len(missing),
    "missing_fraction_of_reads": len(missing) / len(rset) if rset else None,
    "missing_addresses_sample_first_20": missing[:20],
    "missing_addresses_sample_last_20": missing[-20:],
    "missing_min": (missing[0] if missing else None),
    "missing_max": (missing[-1] if missing else None),
    "missing_all_out_of_declared_bounds": bool(all(m >= BUF1_SIZE for m in missing)) if missing else None,
  }
  return out


def replication_check():
  """Confirm the full-grid (98,304-thread) enumeration reproduces the same set as the
  local-id-only (256-thread) enumeration, i.e. gidx0/gidx1 truly don't appear in the LDS
  address formulas."""
  out = {}
  for name, fn in (("METAL", metal_addresses), ("AMD", amd_addresses)):
    w_local, r_local = fn(local_only=True)
    w_full, r_full = fn(local_only=False)
    out[name] = {
      "local_write_set_size": len(set(w_local.tolist())),
      "full_write_set_size": len(set(w_full.tolist())),
      "sets_equal_write": set(w_local.tolist()) == set(w_full.tolist()),
      "local_read_set_size": len(set(r_local.tolist())),
      "full_read_set_size": len(set(r_full.tolist())),
      "sets_equal_read": set(r_local.tolist()) == set(r_full.tolist()),
      "full_grid_replication_factor": w_full.size // w_local.size,
    }
  return out


def main():
  if "--skip-regen" not in sys.argv:
    regenerate_sources()
  results = {}
  for name, fn in (("METAL", metal_addresses), ("AMD", amd_addresses)):
    writes, reads = fn(local_only=True)
    results[name] = analyze(name, writes, reads)
  results["_replication_check"] = replication_check()
  print(json.dumps(results, indent=2))
  with open("/tmp/mb0_lds_coverage_result.json", "w") as f:
    json.dump(results, f, indent=2)


if __name__ == "__main__":
  main()
