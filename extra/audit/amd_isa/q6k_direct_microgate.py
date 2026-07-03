#!/usr/bin/env python3
"""Q6K-2 microgate: prove the HALF-WARP 2-ROW PARTITION (the foundational mapping) for one Q6_K role (ffn_down).

Distinct from q6k_gemv_warp_kernel (which packs 2 K-groups of ONE row into a full 32-lane warp_reduce, 1 row/warp).
Here q6k_halfwarp_partition_kernel maps TWO INDEPENDENT rows onto one 32-lane wave as two 16-lane partitions:
  lanes 0..15 = row A pos 0..15 ; lanes 16..31 = row B pos 0..15
each half: _q6k_weight dequant (verbatim) -> fp32 accumulate over its pos -> HALF-WARP reduce (warp_reduce_sum width=16
with the FULL 32-lane lane; the xor ladder {8,4,2,1} stays within each 16-lane half) -> store out[rowA]/out[rowB]
independently. NO partials buffer, NO external r_* reduce. This gate proves out[A] (even rows) and out[B] (odd rows)
both match the current coop_partial + external .sum route (and the fp32 reference), for ffn_down (4096x12288).

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/q6k_direct_microgate.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
Writes: bench/amd-isa-backend-q6k-direct-correctness/{latest.json,summary.md}
"""
import sys, json, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.llm.gguf import ggml_data_to_tensor
from extra.qk.layout import GGML_Q6_K, Q6_K_BLOCK_ELEMS, Q6_K_BLOCK_BYTES, read_metadata, tensor_shape
from extra.qk.quant.q6_k_gemv_primitive import q6k_halfwarp_partition_kernel, q6k_coop_partial_kernel

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "bench/amd-isa-backend-q6k-direct-correctness"
TOL = 1e-2   # fp-reassoc tolerance (same as the q6_k_gemv_primitive correctness check)

def main(gguf, rows=256):
  OUT.mkdir(parents=True, exist_ok=True)
  meta = read_metadata(pathlib.Path(gguf))
  cand = [x for x in meta.infos if x.typ == GGML_Q6_K and "ffn_down" in x.name and tensor_shape(x) == (4096, 12288)]
  if not cand:
    cand = [x for x in meta.infos if x.typ == GGML_Q6_K and len(tensor_shape(x)) == 2 and tensor_shape(x)[1] % Q6_K_BLOCK_ELEMS == 0]
  if not cand:
    rec = {"verdict": "Q6K2_BLOCKED_HALFWARP", "reason": "no Q6_K ffn_down tensor in gguf"}
    json.dump(rec, open(OUT/"latest.json","w"), indent=2); print(json.dumps(rec)); return rec
  info = cand[0]; full_shape = tensor_shape(info); k = full_shape[1]
  rows = min(rows, full_shape[0]); rows -= rows % 2     # even (2 rows / half-warp)
  k_blocks = k // Q6_K_BLOCK_ELEMS
  byte_start = meta.data_start + info.off
  row_bytes = k_blocks * Q6_K_BLOCK_BYTES
  quant_bytes = rows * row_bytes

  dev = "AMD"; gpath = pathlib.Path(gguf)
  raw = Tensor(gpath); raw_halfs = Tensor(gpath, dtype=dtypes.uint16)
  halfs = raw_halfs[byte_start//2:byte_start//2+quant_bytes//2].to(dev).contiguous().realize()
  Tensor.manual_seed(1337)
  x = Tensor.randn(k, dtype=dtypes.float16, device=dev).realize()

  raw_u8 = raw[byte_start:byte_start+quant_bytes].to(dev).contiguous().realize()
  decoded = ggml_data_to_tensor(raw_u8, rows*k, info.typ).reshape(rows, k).cast(dtypes.float16).realize()
  ref = (decoded.cast(dtypes.float32) * x.reshape(1, k).cast(dtypes.float32)).sum(axis=1).realize()

  # HALF-WARP 2-row partition route: out[row] direct, no partials, no external sum
  hw_out = Tensor.empty(rows, dtype=dtypes.float32, device=dev)
  hw = hw_out.custom_kernel(halfs, x, fxn=q6k_halfwarp_partition_kernel(rows, k))[0].realize()
  # current coop route (partials[rows,16] + external .sum)
  coop_partials = Tensor.empty(rows, 16, dtype=dtypes.float32, device=dev)
  coop = coop_partials.custom_kernel(halfs, x, fxn=q6k_coop_partial_kernel(rows, k, 4))[0].sum(axis=1).realize()

  hw_np, coop_np, ref_np = hw.numpy(), coop.numpy(), ref.numpy()
  import numpy as np
  A, B = slice(0, rows, 2), slice(1, rows, 2)   # row A = even (half 0), row B = odd (half 1)
  def mx(u, v): return float(np.abs(u - v).max())
  rowA_vs_coop = mx(hw_np[A], coop_np[A]); rowB_vs_coop = mx(hw_np[B], coop_np[B])
  rowA_vs_ref = mx(hw_np[A], ref_np[A]);   rowB_vs_ref = mx(hw_np[B], ref_np[B])
  coop_vs_ref = mx(coop_np, ref_np)
  ok = max(rowA_vs_coop, rowB_vs_coop, rowA_vs_ref, rowB_vs_ref) <= TOL
  verdict = "Q6K2_PASS_HALFWARP_PARTITION" if ok else "Q6K2_BLOCKED_HALFWARP"
  rec = {"verdict": verdict, "tensor": info.name, "shape": list(full_shape), "rows_tested": rows, "k": k, "k_blocks": k_blocks,
    "halfwarp_mapping": {"lanes_0_15": "row A (row_pair*2), pos 0..15", "lanes_16_31": "row B (row_pair*2+1), pos 0..15",
      "reduce": "warp_reduce_sum(acc, lane, width=16) with the FULL 32-lane lidx0; xor ladder {8,4,2,1} stays within each 16-lane half (lane^8 never crosses the 16-boundary) -> 2 INDEPENDENT half-warp sums",
      "store": "out[row].store(total) per half -> out[rowA], out[rowB] independent. NO partials buffer, NO external r_* reduce.",
      "primitive": "q6k_halfwarp_partition_kernel (NEW, extra/qk/quant/q6_k_gemv_primitive.py)",
      "why_not_lanepartition": "LanePartition/lane_partition_reduce_sum reduces the WHOLE 32-lane wave to ONE value (words_per_group is only the ADDRESS split, not independent partitions) -> wrong for 2 independent rows. The half-warp reduce is warp_reduce_sum(width=16) over the full lane."},
    "microgate": {"rowA_vs_coop_max_abs": rowA_vs_coop, "rowB_vs_coop_max_abs": rowB_vs_coop,
                  "rowA_vs_ref_max_abs": rowA_vs_ref, "rowB_vs_ref_max_abs": rowB_vs_ref, "coop_vs_ref_max_abs": coop_vs_ref, "tol": TOL},
    "route_label": f"q6k_halfwarp_partition_{rows}_{k}", "no_external_reduce": "out[row] stored in-kernel; no .sum/r_* for this route",
    "dequant_unchanged": "_q6k_weight reused verbatim (same bits/scale/dtype); only the cross-pos reduction moves in-kernel (half-warp)",
    "flag_off_rollback": "new kernel is NOT wired into model.py (Q6K-2 scope = no model wiring) -> the model's Q6_K route is byte-identical by construction; rollback is trivial (kernel unreferenced by the decode path).",
    "vs_prior_full_warp": "supersedes the prior full-warp proof (q6k_gemv_warp_kernel = 2 K-groups of ONE row/warp). THIS gate proves the DIFFERENT foundational mapping: 2 independent rows sharing a warp as two 16-lane partitions.",
    "scope": "Q6K-2: half-warp 2-row partition correctness ONLY. No model wiring, no full decode, no W==D (Q6K-3)."}
  json.dump(rec, open(OUT/"latest.json","w"), indent=2)
  (OUT/"summary.md").write_text(f"# Q6K-2 half-warp 2-row partition microgate\n\n**Verdict:** {verdict}\n\n"
    f"## Mapping\nlanes 0..15 = row A pos 0..15; lanes 16..31 = row B pos 0..15. "
    f"Half-warp reduce = warp_reduce_sum(width=16) over the FULL 32-lane lidx0 (xor {{8,4,2,1}} stays within each 16-lane half). "
    f"Store out[rowA]/out[rowB] independent; no partials, no external r_* reduce.\n\n"
    f"Why not LanePartition: it reduces the WHOLE wave to one value (words_per_group is the address split, not independent partitions).\n\n"
    f"## Microgate ({info.name}, {rows} rows of {full_shape}, both halves)\n"
    f"| comparison | max_abs | tol |\n|---|---|---|\n"
    f"| row A (even) vs coop+sum | {rowA_vs_coop:.3g} | {TOL} |\n| row B (odd) vs coop+sum | {rowB_vs_coop:.3g} | {TOL} |\n"
    f"| row A vs fp32 ref | {rowA_vs_ref:.3g} | {TOL} |\n| row B vs fp32 ref | {rowB_vs_ref:.3g} | {TOL} |\n"
    f"| coop vs ref | {coop_vs_ref:.3g} | {TOL} |\n\nroute label: q6k_halfwarp_partition_{rows}_{k}; no external r_* reduce; "
    f"new kernel unwired (model route byte-identical).\n")
  print(json.dumps({k2: rec[k2] for k2 in ("verdict","tensor","rows_tested","microgate","route_label")}, indent=2))
  print("\nQ6K-2", verdict)
  return rec

if __name__ == "__main__":
  gguf = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
  main(gguf)
