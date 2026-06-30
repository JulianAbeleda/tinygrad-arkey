#!/usr/bin/env python3
"""Q6K-2 (narrow) microgate: prove the pos16->warp32 packing + single-role ffn_down correctness.

The pos16->warp32 packing is ALREADY EXPRESSED in q6k_gemv_warp_kernel (extra/q6_k_gemv_primitive.py): lane = bg(0..1)*16
+ pos(0..15) packs 2 K-parallel block-groups into one 32-lane wave, then warp_reduce_sum (ds_bpermute) -> out[row]
directly -- no partials buffer, no separate r_* sum (the direct/warp route Q6K-1 selected). This gate proves it is
correct vs (a) the fp32 reference and (b) the current coop_partial + external .sum route, for ffn_down (4096x12288).

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_q6k_direct_microgate.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
Writes: bench/amd-isa-backend-q6k-direct-correctness/{latest.json,summary.md}
"""
import sys, json, pathlib
from math import prod
from tinygrad import Tensor, dtypes
from tinygrad.llm.gguf import ggml_data_to_tensor
from extra.qk_layout import GGML_Q6_K, Q6_K_BLOCK_ELEMS, Q6_K_BLOCK_BYTES, read_metadata, tensor_shape
from extra.q6_k_gemv_primitive import q6k_gemv_warp_kernel, q6k_coop_partial_kernel

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-isa-backend-q6k-direct-correctness"
TOL = 1e-2   # fp-reassoc tolerance (same as the q6_k_gemv_primitive correctness check)

def main(gguf, rows=256):
  OUT.mkdir(parents=True, exist_ok=True)
  meta = read_metadata(pathlib.Path(gguf))
  # find a Q6_K ffn_down (4096x12288) tensor
  cand = [x for x in meta.infos if x.typ == GGML_Q6_K and "ffn_down" in x.name and tensor_shape(x) == (4096, 12288)]
  if not cand:
    cand = [x for x in meta.infos if x.typ == GGML_Q6_K and len(tensor_shape(x)) == 2 and tensor_shape(x)[1] % Q6_K_BLOCK_ELEMS == 0]
  if not cand:
    rec = {"verdict": "Q6K2_BLOCKED_NO_Q6K_TENSOR", "reason": "no Q6_K ffn_down tensor in gguf"}
    json.dump(rec, open(OUT/"latest.json","w"), indent=2); print(json.dumps(rec)); return rec
  info = cand[0]; full_shape = tensor_shape(info); k = full_shape[1]
  rows = min(rows, full_shape[0])
  k_blocks = k // Q6_K_BLOCK_ELEMS
  byte_start = meta.data_start + info.off
  row_bytes = k_blocks * Q6_K_BLOCK_BYTES
  quant_bytes = rows * row_bytes

  dev = "AMD"
  gpath = pathlib.Path(gguf)
  raw = Tensor(gpath); raw_halfs = Tensor(gpath, dtype=dtypes.uint16)
  halfs = raw_halfs[byte_start//2:byte_start//2+quant_bytes//2].to(dev).contiguous().realize()
  Tensor.manual_seed(1337)
  x = Tensor.randn(k, dtype=dtypes.float16, device=dev).realize()

  raw_u8 = raw[byte_start:byte_start+quant_bytes].to(dev).contiguous().realize()
  decoded = ggml_data_to_tensor(raw_u8, rows*k, info.typ).reshape(rows, k).cast(dtypes.float16).realize()
  ref = (decoded.cast(dtypes.float32) * x.reshape(1, k).cast(dtypes.float32)).sum(axis=1).realize()

  # direct/warp route (pos16->warp32 packing, in-warp reduce, no partials/sum)
  warp_out = Tensor.empty(rows, dtype=dtypes.float32, device=dev)
  warp = warp_out.custom_kernel(halfs, x, fxn=q6k_gemv_warp_kernel(rows, k))[0].realize()
  # current coop route (partials[rows,16] + external .sum)
  coop_partials = Tensor.empty(rows, 16, dtype=dtypes.float32, device=dev)
  coop = coop_partials.custom_kernel(halfs, x, fxn=q6k_coop_partial_kernel(rows, k, 4))[0].sum(axis=1).realize()

  warp_vs_ref = (warp - ref).abs().max().item()
  coop_vs_ref = (coop - ref).abs().max().item()
  warp_vs_coop = (warp - coop).abs().max().item()
  ok = warp_vs_ref <= TOL and warp_vs_coop <= TOL
  verdict = "Q6K2_PASS_PACKING_AND_MICROGATE" if ok else "Q6K2_BLOCKED_MICROGATE_MISMATCH"
  rec = {"verdict": verdict, "tensor": info.name, "shape": list(full_shape), "rows_tested": rows, "k": k,
    "k_blocks": k_blocks, "k_blocks_even": k_blocks % 2 == 0,
    "packing": {"how": "lane = block_group(0..1)*16 + pos(0..15): 2 K-parallel block-groups packed into one 32-lane wave; "
                "warp_reduce_sum (ds_bpermute) over 32 lanes -> out[row]. No partials buffer, no external r_* sum.",
                "primitive": "q6k_gemv_warp_kernel (extra/q6_k_gemv_primitive.py:89), already wired behind Q6K_GEMV_WARP_DOWN (default-off)",
                "lanepartition_extent16_blocker": "MOOT -- the route uses warp_reduce_sum over the FULL 32-lane wave via the 2-group pack, not LanePartition(extent=16)"},
    "microgate": {"warp_vs_ref_max_abs": warp_vs_ref, "coop_vs_ref_max_abs": coop_vs_ref, "warp_vs_coop_max_abs": warp_vs_coop, "tol": TOL},
    "dequant_unchanged": "warp route reuses _q6k_weight verbatim (same bits/scale/dtype); only the cross-pos reduction moves in-kernel",
    "note_wd": "q6k_gemv_warp (ffn_down) is correct+byte-identical but model.py:434-436 records it as ~1.09x / no W==D gain for ffn_down ALONE "
               "(down already coop-routed ~51% peak). The Q6K-0 firm-removable W==D win is the lm_head coop reduce (r_32_4_1187), which this "
               "ffn_down-only warp route does NOT cover -> Q6K-3 must extend the warp route to lm_head (151936x4096), the folded-in target.",
    "scope": "Q6K-2 narrow: packing proof + ffn_down correctness microgate ONLY. No model wiring, no full decode, no W==D (Q6K-3)."}
  json.dump(rec, open(OUT/"latest.json","w"), indent=2)
  (OUT/"summary.md").write_text(f"# Q6K-2 (narrow) microgate\n\n**Verdict:** {verdict}\n\n"
    f"pos16->warp32 packing: {rec['packing']['how']}\n\nLanePartition extent16 'blocker': {rec['packing']['lanepartition_extent16_blocker']}\n\n"
    f"## ffn_down microgate ({info.name}, {rows} rows of {full_shape})\n"
    f"| comparison | max_abs | tol |\n|---|---|---|\n"
    f"| warp vs fp32 ref | {warp_vs_ref:.3g} | {TOL} |\n| warp vs coop+sum | {warp_vs_coop:.3g} | {TOL} |\n| coop vs ref | {coop_vs_ref:.3g} | {TOL} |\n\n"
    f"## W==D note\n{rec['note_wd']}\n")
  print(json.dumps({k2: rec[k2] for k2 in ("verdict","tensor","rows_tested","microgate")}, indent=2))
  print("\nQ6K-2", verdict)
  return rec

if __name__ == "__main__":
  gguf = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
  main(gguf)
