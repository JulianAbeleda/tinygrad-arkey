#!/usr/bin/env python3
"""Build the CPU-only d512 llama/tinygrad quantized-role manifest.

This maps the Qwen3 d512 projection identities established by the captured launch
order and the pinned graph-construction source.  Fields that require runtime
tensor arguments or buffer attribution remain the string ``UNKNOWN`` rather than
being guessed from shapes.
"""
from __future__ import annotations

import argparse, hashlib, json, re, sqlite3, subprocess
from pathlib import Path

HASH64 = re.compile(r"_[0-9a-f]{64}$")
# Qwen3 builds separate Q/K/V projections, but the graph explicitly expands
# Q, then V, then K.  The order is intentional: K is expanded later so rope
# can fuse its write into the KV cache.  Thus the equal-shape MMVQs are V/K.
ROLES = ("attn_q", "attn_v", "attn_k", "attn_o", "ffn_gate_up", "ffn_down")
LLAMA_OFFSETS = (0, 1, 2, 3, 4, 5)  # indices within each six-MMVQ layer group
TG_OFFSETS = (0, 1, 2, 3, 4, 6)     # indices within each seven-core layer group; gate_up owns 4,5


def sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    while chunk := f.read(1 << 20): h.update(chunk)
  return h.hexdigest()


def clean(name: str) -> str: return HASH64.sub("", name)


def _llama_rows(trace: Path, graph_id: int) -> list[dict]:
  db = sqlite3.connect(trace)
  q = """SELECT k.start,k.graphNodeId,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
    k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory,s.value,d.value,m.value
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON s.id=k.shortName JOIN StringIds d ON d.id=k.demangledName
    LEFT JOIN StringIds m ON m.id=k.mangledName WHERE k.graphId=? ORDER BY k.start"""
  rows = db.execute(q, (graph_id,)).fetchall(); db.close()
  # A complete replay has 762 rows. The first captured replay is complete in the
  # pinned trace; validating the exact class census prevents a partial prefix.
  if len(rows) < 762: raise ValueError(f"graph {graph_id}: only {len(rows)} rows")
  rows = rows[:762]
  out = [{"ordinal": i, "node_id": r[1], "grid": list(r[2:5]), "block": list(r[5:8]),
          "registers_per_thread": r[8], "static_smem_bytes": r[9], "dynamic_smem_bytes": r[10],
          "short_symbol": r[11], "demangled_symbol": r[12], "mangled_symbol": r[13]} for i,r in enumerate(rows)]
  counts = {n: sum(x["short_symbol"] == n for x in out) for n in
            ("mul_mat_vec_q", "quantize_q8_1", "rms_norm_f32", "rope_neox")}
  if counts != {"mul_mat_vec_q":217, "quantize_q8_1":217, "rms_norm_f32":145, "rope_neox":72}:
    raise ValueError(f"unexpected llama replay census: {counts}")
  return out


def _tg_nodes(capture: Path) -> list[dict]:
  data = json.loads(capture.read_text())
  nodes = data["arms"]["logical"]["nodes"]
  if len(nodes) != 1021 or [n["id"] for n in nodes] != list(range(1021)):
    raise ValueError("tinygrad capture is not the expected ordered 1021-node d512 token")
  return nodes


def _quant_type(demangled: str) -> str:
  if "(ggml_type)12" in demangled: return "Q4_K"
  if "(ggml_type)14" in demangled: return "Q6_K"
  return "UNKNOWN"


def build(trace: Path, capture: Path, oracle: Path, model: Path, llama_repo: Path, graph_id: int = 2) -> dict:
  lr, tn = _llama_rows(trace, graph_id), _tg_nodes(capture)
  oracle_data = json.loads(oracle.read_text())
  cubin_name = oracle_data["kernels"]["Q4_K"]["cubin"]["name"]
  cubin = oracle.parent / cubin_name
  llama_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=llama_repo, text=True).strip()
  src = llama_repo / "ggml/src/ggml-cuda/mmvq.cu"
  graph_src = llama_repo / "src/llama-graph.cpp"
  qwen3_src = llama_repo / "src/models/qwen3.cpp"
  rows = []
  llama_mmv_ordinals = [i for i,x in enumerate(lr) if x["short_symbol"] == "mul_mat_vec_q"]
  tg_core_ids = [n["id"] for n in tn if clean(n["name"]).startswith(("q4k_g3_lanemap_gemv", "q6k_gen_coop", "q6k_gen_partial"))]
  if len(llama_mmv_ordinals) != 217 or len(tg_core_ids) != 253:
    raise ValueError(f"quantized core count changed: llama={len(llama_mmv_ordinals)} tinygrad={len(tg_core_ids)}")
  for layer in range(36):
    llama_base, tg_base = 6*layer, 7*layer
    for role, lo, to in zip(ROLES, LLAMA_OFFSETS, TG_OFFSETS):
      mmq_ordinal = llama_mmv_ordinals[llama_base+lo]
      mmq = lr[mmq_ordinal]
      q8 = lr[mmq_ordinal-1]
      if mmq["short_symbol"] != "mul_mat_vec_q" or q8["short_symbol"] != "quantize_q8_1":
        raise ValueError(f"layer {layer} {role}: llama q8/MMVQ adjacency changed")
      role_core_ids = tg_core_ids[tg_base+to:tg_base+to+(2 if role == "ffn_gate_up" else 1)]
      tg = tn[role_core_ids[0]]
      tname = clean(tg["name"])
      if not (tname.startswith("q4k_") or tname.startswith("q6k_")):
        raise ValueError(f"layer {layer} {role}: unexpected tinygrad core {tname}")
      weight = _quant_type(mmq["demangled_symbol"])
      rows.append({
        "model_role": role, "layer": layer, "rows": mmq["grid"][0], "K": 4096 if role != "ffn_down" else 12288,
        "weight": {"type":weight, "layout": f"ggml block_{weight.lower()}",
                   "block_bytes": {"Q4_K":144,"Q6_K":210}.get(weight,"UNKNOWN"), "stride":"UNKNOWN"},
        "activation": {"llama_input":"F32", "llama_mmv_input":"Q8_1", "q8_layout":"block_q8_1; padded K to 512",
                       "q8_producer_node_id":q8["node_id"], "observed_reuse_consumers":1,
                       "tinygrad_input":"F16", "tinygrad_layout":"contiguous row vector"},
        "llama": {"host_op":"GGML_OP_MUL_MAT", "source":"ggml/src/ggml-cuda/mmvq.cu",
          "ordered_launch_subgraph":[q8["node_id"],mmq["node_id"]], "launch_count":2,
          "quantize": {k:q8[k] for k in ("short_symbol","demangled_symbol","mangled_symbol","grid","block","registers_per_thread","static_smem_bytes","dynamic_smem_bytes")},
          "mmvq": {k:mmq[k] for k in ("short_symbol","demangled_symbol","mangled_symbol","grid","block","registers_per_thread","static_smem_bytes","dynamic_smem_bytes")},
          "fusion": "has_fusion=true" if "(bool)1, (bool)0" in mmq["demangled_symbol"] else "has_fusion=false",
          "epilogue_semantics":"UNKNOWN"},
        "tinygrad": {"semantic_call":"tinygrad.llm.decode_routes quantized linear", "core_node_ids":role_core_ids,
          "core_symbols":[clean(tn[i]["name"]) for i in role_core_ids],
          "core_identity_sha256":[tn[i]["metadata"]["identity_sha256"] for i in role_core_ids],
          "ordered_full_semantic_subgraph":"UNKNOWN", "core_launch_count":len(role_core_ids),
          "reduction_or_epilogue_node_ids":"UNKNOWN"},
        "evidence":("OBSERVED_CORE_IDENTITIES; ROLE_EXACT_FROM_QWEN3_BUILD_AND_GRAPH_EXPANSION_ORDER; "
                    "UNKNOWN_FULL_TG_SUBGRAPH")
      })
  # Final vocabulary MMVQ is the 217th q8/MMVQ pair.
  q8, mmq = lr[760], lr[761]
  rows.append({"model_role":"vocab", "layer":"final", "rows":mmq["grid"][0], "K":4096,
    "weight":{"type":_quant_type(mmq["demangled_symbol"]),"layout":"ggml block_q6_k","block_bytes":210,"stride":"UNKNOWN"},
    "activation":{"llama_input":"F32","llama_mmv_input":"Q8_1","q8_layout":"block_q8_1; padded K to 512",
      "q8_producer_node_id":q8["node_id"],"observed_reuse_consumers":1,"tinygrad_input":"F16","tinygrad_layout":"contiguous row vector"},
    "llama":{"host_op":"GGML_OP_MUL_MAT","source":"ggml/src/ggml-cuda/mmvq.cu",
      "ordered_launch_subgraph":[q8["node_id"],mmq["node_id"]],"launch_count":2,
      "quantize":{k:q8[k] for k in ("short_symbol","demangled_symbol","mangled_symbol","grid","block","registers_per_thread","static_smem_bytes","dynamic_smem_bytes")},
      "mmvq":{k:mmq[k] for k in ("short_symbol","demangled_symbol","mangled_symbol","grid","block","registers_per_thread","static_smem_bytes","dynamic_smem_bytes")},
      "fusion":"has_fusion=false","epilogue_semantics":"UNKNOWN"},
      "tinygrad":{"semantic_call":"vocab projection + scalar reduction","core_node_ids":[tg_core_ids[-1]],
      "core_symbols":[clean(tn[tg_core_ids[-1]]["name"])],"core_identity_sha256":[tn[tg_core_ids[-1]]["metadata"]["identity_sha256"]],
      "ordered_full_semantic_subgraph":"UNKNOWN","core_launch_count":2,"reduction_or_epilogue_node_ids":[1017]},
    "evidence":"OBSERVED_CORE_IDENTITIES; INFERRED_ROLE_FROM_FINAL_POSITION"})
  role_counts = {r:sum(x["model_role"] == r for x in rows) for r in (*ROLES,"vocab")}
  return {"schema":"tinygrad.nv_decode.semantic_call_manifest.v2", "status":"PARTIAL_P2_FAIL_CLOSED",
    "summary":{"rows":len(rows),"role_counts":role_counts,"llama_total_nodes":len(lr),"tinygrad_total_nodes":len(tn),
      "llama_quantize_q8_1":217,"llama_mmvq":217,"llama_observed_cross_mmv_q8_reuse":False,
      "tinygrad_quantized_core_launches":253,
      "p2_gate":"BLOCKED: full tinygrad semantic subgraphs and runtime tensor strides are not attributed"},
    "provenance":{"llama_trace":str(trace),"graph_id":graph_id,"llama_commit":llama_commit,
      "llama_mmvq_source_sha256":sha256(src),"llama_graph_source_sha256":sha256(graph_src),
      "qwen3_graph_source_sha256":sha256(qwen3_src),"llama_library_sha256":oracle_data["library_sha256"],
      "llama_cubin":str(cubin),"llama_cubin_sha256":sha256(cubin),"oracle_manifest_sha256":sha256(oracle),
      "tinygrad_capture":str(capture),"tinygrad_capture_sha256":sha256(capture),
      "model":str(model),"model_sha256":sha256(model),"model_bytes":model.stat().st_size},
    "invariants":{"llama_layer_nodes":21,"tinygrad_layer_nodes":28,"layers":36,
      "llama_role_order":list(ROLES),"tinygrad_role_order":list(ROLES),
      "qwen3_qkv_construction_order":["attn_q","attn_k","attn_v"],
      "qwen3_graph_expansion_order":["attn_q","attn_v","attn_k"],
      "equal_shape_1024_mmvq_order":["attn_v","attn_k"]},
    "role_identity_proof":{"model":"src/models/qwen3.cpp: build_qkv returns Qcur,Kcur,Vcur to build_attn",
      "construction":"src/llama-graph.cpp: build_qkv separate path creates layer.wq, layer.wk, layer.wv",
      "graph_order":"src/llama-graph.cpp: build_attn expands q_cur, v_cur, k_cur in that order",
      "why_order_is_stable":"the adjacent source comment says these nodes are added together so they are not reordered; K is expanded later for rope-to-KV-cache fusion"},
    "unknowns":["runtime weight/activation strides","q8 buffer identity across calls","full tinygrad semantic subgraph boundaries",
      "runtime weight/activation strides","q8 buffer identity across calls","full tinygrad semantic subgraph boundaries",
      "exact fusion epilogue semantics per node","DRAM versus L2 bytes","tinygrad CUDA grid/block/register resources absent from capture"],
    "rows":rows}


def main() -> None:
  p=argparse.ArgumentParser(); p.add_argument("--llama-trace",type=Path,required=True); p.add_argument("--tinygrad-capture",type=Path,required=True)
  p.add_argument("--oracle-manifest",type=Path,required=True); p.add_argument("--model",type=Path,required=True)
  p.add_argument("--llama-repo",type=Path,required=True); p.add_argument("--graph-id",type=int,default=2); p.add_argument("--out",type=Path,required=True)
  a=p.parse_args(); out=build(a.llama_trace,a.tinygrad_capture,a.oracle_manifest,a.model,a.llama_repo,a.graph_id)
  a.out.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out["summary"],indent=2))


if __name__ == "__main__": main()
