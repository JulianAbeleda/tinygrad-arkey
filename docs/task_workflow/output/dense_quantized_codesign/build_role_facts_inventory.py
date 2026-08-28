"""Build a dense Q4_K/Q6_K role-facts inventory from GGUF metadata only."""
from __future__ import annotations

import argparse, collections, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tinygrad.llm.gguf import gguf_load_metadata
from tinygrad.llm.model_facts import model_facts_from_gguf_metadata
from tinygrad.llm.model_route_plan import build_model_route_plan

BLOCK_BYTES = {"Q4_K":144, "Q6_K":210}

def build(model_path:Path, *, dense_verified:bool=False) -> dict:
  kv, meta = gguf_load_metadata(model_path)
  facts = model_facts_from_gguf_metadata(kv, meta)
  plan = build_model_route_plan(meta=meta, model_facts=facts)
  info_by_name = {str(name):(tuple(int(x) for x in dims), int(typ), int(offset))
                  for name, dims, typ, offset in meta["tensor_infos"]}
  rows = []
  for fact in facts.tensors:
    if fact.quant_label not in BLOCK_BYTES: raise ValueError(f"unsupported dense source quant {fact.quant_label}: {fact.name}")
    if fact.cols % 256: raise ValueError(f"{fact.name}: K dimension is not divisible by 256")
    dims, typ, offset = info_by_name[fact.name]
    route = plan.primitive(fact.name)
    rows.append({
      "tensor_name":fact.name, "inferred_role":fact.role, "quant_type":fact.quant_label, "typ":typ,
      "rows":fact.rows, "cols":fact.cols, "packed_bytes":fact.rows*(fact.cols//256)*BLOCK_BYTES[fact.quant_label],
      "gguf_offset":int(meta["data_start"])+offset,
      "route_plan":None if route is None else {"family":route.family, "kernel_mode":route.kernel_mode,
        "parts":route.parts, "opts":list(route.opts)},
    })
  if not dense_verified: raise ValueError("dense inventory requires the explicit --dense-verified assertion")
  roles, quants = collections.Counter(row["inferred_role"] or "unresolved" for row in rows), collections.Counter(row["quant_type"] for row in rows)
  return {
    "artifact":"dense-role-facts-inventory", "model_path":str(model_path.resolve()),
    "model_size_bytes":model_path.stat().st_size, "dense_verified":True,
    "source_helpers":["tinygrad.llm.gguf.gguf_load_metadata", "tinygrad.llm.model_facts.model_facts_from_gguf_metadata",
                      "tinygrad.llm.model_route_plan.build_model_route_plan"],
    "model_facts":{"architecture":facts.architecture, "hidden_size":facts.hidden_size,
      "intermediate_size":facts.intermediate_size, "n_heads":facts.n_heads,
      "n_kv_heads":facts.n_kv_heads, "head_dim":facts.head_dim},
    "summary":{"tensor_count":len(rows), "route_plan_entry_count":len(plan),
      "role_counts":dict(sorted(roles.items())), "quant_counts":dict(sorted(quants.items()))},
    "git":{"commit":subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}, "tensors":rows,
  }

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--dense-verified", action="store_true")
  args = parser.parse_args()
  result = build(args.model, dense_verified=args.dense_verified)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")

if __name__ == "__main__": main()
