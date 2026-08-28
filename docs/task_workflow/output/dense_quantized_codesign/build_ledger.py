"""Build the generic dense representation Gate-0 ledger without import-time writes."""
from __future__ import annotations

import argparse, collections, json
from pathlib import Path

FORMATS = {
  "R1": {"id":"S4_G32_P256", "group_size":32, "payload_bytes_per_block":144},
  "R2": {"id":"S4_G64_P256", "group_size":64, "payload_bytes_per_block":136},
  "R3": {"id":"S5_G32_P256", "group_size":32, "payload_bytes_per_block":176},
}
STREAMED_ROLES = ("Q", "K", "V", "O", "gate", "up", "down", "vocab")

def roundup(value:int, alignment:int=256) -> int: return (value+alignment-1)//alignment*alignment

def classify(name:str, source_role:str|None) -> str:
  lowered = name.lower()
  for suffix, role in (("attn_q.weight", "Q"), ("attn_k.weight", "K"), ("attn_v.weight", "V"),
                       ("attn_output.weight", "O"), ("ffn_gate.weight", "gate"), ("ffn_up.weight", "up"),
                       ("ffn_down.weight", "down"), ("output.weight", "vocab"), ("token_embd.weight", "embedding")):
    if lowered.endswith(suffix): return role
  return source_role or "unknown"

def _sum(rows:list[dict], field:str) -> int: return sum(row[field] for row in rows)

def build(src:dict, *, inventory_path:str|None=None, dense_verified:bool=False) -> dict:
  rows:list[dict] = []
  for tensor in src["tensors"]:
    if tensor["cols"] % 256: raise ValueError(f"{tensor['tensor_name']}: cols must be divisible by 256")
    row = {
      "tensor_name":tensor["tensor_name"], "role":classify(tensor["tensor_name"], tensor.get("inferred_role")),
      "source_role":tensor.get("inferred_role"), "source_quant":tensor["quant_type"],
      "rows":tensor["rows"], "cols":tensor["cols"], "weights":tensor["rows"]*tensor["cols"],
      "source_payload_bytes":tensor["packed_bytes"], "source_padded_bytes":roundup(tensor["packed_bytes"]), "candidates":{},
    }
    for key, fmt in FORMATS.items():
      payload = tensor["rows"]*(tensor["cols"]//256)*fmt["payload_bytes_per_block"]
      row["candidates"][key] = {"payload_bytes":payload, "padded_bytes":roundup(payload),
                                 "bits_per_weight":8*fmt["payload_bytes_per_block"]/256}
    rows.append(row)

  aggregates = collections.defaultdict(lambda:collections.defaultdict(int))
  for row in rows:
    for field in ("weights", "source_payload_bytes", "source_padded_bytes"): aggregates[row["role"]][field] += row[field]
    for key in FORMATS:
      for field in ("payload_bytes", "padded_bytes"): aggregates[row["role"]][f"{key}_{field}"] += row["candidates"][key][field]

  streamed = [row for row in rows if row["role"] in STREAMED_ROLES]
  source_streamed = {field:_sum(streamed, f"source_{field}") for field in ("payload_bytes", "padded_bytes")}
  candidate_streamed = {key:{field:_sum([row["candidates"][key] for row in streamed], field)
                              for field in ("payload_bytes", "padded_bytes")} for key in FORMATS}
  population_filters = {
    "R1":lambda row: row["role"] in STREAMED_ROLES,
    "R2":lambda row: row["role"] in STREAMED_ROLES and row["source_quant"] == "Q4_K",
    "R3":lambda row: row["role"] in STREAMED_ROLES and row["source_quant"] == "Q6_K",
  }
  population_exposure = {}
  for key, selected in population_filters.items():
    population = [row for row in rows if selected(row)]
    source_bytes = _sum(population, "source_padded_bytes")
    candidate_bytes = _sum([row["candidates"][key] for row in population], "padded_bytes")
    mixed_bytes = source_streamed["padded_bytes"] - source_bytes + candidate_bytes
    population_exposure[key] = {
      "tensor_count":len(population), "source_population_padded_bytes":source_bytes,
      "candidate_population_padded_bytes":candidate_bytes, "candidate_minus_source_bytes":candidate_bytes-source_bytes,
      "mixed_streamed_padded_bytes":mixed_bytes,
      "mixed_streamed_ratio_to_source":mixed_bytes/source_streamed["padded_bytes"],
    }

  authority = {
    "inventory_path":inventory_path, "inventory_artifact":src.get("artifact"), "inventory_date":src.get("date"),
    "source_model_path":src.get("model_path"), "source_model_size_bytes":src.get("model_size_bytes"),
    "model_facts":src.get("model_facts"), "inventory_git":src.get("git"), "dense_verified":dense_verified,
  }
  return {
    "schema":"dense-quantized-codesign-ledger/v3", "authority":authority,
    "accounting_contract":{"source_block_elements":256, "candidate_alignment_bytes":256,
      "exposure_definition":"streamed projection bytes per decode token; embedding resident only; booked tok/s separate",
      "tied_weight_status":"unproven; embedding and vocab charged independently"},
    "formats":FORMATS,
    "recommended_populations":{"R1":"all projection roles (plumbing only)",
      "R2":"source-Q4_K projection tensors pending recurrent quality",
      "R3":"source-Q6_K projection tensors pending recurrent quality"},
    "role_aggregates":aggregates,
    "resident_aggregate":{"source_padded_bytes":_sum(rows, "source_padded_bytes"),
      **{f"{key}_padded_bytes":_sum([row["candidates"][key] for row in rows], "padded_bytes") for key in FORMATS}},
    "source_streamed_projection_aggregate":source_streamed,
    "streamed_projection_aggregate":candidate_streamed,
    "population_exposure":population_exposure,
    "representative_tensors":{role:next(row for row in rows if row["role"] == role) for role in STREAMED_ROLES},
    "tensor_count":len(rows), "tensors":rows,
  }

def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--dense-verified", action="store_true", help="assert that the supplied model-facts inventory is dense")
  args = parser.parse_args()
  result = build(json.loads(args.input.read_text()), inventory_path=str(args.input), dense_verified=args.dense_verified)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")

if __name__ == "__main__": main()
