"""GPU-free census for NV pp512 program inventories."""
from __future__ import annotations
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED = {"q4_gate_up": 72, "qkv": 108, "q4_attention_o": 36,
            "q4_ffn_down": 18, "q6_ffn_down": 18, "flash_attention": 36,
            "q6_vocabulary": 1}
LLAMA = re.compile(r"(?:^|[_$])_?Z|llama|libggml-cuda|scratchpad/llama", re.I)

def classify(p):
  origin = str(p.get("origin", "")).lower()
  path = str(p.get("binary_path", p.get("path", "")))
  symbol = str(p.get("symbol", p.get("name", "")))
  if origin in {"llama_extracted", "llama", "extracted"} or ".cubin" in path and LLAMA.search(path) or LLAMA.search(symbol):
    return "llama_extracted"
  if origin in {"tinygrad_generated", "tinygrad", "uop", "uops"}: return "tinygrad_generated"
  if origin in {"external_source_compiled", "source_compiled", "nvrtc", "nvcc"}: return "external_source_compiled"
  if p.get("source_identity") or p.get("source_hash"): return "external_source_compiled"
  return "unknown"

def audit(data):
  programs = data.get("programs", data) if isinstance(data, dict) else data
  if not isinstance(programs, list): raise ValueError("inventory must contain a programs list")
  rows=[]; violations=[]; families=Counter(); inv=defaultdict(list); owners=defaultdict(set)
  for i,p in enumerate(programs):
    if not isinstance(p, dict): violations.append(f"program[{i}]:not_object"); continue
    family=str(p.get("family", "unknown")); origin=classify(p); families[origin]+=1
    row={"index":i,"family":family,"origin":origin,"symbol":p.get("symbol",p.get("name")),
         "invocation_identity":p.get("invocation_identity",p.get("invocation_id")),
         "owner":p.get("owner",p.get("owner_id"))}
    rows.append(row)
    if origin == "llama_extracted": violations.append(f"program[{i}]:llama_extracted")
    if not row["invocation_identity"]: violations.append(f"program[{i}]:missing_invocation_identity")
    else:
      key=(family,str(row["invocation_identity"])); inv[key].append(i)
      if row["owner"] is not None: owners[key].add(str(row["owner"]))
    if not row["symbol"]: violations.append(f"program[{i}]:missing_symbol")
  duplicates={f"{f}:{k}":v for (f,k),v in inv.items() if len(v)>1}
  for key,indices in duplicates.items(): violations.append(f"duplicate_physical_invocation:{key}")
  one_owner={f"{f}:{k}":sorted(v) for (f,k),v in owners.items() if len(v)>1}
  for key in one_owner: violations.append(f"one_owner_violation:{key}")
  observed=Counter(r["family"] for r in rows)
  populations={f:{"expected":n,"observed":observed.get(f,0),"pass":observed.get(f,0)==n} for f,n in EXPECTED.items()}
  for f,x in populations.items():
    if not x["pass"]: violations.append(f"population:{f}:expected_{x['expected']}_observed_{x['observed']}")
  return {"schema":"nv_all_native_runtime_census/v1","program_count":len(rows),"programs":rows,
          "origin_counts":dict(sorted(families.items())),"populations":populations,
          "duplicate_physical_invocations":duplicates,"one_owner_violations":one_owner,
          "violations":sorted(set(violations)),"all_native":not violations}

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True,type=Path); ap.add_argument("--out",required=True,type=Path); ap.add_argument("--require-all-native",action="store_true"); a=ap.parse_args()
  report=audit(json.loads(a.input.read_text())); a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
  return 1 if a.require_all_native and not report["all_native"] else 0
if __name__ == "__main__": raise SystemExit(main())
