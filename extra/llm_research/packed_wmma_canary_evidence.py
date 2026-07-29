import json, sys

def main():
  from tinygrad.llm import packed_wmma_prefill as production
  from extra.llm_research.prefill.packed_wmma_production_canary import install_production_qualification_verifier
  install_production_qualification_verifier()

  results = []
  for row in production.PACKED_WMMA_ROUTES:
    passed = production.gate_combo(row.quant, row.role, row.shape)
    r = production.gate_result(row.quant, row.role, row.shape)
    results.append({"quant": row.quant, "role": row.role, "shape": list(row.shape), "passed": bool(passed),
                     "max_abs": r[1] if r is not None else None})
    print(row.quant, row.role, row.shape, "->", r, flush=True)

  print(json.dumps(results, indent=2))
  n_pass = sum(1 for r in results if r["passed"])
  print(f"{n_pass}/{len(results)} PASS (production packed-WMMA rows)")

if __name__ == "__main__":
  main()
