import json, sys

def main():
  from extra.qk.prefill.packed_wmma_prefill_candidates import gate_combo, gate_result

  COMBOS = [
    ("Q4_K", "attn_qo", (512, 5120, 5120)),
    ("Q4_K", "attn_kv", (512, 1024, 5120)),
    ("Q4_K", "ffn_gate_up", (512, 17408, 5120)),
    ("Q4_K", "ffn_down", (512, 5120, 17408)),
    ("Q6_K", "attn_kv", (512, 1024, 5120)),
    ("Q6_K", "ffn_down", (512, 5120, 17408)),
  ]

  results = []
  for quant, role, shape in COMBOS:
    passed = gate_combo(quant, role, shape)
    r = gate_result(quant, role, shape)
    results.append({"quant": quant, "role": role, "shape": list(shape), "passed": bool(passed),
                     "max_abs": r[1] if r is not None else None})
    print(quant, role, shape, "->", r, flush=True)

  print(json.dumps(results, indent=2))
  n_pass = sum(1 for r in results if r["passed"])
  print(f"{n_pass}/{len(results)} PASS")

if __name__ == "__main__":
  main()
