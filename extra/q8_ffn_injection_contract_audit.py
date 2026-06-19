#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib

from tinygrad import Tensor, dtypes
from tinygrad.engine.realize import compile_linear, get_call_arg_uops
from tinygrad.uop.ops import Ops
from extra.q8_ffn_fast_artifact_inject import DIM, HIDDEN, PROD_THREADS, Q4_WORDS, Q8_BYTES, gateup_stub, producer_stub

def program_records(ts:list[Tensor]) -> list[dict]:
  linear, var_vals = Tensor.linear_with_vars(*ts)
  assert not var_vals, f"unexpected symbolic vars: {var_vals}"
  compiled = compile_linear(linear, validate=False)
  out = []
  for call in compiled.src:
    ast = call.src[0]
    if ast.op is not Ops.PROGRAM: continue
    args = get_call_arg_uops(call)
    out.append({
      "name": ast.arg.name,
      "global_size": list(ast.arg.global_size),
      "local_size": list(ast.arg.local_size or []),
      "globals": list(ast.arg.globals),
      "outs": list(ast.arg.outs),
      "ins": list(ast.arg.ins),
      "args": [
        {"slot": i, "shape": list(u.shape), "dtype": str(u.dtype), "op": u.op.name}
        for i, u in enumerate(args)
      ],
    })
  return out

def main() -> None:
  x = Tensor.empty(DIM, dtype=dtypes.float32, device="AMD").contiguous()
  w = Tensor.empty(DIM, dtype=dtypes.float32, device="AMD").contiguous()
  norm = Tensor.empty(DIM, dtype=dtypes.float32, device="AMD").contiguous()
  q8 = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous()
  norm_out, q8_out, *_ = norm.custom_kernel(q8, x, w, fxn=producer_stub)

  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_words = Tensor.empty(Q4_WORDS, dtype=dtypes.uint32, device="AMD").contiguous()
  up_words = Tensor.empty(Q4_WORDS, dtype=dtypes.uint32, device="AMD").contiguous()
  gate_out, up_out, *_ = gate.custom_kernel(up, gate_words, up_words, q8, fxn=gateup_stub)

  records = program_records([norm_out, q8_out]) + program_records([gate_out, up_out])
  by_name = {r["name"]: r for r in records}
  expected = {
    "q8_rmsnorm_side_inject": {
      "globals": [0, 1, 2, 3],
      "outs": [0, 1],
      "ins": [2, 3],
      "artifact_arg_order": ["norm_out", "q8", "x", "w"],
      "artifact_global_size": [1, 1, 1],
      "artifact_local_size": [PROD_THREADS, 1, 1],
    },
    "q8_mmvq_gateup_inject": {
      "globals": [0, 1, 2, 3, 4],
      "outs": [0, 1],
      "ins": [2, 3, 4],
      "artifact_arg_order": ["gate", "up", "gate_words", "up_words", "q8"],
      "artifact_global_size": [HIDDEN, 2, 1],
      "artifact_local_size": [32, 4, 1],
    },
  }
  checks = {}
  for name, exp in expected.items():
    got = by_name.get(name)
    checks[name] = {
      "present": got is not None,
      "globals_match": got is not None and got["globals"] == exp["globals"],
      "outs_match": got is not None and got["outs"] == exp["outs"],
      "ins_match": got is not None and got["ins"] == exp["ins"],
      "placeholder_launch": {"global_size": got["global_size"], "local_size": got["local_size"]} if got else None,
      "artifact_launch": {"global_size": exp["artifact_global_size"], "local_size": exp["artifact_local_size"]},
    }

  res = {
    "date": "2026-06-19",
    "phase": "A3-contract-audit",
    "records": records,
    "expected": expected,
    "checks": checks,
    "verdict": "PASS" if all(all(v for k, v in c.items() if k not in {"placeholder_launch", "artifact_launch"}) for c in checks.values()) else "FAIL",
    "note": "No artifact runtime execution. This validates the Tensor PROGRAM buffer contract only; launch dims still require Q8ArtifactRunner override for gate/up.",
  }
  out = pathlib.Path("bench/q8-ffn-handwritten-oracle/fast_artifact_contract_audit.json")
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(res, indent=2) + "\n")
  print(json.dumps(res, indent=2))
  if res["verdict"] != "PASS": raise SystemExit(1)

if __name__ == "__main__":
  main()
