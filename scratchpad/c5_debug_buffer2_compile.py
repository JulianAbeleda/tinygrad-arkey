#!/usr/bin/env python3
"""Debug driver: reproduce the NV buffer2 child compile in-process (no spawn, no GPU).

Mirrors scratchpad/c5_nv_canonical_lane_probe.py's payload mint and admission, then calls
prepare_current_prefill_compile directly so the render failure can be inspected.
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import tinygrad.renderer.ptx as ptx_mod
from extra.llm_research.runtime_specs import derive_packed_weight_candidate
from extra.llm_research.prefill.current_prefill_execution_adapter import prepare_current_prefill_compile

MINT_PATH = pathlib.Path(
  "bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-sm120-v1/candidate-set.json")

_orig_render = ptx_mod.PTXRenderer.render


def _debug_render(self, uops):
  if "WMMA" in [u.op.name for u in uops]:
    print("WMMA node shapes:")
    for u in uops:
      if u.op is not None and u.op.name == "WMMA":
        print(f"  WMMA {u.dtype} srcs={[(x.op, str(x.dtype)[:30]) for x in u.src]}")
    print("SPECIAL nodes:")
    for u in uops:
      if u.op is not None and u.op.name == "SPECIAL":
        print(f"  SPECIAL arg={u.arg} vmax={u.vmax} dtype={u.dtype}")
  try:
    return _orig_render(self, uops)
  except Exception:
    from collections import Counter
    from tinygrad.dtype import PtrDType
    kinds = Counter(u.op for u in uops)
    print("PTX render op histogram:", dict(kinds))
    print("vec-dtype nodes:")
    for u in uops:
      if getattr(u.dtype, "vcount", 1) > 1 or isinstance(u.dtype, PtrDType):
        print(f"  {u.op} {u.dtype} (base={getattr(u.dtype, 'base', None)}) srcs={[x.op for x in u.src]} arg={str(u.arg)[:80]}")
    print("GEP nodes:")
    for u in uops:
      if u.op is not None and u.op.name == "GEP":
        print(f"  GEP {u.dtype} srcs={[(x.op, str(x.dtype)[:28]) for x in u.src]} arg={u.arg}")
    print("WMMA nodes:")
    for u in uops:
      if u.op is not None and u.op.name == "WMMA":
        print(f"  WMMA {u.dtype} srcs={[(x.op, str(x.dtype)[:28]) for x in u.src]} arg0={str(u.arg[0])[:40]}")
    raise


def main() -> None:
  ptx_mod.PTXRenderer.render = _debug_render
  artifact = json.loads(MINT_PATH.read_text())
  payload = next(e["payload"] for e in artifact["entries"]
                 if e["payload"]["workload"]["role"] == "attn_kv")
  entry = derive_packed_weight_candidate(payload, "Q4_K")
  final_payload = entry.to_json()["payload"]
  canonical_identity = entry.canonical_identity
  print(f"canonical_identity={canonical_identity}")
  try:
    program, evidence = prepare_current_prefill_compile(final_payload, canonical_identity, device="CUDA")
    print("COMPILE OK")
    print("source sha256:", evidence.get("source_sha256"))
    print("binary sha256:", evidence.get("binary_sha256"))
    src = next((u.arg for u in program.src if u.op.name == "SOURCE" and isinstance(u.arg, str)), None)
    if src is not None:
      out = pathlib.Path("/tmp/buffer2_ptx.txt")
      out.write_text(src)
      print("PTX source written to", out)
  except Exception:
    traceback.print_exc()


if __name__ == "__main__":
  main()
