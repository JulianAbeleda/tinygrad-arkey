#!/usr/bin/env python3
"""T4: dump the KV loop body of the SHIPPED fused attention kernel and classify sync/sched."""
from __future__ import annotations
import json, re, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.uop.ops import Ops
from tinygrad.llm.fused_attention import custom_kernel_attention
from extra.qk.attention_harness_common import candidate_context
from extra.qk.mmq_compile_evidence import disassemble_amdgpu, parse_amdgpu_metadata
from kv_tile_amortization_probe import parse_disasm, find_kv_loop

OUT = Path(__file__).resolve().parent

def go(label="8b", hq=32, hkv=8, kv=512):
  profile = f"qwen3_{label}_q4k_m_gfx1100"
  strategy = "FULL_RESIDENT_OVERLAY" if label == "8b" else "BOUNDED_PACKED_TILES"
  ctx = candidate_context(profile, strategy, hq, hkv, kv)
  q = Tensor.empty(1, hq, ctx.q_tokens, 128, dtype=dtypes.float16, device="AMD")
  k = Tensor.empty(1, hkv, kv, 128, dtype=dtypes.float16, device="AMD")
  v = Tensor.empty(1, hkv, kv, 128, dtype=dtypes.float16, device="AMD")
  out = custom_kernel_attention(q, k, v, scale=128 ** -0.5, causal=True, ctx=ctx)
  schedule = out.schedule_linear()
  calls = [c for c in schedule.src if c.op is Ops.CALL and "q16_grid_hd128" in str(getattr(c.src[0].arg, "name", ""))]
  program = to_program(calls[0].src[0], HIPRenderer(Target.parse("AMD:HIP:gfx1100")))
  for i, s in enumerate(program.src):
    print(f"--- program.src[{i}] op={s.op} argtype={type(s.arg).__name__}")
  srcstr = program.src[3].arg
  (OUT/"t4_source.txt").write_text(srcstr if isinstance(srcstr, str) else repr(srcstr))
  print("SOURCE HEAD:")
  print("\n".join(str(srcstr).splitlines()[:25]))
  binary = Device["AMD"].compiler.compile(srcstr)
  print("compiler class:", type(Device["AMD"].compiler).__name__)
  disasm = disassemble_amdgpu(binary)[0]
  (OUT/"t4_full.disasm").write_text(disasm)
  insts, base = parse_disasm(disasm)
  start, end, span = find_kv_loop(insts, base)
  body = insts[start:end+1]
  print(f"loop span {span}  instrs={len(body)}  wmma={sum(1 for _,m,_,_ in body if m.startswith('v_wmma'))}")
  with open(OUT/"t4_body.txt","w") as f:
    for pc,m,o,t in body: f.write(f"{pc:08x} {m} {o}\n")
  c = Counter(m for _,m,_,_ in body)
  print("sync/sched:", {k:v for k,v in c.items() if k.startswith(("s_waitcnt","s_delay_alu","s_clause"))})
  print("total loads:", sum(v for k,v in c.items() if k.startswith("global_load")))
  return body

if __name__ == "__main__": go()
