#!/usr/bin/env python3
"""Static probe: what fraction of the fused prefill attention KV-loop body is fixed overhead?

Hypothesis under test (docs/prefill-current-state.md next-lever #1): the KV loop body is ONE
16-token WMMA tile, so the per-iteration fixed costs (P LDS repack + barrier, row max/sum XOR
shuffles, online rescale of all acc blocks) are amortized over only 16 KV tokens -- and that,
not a context-dependent regression, is why our attention runs ~1.5-1.8x less efficiently than
llama's flash_attn_ext.

This probe does NOT run anything on the GPU. It compiles the production emitter's kernel, finds
the KV loop body in the real gfx1100 disassembly, and classifies every instruction in it.
"""
from __future__ import annotations
import json, re, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.uop.ops import Ops, SharedAttentionCandidateContext
from tinygrad.llm.fused_attention import custom_kernel_attention
from extra.qk.attention_harness_common import candidate_context
from extra.qk.mmq_compile_evidence import disassemble_amdgpu, parse_amdgpu_metadata

# The two production geometries (8B / 14B), pp512 first chunk. kv_tokens does not change the
# loop BODY (the loop is a real runtime UOp.range over (kv_tokens+15)//16) -- only its trip count.
GEOMETRIES = (("8b", 32, 8), ("14b", 40, 8))

INST_CLASS = (
  ("wmma",      re.compile(r"^v_wmma_")),
  ("ds_store",  re.compile(r"^ds_store")),
  ("ds_load",   re.compile(r"^ds_(load|read)")),
  ("barrier",   re.compile(r"^s_barrier")),
  ("lds_wait",  re.compile(r"^s_waitcnt_lgkmcnt|^s_waitcnt.*lgkm")),
  ("shuffle",   re.compile(r"^(ds_swizzle|ds_permute|ds_bpermute|v_permlane|.*_dpp)")),
  ("transcend", re.compile(r"^v_(exp|log|rcp|rsq)_")),
  ("global_ld", re.compile(r"^(global_load|buffer_load|flat_load)")),
  ("global_st", re.compile(r"^(global_store|buffer_store|flat_store)")),
  ("vmem_wait", re.compile(r"^s_waitcnt$|^s_waitcnt\s+vmcnt")),
  ("valu",      re.compile(r"^v_")),
  ("salu",      re.compile(r"^s_")),
)
# "useful" = the tensor-core math the kernel exists to do. Everything else in the loop body is
# per-tile plumbing whose cost is what a wider KV tile would amortize.
USEFUL = {"wmma"}


def classify(mnemonic: str) -> str:
  for name, pattern in INST_CLASS:
    if pattern.match(mnemonic):
      return name
  return "other"


# llvm-objdump for amdgpu puts the PC in a trailing comment: "<mnem> <ops>  // <pc>: <bytes>".
# Branch targets appear as a symbol-relative annotation "<kernel+0xNNN>" on the same line.
_INST = re.compile(r"^\s*([a-z][a-z0-9_]*)\s*(.*?)\s*//\s*([0-9A-Fa-f]+):\s*[0-9A-Fa-f ]+$")
_BRTARGET = re.compile(r"<[A-Za-z_][A-Za-z0-9_]*\+0x([0-9a-f]+)>")
_KERNEL_START = re.compile(r"^([0-9a-f]+)\s+<([A-Za-z_][A-Za-z0-9_]*)>:")


def parse_disasm(text: str) -> tuple[list[tuple[int, str, str, int | None]], int]:
  """-> ([(pc, mnemonic, operands, branch_target_or_None)], kernel_base_pc)."""
  out, base = [], None
  for line in text.splitlines():
    if base is None and (m := _KERNEL_START.match(line.strip())):
      base = int(m.group(1), 16)
    # the "<sym+0xNNN>" annotation sits AFTER the byte column, so strip it before matching _INST
    target = int(t.group(1), 16) if (t := _BRTARGET.search(line)) else None
    if (m := _INST.match(_BRTARGET.sub("", line).rstrip())) is None:
      continue
    out.append((int(m.group(3), 16), m.group(1), m.group(2).strip(), target))
  if base is None:
    raise RuntimeError("could not locate kernel symbol start in disassembly")
  return out, base


def find_kv_loop(insts, base: int) -> tuple[int, int, str]:
  """The KV loop is the backward-branch span containing the most WMMAs.

  A backward s_branch/s_cbranch (annotated target < its own pc) is a loop bottom. The Hd-block
  loops are trace-time unrolled Python `for` loops, so they are straight-line code and cannot be
  confused for this one -- the only real branch-formed loop is the KV tile range.
  """
  by_pc = {pc: i for i, (pc, _, _, _) in enumerate(insts)}
  best = None
  for idx, (pc, mnem, _ops, target) in enumerate(insts):
    if target is None or not mnem.startswith(("s_branch", "s_cbranch")):
      continue
    abs_target = base + target
    if abs_target >= pc or abs_target not in by_pc:   # forward branch / not a loop bottom
      continue
    start = by_pc[abs_target]
    wmmas = sum(1 for _, mn, _, _ in insts[start:idx + 1] if mn.startswith("v_wmma_"))
    if wmmas and (best is None or wmmas > best[2]):
      best = (start, idx, wmmas, f"0x{abs_target:x}..0x{pc:x}")
  if best is None:
    raise RuntimeError("no backward branch containing WMMAs found -- loop may be fully unrolled")
  return best[0], best[1], best[3]


def probe(label: str, hq: int, hkv: int, kv: int = 512) -> dict:
  """Compile the PRODUCTION route (custom_kernel_attention, exactly as the model calls it)."""
  profile = f"qwen3_{label}_q4k_m_gfx1100"
  strategy = "FULL_RESIDENT_OVERLAY" if label == "8b" else "BOUNDED_PACKED_TILES"
  ctx = candidate_context(profile, strategy, hq, hkv, kv)
  q = Tensor.empty(1, hq, ctx.q_tokens, 128, dtype=dtypes.float16, device="AMD")
  k = Tensor.empty(1, hkv, kv, 128, dtype=dtypes.float16, device="AMD")
  v = Tensor.empty(1, hkv, kv, 128, dtype=dtypes.float16, device="AMD")
  out = custom_kernel_attention(q, k, v, scale=128 ** -0.5, causal=True, ctx=ctx)
  schedule = out.schedule_linear()
  calls = [c for c in schedule.src if c.op is Ops.CALL
           and "q16_grid_hd128" in str(getattr(c.src[0].arg, "name", ""))]
  if len(calls) != 1:
    raise RuntimeError(f"expected exactly one fused-attention CALL, got {len(calls)}")
  program = to_program(calls[0].src[0], HIPRenderer(Target.parse("AMD:HIP:gfx1100")))
  binary = Device["AMD"].compiler.compile(program.src[3].arg)
  disasm = disassemble_amdgpu(binary)[0]
  meta = parse_amdgpu_metadata(binary)
  insts, base = parse_disasm(disasm)
  start, end, span = find_kv_loop(insts, base)
  body = insts[start:end + 1]
  counts = Counter(classify(mnem) for _, mnem, _, _ in body)
  useful = sum(v for k, v in counts.items() if k in USEFUL)
  total = sum(counts.values())
  return {"route": label, "Hq": hq, "Hkv": hkv, "vgpr": meta.get("vgprs"), "lds_bytes": meta.get("lds_bytes"),
          "spills": (meta.get("vgpr_spills"), meta.get("sgpr_spills")),
          "kernel_instructions": len(insts), "kv_loop_span": span, "kv_loop_instructions": total,
          "useful_wmma": useful, "overhead_instructions": total - useful,
          "overhead_fraction": round((total - useful) / total, 4),
          "by_class": dict(counts.most_common()),
          "disasm_sha_len": len(disasm)}


if __name__ == "__main__":
  results = [probe(*g) for g in GEOMETRIES]
  out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
  print(json.dumps(results, indent=2))
  if out:
    out.write_text(json.dumps(results, indent=2) + "\n")
