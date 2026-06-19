#!/usr/bin/env python3
"""TPE-7b — TensileRunner: a runtime object conforming to the HCQGraph protocol, so the extracted Tensile kernel can
serve as a graph node. Research-only; no model route, no defaults, decode untouched, no ops_amd.py edit (it's a
probe-local subclass).

HCQGraph (graph/hcq.py) needs from a runtime: `.dev`, `.kernargs_alloc_size`, and
`fill_kernargs(bufs, vars, argsbuf) -> HCQArgsState`; it takes launch dims from the PROGRAM UOp's ProgramInfo and
execs the returned args_state. TensileRunner provides a rebindable Tensile-layout fill_kernargs (writes the captured
128B template, substitutes the CURRENT call's buffer VAs at the fixed offsets), validated against the exact protocol.

  run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_tensile_runtime.py
"""
from __future__ import annotations
import json, struct, pathlib
from tinygrad import Tensor, Device, dtypes
from tinygrad.runtime.support.hcq import HCQArgsState
from extra.qk_tensile_hcq_launch import NamedAMDProgram, kd_offset, unbundle

# Tensile kernarg pointer offsets (same for all rocBLAS HHS roles): D@16, C@24, A@32, B@40.
# bufs order convention for the graph node: (out=C/D, A, B).
PTR_OFFSETS = {"D": 16, "C": 24, "A": 32, "B": 40}

class TensileRunner(NamedAMDProgram):
  """NamedAMDProgram (named-descriptor) + HCQGraph-protocol rebindable Tensile fill_kernargs."""
  def __init__(self, dev, role, cap, elf):
    self._template = bytes(cap["kernarg_bytes"]); self._role = role
    gx,gy,gz = cap["global"]; lx,ly,lz = cap["local"]
    self.tensile_global = (gx//lx, gy//ly, gz//lz); self.tensile_local = (lx,ly,lz)
    super().__init__(dev, f"tensile_{role}", elf, kd_offset(elf, cap["kernel_symbol"]), self._template)
  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, timeout=None):
    # ignore the host kernel's launch dims/vals; FORCE the Tensile grid. bufs = (out, A, B) from ast.arg.globals.
    return super().__call__(*bufs, global_size=self.tensile_global, local_size=self.tensile_local, vals=(), wait=wait, timeout=timeout)
  def fill_kernargs(self, bufs, vars=(), kernargs=None):
    """HCQGraph signature. bufs = (out, A, B) HCQBuffers; write Tensile layout with their VAs (rebindable)."""
    ab = kernargs or self.dev.kernargs_buf.offset(
      offset=self.dev.kernargs_offset_allocator.alloc(self.kernargs_alloc_size, 8), size=self.kernargs_alloc_size)
    buf = bytearray(self._template)
    out_va, a_va, b_va = bufs[0].va_addr, bufs[1].va_addr, bufs[2].va_addr
    struct.pack_into("<Q", buf, PTR_OFFSETS["D"], out_va); struct.pack_into("<Q", buf, PTR_OFFSETS["C"], out_va)
    struct.pack_into("<Q", buf, PTR_OFFSETS["A"], a_va);   struct.pack_into("<Q", buf, PTR_OFFSETS["B"], b_va)
    ab.cpu_view().view(size=len(buf), fmt='B')[:] = buf
    return HCQArgsState(ab, self, tuple(bufs), vals=tuple(vars))

def graph_protocol_launch(dev, runner, out, A, B):
  """exactly mirror HCQGraph: fill_kernargs(bufs) -> exec(args, dims) in one queue."""
  args = runner.fill_kernargs((out.uop.buffer._buf, A.uop.buffer._buf, B.uop.buffer._buf), ())
  q = dev.hw_compute_queue_t().wait(dev.timeline_signal, dev.timeline_value - 1).memory_barrier()
  q.exec(runner, args, runner.tensile_global, runner.tensile_local)
  q.signal(dev.timeline_signal, dev.next_timeline()).submit(dev); dev.synchronize()

def main():
  assert Device.DEFAULT == "AMD"; dev = Device[Device.DEFAULT]
  caps = {json.loads(l)["role"]: json.loads(l) for l in open("bench/qk-tensile-extraction/kernarg_all.jsonl")}
  elf = unbundle()
  shapes = {"ffn_gate_up": (4096,12288), "ffn_down": (12288,4096), "attn_q_o": (4096,4096)}  # (in/k, out/n); m=T=512
  T = 512; rows=[]
  for role,(K,N) in shapes.items():
    runner = TensileRunner(dev, role, caps[role], elf)   # one runner per role (the would-be graph node)
    ok=[]
    for i in range(3):                                    # rebind to distinct buffers via the protocol
      Tensor.manual_seed(7+i)
      A = Tensor.randn(K, T, dtype=dtypes.half).contiguous().realize()   # [in,T]
      B = Tensor.randn(N, K, dtype=dtypes.half).contiguous().realize()   # W[out,in]
      C = Tensor.zeros(N, T, dtype=dtypes.half).contiguous().realize()   # [out,T]
      oracle = (B.float() @ A.float()).realize(); dev.synchronize()
      graph_protocol_launch(dev, runner, C, A, B)
      rel = ((C.float()-oracle).abs().max()/(oracle.float().abs().max()+1e-6)).item()
      ok.append(rel<2e-2)
    rows.append({"role":role, "all_correct":all(ok), "rebindings":len(ok)})
    print(f"  {role:14s} graph-protocol fill+exec, {len(ok)} rebindings: {'OK' if all(ok) else 'BAD'}")
  allok = all(r["all_correct"] for r in rows)
  res = dict(schema="qk_tensile_runtime_v1", phase="TPE-7b", roles=rows,
             conforms_hcqgraph_protocol=allok,
             protocol=dict(provides=[".dev",".kernargs_alloc_size","fill_kernargs(bufs,vars,argsbuf)","named-descriptor exec"],
                           dims_source="PROGRAM-UOp ProgramInfo (the TPE-7c blocker: must carry Tensile global/local)",
                           kernarg="Tensile layout from bufs VAs @16/24/32/40 (overrides CLikeArgsState pointers-first)"),
             verdict="PASS" if allok else "KILL",
             note="TensileRunner conforms to the HCQGraph runtime contract for all 3 roles via rebindable Tensile "
                  "fill_kernargs. Remaining for in-model capture (TPE-7c): construct/replace the precompiled Ops.PROGRAM "
                  "UOp carrying Tensile dims so HCQGraph enqueues with the right launch geometry + this runner.")
  pathlib.Path("bench/qk-tensile-extraction/runtime.json").write_text(json.dumps(res,indent=2))
  print(json.dumps(res,indent=2)); print("\nTPE-7b VERDICT:", res["verdict"])

if __name__ == "__main__":
  main()
