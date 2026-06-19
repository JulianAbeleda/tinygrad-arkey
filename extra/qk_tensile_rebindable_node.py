#!/usr/bin/env python3
"""TPE-7a — keystone: prove a single extracted Tensile kernel node is REBINDABLE — re-fed DIFFERENT (A,B,C) buffers
per launch via a graph-style fill_kernargs(bufs), correct every time.

This is the capability the in-model JIT route depends on: one captured graph node must serve (a) different layers
(different weight/activation buffers) and (b) JIT replay with re-bound input buffers. HCQGraph fills kernargs from the
CURRENT call's buffers; the stock CLikeArgsState writes pointers-first, but Tensile needs them at fixed offsets
16=D/24=C/32=A/40=B with the captured scalars/strides/WGM kept. We model that: bind_tensile_kernargs(argsbuf, A,B,C)
writes the captured 128B template + substitutes the 4 VAs. Then we launch the SAME program against several distinct
buffer sets and verify each.

  run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_tensile_rebindable_node.py
"""
from __future__ import annotations
import json, struct, pathlib
from tinygrad import Tensor, Device, dtypes
from tinygrad.runtime.support.hcq import HCQArgsState
from extra.qk_tensile_hcq_launch import NamedAMDProgram, kd_offset, unbundle

T, IN, FF = 512, 4096, 12288   # ffn_gate/up shape (m=512,n=12288,k=4096): A=[in,T], B=W[out,in], C=[out,T]

def bind_tensile_kernargs(argsbuf, raw_template:bytes, A, B, C):
  """graph-protocol fill: write the captured Tensile kernarg, substitute the CURRENT buffers' VAs (rebindable)."""
  buf = bytearray(raw_template); va = lambda t: t.uop.buffer._buf.va_addr
  struct.pack_into("<Q", buf, 16, va(C)); struct.pack_into("<Q", buf, 24, va(C))   # D, C
  struct.pack_into("<Q", buf, 32, va(A)); struct.pack_into("<Q", buf, 40, va(B))   # A, B
  argsbuf.cpu_view().view(size=len(buf), fmt='B')[:] = buf
  return argsbuf

def launch_rebound(dev, prg, raw_template, A, B, C, gws, lws):
  # fresh kernarg buffer + rebind to THESE buffers (models a per-call/per-replay kernarg fill)
  ab = dev.kernargs_buf.offset(offset=dev.kernargs_offset_allocator.alloc(prg.kernargs_alloc_size, 8), size=prg.kernargs_alloc_size)
  bind_tensile_kernargs(ab, raw_template, A, B, C)
  args = HCQArgsState(ab, prg, (), ())
  q = dev.hw_compute_queue_t().wait(dev.timeline_signal, dev.timeline_value - 1).memory_barrier()
  q.exec(prg, args, gws, lws)
  q.signal(dev.timeline_signal, dev.next_timeline()).submit(dev); dev.synchronize()

def main():
  assert Device.DEFAULT == "AMD"; dev = Device[Device.DEFAULT]
  cap = {json.loads(l)["role"]: json.loads(l) for l in open("bench/qk-tensile-extraction/kernarg_all.jsonl")}["ffn_gate_up"]
  raw_template = bytes(cap["kernarg_bytes"]); sym = cap["kernel_symbol"]
  gx,gy,gz = cap["global"]; lx,ly,lz = cap["local"]; gws=(gx//lx,gy//ly,gz//lz); lws=(lx,ly,lz)
  elf = unbundle()
  prg = NamedAMDProgram(dev, "tensile_rebind", elf, kd_offset(elf, sym), raw_template)  # built ONCE

  # several DISTINCT buffer sets (different weights+activations) -> one node, rebound each time
  rows=[]
  for i in range(4):
    Tensor.manual_seed(100+i)
    A = Tensor.randn(IN, T, dtype=dtypes.half).contiguous().realize()
    B = Tensor.randn(FF, IN, dtype=dtypes.half).contiguous().realize()
    C = Tensor.zeros(FF, T, dtype=dtypes.half).contiguous().realize()
    oracle = (B.float() @ A.float()).realize(); dev.synchronize()          # C[out,T] = W[out,in] @ A[in,T]
    launch_rebound(dev, prg, raw_template, A, B, C, gws, lws)
    rel = ((C.float()-oracle).abs().max()/(oracle.float().abs().max()+1e-6)).item()
    rows.append({"binding": i, "rel_err": round(rel,6), "correct": rel<2e-2})
    print(f"  binding {i}: rel_err {rel:.6f} {'OK' if rel<2e-2 else 'BAD'}")

  # also re-launch binding 0 a 2nd time to confirm stable replay on the same buffers
  Tensor.manual_seed(100); A0=Tensor.randn(IN,T,dtype=dtypes.half).contiguous().realize()
  B0=Tensor.randn(FF,IN,dtype=dtypes.half).contiguous().realize(); C0=Tensor.zeros(FF,T,dtype=dtypes.half).contiguous().realize()
  o0=(B0.float()@A0.float()).realize(); dev.synchronize()
  launch_rebound(dev, prg, raw_template, A0, B0, C0, gws, lws); dev.synchronize()
  replay_rel = ((C0.float()-o0).abs().max()/(o0.float().abs().max()+1e-6)).item()

  all_ok = all(r["correct"] for r in rows) and replay_rel < 2e-2
  distinct_results = len({r["rel_err"] for r in rows}) >= 3      # different buffers -> genuinely different work
  res = dict(schema="qk_tensile_rebindable_node_v1", phase="TPE-7a", program_built_once=True,
             bindings=rows, replay_rel_err=round(replay_rel,6), distinct_bindings_ok=distinct_results,
             gates=dict(all_bindings_correct=all_ok, one_node_many_buffers=True, replay_stable=replay_rel<2e-2),
             verdict="PASS" if all_ok else "KILL",
             note="one NamedAMDProgram node, rebound to 4 distinct (A,B,C) sets via graph-style fill_kernargs(bufs); "
                  "proves the captured node can serve different layers / JIT-replayed input buffers. Keystone for TPE-7b/c.")
  pathlib.Path("bench/qk-tensile-extraction/rebindable_node.json").write_text(json.dumps(res,indent=2))
  print(json.dumps(res,indent=2)); print("\nTPE-7a VERDICT:", res["verdict"])

if __name__ == "__main__":
  main()
