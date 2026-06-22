#!/usr/bin/env python3
"""Route B B4 -- the EXTERNAL-precompiled-AMDGCN-kernel-as-JIT-graph-node capability.

Injects the B3 owned decode-attention tile (extra/qk_owned_flash_decode.hip -- llama-style flash decode, tinygrad's
NATIVE K/V layout [Hkv,MAXC,Hd], NO repack) into tinygrad's decode JIT graph as Ops.PROGRAM nodes via
Tensor.custom_kernel, by handing custom_kernel a FULLY-FORMED Ops.PROGRAM whose src carries a PRECOMPILED BINARY
(our .co ELF). This is NOT Route-A codegen (we do not make the renderer emit the tile); it is the runtime/graph
scheduling capability that B3 named as the single blocker to W==D.

Mechanism (audited; see docs/decode-attention-route-b-b4-external-graph-node-scope-20260621.md):
  - A 5-src PROGRAM (SINK, DEVICE, LINEAR, SOURCE, BINARY) + an explicit ProgramInfo arg SKIPS codegen
    (codegen/__init__.py:226 keeps an explicit ProgramInfo; pm_to_program matches only incomplete PROGRAMs).
  - get_runtime builds the AMDProgram straight from the BINARY (engine/realize.py:114); HCQGraph schedules it as one
    exec node (runtime/graph/hcq.py:175).
  - CLikeArgsState lays out [ptrs in bufs order][vars as 4B] -> matches the kernel ABI with bufs=[Q,K,V,part,meta]
    and the single per-step var start_pos; S and scale are BAKED (compile-time), kernel uses n_valid=start_pos+1.

  run (standalone fixed-shape correctness + JIT capture/replay proof):
    DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_owned_flash_decode_graph_node.py [S]
"""
from __future__ import annotations
import re, struct, subprocess, pathlib, hashlib, math
import numpy as np
from tinygrad import Tensor, dtypes, UOp
from tinygrad.uop.ops import Ops, KernelInfo, ProgramInfo
from tinygrad.renderer import Estimates

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "extra/qk_owned_flash_decode.hip"
HIPCC = "/opt/rocm-7.2.4/bin/hipcc"
BUNDLER = "/opt/rocm-7.2.4/llvm/bin/clang-offload-bundler"
Hd, Hq, Hkv, G = 128, 32, 8, 4
SCALE = 1.0 / math.sqrt(Hd)

# ---- specialize a single B3 kernel into its own source (S/scale baked, start_pos arg) -> single-kernel ELF ----
def _preamble(src:str, maxc:int) -> str:
  pre = src[:src.index('extern "C"')]   # #include + #define block before the first kernel
  return pre.replace("#define MAXC 4096", f"#define MAXC {maxc}")   # bake the model's actual max_context (KV stride)

def _extract(src:str, sym:str) -> str:
  m = re.search(r'extern "C" __global__ void '+sym+r'\b.*?\n\}\n', src, re.S)
  assert m, f"kernel {sym} not found"
  return m.group(0)

def _specialize_tile(src:str, S:int, maxc:int) -> str:
  body = _extract(src, "owned_flash_tile_gqa")
  # drop the (int n_valid, int S, float scale) params -> (int start_pos); bake constants at body top
  body = body.replace("int n_valid, int S, float scale)", "int start_pos)")
  inject = (f"\n  const int n_valid = start_pos + 1;   // T=1 decode (B4 graph-node specialization)\n"
            f"  const int S = {S};\n  const float scale = {SCALE!r}f;\n")
  body = body.replace("{\n", "{"+inject, 1)
  return _preamble(src, maxc) + "#define TK 16\n" + body

def _specialize_combine(src:str, S:int, maxc:int) -> str:
  body = _extract(src, "owned_flash_combine")
  body = body.replace("float* __restrict__ out, int S)", "float* __restrict__ out)")
  body = body.replace("{\n", "{\n  const int S = "+str(S)+";\n", 1)
  return _preamble(src, maxc) + body

def _compile(source:str, tag:str) -> bytes:
  h = hashlib.sha256(source.encode()).hexdigest()[:12]
  elf = pathlib.Path(f"/tmp/b4_{tag}_{h}.elf")
  if not elf.exists():
    hip = pathlib.Path(f"/tmp/b4_{tag}_{h}.hip"); hip.write_text(source)
    co = f"/tmp/b4_{tag}_{h}.co"
    subprocess.run([HIPCC, "--offload-arch=gfx1100", "--genco", "-O3", "-D__AMDGCN_WAVEFRONT_SIZE=32",
                    str(hip), "-o", co], check=True, capture_output=True)
    subprocess.run([BUNDLER, "--type=o", "--unbundle", f"--input={co}", f"--output={elf}",
                    "--targets=hipv4-amdgcn-amd-amdhsa--gfx1100"], check=True, capture_output=True)
  return elf.read_bytes()

# ---- build the precompiled Ops.PROGRAM node ----
def _make_program(name:str, elf:bytes, placeholders:list[UOp], scalar_vars:tuple[UOp,...],
                  grid:tuple[int,int,int], block:tuple[int,int,int], outs:tuple[int,...], ins:tuple[int,...],
                  group_seg:int, est_ops:int, est_mem:int) -> UOp:
  # SINK is inert (we set ProgramInfo explicitly + BINARY present -> no codegen); it carries KernelInfo estimates and
  # references the buffer placeholders + scalar vars so the body is structurally valid.
  sink = UOp.sink(*[p for p in placeholders], *scalar_vars,
                  arg=KernelInfo(name=name, estimates=Estimates(ops=est_ops, mem=est_mem)))
  pinfo = ProgramInfo(name=name, global_size=grid, local_size=block, vars=scalar_vars,
                      globals=tuple(range(len(placeholders))), outs=outs, ins=ins, aux=())
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=()),
                               UOp(Ops.SOURCE, arg=""), UOp(Ops.BINARY, arg=elf)), arg=pinfo)

# group_segment_fixed_size from the kernel descriptor (LDS bytes) -- needed for rsrc2; read from ELF.
def _group_seg(elf:bytes) -> int:
  from tinygrad.runtime.support.elf import elf_loader
  import ctypes
  from tinygrad.runtime.autogen import amdgpu_kd
  image, sections, _ = elf_loader(elf)
  rodata = next(sh.header.sh_addr for sh in sections if sh.name == ".rodata")
  dsz = ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)
  desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[rodata:rodata+dsz]))
  return desc.group_segment_fixed_size

# ---- the public entry: returns the [Hq,Hd] fp32 attention output Tensor via two graph nodes ----
import functools
@functools.lru_cache(maxsize=None)
def _kernels(S:int, maxc:int):
  """Compile (cached) the specialized single-kernel ELFs for split count S and the model's max_context (=KV stride).
  Memoized so the per-layer model route does not recompile/re-read during JIT capture."""
  src = SRC.read_text()
  tile_elf = _compile(_specialize_tile(src, S, maxc), f"tile_s{S}_m{maxc}")
  comb_elf = _compile(_specialize_combine(src, S, maxc), f"comb_s{S}_m{maxc}")
  return tile_elf, comb_elf, _group_seg(tile_elf), _group_seg(comb_elf)

def amdgcn_flash_decode(Q:Tensor, K:Tensor, V:Tensor, start_pos_var:UOp, S:int=48, MAXC:int=4096) -> Tensor:
  """Q:[Hq,Hd] fp16, K/V:[Hkv,MAXC,Hd] fp16 (native layout, MAXC=model max_context = KV stride), start_pos_var:
  unbound 'start_pos' DEFINE_VAR. Returns out:[Hq,Hd] fp32. Two precompiled graph nodes: tile -> combine."""
  tile_elf, comb_elf, tile_lds, comb_lds = _kernels(int(S), int(MAXC))

  part = Tensor.empty(Hq*S*Hd, dtype=dtypes.float32)
  meta = Tensor.empty(Hq*S*2, dtype=dtypes.float32)
  out  = Tensor.empty(Hq*Hd, dtype=dtypes.float32)
  Qf, Kf, Vf = Q.reshape(Hq*Hd), K.reshape(Hkv*MAXC*Hd), V.reshape(Hkv*MAXC*Hd)

  # tile: bufs = [Q, K, V, part, meta] (kernel arg order); writes part(3), meta(4); reads Q(0),K(1),V(2)
  def tile_fxn(*ph):
    return _make_program("owned_flash_tile_gqa", tile_elf, list(ph), (start_pos_var,),
                         (Hkv, S, 1), (128, 1, 1), outs=(3, 4), ins=(0, 1, 2),
                         group_seg=tile_lds, est_ops=Hq*MAXC*Hd*2, est_mem=Hkv*MAXC*Hd*2*2)
  r = Tensor.custom_kernel(Qf, Kf, Vf, part, meta, fxn=tile_fxn)
  part2, meta2 = r[3], r[4]

  # combine: bufs = [part, meta, out]; writes out(2); reads part(0),meta(1)
  def comb_fxn(*ph):
    return _make_program("owned_flash_combine", comb_elf, list(ph), (),
                         (Hq, 1, 1), (32, 1, 1), outs=(2,), ins=(0, 1),
                         group_seg=comb_lds, est_ops=Hq*S*Hd, est_mem=Hq*S*Hd*4)
  rc = Tensor.custom_kernel(part2, meta2, out, fxn=comb_fxn)
  return rc[2].reshape(Hq, Hd)


MAXC_TEST = 4096
def _numpy_ref(Q, K, V, nvalid):
  ref = np.zeros((Hq, Hd), np.float32)
  for h in range(Hq):
    kvh = h // G
    sc = (Q[h:h+1].astype(np.float32) @ K[kvh, :nvalid].astype(np.float32).T)[0] * SCALE
    p = np.exp(sc - sc.max()); p /= p.sum()
    ref[h] = p @ V[kvh, :nvalid].astype(np.float32)
  return ref


def main():
  import sys
  from tinygrad import Device, TinyJit
  assert Device.DEFAULT == "AMD"
  S = int(sys.argv[1]) if len(sys.argv) > 1 else 48
  rng = np.random.default_rng(0)
  Qn = rng.standard_normal((Hq, Hd)).astype(np.float16)
  Kn = (rng.standard_normal((Hkv, MAXC_TEST, Hd))*0.5).astype(np.float16)
  Vn = (rng.standard_normal((Hkv, MAXC_TEST, Hd))*0.5).astype(np.float16)
  Qt, Kt, Vt = Tensor(Qn).realize(), Tensor(Kn).realize(), Tensor(Vn).realize()
  vsp = UOp.variable("start_pos", 0, MAXC_TEST-1)

  def run(sp):  # sp = vsp.bind(n)
    # carry the bound start_pos into var_vals via a tiny real op (the model carries it through the KV-cache store);
    # the kernel itself uses the UNBOUND twin vsp (same expr 'start_pos').
    carry = Tensor.ones(MAXC_TEST, dtype=dtypes.float32)[0:sp].sum().reshape(1, 1) * 0.0
    return (amdgcn_flash_decode(Qt, Kt, Vt, vsp, S, MAXC_TEST) + carry).realize()

  jf = TinyJit(run)
  for call_i, n in enumerate((1023, 1023, 511)):   # eager / capture / replay(diff start_pos)
    out = jf(vsp.bind(n)).numpy()
    ref = _numpy_ref(Qn, Kn, Vn, n+1)
    rel = float(np.abs(out-ref).max()/(np.abs(ref).max()+1e-6))
    rmse = float(np.sqrt(((out-ref)**2).mean())/(np.sqrt((ref**2).mean())+1e-9))
    tag = ("eager", "capture", "replay")[call_i]
    print(f"  [{tag}] start_pos={n}: rel_max={rel:.3e} rel_rmse={rmse:.3e} {'OK' if rmse<=1e-3 else 'FAIL'}")

  # prove the kernels were captured as Ops.PROGRAM graph nodes
  assert jf.captured is not None, "TinyJit captured nothing"
  names = [u.src[0].arg.name for u in jf.captured.linear.toposort()
           if u.op is Ops.CALL and len(u.src) and u.src[0].op is Ops.PROGRAM]
  print("  captured PROGRAM nodes:", [n for n in names if n.startswith("owned_flash")])

if __name__ == "__main__":
  main()
