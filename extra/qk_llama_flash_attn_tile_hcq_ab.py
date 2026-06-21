#!/usr/bin/env python3
"""Route B B1.2-1.4 — vendored llama flash_attn_tile launched through tinygrad's HCQ, local A/B vs gqa_coop_vec.

NON-PROMOTABLE reference oracle (family reference_oracle): proves whether the llama-class decode-attention tile WINS
when dispatched by tinygrad's runtime (not just in llama's rocprofv3 trace). Capture-and-replay (Tensile precedent
extra/qk_tensile_hcq_launch.py): the exact kernarg VALUES + geometry were captured from a real ggml-hip decode
(extra/qk_llama_fattn_kernarg_capture.cpp -> bench/qk-llama-hcq-tile/capture_decode_ctx1024.json); here we load the
already-extracted gfx1100 .co, rebuild the 37-arg kernarg at AMD-ABI offsets, patch the 8 pointers to tinygrad
Buffers, and launch the tile + combine. Correctness vs numpy GQA softmax, then A/B vs gqa_coop_vec.

  run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_llama_flash_attn_tile_hcq_ab.py
"""
from __future__ import annotations
import ctypes, json, struct, weakref, math, time
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import round_up, getenv
from tinygrad.device import Buffer, BufferSpec
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.autogen import amdgpu_kd, hsa
from tinygrad.runtime.ops_amd import AMDProgram
from tinygrad.runtime.support.hcq import HCQProgram, HCQArgsState

CFG = "bench/qk-llama-hcq-tile/capture_decode_ctx1024.json"

# --- the 37-arg flash_attn_tile signature (name, size, align, is_ptr) in declaration order (fattn-tile.cuh:788-811) ---
TILE_SPEC = [("Q",8,8,1),("K",8,8,1),("V",8,8,1),("mask",8,8,1),("sinks",8,8,1),("KV_max",8,8,1),("dst",8,8,1),
  ("dst_meta",8,8,1),("scale",4,4,0),("max_bias",4,4,0),("m0",4,4,0),("m1",4,4,0),("n_head_log2",4,4,0),
  ("logit_softcap",4,4,0),("ne00",4,4,0),("ne01",12,4,0),("ne02",4,4,0),("ne03",4,4,0),("nb01",4,4,0),("nb02",4,4,0),
  ("nb03",4,4,0),("ne10",4,4,0),("ne11",4,4,0),("ne12",4,4,0),("ne13",4,4,0),("nb11",4,4,0),("nb12",4,4,0),
  ("nb13",8,8,0),("nb21",4,4,0),("nb22",4,4,0),("nb23",8,8,0),("ne31",4,4,0),("ne32",4,4,0),("ne33",4,4,0),
  ("nb31",4,4,0),("nb32",4,4,0),("nb33",8,8,0)]

def abi_offsets(spec):
  off, offs = 0, []
  for _, sz, al, _ in spec:
    off = round_up(off, al); offs.append(off); off += sz
  return offs, off

def write_hidden(ka, ksize, grid, block):
  """Fill the COV5 implicit-arg block (trailing 256 bytes) so the kernel's gridDim/blockDim reads are valid. tinygrad
  does not populate these (its own kernels don't use them); the vendored llama kernel reads gridDim.y (parallel_blocks).
  Offsets (relative to ksize-256) verified from the .co amdhsa.kernels metadata."""
  if ksize < 256: return                                             # kernel has no full COV5 hidden block
  h = ksize - 256
  struct.pack_into("<III", ka, h+0, grid[0], grid[1], grid[2])        # hidden_block_count_x/y/z (workgroup grid)
  struct.pack_into("<HHH", ka, h+12, block[0], block[1], block[2])    # hidden_group_size_x/y/z
  struct.pack_into("<HHH", ka, h+18, 0, 0, 0)                          # hidden_remainder (grid divides evenly)
  struct.pack_into("<qqq", ka, h+40, 0, 0, 0)                         # hidden_global_offset_x/y/z
  struct.pack_into("<H", ka, h+64, 3)                                 # hidden_grid_dims

def kd_offset(elf:bytes, sym:str) -> int:
  e_shoff = struct.unpack_from("<Q", elf, 0x28)[0]; shent = struct.unpack_from("<H", elf, 0x3a)[0]
  shnum = struct.unpack_from("<H", elf, 0x3c)[0]; target = sym.encode()
  for s in range(shnum):
    sh = e_shoff + s*shent; sht = struct.unpack_from("<I", elf, sh+4)[0]
    if sht not in (2, 11): continue
    o = struct.unpack_from("<Q", elf, sh+0x18)[0]; size = struct.unpack_from("<Q", elf, sh+0x20)[0]
    link = struct.unpack_from("<I", elf, sh+0x28)[0]; esz = struct.unpack_from("<Q", elf, sh+0x38)[0]
    stroff = struct.unpack_from("<Q", elf, e_shoff+link*shent+0x18)[0]
    for i in range(0, size, esz):
      e = o+i; st_name = struct.unpack_from("<I", elf, e)[0]; st_value = struct.unpack_from("<Q", elf, e+8)[0]
      nm = elf[stroff+st_name:elf.index(b"\x00", stroff+st_name)]
      if nm == target: return st_value
  raise ValueError(f"{sym} not found")

class NamedAMDProgram(AMDProgram):
  """Load a named kernel out of a multi-kernel .co (skip AMDProgram's first-.rodata pick) and launch a prebuilt
  kernarg blob. dynamic_lds adds to the descriptor's static group segment (combine needs parallel_blocks*8)."""
  def __init__(self, dev, name, lib, kd_off, raw_kernarg, dynamic_lds=0):
    self.dev, self.name, self.lib, self._raw = dev, name, lib, raw_kernarg
    image, _, relocs = elf_loader(self.lib)
    for ao, rso, typ, addent in relocs:
      if typ == 5: image[ao:ao+8] = struct.pack('<q', rso - ao + addent)
      else: raise RuntimeError(f"reloc {typ}")
    self.lib_gpu = self.dev.allocator.alloc(round_up(image.nbytes, 0x1000), bs:=BufferSpec(nolru=True))
    self.dev.allocator._copyin(self.lib_gpu, image); self.dev.synchronize()
    dsz = ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)
    desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[kd_off:kd_off+dsz]))
    self.group_segment_size = desc.group_segment_fixed_size + dynamic_lds
    self.private_segment_size = desc.private_segment_fixed_size
    self.kernargs_segment_size = desc.kernarg_size
    lds_size = ((self.group_segment_size + 511)//512) & 0x1FF
    self.dev._ensure_has_local_memory(self.private_segment_size)
    self.wave32 = desc.kernel_code_properties & 0x400 == 0x400
    self.rsrc1 = desc.compute_pgm_rsrc1 | ((1<<20) if self.dev.target[0]==11 else 0)
    self.rsrc2 = desc.compute_pgm_rsrc2 | (lds_size<<15); self.rsrc3 = desc.compute_pgm_rsrc3
    self.aql_prog_addr = self.lib_gpu.va_addr + kd_off
    self.prog_addr = self.lib_gpu.va_addr + kd_off + desc.kernel_code_entry_byte_offset
    self.enable_dispatch_ptr = desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR
    self.enable_private_segment_sgpr = desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER
    add = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t) if self.enable_dispatch_ptr else 0
    alloc = max(self.kernargs_segment_size, len(raw_kernarg)) + add
    HCQProgram.__init__(self, HCQArgsState, self.dev, self.name, kernargs_alloc_size=alloc, lib=self.lib,
                        base=self.lib_gpu.va_addr)
    weakref.finalize(self, self._fini, self.dev, self.lib_gpu, bs)
  def fill_kernargs(self, bufs, vals=(), kernargs=None):
    ab = kernargs or self.dev.kernargs_buf.offset(offset=self.dev.kernargs_offset_allocator.alloc(self.kernargs_alloc_size, 8), size=self.kernargs_alloc_size)
    ab.cpu_view().view(size=len(self._raw), fmt='B')[:] = bytearray(self._raw)
    return HCQArgsState(ab, self, tuple(bufs), vals=tuple(vals))

def buf(nbytes):
  b = Buffer("AMD", nbytes, dtypes.uint8).ensure_allocated(); return b
def va(b): return b._buf.va_addr

def main():
  assert Device.DEFAULT == "AMD"
  dev = Device[Device.DEFAULT]
  cfg = json.load(open(CFG))
  elf = open(cfg["co_path"], "rb").read()
  tile_args = [bytes(a) for a in cfg["tile"]["args"]]
  offs, ksize_explicit = abi_offsets(TILE_SPEC)
  # validate the 8 pointers land at 0,8,..,56
  assert offs[:8] == [0,8,16,24,32,40,48,56], offs[:8]

  # ---- shape constants (from capture) ----
  Hd, Hq, Hkv = 128, 32, 8
  KV = struct.unpack("<i", tile_args[22])[0]            # ne11 (padded KV, 1280)
  PB = cfg["tile"]["grid_workgroups"][1]                 # parallel_blocks (20)
  tg, tb = cfg["tile"]["grid_workgroups"], cfg["tile"]["block"]
  cg, cb = cfg["combine"]["grid_workgroups"], cfg["combine"]["block"]
  NVALID = 1024                                          # decode depth; positions >= are masked
  scale = struct.unpack("<f", tile_args[8])[0]
  rng = np.random.default_rng(0)
  # ---- inputs (numpy), laid out to match captured strides ----
  Q  = rng.standard_normal((Hq, Hd)).astype(np.float32)                  # nb02=512 -> contiguous heads
  Kf = (rng.standard_normal((KV, Hkv, Hd))*0.5).astype(np.float16)       # nb11=2048(pos),nb12=256(head)
  Vf = (rng.standard_normal((KV, Hkv, Hd))*0.5).astype(np.float16)
  mask = np.zeros((KV,), np.float16); mask[NVALID:] = np.float16(-np.inf) # additive, flat [KV]
  # ---- numpy reference: GQA softmax(QK/sqrt(d)+mask) V ----
  ref = np.zeros((Hq, Hd), np.float32)
  for h in range(Hq):
    kvh = h // (Hq//Hkv)
    sc = (Q[h:h+1] @ Kf[:,kvh,:].astype(np.float32).T)[0]*scale + mask.astype(np.float32)
    sc = sc[:NVALID]; p = np.exp(sc-sc.max()); p/=p.sum()
    ref[h] = p @ Vf[:NVALID,kvh,:].astype(np.float32)

  # ---- device buffers ----
  bQ, bK, bV, bM = buf(Q.nbytes), buf(Kf.nbytes), buf(Vf.nbytes), buf(mask.nbytes)
  bQ.copyin(memoryview(np.ascontiguousarray(Q))); bK.copyin(memoryview(np.ascontiguousarray(Kf)))
  bV.copyin(memoryview(np.ascontiguousarray(Vf))); bM.copyin(memoryview(np.ascontiguousarray(mask)))
  bDtmp = buf(PB*Hq*Hd*4)            # VKQ_parts: parallel_blocks * ggml_nelements(KQV=DV*1*Hq*1)
  bMeta = buf(Hq*PB*8)              # VKQ_meta: float2 [Hq*parallel_blocks]
  bDst  = buf(Hq*Hd*4)             # final fp32 [Hq,Hd]

  # ---- build tile kernarg ----
  ka = bytearray(max(ksize_explicit, 0))
  for (nm,sz,al,isp), o, val in zip(TILE_SPEC, offs, tile_args): ka[o:o+sz] = val
  ptrs = {"Q":va(bQ),"K":va(bK),"V":va(bV),"mask":va(bM),"sinks":0,"KV_max":0,"dst":va(bDtmp),"dst_meta":va(bMeta)}
  for i,(nm,sz,al,isp) in enumerate(TILE_SPEC):
    if isp: struct.pack_into("<Q", ka, offs[i], ptrs[nm])
  tkd = kd_offset(elf, cfg["tile_kd_symbol"])
  tile = NamedAMDProgram(dev, "flash_attn_tile", elf, tkd, bytes(ka))
  ka_full = bytearray(max(tile.kernargs_segment_size, len(ka))); ka_full[:len(ka)] = ka
  write_hidden(ka_full, tile.kernargs_segment_size, tg, tb); tile._raw = bytes(ka_full)
  print(f"  tile kernarg_size(desc)={tile.kernargs_segment_size} explicit={len(ka)} lds={tile.group_segment_size}")

  # ---- build combine kernarg (4 args: VKQ_parts, VKQ_meta, dst, parallel_blocks) ----
  ck = bytearray(32)
  struct.pack_into("<Q", ck, 0, va(bDtmp)); struct.pack_into("<Q", ck, 8, va(bMeta))
  struct.pack_into("<Q", ck, 16, va(bDst)); struct.pack_into("<i", ck, 24, PB)
  cdyn = PB*8   # dynamic LDS = parallel_blocks*sizeof(float2)
  ckd = kd_offset(elf, cfg["combine_kd_symbol"])
  comb = NamedAMDProgram(dev, "flash_attn_combine", elf, ckd, bytes(ck), dynamic_lds=cdyn)
  ck_full = bytearray(max(comb.kernargs_segment_size, len(ck))); ck_full[:len(ck)] = ck
  write_hidden(ck_full, comb.kernargs_segment_size, cg, cb); comb._raw = bytes(ck_full)
  print(f"  combine kernarg_size(desc)={comb.kernargs_segment_size} lds={comb.group_segment_size}")

  def run():
    tile(global_size=tuple(tg), local_size=tuple(tb), wait=True, timeout=10000)
    comb(global_size=tuple(cg), local_size=tuple(cb), wait=True, timeout=10000)
    dev.synchronize()
  run()
  ob = bytearray(Hq*Hd*4); bDst.copyout(memoryview(ob))
  out = np.frombuffer(bytes(ob), np.float32).reshape(Hq, Hd)
  rel = float(np.abs(out-ref).max() / (np.abs(ref).max()+1e-6))
  rmse = float(np.sqrt(((out-ref)**2).mean()) / (np.sqrt((ref**2).mean())+1e-9))
  print(f"correctness: rel_max={rel:.4e} rel_rmse={rmse:.4e}  (KV={KV} valid={NVALID} pb={PB} scale={scale:.5f})")
  print(f"  out[0,:4]={out[0,:4]}  ref[0,:4]={ref[0,:4]}")
  return rel, rmse

if __name__ == "__main__":
  main()
