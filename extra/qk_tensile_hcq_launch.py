#!/usr/bin/env python3
"""TPE-3 — minimal HCQ launch proof: run the selected rocBLAS Tensile ffn_gate/up kernel from tinygrad HCQ on
tinygrad-owned buffers, no HIP runtime, no copies.

Pieces:
- named-descriptor loader: a probe-local AMDProgram subclass that resolves <kernel>.kd from the multi-kernel object
  (AMDProgram normally uses the FIRST .rodata descriptor);
- exact kernarg: the 128-byte buffer CAPTURED from a separate HIP-only rocBLAS run (extra/qk_tensile_kernarg_capture
  -> /tmp/kernarg.json), with the 4 Address VAs substituted by tinygrad buffer pointers (removes WGM guesswork);
- launch grid(512,96,1)/wg(128,1,1) == num_workgroups(4,96,1) via the HCQ compute queue; verify vs fp16 oracle.

The GEMM (from the captured strides): col-major C[512,12288] = A[512,4096]*B[4096,12288], alpha=1/beta=0.
  setup: DEV=AMD; the .co was unbundled to ELF via clang-offload-bundler (host tool, no in-proc HIP).
  run:   DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_tensile_hcq_launch.py
"""
from __future__ import annotations
import ctypes, json, struct, subprocess, pathlib, weakref
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import round_up
from tinygrad.device import BufferSpec
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.autogen import amdgpu_kd, hsa
from tinygrad.runtime.ops_amd import AMDProgram
from tinygrad.runtime.support.hcq import HCQArgsState

CO = "/opt/rocm-7.2.4/lib/rocblas/library/TensileLibrary_Type_HH_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx1100.co"
BUNDLER = "/opt/rocm-7.2.4/llvm/bin/clang-offload-bundler"

def unbundle() -> bytes:
  out = "/tmp/tpe3_kernel.elf"
  subprocess.run([BUNDLER, "--type=o", "--unbundle", f"--input={CO}", f"--output={out}",
                  "--targets=hipv4-amdgcn-amd-amdhsa--gfx1100"], check=True, capture_output=True)
  return pathlib.Path(out).read_bytes()

def kd_offset(elf:bytes, sym:str) -> int:
  # find st_value of `<sym>.kd` in SHT_SYMTAB(2)/SHT_DYNSYM(11)
  e_shoff = struct.unpack_from("<Q", elf, 0x28)[0]; shent = struct.unpack_from("<H", elf, 0x3a)[0]; shnum = struct.unpack_from("<H", elf, 0x3c)[0]
  target = (sym + ".kd").encode()
  for s in range(shnum):
    sh = e_shoff + s*shent; sht = struct.unpack_from("<I", elf, sh+4)[0]
    if sht not in (2, 11): continue
    off = struct.unpack_from("<Q", elf, sh+0x18)[0]; size = struct.unpack_from("<Q", elf, sh+0x20)[0]
    link = struct.unpack_from("<I", elf, sh+0x28)[0]; esz = struct.unpack_from("<Q", elf, sh+0x38)[0]
    strsh = e_shoff + link*shent; stroff = struct.unpack_from("<Q", elf, strsh+0x18)[0]
    for i in range(0, size, esz):
      e = off + i; st_name = struct.unpack_from("<I", elf, e)[0]; st_value = struct.unpack_from("<Q", elf, e+8)[0]
      nm = elf[stroff+st_name:elf.index(b"\x00", stroff+st_name)]
      if nm == target: return st_value
  raise ValueError(f"{sym}.kd not found in symtab")

class NamedAMDProgram(AMDProgram):
  def __init__(self, dev, name:str, lib:bytes, kd_off:int, raw_kernarg:bytes):
    self.dev, self.name, self.lib, self._raw = dev, name, lib, raw_kernarg
    image, sections, relocs = elf_loader(self.lib)
    rodata_entry = kd_off                                   # <-- named descriptor, not first .rodata
    for ao, rso, typ, addent in relocs:
      if typ == 5: image[ao:ao+8] = struct.pack('<q', rso - ao + addent)
      else: raise RuntimeError(f"unknown AMD reloc {typ}")
    self.lib_gpu = self.dev.allocator.alloc(round_up(image.nbytes, 0x1000), bs:=BufferSpec(nolru=True))
    self.dev.allocator._copyin(self.lib_gpu, image); self.dev.synchronize()
    dsz = ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)
    desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[rodata_entry:rodata_entry+dsz]))
    self.group_segment_size = desc.group_segment_fixed_size; self.private_segment_size = desc.private_segment_fixed_size
    self.kernargs_segment_size = desc.kernarg_size
    lds_size = ((self.group_segment_size + 511)//512) & 0x1FF
    self.dev._ensure_has_local_memory(self.private_segment_size)
    self.wave32 = desc.kernel_code_properties & 0x400 == 0x400
    self.rsrc1 = desc.compute_pgm_rsrc1 | ((1<<20) if self.dev.target[0]==11 else 0)
    self.rsrc2 = desc.compute_pgm_rsrc2 | (lds_size<<15); self.rsrc3 = desc.compute_pgm_rsrc3
    self.aql_prog_addr = self.lib_gpu.va_addr + rodata_entry
    self.prog_addr = self.lib_gpu.va_addr + rodata_entry + desc.kernel_code_entry_byte_offset
    self.enable_dispatch_ptr = desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR
    self.enable_private_segment_sgpr = desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER
    add = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t) if self.enable_dispatch_ptr else 0
    # NOTE: this kernel's .kd kernarg_size field reads 0; the real size (128) is in the metadata. Allocate >= the
    # captured raw kernarg so we never write past the allocation (TPE-2 metadata kernarg_segment_size=128).
    alloc = max(self.kernargs_segment_size, len(raw_kernarg)) + add
    # skip AMDProgram.__init__; call HCQProgram.__init__ (grandparent) directly
    from tinygrad.runtime.support.hcq import HCQProgram
    HCQProgram.__init__(self, HCQArgsState, self.dev, self.name, kernargs_alloc_size=alloc,
                        lib=self.lib, base=self.lib_gpu.va_addr)
    weakref.finalize(self, self._fini, self.dev, self.lib_gpu, bs)
    self.contract = dict(kernargs_segment_size=self.kernargs_segment_size, group=self.group_segment_size,
                         enable_dispatch_ptr=bool(self.enable_dispatch_ptr), wave32=self.wave32)
  def fill_kernargs(self, bufs, vals=(), kernargs=None):
    ab = kernargs or self.dev.kernargs_buf.offset(offset=self.dev.kernargs_offset_allocator.alloc(self.kernargs_alloc_size, 8), size=self.kernargs_alloc_size)
    ab.cpu_view().view(size=len(self._raw), fmt='B')[:] = bytearray(self._raw)   # raw Tensile kernarg
    return HCQArgsState(ab, self, tuple(bufs), vals=tuple(vals))

def main():
  assert Device.DEFAULT == "AMD"
  dev = Device[Device.DEFAULT]
  cap = json.load(open("/tmp/kernarg.json")); raw = bytearray(cap["kernarg_bytes"]); assert len(raw) == 128
  sym = json.load(open("bench/qk-tensile-extraction/selection.json"))["selected"]["rocblas"]["kernel_symbol"]
  # GEMM (col-major): C[512,12288]=A[512,4096]*B[4096,12288]. tinygrad row-major: A_t[K,M]=colmajA, B_t[N,K]=colmajB, C_t[N,M]=colmajC
  Tensor.manual_seed(0)
  A_t = Tensor.randn(4096, 512, dtype=dtypes.half).contiguous().realize()       # col-major A (M=512,K=4096)
  B_t = Tensor.randn(12288, 4096, dtype=dtypes.half).contiguous().realize()     # col-major B (K=4096,N=12288)
  C_t = Tensor.zeros(12288, 512, dtype=dtypes.half).contiguous().realize()      # col-major C output
  oracle = (B_t.float() @ A_t.float()).realize()                                # [N,M] = colmaj C[m,n]
  dev.synchronize()
  va = lambda t: t.uop.buffer._buf.va_addr
  struct.pack_into("<Q", raw, 16, va(C_t)); struct.pack_into("<Q", raw, 24, va(C_t))  # AddressD, AddressC
  struct.pack_into("<Q", raw, 32, va(A_t)); struct.pack_into("<Q", raw, 40, va(B_t))  # AddressA, AddressB

  elf = unbundle(); kd = kd_offset(elf, sym)
  prg = NamedAMDProgram(dev, "tensile_ffn_gate_up", elf, kd, bytes(raw))
  res = {"phase":"TPE-3","kernel_symbol":sym[:60]+"...","kd_offset":hex(kd),"contract":prg.contract,
         "launch":{"global":[4,96,1],"local":[128,1,1]},"no_hip_runtime":True,"no_copies":True}
  errs=[]
  for i in range(6):
    C_t.uop.buffer._buf  # ensure realized
    prg(global_size=(4,96,1), local_size=(128,1,1), wait=True, timeout=10000)
    dev.synchronize()
    diff = (C_t.float() - oracle).abs(); rel = (diff.max()/(oracle.abs().max()+1e-6)).item()
    errs.append({"run":i, "max_abs":round(diff.max().item(),4), "rel_err":round(rel,5)})
  res["runs"]=errs; res["rel_err_last"]=errs[-1]["rel_err"]; res["correct"]=all(e["rel_err"]<2e-2 for e in errs)
  res["stable"]=len({round(e["rel_err"],4) for e in errs})<=2
  res["verdict"]="PASS" if (res["correct"] and res["stable"]) else "KILL"
  out=pathlib.Path("bench/qk-tensile-extraction/hcq_launch.json"); out.write_text(json.dumps(res,indent=2))
  print(json.dumps(res,indent=2)); print("\nTPE-3 VERDICT:", res["verdict"])

if __name__ == "__main__":
  main()
