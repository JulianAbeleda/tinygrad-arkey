#!/usr/bin/env python3
"""TPE-2 — launch-contract extraction for a selected Tensile GEMM kernel.

Unbundles a Tensile .co (compressed clang offload bundle, 'CCOB') to the gfx1100 AMDGPU ELF, finds the
NT_AMDGPU_METADATA note, msgpack-decodes it, and emits the machine-readable launch contract for the selected kernel:
kernarg byte layout (.args: offset/size/value_kind/address_space), kernarg_segment_size/align, group/private segment,
sgpr/vgpr/wavefront, and the .kd descriptor symbol. No HIP runtime, no msgpack/pyelftools deps (hand-rolled).

  PYTHONPATH=. .venv/bin/python extra/qk_tensile_contract.py
"""
from __future__ import annotations
import json, struct, subprocess, pathlib, sys

BUNDLER = "/opt/rocm-7.2.4/llvm/bin/clang-offload-bundler"
ROCBLAS_LIB = "/opt/rocm-7.2.4/lib/rocblas/library"

# ---- minimal msgpack decoder (maps/arrays/str/int/bool/nil/bin) ----
def mp_decode(b:memoryview, i:int=0):
  c = b[i]; i += 1
  if c < 0x80: return c, i                              # positive fixint
  if c >= 0xe0: return c - 0x100, i                     # negative fixint
  if 0x80 <= c <= 0x8f: return _mp_map(b, i, c & 0xf)
  if 0x90 <= c <= 0x9f: return _mp_arr(b, i, c & 0xf)
  if 0xa0 <= c <= 0xbf: return _mp_str(b, i, c & 0x1f)
  if c == 0xc0: return None, i
  if c == 0xc2: return False, i
  if c == 0xc3: return True, i
  if c == 0xcc: return b[i], i+1
  if c == 0xcd: return struct.unpack_from(">H", b, i)[0], i+2
  if c == 0xce: return struct.unpack_from(">I", b, i)[0], i+4
  if c == 0xcf: return struct.unpack_from(">Q", b, i)[0], i+8
  if c == 0xd9: n = b[i]; return _mp_str(b, i+1, n)
  if c == 0xda: n = struct.unpack_from(">H", b, i)[0]; return _mp_str(b, i+2, n)
  if c == 0xdb: n = struct.unpack_from(">I", b, i)[0]; return _mp_str(b, i+4, n)
  if c == 0xdc: n = struct.unpack_from(">H", b, i)[0]; return _mp_arr(b, i+2, n)
  if c == 0xdd: n = struct.unpack_from(">I", b, i)[0]; return _mp_arr(b, i+4, n)
  if c == 0xde: n = struct.unpack_from(">H", b, i)[0]; return _mp_map(b, i+2, n)
  if c == 0xdf: n = struct.unpack_from(">I", b, i)[0]; return _mp_map(b, i+4, n)
  raise ValueError(f"msgpack byte {hex(c)} @ {i-1} unsupported")
def _mp_str(b, i, n): return bytes(b[i:i+n]).decode("utf-8", "replace"), i+n
def _mp_arr(b, i, n):
  out=[]
  for _ in range(n): v,i = mp_decode(b,i); out.append(v)
  return out, i
def _mp_map(b, i, n):
  out={}
  for _ in range(n): k,i = mp_decode(b,i); v,i = mp_decode(b,i); out[k]=v
  return out, i

# ---- ELF64: collect kernels from ALL NT_AMDGPU_METADATA notes (Tensile emits one note per kernel) ----
def amdgpu_kernels(elf:bytes) -> list:
  assert elf[:4] == b"\x7fELF", "not ELF"
  e_shoff = struct.unpack_from("<Q", elf, 0x28)[0]
  e_shentsize = struct.unpack_from("<H", elf, 0x3a)[0]
  e_shnum = struct.unpack_from("<H", elf, 0x3c)[0]
  kerns = []
  for s in range(e_shnum):
    sh = e_shoff + s*e_shentsize
    if struct.unpack_from("<I", elf, sh+4)[0] != 7: continue   # SHT_NOTE
    off = struct.unpack_from("<Q", elf, sh+0x18)[0]; size = struct.unpack_from("<Q", elf, sh+0x20)[0]
    p = off; end = off+size
    while p < end:
      namesz, descsz, ntype = struct.unpack_from("<III", elf, p); p += 12
      name = elf[p:p+namesz].rstrip(b"\x00"); p += (namesz+3)&~3
      desc = elf[p:p+descsz]; p += (descsz+3)&~3
      if name == b"AMDGPU" and ntype == 32:                    # NT_AMDGPU_METADATA
        md,_ = mp_decode(memoryview(desc)); kerns.extend(md.get("amdhsa.kernels", []))
  if not kerns: raise ValueError("no NT_AMDGPU_METADATA kernels")
  return kerns

def main():
  sym = json.load(open("bench/qk-tensile-extraction/selection.json"))["selected"]["rocblas"]["kernel_symbol"]
  co = f"{ROCBLAS_LIB}/TensileLibrary_Type_HH_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx1100.co"  # Ailk_Bljk layout
  elf_path = "/tmp/tpe2_kernel.elf"
  subprocess.run([BUNDLER, "--type=o", "--unbundle", f"--input={co}", f"--output={elf_path}",
                  "--targets=hipv4-amdgcn-amd-amdhsa--gfx1100"], check=True, capture_output=True)
  elf = pathlib.Path(elf_path).read_bytes()
  kerns = amdgpu_kernels(elf)
  k = next((x for x in kerns if x.get(".name") == sym), None)
  if k is None: print(f"kernel {sym[:40]} not found in metadata ({len(kerns)} kernels)"); sys.exit(2)
  args = [{"offset": a.get(".offset"), "size": a.get(".size"), "value_kind": a.get(".value_kind"),
           "address_space": a.get(".address_space"), "name": a.get(".name")} for a in k.get(".args", [])]
  ptr_args = [a for a in args if a["value_kind"] in ("global_buffer", "dynamic_shared_pointer")]
  hidden = [a for a in args if str(a["value_kind"]).startswith("hidden")]
  import hashlib
  contract = {
    "schema": "qk_tensile_contract_v1", "phase": "TPE-2", "role": "ffn_gate/up", "shape": {"M":512,"N":12288,"K":4096},
    "library": "rocBLAS", "code_object": {"co_path": co, "unbundled_target": "hipv4-amdgcn-amd-amdhsa--gfx1100",
      "co_sha256_16": hashlib.sha256(pathlib.Path(co).read_bytes()).hexdigest()[:16]},
    "kernel_symbol": sym, "descriptor_symbol": sym + ".kd",
    "kernarg_segment_size": k.get(".kernarg_segment_size"), "kernarg_segment_align": k.get(".kernarg_segment_align"),
    "group_segment_fixed_size": k.get(".group_segment_fixed_size"), "private_segment_fixed_size": k.get(".private_segment_fixed_size"),
    "sgpr_count": k.get(".sgpr_count"), "vgpr_count": k.get(".vgpr_count"), "wavefront_size": k.get(".wavefront_size"),
    "max_flat_workgroup_size": k.get(".max_flat_workgroup_size"),
    "launch_geometry": {"grid": [512,96,1], "workgroup": [128,1,1], "source": "TPE-1 rocprofv3 trace, fixed shape"},
    "workspace": "none (SU0_SUM0_SUS0 = no StreamK/GSU; scratch=0; single kernel)",
    "n_args": len(args), "n_pointer_args": len(ptr_args), "n_hidden_args": len(hidden),
    "args": args,
  }
  p = pathlib.Path("bench/qk-tensile-extraction/ffn_gate_up_contract.json"); p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(json.dumps(contract, indent=2))
  print(f"kernarg_segment_size={contract['kernarg_segment_size']} align={contract['kernarg_segment_align']} "
        f"group={contract['group_segment_fixed_size']} vgpr={contract['vgpr_count']} sgpr={contract['sgpr_count']} wf={contract['wavefront_size']}")
  print(f"args={len(args)} (pointers={len(ptr_args)}, hidden={len(hidden)})")
  print("--- arg layout ---")
  for a in args: print(f"  off{a['offset']:>4} sz{a['size']:>2} {a['value_kind']:<22} {a['address_space'] or '':<8} {a['name'] or ''}")

if __name__ == "__main__":
  main()
