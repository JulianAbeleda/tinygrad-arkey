#!/usr/bin/env python3
"""P1 loader-only smoke for the funded decode MMVQ project.

Loads selected llama.cpp MMVQ kernel descriptors from the gfx1100 AMDGPU
object through tinygrad HCQ. It does not launch kernels.
"""
from __future__ import annotations

import ctypes, json, pathlib, struct, weakref
from typing import Any

from tinygrad import Device
from tinygrad.device import BufferSpec
from tinygrad.helpers import round_up
from tinygrad.runtime.autogen import amdgpu_kd, hsa
from tinygrad.runtime.ops_amd import AMDProgram
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.hcq import HCQArgsState, HCQProgram

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-mmvq-large-project"
OBJ = pathlib.Path("/home/ubuntu/env/llama.cpp/build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/mmvq.cu.o.0.hipv4-amdgcn-amd-amdhsa--gfx1100")
INV = OUT / "contract_inventory.json"


def kd_offset(elf: bytes, symbol: str) -> int:
  target = (symbol + ".kd").encode()
  e_shoff = struct.unpack_from("<Q", elf, 0x28)[0]
  shent = struct.unpack_from("<H", elf, 0x3a)[0]
  shnum = struct.unpack_from("<H", elf, 0x3c)[0]
  for s in range(shnum):
    sh = e_shoff + s * shent
    sht = struct.unpack_from("<I", elf, sh + 4)[0]
    if sht not in (2, 11):
      continue
    off = struct.unpack_from("<Q", elf, sh + 0x18)[0]
    size = struct.unpack_from("<Q", elf, sh + 0x20)[0]
    link = struct.unpack_from("<I", elf, sh + 0x28)[0]
    esz = struct.unpack_from("<Q", elf, sh + 0x38)[0]
    strsh = e_shoff + link * shent
    stroff = struct.unpack_from("<Q", elf, strsh + 0x18)[0]
    for i in range(0, size, esz):
      e = off + i
      st_name = struct.unpack_from("<I", elf, e)[0]
      st_value = struct.unpack_from("<Q", elf, e + 8)[0]
      name = elf[stroff + st_name:elf.index(b"\x00", stroff + st_name)]
      if name == target:
        return st_value
  raise ValueError(f"{symbol}.kd not found")


class LoadOnlyNamedAMDProgram(AMDProgram):
  def __init__(self, dev: Any, name: str, lib: bytes, kd_off: int, metadata_kernarg_size: int):
    self.dev, self.name, self.lib = dev, name, lib
    image, sections, relocs = elf_loader(self.lib)
    for ao, rso, typ, addend in relocs:
      if typ == 5:
        image[ao:ao + 8] = struct.pack("<q", rso - ao + addend)
      else:
        raise RuntimeError(f"unknown AMD reloc {typ}")
    self.lib_gpu = self.dev.allocator.alloc(round_up(image.nbytes, 0x1000), bs := BufferSpec(nolru=True))
    self.dev.allocator._copyin(self.lib_gpu, image)
    self.dev.synchronize()

    dsz = ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)
    desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[kd_off:kd_off + dsz]))
    self.group_segment_size = desc.group_segment_fixed_size
    self.private_segment_size = desc.private_segment_fixed_size
    self.kernargs_segment_size = desc.kernarg_size
    self.dev._ensure_has_local_memory(self.private_segment_size)
    lds_size = ((self.group_segment_size + 511) // 512) & 0x1FF
    self.wave32 = desc.kernel_code_properties & 0x400 == 0x400
    self.rsrc1 = desc.compute_pgm_rsrc1 | ((1 << 20) if self.dev.target[0] == 11 else 0)
    self.rsrc2 = desc.compute_pgm_rsrc2 | (lds_size << 15)
    self.rsrc3 = desc.compute_pgm_rsrc3
    self.aql_prog_addr = self.lib_gpu.va_addr + kd_off
    self.prog_addr = self.lib_gpu.va_addr + kd_off + desc.kernel_code_entry_byte_offset
    self.enable_dispatch_ptr = desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR
    self.enable_private_segment_sgpr = desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER
    add = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t) if self.enable_dispatch_ptr else 0
    alloc = max(self.kernargs_segment_size, metadata_kernarg_size) + add
    HCQProgram.__init__(self, HCQArgsState, self.dev, self.name, kernargs_alloc_size=alloc, lib=self.lib, base=self.lib_gpu.va_addr)
    weakref.finalize(self, self._fini, self.dev, self.lib_gpu, bs)
    self.p1_contract = {
      "kd_offset": kd_off,
      "desc_kernarg_size": self.kernargs_segment_size,
      "metadata_kernarg_size": metadata_kernarg_size,
      "kernargs_alloc_size": self.kernargs_alloc_size,
      "group_segment_size": self.group_segment_size,
      "private_segment_size": self.private_segment_size,
      "wave32": self.wave32,
      "rsrc1": self.rsrc1,
      "rsrc2": self.rsrc2,
      "rsrc3": self.rsrc3,
      "aql_prog_addr": self.aql_prog_addr,
      "prog_addr": self.prog_addr,
      "enable_dispatch_ptr": bool(self.enable_dispatch_ptr),
      "enable_private_segment_sgpr": bool(self.enable_private_segment_sgpr),
    }


def select_targets(inv: dict[str, Any]) -> list[dict[str, Any]]:
  targets = []
  for type_name in ("Q4_K", "Q6_K"):
    cands = [
      c for c in inv["candidates"]
      if c["type_name"] == type_name and c["ncols_dst"] == 1 and not c["template_bool_0"] and not c["template_bool_1"]
    ]
    if not cands:
      raise RuntimeError(f"no low-VGPR ncols=1 target for {type_name}")
    targets.append(sorted(cands, key=lambda c: c["metadata"]["vgpr_count"])[0])
  return targets


def main() -> None:
  if Device.DEFAULT != "AMD":
    raise RuntimeError(f"P1 requires DEV=AMD, got {Device.DEFAULT!r}")
  OUT.mkdir(parents=True, exist_ok=True)
  inv = json.loads(INV.read_text())
  elf = OBJ.read_bytes()
  dev = Device[Device.DEFAULT]
  rows = []
  errors = []
  for tgt in select_targets(inv):
    try:
      kd = kd_offset(elf, tgt["name"])
      prg = LoadOnlyNamedAMDProgram(dev, f"llama_mmvq_p1_{tgt['type_name'].lower()}", elf, kd, tgt["metadata"]["kernarg_segment_size"])
      rows.append({
        "type_name": tgt["type_name"],
        "ncols_dst": tgt["ncols_dst"],
        "template_bool_0": tgt["template_bool_0"],
        "template_bool_1": tgt["template_bool_1"],
        "symbol": tgt["name"],
        "metadata": tgt["metadata"],
        "load_contract": prg.p1_contract,
        "loaded": True,
      })
    except Exception as exc:  # P1 is a smoke: report all loader errors.
      errors.append({"type_name": tgt["type_name"], "symbol": tgt["name"], "error": repr(exc)})
  result = {
    "schema": "decode_mmvq_large_project_p1_loader_smoke_v1",
    "date": "2026-06-19",
    "phase": "P1_loader_smoke",
    "object": str(OBJ),
    "targets": rows,
    "errors": errors,
    "no_kernel_launch": True,
    "no_model_route_change": True,
    "no_in_process_hip_runtime": True,
    "verdict": "PASS" if len(rows) == 2 and not errors else "KILL",
    "next": "P2 kernarg and launch capture" if len(rows) == 2 and not errors else "Stop source/object import or fix loader boundary",
  }
  (OUT / "p1_loader_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
  summary = [
    "# Decode MMVQ large project P1 loader smoke",
    "",
    f"- verdict: `{result['verdict']}`",
    f"- loaded targets: `{len(rows)}`",
    f"- errors: `{len(errors)}`",
    f"- no kernel launch: `{result['no_kernel_launch']}`",
    "",
  ]
  for row in rows:
    lc = row["load_contract"]
    summary.append(
      f"- `{row['type_name']}` descriptor loaded: kd `{hex(lc['kd_offset'])}`, "
      f"kernarg alloc `{lc['kernargs_alloc_size']}`, wave32 `{lc['wave32']}`, "
      f"rsrc `({lc['rsrc1']},{lc['rsrc2']},{lc['rsrc3']})`"
    )
  for err in errors:
    summary.append(f"- `{err['type_name']}` error: `{err['error']}`")
  summary.append("")
  (OUT / "p1_loader_smoke_summary.md").write_text("\n".join(summary))
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
