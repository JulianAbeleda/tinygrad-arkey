#!/usr/bin/env python3
"""P3 standalone correctness for imported llama Q4_K MMVQ through tinygrad HCQ."""
from __future__ import annotations

import ctypes, json, pathlib, struct, weakref

import numpy as np

from tinygrad import Tensor, Device, dtypes
from tinygrad.device import BufferSpec
from tinygrad.helpers import round_up
from tinygrad.runtime.autogen import amdgpu_kd, hsa
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.hcq import HCQArgsState, HCQProgram
from tinygrad.runtime.ops_amd import AMDProgram

from extra.q8_ffn_handwritten_oracle import q4_ref_rows, q8_blocks
from extra.qk_layout import GGML_Q4_K, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, read_metadata, tensor_shape
from extra.qk_paths import DEFAULT_MODEL_GGUF

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-mmvq-large-project"
OBJ = pathlib.Path("/home/ubuntu/env/llama.cpp/build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/mmvq.cu.o.0.hipv4-amdgcn-amd-amdhsa--gfx1100")
MODEL = pathlib.Path(DEFAULT_MODEL_GGUF)


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


class RawKernargAMDProgram(AMDProgram):
  def __init__(self, dev, name: str, lib: bytes, kd_off: int, raw: bytes):
    self.dev, self.name, self.lib, self._raw = dev, name, lib, bytes(raw)
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
    HCQProgram.__init__(self, HCQArgsState, self.dev, self.name, kernargs_alloc_size=max(self.kernargs_segment_size, len(self._raw)) + add,
                        lib=self.lib, base=self.lib_gpu.va_addr)
    weakref.finalize(self, self._fini, self.dev, self.lib_gpu, bs)

  def fill_kernargs(self, bufs, vals=(), kernargs=None):
    ab = kernargs or self.dev.kernargs_buf.offset(offset=self.dev.kernargs_offset_allocator.alloc(self.kernargs_alloc_size, 8),
                                                  size=self.kernargs_alloc_size)
    ab.cpu_view().view(size=len(self._raw), fmt="B")[:] = bytearray(self._raw)
    return HCQArgsState(ab, self, tuple(bufs), vals=tuple(vals))


def q4_tensor_bytes(name: str) -> tuple[bytes, int, int]:
  meta = read_metadata(MODEL)
  info = next(i for i in meta.infos if i.name == name)
  if info.typ != GGML_Q4_K:
    raise ValueError(f"{name} is not Q4_K")
  rows, k = tensor_shape(info)
  start = meta.data_start + info.off
  nbytes = rows * (k // Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES
  with MODEL.open("rb") as f:
    f.seek(start)
    return f.read(nbytes), rows, k


def main() -> None:
  if Device.DEFAULT != "AMD":
    raise RuntimeError(f"P3 requires DEV=AMD, got {Device.DEFAULT!r}")
  OUT.mkdir(parents=True, exist_ok=True)
  cap_all = json.loads((OUT / "p2_kernarg_capture.json").read_text())
  cap = cap_all["selected"]["q4_attn_q_or_o"]
  raw = bytearray(cap["kernarg_bytes"])
  q4_name = "blk.0.attn_output.weight"
  q4, rows, k = q4_tensor_bytes(q4_name)
  Tensor.manual_seed(3)
  x = Tensor.randn(k, dtype=dtypes.float32).numpy().astype(np.float32)
  q8 = q8_blocks(x)
  q8_deq = np.frombuffer(q8, dtype=np.uint8)  # only for artifact sizing; ref uses q8_dequant below
  # Reconstruct the dequantized q8 values from the packed llama block_q8_1 bytes.
  vals = []
  for off in range(0, len(q8), 36):
    d = np.frombuffer(q8[off:off + 2], dtype=np.float16).astype(np.float32)[0]
    vals.append(np.frombuffer(q8[off + 4:off + 36], dtype=np.int8).astype(np.float32) * d)
  xq = np.concatenate(vals).astype(np.float32)
  ref = q4_ref_rows(q4, rows, k, xq)

  q4_t = Tensor(np.frombuffer(q4, dtype=np.uint32).copy(), dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8_t = Tensor(np.frombuffer(q8, dtype=np.uint8).copy(), dtype=dtypes.uint8, device="AMD").contiguous().realize()
  out_t = Tensor.zeros(rows, dtype=dtypes.float32, device="AMD").contiguous().realize()
  Device["AMD"].synchronize()

  va = lambda t: t.uop.buffer._buf.va_addr
  struct.pack_into("<Q", raw, 0, va(q4_t))
  struct.pack_into("<Q", raw, 8, va(q8_t))
  struct.pack_into("<Q", raw, 16, 0)
  struct.pack_into("<Q", raw, 56, va(out_t))

  elf = OBJ.read_bytes()
  prg = RawKernargAMDProgram(Device["AMD"], "llama_mmvq_q4_p3", elf, kd_offset(elf, cap["kernel_symbol"]), bytes(raw))
  times = []
  for _ in range(8):
    times.append(prg(q4_t.uop.buffer._buf, q8_t.uop.buffer._buf, out_t.uop.buffer._buf,
                     global_size=tuple(cap["num_workgroups"]), local_size=tuple(cap["local"]), wait=True, timeout=10000))
  got = out_t.numpy()
  diff = np.abs(got - ref)
  result = {
    "schema": "decode_mmvq_large_project_p3_q4_correctness_v1",
    "date": "2026-06-19",
    "phase": "P3_Q4_standalone_correctness",
    "tensor": q4_name,
    "rows": rows,
    "k": k,
    "q4_bytes": len(q4),
    "q8_bytes": int(q8_deq.size),
    "kernel_symbol": cap["kernel_symbol"],
    "launch": {"num_workgroups": cap["num_workgroups"], "local": cap["local"]},
    "times_ms": times,
    "median_ms": float(np.median(times)),
    "max_abs": float(diff.max()),
    "mean_abs": float(diff.mean()),
    "max_rel": float((diff / np.maximum(np.abs(ref), 1e-6)).max()),
    "correct": bool(diff.max() < 2e-2),
    "verdict": "PASS" if diff.max() < 2e-2 else "KILL",
  }
  (OUT / "p3_q4_correctness.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
