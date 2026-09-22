"""Research-only, fail-closed Qwen3-8B pp512 gate/up Q4_K IMMA binding."""
from __future__ import annotations

from dataclasses import dataclass
from tinygrad import Tensor, dtypes
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from extra.llm_research.prefill.nv_q8_compact_producer_gate import SRC as Q8_SOURCE, SRC_FP16 as Q8_SOURCE_FP16
from extra.llm_research.prefill.nv_q4_imma_provider import M, N, K, PARTIAL_SLOTS, Provider, compile_provider, provider_programs
from extra.llm_research.prefill.nv_native_program_uop import call_native, native_nv_program

LEGAL_ROLES = frozenset(("ffn_gate", "ffn_up"))
_BINDINGS: dict[str, "PP512Binding"] = {}


def supports(*, model_family:str, role:str, weight_type:str, m:int, n:int, k:int, device:str) -> bool:
  return (model_family == "qwen3_8b" and role in LEGAL_ROLES and weight_type == "Q4_K" and
          (m, n, k) == (M, N, K) and device == "NV")


@dataclass
class PP512Binding:
  provider: Provider
  producer: object
  producer_fp16: object
  main_program: object
  fixup_program: object
  map_tensor: Tensor
  q8: list[Tensor]
  scales: list[Tensor]
  sums: list[Tensor]
  outputs: list[Tensor]
  partials: Tensor
  ids: Tensor
  partial_epoch: Tensor | None = None
  id_epoch: Tensor | None = None
  cursor: int = 0

  @classmethod
  def compile(cls, dev) -> "PP512Binding":
    provider = compile_provider(dev)
    qlib = NVRTCCompiler(dev.arch, ptx=False, cache_key="q8_pp512_binding_v1").compile(Q8_SOURCE)
    producer = native_nv_program("q8_compact", qlib, global_size=(M,8,1), local_size=(128,1,1),
      globals=(0,1,2,3), outs=(1,2,3), ins=(0,))
    qlib_fp16 = NVRTCCompiler(dev.arch, ptx=False, cache_key="nv_q8_compact_fp16_input_v1").compile(Q8_SOURCE_FP16)
    producer_fp16 = native_nv_program("q8_compact_fp16", qlib_fp16, global_size=(M,8,1), local_size=(128,1,1),
      globals=(0,1,2,3), outs=(1,2,3), ins=(0,))
    main_program, fixup_program = provider_programs(provider)
    map_tensor = Tensor(provider.slotmap, device="NV").contiguous().realize()
    partials = Tensor.empty(PARTIAL_SLOTS*128*128,dtype=dtypes.float32,device="NV").realize()
    ids = Tensor.empty(PARTIAL_SLOTS,dtype=dtypes.int32,device="NV").realize()
    return cls(provider, producer, producer_fp16, main_program, fixup_program, map_tensor,
      [], [], [], [], partials, ids)

  def prepare_outputs(self, count:int) -> None:
    while len(self.outputs) < count:
      self.outputs.append(Tensor.empty(M*N,dtype=dtypes.float32,device="NV").realize())
      self.q8.append(Tensor.empty(M*K,dtype=dtypes.int8,device="NV").realize())
      self.scales.append(Tensor.empty(M*(K//32),dtype=dtypes.float32,device="NV").realize())
      self.sums.append(Tensor.empty(M*(K//32),dtype=dtypes.float32,device="NV").realize())

  def begin_trace(self) -> None:
    # One physical workspace is safe only when every opaque access threads the
    # returned completion epoch. Reset to the raw base once per graph build;
    # TinyJit replay uses the already captured dependency chain.
    self.cursor, self.partial_epoch, self.id_epoch = 0, self.partials, self.ids

  def project(self, x:Tensor, words:Tensor, *, model_family:str, role:str, weight_type:str="Q4_K", wait:bool=False):
    if not supports(model_family=model_family, role=role, weight_type=weight_type,
                    m=x.shape[0], n=N, k=x.shape[1], device=x.device):
      raise ValueError("unsupported Q4 IMMA research route")
    if x.dtype not in (dtypes.float16, dtypes.float32) or words.dtype != dtypes.uint32:
      raise ValueError("Q4 IMMA research route requires fp16/fp32 input and canonical uint32 Q4_K words")
    x = x if x.uop.has_precompiled_output_identity() else x.contiguous()
    words = words.contiguous()
    if self.cursor >= len(self.outputs): self.prepare_outputs(self.cursor+1)
    if self.partial_epoch is None or self.id_epoch is None:
      raise RuntimeError("begin_trace must establish the ordered scratch epoch before projection")
    slot=self.cursor; out=self.outputs[slot]; self.cursor += 1
    producer = self.producer_fp16 if x.dtype == dtypes.float16 else self.producer
    x, q8, scales, sums = x.uop_program(self.q8[slot], self.scales[slot], self.sums[slot], fxn=lambda *_: producer)
    out, partials, ids, words, q8, scales, sums = out.uop_program(
      self.partial_epoch, self.id_epoch, words, q8, scales, sums, fxn=lambda *_: self.main_program)
    out, partials, _ = out.uop_program(partials, self.map_tensor, fxn=lambda *_: self.fixup_program)
    self.partial_epoch, self.id_epoch = partials, ids
    return out.reshape(M, N)



def binding_for(device:str="NV") -> PP512Binding:
  if device != "NV": raise ValueError("Q4 IMMA research binding is NV-only")
  if device not in _BINDINGS:
    from tinygrad import Device
    _BINDINGS[device] = PP512Binding.compile(Device[device])
  return _BINDINGS[device]
