"""Default-off compiler-owned Qwen3-8B pp512 Q4_K/Q8_1 IMMA binding.

Unlike ``nv_q4_imma_pp512_binding``, this route has no opaque main/fixup
program and no Stream-K partial workspace.  The only native sidecar is the
qualified, exact Q8 producer.  Its packed record and the canonical Q4_K model
parameter are movement-only carriers into an ordinary tinygrad matmul; Gate A
then lowers that matmul to one compiler-owned direct-output IMMA PROGRAM.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from types import MappingProxyType
from typing import Mapping

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.packed_weight import (PackedWeightTransform, Q4KInt8FragmentProvider,
  Q8ActivationRecordTransform, Q8Int8FragmentProvider, Q4KQ8GroupAccumulatorContract)
from tinygrad.codegen.opt.postrange import warmstart_key
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.kernel_vocabulary import KernelLDSWindow, KernelTileGeometry
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
from extra.llm_research.prefill.nv_q8_compact_producer_gate import SRC_FP16
from extra.llm_research.prefill.nv_tile_major_q8_1_record import TileMajorQ8ActivationRecordTransform, TileMajorActivationCarrierSpec, tile_major_q8_carrier

M, N, K, TILE_K = 512, 12288, 4096, 64
@dataclass(frozen=True)
class CompilerQ4ScheduleConfig:
  tile_m:int=128; tile_n:int=128; tile_k:int=64; warp_m:int=2; warp_n:int=4; threads:int=256
  def validate(self):
    if self.tile_m<=0 or self.tile_n<=0 or self.tile_k<=0 or self.warp_m*self.warp_n!=8 or self.threads!=256 or M%self.tile_m or N%self.tile_n or K%self.tile_k: raise ValueError("invalid Q4 schedule")
DEFAULT_SCHEDULE=CompilerQ4ScheduleConfig()
PROJECTIONS_PER_MODEL = 72
LEGAL_ROLES = frozenset(("ffn_gate", "ffn_up"))
RECORD_BYTES = M*K + 2*M*(K//32)*4
RECORD_U32 = RECORD_BYTES//4
_BINDINGS: dict[str, "CompilerPP512Binding"] = {}
_UNROLL_LOOP = "  for (int Ridx0 = 0; Ridx0 < 64; Ridx0++) {"


def supports(*, model_family:str, role:str, weight_type:str, m:int, n:int, k:int, device:str) -> bool:
  return (model_family == "qwen3_8b" and role in LEGAL_ROLES and weight_type == "Q4_K" and
          (m, n, k) == (M, N, K) and device == "NV")


@dataclass(frozen=True)
class _Context:
  schema_version: str
  canonical_identity: str
  geometry: KernelTileGeometry
  packed_weight: PackedWeightTransform
  packed_fragment_provider: Q4KInt8FragmentProvider
  packed_activation: Q8ActivationRecordTransform
  packed_activation_provider: Q8Int8FragmentProvider
  group_accumulator: Q4KQ8GroupAccumulatorContract
  pipeline: None = None


def _weight_carrier(words:Tensor, transform:PackedWeightTransform) -> Tensor:
  blocks, halfwords = transform.rows*transform.blocks_per_row, int(transform.block_bytes)//2
  return words.bitcast(dtypes.uint16).reshape(blocks, halfwords).pad(((0, 0), (0, 128-halfwords))) \
    .reshape(blocks, 128, 1).expand(blocks, 128, 2).reshape(transform.rows, transform.k).bitcast(dtypes.float16).cast(dtypes.int8)


def _activation_carrier(record:Tensor, transform:Q8ActivationRecordTransform) -> Tensor:
  return record.bitcast(dtypes.uint16)[:transform.values_bytes//2].reshape(transform.rows, transform.k//2) \
    .reshape(transform.rows, transform.k//2, 1).expand(transform.rows, transform.k//2, 2) \
    .reshape(transform.rows, transform.k).bitcast(dtypes.float16).cast(dtypes.int8)


def _record_source() -> str:
  old = ("void q8_compact_fp16(const half* __restrict__ x, signed char* __restrict__ q,\n"
         " float* __restrict__ scales,float* __restrict__ sums) {")
  new = ("void q8_compact_record_fp16(const half* __restrict__ x, unsigned int* __restrict__ record) {\n"
         f" signed char* __restrict__ q=(signed char*)record;\n"
         f" float* __restrict__ scales=(float*)(q+{M*K});\n"
         f" float* __restrict__ sums=scales+{M*(K//32)};")
  if old not in SRC_FP16: raise RuntimeError("Q8 producer source ABI changed")
  return SRC_FP16.replace(old, new)


def _research_unroll_program(program:UOp, dev) -> UOp:
  """Apply the counter-selected source scheduling discriminator.

  This is nested behind the already-default-off compiler route and accepts one
  exact factor.  Source and binary are replaced together; ProgramInfo, launch
  ABI, candidate context, and symbol remain compiler-owned.
  """
  raw = os.environ.get("NV_COMPILER_Q4_IMMA_UNROLL")
  if raw is None: return program
  if raw != "4": raise RuntimeError("NV_COMPILER_Q4_IMMA_UNROLL only admits the qualified factor 4")
  sources = [u for u in program.src if u.op is Ops.SOURCE and isinstance(u.arg, str)]
  binaries = [u for u in program.src if u.op is Ops.BINARY and isinstance(u.arg, bytes)]
  if len(sources) != 1 or len(binaries) != 1 or sources[0].arg.count(_UNROLL_LOOP) != 1:
    raise RuntimeError("compiler Q4 main source no longer has the exact unroll discriminator ABI")
  source = sources[0].arg.replace(_UNROLL_LOOP, f"  #pragma unroll {raw}\n{_UNROLL_LOOP}")
  binary = NVRTCCompiler(dev.arch, ptx=False, cache_key="nv_compiler_q4_gateup_k64_unroll4_v1").compile(source)
  return program.replace(src=tuple(u.replace(arg=source) if u is sources[0] else
    u.replace(arg=binary) if u is binaries[0] else u for u in program.src))


@dataclass(frozen=True)
class CompilerPP512Binding:
  producer: object
  main_program: object
  transform: PackedWeightTransform
  activation: Q8ActivationRecordTransform
  context: _Context
  warmstart: Mapping
  warmstart_contexts: Mapping
  @classmethod
  def compile(cls, dev, config:CompilerQ4ScheduleConfig=DEFAULT_SCHEDULE, *, compact_q8:bool=False) -> "CompilerPP512Binding":
    config.validate()
    wt = PackedWeightTransform("Q4_K", N, K)
    at = TileMajorQ8ActivationRecordTransform(M, K) if compact_q8 else Q8ActivationRecordTransform(M, K)
    wp, ap = Q4KInt8FragmentProvider(wt), Q8Int8FragmentProvider(at)
    accum = Q4KQ8GroupAccumulatorContract(wp, ap)
    stride = 80
    a_end=config.tile_m*stride; b_end=(config.tile_m+config.tile_n)*stride
    if b_end > 256*stride: raise ValueError("Q4 schedule exceeds shared LDS window budget")
    geometry = KernelTileGeometry((config.tile_m, config.tile_n, config.tile_k), (config.warp_m, config.warp_n), config.threads, 32,
      (KernelLDSWindow("A", 0, a_end, stride), KernelLDSWindow("B", a_end, b_end, stride)))
    identity = hashlib.sha256(repr(("compact_q8" if compact_q8 else "flat_q8", config, geometry, wp.identity, ap.identity, accum.abi)).encode()).hexdigest()
    context = _Context("boltbeam.full_kernel_candidate.v1", identity, geometry, wt, wp, at, ap, accum)
    key = warmstart_key({M, N}, K, wt.storage_dtype)
    lib = NVRTCCompiler(dev.arch, ptx=False, cache_key="nv_q8_compact_record_fp16_v1").compile(_record_source())
    producer = native_nv_program("q8_compact_record_fp16", lib, global_size=(M, 8, 1), local_size=(128, 1, 1),
                                 globals=(0, 1), outs=(1,), ins=(0,))
    warmstart, warmstart_contexts = {key:(Opt(OptOps.TC, 0, (-1, 2, 1)),)}, {key:context}

    # Compile the ordinary carrier matmul once, then retain its compiler-owned
    # PROGRAM as the reusable model asset.  This is not the old handwritten
    # provider: source, binary, launch ABI, tensor-core ledger, and candidate
    # identity all come from the normal tinygrad compiler.
    from tinygrad.codegen import to_program_cache
    from tinygrad.codegen.opt.postrange import warmstart_candidate_state
    record_probe = Tensor.empty(at.storage_units, dtype=dtypes.uint32, device="NV").realize()
    words_probe = Tensor.empty(wt.packed_bytes//4, dtype=dtypes.uint32, device="NV").realize()
    with warmstart_candidate_state(warmstart, warmstart_contexts):
      activation_probe = tile_major_q8_carrier(record_probe, TileMajorActivationCarrierSpec(at)) if compact_q8 else record_probe
      activation_probe = activation_probe if compact_q8 else _activation_carrier(activation_probe, at)
      activation_probe.matmul(_weight_carrier(words_probe, wt).transpose(), dtype=dtypes.int).cast(dtypes.float).contiguous().realize()
    matching = [program for program in to_program_cache.values() if program.op is Ops.PROGRAM and program.src and
                getattr(program.src[0].arg, "candidate_context", None) is not None and
                program.src[0].arg.candidate_context.canonical_identity == identity]
    if len(set(matching)) != 1: raise RuntimeError(f"expected one compiler-generated Q4_K PROGRAM, found {len(set(matching))}")
    compiled_program = _research_unroll_program(matching[0], dev)
    # A cached compiler PROGRAM retains its lowered SINK for diagnostics.  A
    # reusable invocation must be opaque to the next scheduling pass, while
    # retaining the exact compiler-emitted ProgramInfo/source/binary and the
    # candidate-bearing KernelInfo identity.
    main_program = compiled_program.replace(src=(UOp(Ops.SINK, arg=compiled_program.src[0].arg), compiled_program.src[1],
      UOp(Ops.LINEAR), *compiled_program.src[3:]))
    if main_program.arg.outs != (0,) or main_program.arg.ins != (1, 2):
      raise RuntimeError(f"compiler-generated Q4_K PROGRAM has unexpected ABI {main_program.arg}")
    return cls(producer, main_program, wt, at, context,
               MappingProxyType(warmstart), MappingProxyType(warmstart_contexts))

  @property
  def candidate_identity(self) -> str: return self.context.canonical_identity

  def install_warmstart(self, model) -> None:
    """Add only this exact K64 row to the model's normal compiler scope."""
    opts, contexts = dict(model._packed_wmma_warmstart or {}), dict(model._packed_wmma_warmstart_contexts or {})
    for key, value in self.warmstart.items():
      if key in opts and opts[key] != value: raise RuntimeError("compiler Q4_K warmstart key collides with another route")
      opts[key] = value
    for key, value in self.warmstart_contexts.items():
      if key in contexts and contexts[key] != value: raise RuntimeError("compiler Q4_K context key collides with another route")
      contexts[key] = value
    model._packed_wmma_warmstart, model._packed_wmma_warmstart_contexts = opts, contexts

  def new_capture(self) -> "CompilerPP512Capture": return CompilerPP512Capture(self)

  # Compatibility for the retained one-process research gate. These methods
  # no longer reserve device buffers or mutate the device-global compiled
  # asset; production model attachment uses ``new_capture`` below.
  def prepare_records(self, count:int) -> None:
    if count not in (PROJECTIONS_PER_MODEL, PROJECTIONS_PER_MODEL//2):
      raise ValueError(f"exact route requires {PROJECTIONS_PER_MODEL} projections (or isolated gate-only {PROJECTIONS_PER_MODEL//2})")

  @property
  def records(self) -> range: return range(PROJECTIONS_PER_MODEL)

  @property
  def outputs(self) -> range: return range(PROJECTIONS_PER_MODEL)

  def begin_trace(self) -> None: pass

  def project(self, x:Tensor, words:Tensor, *, model_family:str, role:str, weight_type:str="Q4_K", wait:bool=False) -> Tensor:
    return _project(self, x, words, model_family=model_family, role=role, weight_type=weight_type, wait=wait)


@dataclass
class CompilerPP512Capture:
  """Per-model/per-JIT trace identity for one immutable compiler asset.

  Projection buffers are deliberately not retained here.  Each graph build
  creates lazy BUFFER UOps and transfers their ownership to the scheduler;
  captured graphs retain the resulting allocations themselves.  Therefore a
  second model or TinyJit cannot reset a cursor and silently alias the first
  graph's mutable record/output storage.
  """
  asset: CompilerPP512Binding
  trace_epoch: int = 0
  cursor: int = 0

  @property
  def candidate_identity(self) -> str: return self.asset.candidate_identity

  @property
  def transform(self) -> PackedWeightTransform: return self.asset.transform

  @property
  def producer(self): return self.asset.producer

  def prepare_records(self, count:int) -> None: self.asset.prepare_records(count)

  @property
  def records(self) -> range: return range(self.cursor)

  @property
  def outputs(self) -> range: return range(self.cursor)

  def begin_trace(self) -> None:
    self.trace_epoch, self.cursor = self.trace_epoch+1, 0

  def project(self, x:Tensor, words:Tensor, *, model_family:str, role:str, weight_type:str="Q4_K", wait:bool=False) -> Tensor:
    if self.trace_epoch == 0: raise RuntimeError("begin_trace must establish a capture-local epoch before projection")
    if self.cursor >= PROJECTIONS_PER_MODEL: raise RuntimeError("compiler Q4 IMMA trace exceeded exact 72-projection census")
    self.cursor += 1
    return _project(self.asset, x, words, model_family=model_family, role=role, weight_type=weight_type, wait=wait)

  def project_pair(self, x:Tensor, gate_words:Tensor, up_words:Tensor, *, model_family:str,
                   weight_type:str="Q4_K") -> tuple[Tensor, Tensor]:
    return (self.project(x, gate_words, model_family=model_family, role="ffn_gate", weight_type=weight_type),
            self.project(x, up_words, model_family=model_family, role="ffn_up", weight_type=weight_type))

  def project_from_record(self, record:Tensor, words:Tensor, *, model_family:str,
                          role:str, weight_type:str="Q4_K") -> Tensor:
    """Research-only main entry for an externally produced canonical Q8 record."""
    if self.trace_epoch == 0: raise RuntimeError("begin_trace must establish a capture-local epoch before projection")
    if self.cursor >= PROJECTIONS_PER_MODEL: raise RuntimeError("compiler Q4 IMMA trace exceeded exact census")
    if record.dtype != dtypes.uint32 or words.dtype != dtypes.uint32:
      raise ValueError("external record route requires uint32 record and Q4 words")
    if role not in ("ffn_gate", "ffn_up"): raise ValueError("unsupported role")
    self.cursor += 1
    out = Tensor.empty(M*N, dtype=dtypes.float32, device=words.device)
    out, record, words = out.uop_program(record, words, fxn=lambda *_: self.asset.main_program)
    return out.reshape(M, N)


def _project(binding:CompilerPP512Binding, x:Tensor, words:Tensor, *, model_family:str, role:str,
             weight_type:str="Q4_K", wait:bool=False) -> Tensor:
    del wait
    if not supports(model_family=model_family, role=role, weight_type=weight_type,
                    m=x.shape[0], n=N, k=x.shape[1], device=x.device):
      raise ValueError("unsupported compiler Q4 IMMA research route")
    if x.dtype != dtypes.float16 or words.dtype != dtypes.uint32:
      raise ValueError("compiler Q4 IMMA route requires fp16 activation and canonical uint32 Q4_K words")
    # Lazy, graph-owned buffers are the key ownership difference from the old
    # 72-entry device-global pool.  They remain unallocated until scheduling,
    # allowing the normal memory planner to reason about their lifetimes.
    record = Tensor.empty(RECORD_U32, dtype=dtypes.uint32, device=x.device)
    out = Tensor.empty(M*N, dtype=dtypes.float32, device=x.device)
    # B1 preserves an enclosing FUNCTION-owned x here.  The returned AFTER on
    # record is the sole producer dependency consumed by the generated main.
    _, record = x.uop_program(record, fxn=lambda *_: binding.producer)
    out, record, words = out.uop_program(record, words, fxn=lambda *_: binding.main_program)
    return out.reshape(M, N)


def binding_for(device:str="NV") -> CompilerPP512Binding:
  if device != "NV": raise ValueError("compiler Q4 IMMA research binding is NV-only")
  if device not in _BINDINGS: _BINDINGS[device] = CompilerPP512Binding.compile(Device[device])
  return _BINDINGS[device]
