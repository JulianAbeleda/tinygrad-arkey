"""Default-off compiler-owned Qwen3-8B pp512 Q6_K V/down binding.

The route consumes canonical Q6_K halfwords and a scheduler-owned compact-Q8
record.  It admits only the model's Q6 population: 18 attention V and 18 FFN
down projections.  Q4 V/down tensors are deliberately outside the contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Mapping

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.packed_weight import (PackedWeightTransform, Q6KInt8FragmentProvider,
  Q8ActivationRecordTransform, Q8Int8FragmentProvider, Q6KQ8SubgroupAccumulatorContract)
from tinygrad.codegen.opt.postrange import warmstart_key
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.kernel_vocabulary import KernelLDSWindow, KernelTileGeometry
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
from extra.llm_research.prefill.nv_q8_compact_producer_gate import SRC_FP16

M, TILE_K = 512, 64
ROLE_SHAPES = MappingProxyType({"attn_v": (1024, 4096), "ffn_down": (4096, 12288)})
ROLE_COUNTS = MappingProxyType({"attn_v": 18, "ffn_down": 18})
PROJECTIONS_PER_MODEL = sum(ROLE_COUNTS.values())
_BINDINGS: dict[str, "CompilerQ6PP512Binding"] = {}


def supports(*, model_family:str, role:str, weight_type:str, m:int, n:int, k:int, device:str) -> bool:
  return (model_family == "qwen3_8b" and weight_type == "Q6_K" and m == M and
          ROLE_SHAPES.get(role) == (n, k) and device == "NV")


@dataclass(frozen=True)
class _Context:
  schema_version: str
  canonical_identity: str
  geometry: KernelTileGeometry
  packed_weight: PackedWeightTransform
  packed_fragment_provider: Q6KInt8FragmentProvider
  packed_activation: Q8ActivationRecordTransform
  packed_activation_provider: Q8Int8FragmentProvider
  group_accumulator: Q6KQ8SubgroupAccumulatorContract
  pipeline: None = None


def _weight_carrier(halfs:Tensor, transform:PackedWeightTransform) -> Tensor:
  blocks = transform.rows*transform.blocks_per_row
  return halfs.reshape(blocks, 105).pad(((0, 0), (0, 23))).reshape(blocks, 128, 1).expand(blocks, 128, 2) \
    .reshape(transform.rows, transform.k).bitcast(dtypes.float16).cast(dtypes.int8)


def _activation_carrier(record:Tensor, transform:Q8ActivationRecordTransform) -> Tensor:
  return record.bitcast(dtypes.uint16)[:transform.values_bytes//2].reshape(transform.rows, transform.k//2, 1) \
    .expand(transform.rows, transform.k//2, 2).reshape(transform.rows, transform.k).bitcast(dtypes.float16).cast(dtypes.int8)


def _record_source(k:int, name:str) -> str:
  old = ("void q8_compact_fp16(const half* __restrict__ x, signed char* __restrict__ q,\n"
         " float* __restrict__ scales,float* __restrict__ sums) {")
  new = (f"void {name}(const half* __restrict__ x, unsigned int* __restrict__ record) {{\n"
         " signed char* __restrict__ q=(signed char*)record;\n"
         f" float* __restrict__ scales=(float*)(q+{M*k});\n"
         f" float* __restrict__ sums=scales+{M*(k//32)};")
  if old not in SRC_FP16: raise RuntimeError("Q8 producer source ABI changed")
  src = SRC_FP16.replace(old, new)
  if "base=row*4096+i;" not in src or "int g=row*128+seg*16+t/8;" not in src:
    raise RuntimeError("Q8 producer shape spelling changed")
  return src.replace("base=row*4096+i;", f"base=row*{k}+i;") \
            .replace("int g=row*128+seg*16+t/8;", f"int g=row*{k//32}+seg*16+t/8;")


@dataclass(frozen=True)
class _RoleAsset:
  role: str
  producer: object
  main_program: object
  transform: PackedWeightTransform
  activation: Q8ActivationRecordTransform
  context: _Context
  warmstart_key: tuple

  @property
  def candidate_identity(self) -> str: return self.context.canonical_identity


def _compile_role(dev, role:str) -> _RoleAsset:
  n, k = ROLE_SHAPES[role]
  wt, at = PackedWeightTransform("Q6_K", n, k), Q8ActivationRecordTransform(M, k)
  wp, ap = Q6KInt8FragmentProvider(wt), Q8Int8FragmentProvider(at)
  accumulator = Q6KQ8SubgroupAccumulatorContract(wp, ap)
  stride = TILE_K+(TILE_K//16)*4
  geometry = KernelTileGeometry((64, 32, TILE_K), (2, 2), 128, 32,
    (KernelLDSWindow("A", 0, 64*stride, stride), KernelLDSWindow("B", 64*stride, 96*stride, stride)))
  identity = hashlib.sha256(repr((geometry, wp.identity, ap.identity, accumulator.abi)).encode()).hexdigest()
  context = _Context("boltbeam.full_kernel_candidate.v1", identity, geometry, wt, wp, at, ap, accumulator)
  key = warmstart_key({M, n}, k, {wt.storage_dtype, at.storage_dtype})
  name = f"q8_compact_record_fp16_q6_{role}"
  lib = NVRTCCompiler(dev.arch, ptx=False, cache_key=f"{name}_v1").compile(_record_source(k, name))
  producer = native_nv_program(name, lib, global_size=(M, k//512, 1), local_size=(128, 1, 1),
                               globals=(0, 1), outs=(1,), ins=(0,))

  from tinygrad.codegen import to_program_cache
  from tinygrad.codegen.opt.postrange import warmstart_candidate_state
  record_probe = Tensor.empty((M*k + 2*M*(k//32)*4)//4, dtype=dtypes.uint32, device="NV").realize()
  halfs_probe = Tensor.empty(wt.packed_bytes//2, dtype=dtypes.uint16, device="NV").realize()
  with warmstart_candidate_state({key:(Opt(OptOps.TC, 0, (-1, 2, 1)),)}, {key:context}):
    _activation_carrier(record_probe, at).matmul(_weight_carrier(halfs_probe, wt).transpose(), dtype=dtypes.int) \
      .cast(dtypes.float).contiguous().realize()
  matching = [program for program in to_program_cache.values() if program.op is Ops.PROGRAM and program.src and
              getattr(program.src[0].arg, "candidate_context", None) is not None and
              program.src[0].arg.candidate_context.canonical_identity == identity]
  if len(set(matching)) != 1: raise RuntimeError(f"expected one compiler Q6 {role} PROGRAM, found {len(set(matching))}")
  compiled = matching[0]
  main_program = compiled.replace(src=(UOp(Ops.SINK, arg=compiled.src[0].arg), compiled.src[1], UOp(Ops.LINEAR), *compiled.src[3:]))
  expected_global = (n//32, M//64, 1)
  if main_program.arg.outs != (0,) or main_program.arg.ins != (1, 2):
    raise RuntimeError(f"compiler Q6 {role} PROGRAM has unexpected ABI {main_program.arg}")
  if (main_program.arg.global_size, main_program.arg.local_size) != (expected_global, (32, 2, 2)):
    raise RuntimeError(f"compiler Q6 {role} PROGRAM lost qualified geometry: {main_program.arg}")
  return _RoleAsset(role, producer, main_program, wt, at, context, key)


@dataclass(frozen=True)
class CompilerQ6PP512Binding:
  roles: Mapping[str, _RoleAsset]
  warmstart: Mapping
  warmstart_contexts: Mapping

  @classmethod
  def compile(cls, dev) -> "CompilerQ6PP512Binding":
    roles = {role:_compile_role(dev, role) for role in ROLE_SHAPES}
    warmstart = {asset.warmstart_key:(Opt(OptOps.TC, 0, (-1, 2, 1)),) for asset in roles.values()}
    contexts = {asset.warmstart_key:asset.context for asset in roles.values()}
    if len(warmstart) != len(roles): raise RuntimeError("Q6 V/down warmstart keys collide")
    return cls(MappingProxyType(roles), MappingProxyType(warmstart), MappingProxyType(contexts))

  @property
  def candidate_identities(self) -> Mapping[str,str]:
    return MappingProxyType({role:asset.candidate_identity for role,asset in self.roles.items()})

  def install_warmstart(self, model) -> None:
    opts, contexts = dict(model._packed_wmma_warmstart or {}), dict(model._packed_wmma_warmstart_contexts or {})
    for key, value in self.warmstart.items():
      if key in opts and opts[key] != value: raise RuntimeError("compiler Q6 warmstart key collides with another route")
      opts[key] = value
    for key, value in self.warmstart_contexts.items():
      if key in contexts and contexts[key] != value: raise RuntimeError("compiler Q6 context key collides with another route")
      contexts[key] = value
    model._packed_wmma_warmstart, model._packed_wmma_warmstart_contexts = opts, contexts

  def new_capture(self) -> "CompilerQ6PP512Capture": return CompilerQ6PP512Capture(self)

  def prepare_records(self, count:int) -> None:
    if count != PROJECTIONS_PER_MODEL: raise ValueError(f"exact Q6 route requires {PROJECTIONS_PER_MODEL} projections")


@dataclass
class CompilerQ6PP512Capture:
  asset: CompilerQ6PP512Binding
  trace_epoch: int = 0
  cursors: dict[str,int] | None = None

  def prepare_records(self, count:int) -> None: self.asset.prepare_records(count)
  def begin_trace(self) -> None: self.trace_epoch, self.cursors = self.trace_epoch+1, {role:0 for role in ROLE_SHAPES}

  def project(self, x:Tensor, halfs:Tensor, *, model_family:str, role:str, weight_type:str="Q6_K", wait:bool=False) -> Tensor:
    if self.trace_epoch == 0 or self.cursors is None: raise RuntimeError("begin_trace must establish a Q6 capture-local epoch")
    if role not in ROLE_COUNTS or self.cursors[role] >= ROLE_COUNTS[role]:
      raise RuntimeError(f"compiler Q6 trace exceeded exact {role} census")
    self.cursors[role] += 1
    return _project(self.asset, x, halfs, model_family=model_family, role=role, weight_type=weight_type, wait=wait)


def _project(binding:CompilerQ6PP512Binding, x:Tensor, halfs:Tensor, *, model_family:str, role:str,
             weight_type:str="Q6_K", wait:bool=False) -> Tensor:
  del wait
  n, k = ROLE_SHAPES.get(role, (0, 0))
  if not supports(model_family=model_family, role=role, weight_type=weight_type,
                  m=x.shape[0], n=n, k=x.shape[1], device=x.device):
    raise ValueError("unsupported compiler Q6 V/down research route")
  if x.dtype != dtypes.float16 or halfs.dtype != dtypes.uint16:
    raise ValueError("compiler Q6 route requires fp16 activation and canonical uint16 Q6_K halfwords")
  asset = binding.roles[role]
  record_u32 = (M*k + 2*M*(k//32)*4)//4
  record = Tensor.empty(record_u32, dtype=dtypes.uint32, device=x.device)
  _, record = x.uop_program(record, fxn=lambda *_:asset.producer)
  # FFN-down is immediately consumed by the block residual.  Leaving it as an
  # ordinary lazy matmul lets that epilogue rewrite the admitted SINK and lose
  # the exact paired-IMMA context.  Pin the already compiler-produced PROGRAM
  # at this boundary; both buffers remain lazy graph-owned allocations.
  if role == "ffn_down":
    out = Tensor.empty(M*n, dtype=dtypes.float32, device=x.device)
    out, record, halfs = out.uop_program(record, halfs, fxn=lambda *_:asset.main_program)
    return out.reshape(M, n)
  return _activation_carrier(record, asset.activation).matmul(
    _weight_carrier(halfs, asset.transform).transpose(), dtype=dtypes.int).cast(dtypes.float).contiguous()


def binding_for(device:str="NV") -> CompilerQ6PP512Binding:
  if device != "NV": raise ValueError("compiler Q6 research binding is NV-only")
  if device not in _BINDINGS: _BINDINGS[device] = CompilerQ6PP512Binding.compile(Device[device])
  return _BINDINGS[device]
