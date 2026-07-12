#!/usr/bin/env python3
"""LDS-staged WMMA primitive spec for ffn_gate_up.

This is the declarative contract for reducing the current raw LDS oracle into a
compiler-owned primitive. It may describe and validate the existing schedule, but
it must not lower through extra.qk.prefill.wmma or return route-local instruction
lists. The generated implementation must reuse the existing LDS/DBUF codegen and
AMD renderer substrate instead of creating a second LDS lowerer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


LDS2_OWNERSHIP_CLASSIFICATION = "compiler_primitive_spec_owned__asm_backend_atom"
LDS2_DEFAULT_SELECTION_LABEL = "S9_COMPLETE_KEEP_OPT_IN"


@dataclass(frozen=True)
class LDS2RegLayout:
  accumulator: str = "wmma_accum_wm_x_wn_8_vgprs"
  cooperative_load: str = "a_b_b128_temp_vectors"
  preload: str = "plrab_a_b_when_k_substeps_2"

  def to_json(self) -> dict[str, Any]:
    return {"accumulator": self.accumulator, "cooperative_load": self.cooperative_load, "preload": self.preload}

  @classmethod
  def from_json(cls, data: dict[str, Any]) -> "LDS2RegLayout":
    return cls(accumulator=data.get("accumulator", cls.accumulator),
               cooperative_load=data.get("cooperative_load", cls.cooperative_load),
               preload=data.get("preload", cls.preload))


@dataclass(frozen=True)
class LDS2MemoryLayout:
  operand_a: str = "global_row_major_fp16_to_lds"
  operand_b: str = "global_row_major_bt_fp16_to_lds"
  lds_store: str = "packed_ds_store_b128"
  lds_load: str = "typed_ds_load_b128_same_slot"

  def to_json(self) -> dict[str, Any]:
    return {"operand_a": self.operand_a, "operand_b": self.operand_b, "lds_store": self.lds_store, "lds_load": self.lds_load}

  @classmethod
  def from_json(cls, data: dict[str, Any]) -> "LDS2MemoryLayout":
    return cls(operand_a=data.get("operand_a", cls.operand_a), operand_b=data.get("operand_b", cls.operand_b),
               lds_store=data.get("lds_store", cls.lds_store), lds_load=data.get("lds_load", cls.lds_load))


@dataclass(frozen=True)
class LDS2WaitPolicy:
  name: str = "vmem_to_lds_then_lgkm_to_wmma"
  waitcnt_policy: str = "targeted_vmcnt"

  def to_json(self) -> dict[str, Any]:
    return {"name": self.name, "waitcnt_policy": self.waitcnt_policy}

  @classmethod
  def from_json(cls, data: dict[str, Any]) -> "LDS2WaitPolicy":
    return cls(name=data.get("name", cls.name), waitcnt_policy=data.get("waitcnt_policy", cls.waitcnt_policy))


@dataclass(frozen=True)
class LDS2Cadence:
  buffers: int = 2
  stage_order: str = "load_store_wait_wmma"
  dbuf_cadence: str = "s9_opt_in_not_promoted"

  def to_json(self) -> dict[str, Any]:
    return {"buffers": self.buffers, "stage_order": self.stage_order, "dbuf_cadence": self.dbuf_cadence}

  @classmethod
  def from_json(cls, data: dict[str, Any]) -> "LDS2Cadence":
    return cls(buffers=data.get("buffers", cls.buffers), stage_order=data.get("stage_order", cls.stage_order),
               dbuf_cadence=data.get("dbuf_cadence", cls.dbuf_cadence))


@dataclass(frozen=True)
class LDS2LifecycleTemplate:
  name: str = "s9_lds2_ffn_gate_up"
  stages: tuple[str, ...] = ("global_load", "lds_store", "wait_lgkm", "wmma", "advance_k")
  backend_atom: str = "asm_backend_atom"

  def to_json(self) -> dict[str, Any]:
    return {"name": self.name, "stages": list(self.stages), "backend_atom": self.backend_atom}

  @classmethod
  def from_json(cls, data: dict[str, Any]) -> "LDS2LifecycleTemplate":
    return cls(name=data.get("name", cls.name), stages=tuple(data.get("stages", cls.stages)),
               backend_atom=data.get("backend_atom", cls.backend_atom))


@dataclass(frozen=True)
class DBUFEpochPrimitive:
  """Narrow hand-coded DBUF epoch coordinator for the S9/S10 hybrid route.

  This is the intentionally hard part that remains inside the backend atom:
  warm one slot, consume the current epoch, produce the next epoch into the
  alternate slot, and drain the final slot. Everything around it can be owned by
  schedule/spec/search without pretending the DBUF epoch choreography is pure
  generated code.
  """
  name: str = "s9_dbuf_epoch_coordinator"
  owner: str = "hand_coded_backend_primitive"
  nbuf: int = 2
  slot_expr: str = "epoch % 2"
  prologue: tuple[str, ...] = ("produce epoch0 -> slot0",)
  body: tuple[str, ...] = (
    "consume epoch i -> slot(i % 2)",
    "produce epoch i+1 -> slot((i+1) % 2)",
    "barrier before produced slot is consumed",
  )
  tail: tuple[str, ...] = ("consume final produced epoch",)
  reusable_contract: str = "parameterized_by_role_tile_layout_wait_policy"

  def to_json(self) -> dict[str, Any]:
    return {
      "name": self.name, "owner": self.owner, "nbuf": self.nbuf, "slot_expr": self.slot_expr,
      "prologue": list(self.prologue), "body": list(self.body), "tail": list(self.tail),
      "reusable_contract": self.reusable_contract,
      "classification": "hand_coded_dbuf_epoch_primitive",
    }

  @classmethod
  def from_json(cls, data: dict[str, Any]) -> "DBUFEpochPrimitive":
    return cls(name=data.get("name", cls.name), owner=data.get("owner", cls.owner),
               nbuf=data.get("nbuf", cls.nbuf), slot_expr=data.get("slot_expr", cls.slot_expr),
               prologue=tuple(data.get("prologue", cls.prologue)), body=tuple(data.get("body", cls.body)),
               tail=tuple(data.get("tail", cls.tail)),
               reusable_contract=data.get("reusable_contract", cls.reusable_contract))


@dataclass(frozen=True)
class WMMALDSSpec:
  m: int
  n: int
  k: int
  tile_m: int
  tile_n: int
  tile_k: int
  waves_m: int
  waves_n: int
  wm: int
  wn: int
  threads: int
  pad: int
  dbuf: int
  plra: int = 0
  plrab: int = 0
  leanaddr: int = 0
  dshalf: int = 0
  operand_a: str = "global_row_major_fp16_to_lds"
  operand_b: str = "global_row_major_bt_fp16_to_lds"
  wait_policy: str = "vmem_to_lds_then_lgkm_to_wmma"
  target: str = "amd_gfx1100"
  reg_layout: LDS2RegLayout = field(default_factory=LDS2RegLayout)
  memory_layout: LDS2MemoryLayout = field(default_factory=LDS2MemoryLayout)
  wait: LDS2WaitPolicy = field(default_factory=LDS2WaitPolicy)
  cadence: LDS2Cadence = field(default_factory=LDS2Cadence)
  lifecycle: LDS2LifecycleTemplate = field(default_factory=LDS2LifecycleTemplate)
  dbuf_epoch_primitive: DBUFEpochPrimitive = field(default_factory=DBUFEpochPrimitive)
  selection_label: str = LDS2_DEFAULT_SELECTION_LABEL

  @property
  def k_substeps(self) -> int:
    return self.tile_k // 16

  @property
  def cpr(self) -> int:
    return self.tile_k // 8

  @property
  def row_stride(self) -> int:
    return self.threads // self.cpr

  @property
  def loads_a(self) -> int:
    return self.tile_m // self.row_stride

  @property
  def loads_b(self) -> int:
    return self.tile_n // self.row_stride

  @property
  def stride_a_bytes(self) -> int:
    return self.tile_k * 2 + self.pad

  @property
  def stride_b_bytes(self) -> int:
    return self.tile_k * 2 + self.pad

  @property
  def lds_a_bytes(self) -> int:
    return self.stride_a_bytes * self.tile_m

  @property
  def lds_buffer_bytes(self) -> int:
    return self.lds_a_bytes + self.stride_b_bytes * self.tile_n

  @property
  def lds_buffers(self) -> int:
    return 2 if self.dbuf else 1

  @property
  def lds_total_bytes(self) -> int:
    return self.lds_buffer_bytes * self.lds_buffers

  @property
  def accum_vgprs(self) -> int:
    return self.wm * self.wn * 8

  @property
  def coop_temp_vgprs(self) -> int:
    return (self.loads_a + self.loads_b) * 4

  @property
  def plr_mode(self) -> str:
    if self.plrab: return "A+B"
    if self.plra: return "A"
    return "none"

  def legality_errors(self) -> list[str]:
    errors: list[str] = []
    if self.threads != self.waves_m * self.waves_n * 32:
      errors.append("threads must equal waves_m*waves_n*32")
    if self.tile_m != self.waves_m * self.wm * 16:
      errors.append("tile_m must equal waves_m*wm*16")
    if self.tile_n != self.waves_n * self.wn * 16:
      errors.append("tile_n must equal waves_n*wn*16")
    if self.tile_k % 16 != 0:
      errors.append("tile_k must be a multiple of 16")
    if self.tile_k <= 0 or self.cpr <= 0:
      errors.append("tile_k must be positive")
    elif self.threads % self.cpr != 0:
      errors.append("threads must be divisible by tile_k/8")
    elif self.tile_m % self.row_stride != 0 or self.tile_n % self.row_stride != 0:
      errors.append("tile_m and tile_n must be divisible by cooperative row_stride")
    if self.m % self.tile_m != 0 or self.n % self.tile_n != 0 or self.k % self.tile_k != 0:
      errors.append("m/n/k must be divisible by tile_m/tile_n/tile_k")
    if self.lds_total_bytes > 65536:
      errors.append(f"LDS overflow: {self.lds_total_bytes} > 65536")
    if self.dshalf:
      errors.append("dshalf is an incorrect throughput probe, not a valid primitive")
    if self.plra and self.k_substeps != 2:
      errors.append("PLRA requires tile_k/16 == 2")
    if self.plrab and self.k_substeps != 2:
      errors.append("PLRAB requires tile_k/16 == 2")
    return errors

  def ownership_classification(self) -> str:
    return LDS2_OWNERSHIP_CLASSIFICATION

  def to_json(self) -> dict[str, Any]:
    return {
      "m": self.m, "n": self.n, "k": self.k, "tile_m": self.tile_m, "tile_n": self.tile_n,
      "tile_k": self.tile_k, "waves_m": self.waves_m, "waves_n": self.waves_n, "wm": self.wm,
      "wn": self.wn, "threads": self.threads, "pad": self.pad, "dbuf": self.dbuf, "plra": self.plra,
      "plrab": self.plrab, "leanaddr": self.leanaddr, "dshalf": self.dshalf,
      "operand_a": self.operand_a, "operand_b": self.operand_b, "wait_policy": self.wait_policy,
      "reg_layout": self.reg_layout.to_json(), "memory_layout": self.memory_layout.to_json(),
      "wait": self.wait.to_json(), "cadence": self.cadence.to_json(), "lifecycle": self.lifecycle.to_json(),
      "dbuf_epoch_primitive": self.dbuf_epoch_primitive.to_json(),
      "selection_label": self.selection_label, "ownership_classification": self.ownership_classification(),
      "target": self.target, "k_substeps": self.k_substeps, "row_stride": self.row_stride,
      "loads_a": self.loads_a, "loads_b": self.loads_b, "stride_a_bytes": self.stride_a_bytes,
      "stride_b_bytes": self.stride_b_bytes, "lds_a_bytes": self.lds_a_bytes,
      "lds_buffer_bytes": self.lds_buffer_bytes, "lds_buffers": self.lds_buffers,
      "lds_total_bytes": self.lds_total_bytes, "accum_vgprs": self.accum_vgprs,
      "coop_temp_vgprs": self.coop_temp_vgprs, "plr_mode": self.plr_mode,
      "legality_errors": self.legality_errors(),
    }

  @classmethod
  def from_json(cls, data: dict[str, Any] | str) -> "WMMALDSSpec":
    if isinstance(data, str): data = json.loads(data)
    fields = {
      "m", "n", "k", "tile_m", "tile_n", "tile_k", "waves_m", "waves_n", "wm", "wn", "threads", "pad", "dbuf",
      "plra", "plrab", "leanaddr", "dshalf", "operand_a", "operand_b", "wait_policy", "target", "selection_label",
    }
    kwargs = {key: data[key] for key in fields if key in data}
    if "reg_layout" in data: kwargs["reg_layout"] = LDS2RegLayout.from_json(data["reg_layout"])
    if "memory_layout" in data: kwargs["memory_layout"] = LDS2MemoryLayout.from_json(data["memory_layout"])
    if "wait" in data: kwargs["wait"] = LDS2WaitPolicy.from_json(data["wait"])
    if "cadence" in data: kwargs["cadence"] = LDS2Cadence.from_json(data["cadence"])
    if "lifecycle" in data: kwargs["lifecycle"] = LDS2LifecycleTemplate.from_json(data["lifecycle"])
    if "dbuf_epoch_primitive" in data: kwargs["dbuf_epoch_primitive"] = DBUFEpochPrimitive.from_json(data["dbuf_epoch_primitive"])
    return cls(**kwargs)

  @classmethod
  def from_prefill_schedule(cls, prefill_spec) -> "WMMALDSSpec | None":
    return extract_wmma_lds_spec(prefill_spec)


@dataclass(frozen=True)
class LDSWindow:
  operand: str
  buffer: int
  base: int
  bytes: int
  rows: int
  row_stride_bytes: int
  vector_bytes: int = 16

  @property
  def end(self) -> int:
    return self.base + self.bytes

  @property
  def vectors_per_row(self) -> int:
    return self.row_stride_bytes // self.vector_bytes

  @property
  def total_vectors(self) -> int:
    return self.rows * self.vectors_per_row

  def to_json(self) -> dict[str, Any]:
    return {
      "operand": self.operand, "buffer": self.buffer, "base": self.base, "end": self.end,
      "bytes": self.bytes, "rows": self.rows, "row_stride_bytes": self.row_stride_bytes,
      "vector_bytes": self.vector_bytes, "vectors_per_row": self.vectors_per_row,
      "total_vectors": self.total_vectors,
    }


def wmma_lds_slot_identity_proof(spec: WMMALDSSpec, *, active_buffers: int | None = None) -> dict[str, Any]:
  """Static H1/H2 proof for packed LDS windows.

  H1 is the packed global_load_b128 -> ds_store_b128 staging shape. H2 is the
  slot identity that the ds_load_b128 side consumes the same typed A/B byte
  windows. This deliberately proves materialized-offset single-buffer transport
  as the baseline; DBUF promotion must pass the same proof with two active
  buffers before cadence work is allowed.
  """
  if not isinstance(spec, WMMALDSSpec):
    raise TypeError(f"wmma_lds_slot_identity_proof expected WMMALDSSpec, got {type(spec).__name__}")
  buffers = spec.lds_buffers if active_buffers is None else active_buffers
  errors: list[str] = []
  if buffers not in (1, 2): errors.append(f"active_buffers must be 1 or 2, got {buffers}")
  if spec.stride_a_bytes % 16 != 0: errors.append("A row stride must be b128-aligned")
  if spec.stride_b_bytes % 16 != 0: errors.append("B row stride must be b128-aligned")
  if spec.lds_a_bytes % 16 != 0 or spec.lds_buffer_bytes % 16 != 0:
    errors.append("A/B LDS window boundaries must be b128-aligned")
  windows: list[LDSWindow] = []
  if not errors:
    for buf in range(buffers):
      base = buf * spec.lds_buffer_bytes
      windows.append(LDSWindow("A", buf, base, spec.lds_a_bytes, spec.tile_m, spec.stride_a_bytes))
      windows.append(LDSWindow("B", buf, base + spec.lds_a_bytes, spec.stride_b_bytes * spec.tile_n, spec.tile_n, spec.stride_b_bytes))
    for i, lhs in enumerate(windows):
      if lhs.base % 16 or lhs.end % 16: errors.append(f"{lhs.operand}{lhs.buffer} window is not b128 aligned")
      for rhs in windows[i+1:]:
        if max(lhs.base, rhs.base) < min(lhs.end, rhs.end):
          errors.append(f"{lhs.operand}{lhs.buffer} overlaps {rhs.operand}{rhs.buffer}")
    if buffers * spec.lds_buffer_bytes > 65536:
      errors.append(f"active LDS footprint overflow: {buffers * spec.lds_buffer_bytes} > 65536")
  vectors_by_operand = {
    "A": spec.tile_m * (spec.stride_a_bytes // 16),
    "B": spec.tile_n * (spec.stride_b_bytes // 16),
  }
  return {
    "schema": "wmma-lds-slot-identity-proof.v1",
    "active_buffers": buffers,
    "spec_dbuf": spec.dbuf,
    "materialized_offsets_baseline": True,
    "ds_immediate_folding_required": False,
    "packed_vector_bytes": 16,
    "lds_buffer_bytes": spec.lds_buffer_bytes,
    "active_lds_bytes": buffers * spec.lds_buffer_bytes,
    "windows": [w.to_json() for w in windows],
    "vectors_by_operand_per_buffer": vectors_by_operand,
    "expected_stage_vectors_per_buffer": vectors_by_operand["A"] + vectors_by_operand["B"],
    "slot_identity": "A and B ds_load_b128 windows consume the same typed byte intervals staged by ds_store_b128",
    "dbuf_slot_identity_proven": buffers == 2 and not errors,
    "dbuf_cadence_proven": False,
    "ok": not errors,
    "errors": errors,
  }


def wmma_lds_layout_key(spec: WMMALDSSpec, role: str) -> dict[str, Any]:
  """Return the static WMMA operand layout contract for one LDS-staged role."""
  if not isinstance(spec, WMMALDSSpec):
    raise TypeError(f"wmma_lds_layout_key expected WMMALDSSpec, got {type(spec).__name__}")
  if role not in ("A", "B"): raise ValueError(f"unsupported WMMA LDS role {role!r}")
  lds_layout = spec.memory_layout.operand_a if role == "A" else spec.memory_layout.operand_b
  return {
    "role": role,
    "operand": "src0" if role == "A" else "src1",
    "lds_layout": lds_layout,
    "wmma_contract": "rdna3_wmma_f32_16x16x16_f16",
    "fragment_shape": [16, 16],
    "lane_map_id": "rdna3_wmma_f32_16x16x16_f16_lds2_static",
    "lane_count": 32,
    "lane_replication": "A_lanes_16_31_replicate" if role == "A" else None,
    "per_lane_elements": 16,
    "vector_bytes": 16,
    "lds_row_stride_bytes": spec.stride_a_bytes if role == "A" else spec.stride_b_bytes,
  }


def extract_wmma_lds_spec(prefill_spec) -> WMMALDSSpec | None:
  if prefill_spec.route_family != "lds": return None
  spec = WMMALDSSpec(
    m=prefill_spec.m, n=prefill_spec.n, k=prefill_spec.k, tile_m=prefill_spec.tile_m,
    tile_n=prefill_spec.tile_n, tile_k=prefill_spec.tile_k, waves_m=prefill_spec.waves_m,
    waves_n=prefill_spec.waves_n, wm=prefill_spec.wm, wn=prefill_spec.wn, threads=prefill_spec.threads,
    pad=prefill_spec.pad, dbuf=prefill_spec.dbuf, plra=prefill_spec.plra, plrab=prefill_spec.plrab,
    leanaddr=prefill_spec.leanaddr, target=prefill_spec.target,
    memory_layout=LDS2MemoryLayout(), wait=LDS2WaitPolicy(waitcnt_policy=prefill_spec.waitcnt_policy),
    cadence=LDS2Cadence(buffers=2 if prefill_spec.dbuf else 1))
  return None if spec.legality_errors() else spec


def lower_wmma_lds_spec(spec: WMMALDSSpec) -> Any:
  if not isinstance(spec, WMMALDSSpec):
    raise TypeError(f"lower_wmma_lds_spec expected WMMALDSSpec, got {type(spec).__name__}")
  errors = spec.legality_errors()
  if errors:
    raise NotImplementedError(
      "unsupported LDS WMMA primitive spec: " + "; ".join(errors) +
      ". No fallback to extra.qk.prefill.wmma.build_gemm_lds2 was attempted."
    )
  raise NotImplementedError(
    "Generated LDS WMMA primitive lowering is not implemented yet. "
    "This opt-in seam intentionally fails closed and does not call "
    "extra.qk.prefill.wmma.build_gemm_lds2."
  )


def wmma_lds_postrange_opts(spec: WMMALDSSpec, *, unr: int = 2, cooperative_waves: bool = False):
  from tinygrad.codegen.opt import Opt, OptOps
  opts = [
    Opt(OptOps.TC, 0, (-1, 2, 1)),
    Opt(OptOps.UPCAST, 0, spec.wm),
    Opt(OptOps.UPCAST, 1, spec.wn),
  ]
  # TC materializes one hardware wave. The strict full-kernel candidate owns
  # the complete cooperative workgroup, so retain its remaining wave factor as
  # a generated LOCAL axis instead of silently compiling a one-wave kernel.
  if cooperative_waves:
    wave_count = spec.waves_m * spec.waves_n
    if spec.threads != wave_count * 32: raise ValueError("cooperative wave geometry does not account for spec threads")
    if wave_count > 1: opts.append(Opt(OptOps.LOCAL, 0, wave_count))
  opts.append(Opt(OptOps.UNROLL, 0, unr))
  return tuple(opts)


def wmma_lds_generated_env_defaults(spec: WMMALDSSpec) -> dict[str, str]:
  if not isinstance(spec, WMMALDSSpec):
    raise TypeError(f"wmma_lds_generated_env_defaults expected WMMALDSSpec, got {type(spec).__name__}")
  # This is the existing single-buffer LDS substrate. DBUF is deliberately absent here: prior commits proved DBUF
  # needs separate slot-cadence and DS-offset proofs before it can be promoted.
  return {
    "AMD_ISA_WMMA_B128_FRAG": "1",
    "AMD_ISA_REG_ACCUM": "1",
    "PREFILL_TC_LOCAL_STAGE": "both",
    "PREFILL_TC_LOCAL_STAGE_WITH_LOCAL": "1",
    "PREFILL_TC_LOCAL_STAGE_B_TILEKEY": "1",
    "PREFILL_LDS_PACK_WITHLOCAL_B128": "1",
  }


def wmma_lds_lowering_insertion_point() -> dict[str, Any]:
  return {
    "route_spec_source": "extra/qk/prefill_schedule_spec.py::describe_prefill_schedule",
    "current_raw_lowering": "extra/qk/prefill_schedule_spec.py::emit_prefill_gemm_from_spec -> "
                            "extra/qk/prefill_graph_gemm_route.py::_emit_schedule -> "
                            "extra/qk/prefill/wmma.py::build_gemm_lds2",
    "first_generated_diversion": "extra/qk/prefill_schedule_spec.py::emit_prefill_gemm_from_spec",
    "diversion_predicate": 'PrefillGEMMScheduleSpec.route_family == "lds"',
    "primitive_spec": "extra/qk/wmma_lds_spec.py::WMMALDSSpec",
    "primitive_lowerer": "extra/qk/wmma_lds_spec.py::lower_wmma_lds_spec",
    "generated_transport": "extra/qk/prefill_graph_gemm_route.py::route_pf16_graph_gemm -> ordinary generated matmul",
    "generated_transport_env": "extra/qk/wmma_lds_spec.py::wmma_lds_generated_env_defaults",
    "generated_transport_opts": "extra/qk/wmma_lds_spec.py::wmma_lds_postrange_opts",
    "oracle_role": "ffn_gate_up",
    "reuse_existing_substrate": [
      "docs/prefill-lessons-ledger.md",
      "docs/prefill-lessons-ledger.md",
      "docs/prefill-lessons-ledger.md",
      "tinygrad/codegen/opt/postrange.py cooperative LDS/DBUF staging machinery",
      "tinygrad/renderer/isa/amd.py LDSAddr/decompose_lds_index and DS_LOAD_B128/DS_STORE_B128 lowering",
      "extra/qk/prefill/native_isa_l4_stream_probe.py structural LDS/DBUF probe",
      "extra/qk/prefill/kernel_lifecycle_trace.py route lifecycle trace",
    ],
    "do_not_copy": [
      "extra/qk/prefill/wmma.py::build_gemm_lds2 instruction list",
      "route-local UOp(Ops.INS, ...) full-kernel body",
      "hard-coded register layout without a declarative spec",
      "parallel LDS lowering that bypasses existing postrange.py and AMD renderer LDS primitives",
    ],
  }
