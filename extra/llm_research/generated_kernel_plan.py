"""Tinygrad-side admission for BoltBeam primitive-graph candidates."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json
from typing import Any, Mapping

SCHEMA_VERSION = "boltbeam.full_kernel_candidate.v3"
PLAN_KIND = "tinygrad_primitive_graph.v1"
SUPPORTED_OPS = frozenset({"global_load", "global_store", "workgroup_load", "workgroup_store", "workgroup_barrier", "abs", "add", "sub",
  "mul", "div", "precise_div", "select", "round_away", "clamp", "cast", "bitcast", "and", "or", "shift", "subgroup_shuffle",
  "subgroup_reduce_max", "subgroup_reduce_sum", "pack_bytes", "unpack_bits", "matrix_mma", "online_softmax"})

class PlanError(ValueError): pass

def canonical_json(value: Any) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

def candidate_hash(value: Any) -> str: return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()

@dataclass(frozen=True)
class PrimitivePlan:
  candidate: Mapping[str, Any]
  @property
  def workload(self): return self.candidate["workload"]
  @property
  def schedule(self): return self.candidate["schedule"]
  @property
  def parameters(self): return self.schedule["parameters"]
  @property
  def dispatch(self): return tuple(self.schedule["launch"]["dispatch"])
  @property
  def workgroup(self): return tuple(self.schedule["launch"]["workgroup"])

def validate(candidate: Mapping[str, Any], expected_hash: str | None = None) -> PrimitivePlan:
  if not isinstance(candidate, Mapping) or candidate.get("schema_version") != SCHEMA_VERSION: raise PlanError(f"candidate must use {SCHEMA_VERSION}")
  row = json.loads(canonical_json(candidate))
  if expected_hash is not None and candidate_hash(row) != expected_hash: raise PlanError("candidate hash does not match canonical bytes")
  workload, schedule = row.get("workload"), row.get("schedule")
  if not isinstance(workload, dict) or workload.get("operation") != "quantize_q8_1_ds4": raise PlanError("unsupported primitive workload")
  if not isinstance(schedule, dict) or schedule.get("plan_kind") != PLAN_KIND: raise PlanError("unsupported primitive plan kind")
  launch = schedule.get("launch", {})
  if any(not isinstance(launch.get(key), list) or len(launch[key]) != 3 or any(not isinstance(x, int) or x <= 0 for x in launch[key]) for key in ("dispatch", "workgroup")):
    raise PlanError("primitive launch requires positive dispatch/workgroup triples")
  nodes = schedule.get("nodes")
  if not isinstance(nodes, list) or not nodes: raise PlanError("primitive graph requires nodes")
  available = set(workload.get("operands", {})) | {axis.get("name") for axis in schedule.get("axes", []) if isinstance(axis, dict)}
  for node in nodes:
    if not isinstance(node, dict) or set(node) != {"id", "op", "inputs", "attrs"}: raise PlanError("invalid primitive node")
    if node["op"] not in SUPPORTED_OPS: raise PlanError(f"unsupported primitive op {node['op']!r}")
    if node["id"] in available or any(ref not in available for ref in node["inputs"]): raise PlanError("primitive graph is not a forward DAG")
    available.add(node["id"])
  params = schedule.get("parameters")
  required = {"records_per_workgroup", "subgroups_per_record", "values_per_lane", "store_width_bytes", "record_order"}
  if not isinstance(params, dict) or set(params) != required: raise PlanError("DS4 plan parameters are incomplete")
  if params["records_per_workgroup"] * params["subgroups_per_record"] != 4: raise PlanError("DS4 workgroup must account for four target subgroups")
  if params["subgroups_per_record"] * params["values_per_lane"] != 4: raise PlanError("DS4 ownership must cover 128 values per record")
  if params["record_order"] != "segment_major": raise PlanError("DS4 output must remain segment-major")
  return PrimitivePlan(row)

def _ds4_program(plan: PrimitivePlan):
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import Ops
  from extra.llm_research.prefill.nv_q8_ds4_native_producer import emit_ds4_q8_producer, emit_ds4_q8_producer_llama
  from extra.llm_research.prefill.nv_q8_ds4_native_producer_spec import DS4ProducerSpec
  shape = plan.workload["shape"]; spec = DS4ProducerSpec(M=shape["rows"], K=shape["k"])
  emitter = emit_ds4_q8_producer_llama(spec) if plan.parameters["records_per_workgroup"] == 4 else emit_ds4_q8_producer(spec)
  x = Tensor.empty(spec.M*spec.K, dtype=dtypes.float16, device="NV")
  output = Tensor.empty(spec.records*36, dtype=dtypes.uint32, device="NV").uop_program(x, fxn=emitter)[0]
  linear = output.schedule_linear()
  calls = [u for u in linear.src if u.op is Ops.CALL]
  if len(calls) != 1: raise PlanError(f"DS4 primitive plan must lower to one call, got {len(calls)}")
  try: program = to_program(calls[0].src[0], Device["NV"].renderer)
  except Exception as exc: raise PlanError(f"compiler lowering failed: {type(exc).__name__}: {exc}") from exc
  source = next((u.arg for u in program.src if u.op is Ops.SOURCE), None)
  binary = next((u.arg for u in program.src if u.op is Ops.BINARY), None)
  if not isinstance(source, str) or not isinstance(binary, bytes): raise PlanError("compiled DS4 program lacks source or binary")
  return program, source, binary, spec, emitter

def compile_ds4(payload: Mapping[str, Any], plan: PrimitivePlan) -> Mapping[str, Any]:
  try: program, source, binary, _spec, _emitter = _ds4_program(plan)
  except PlanError: raise
  return {"compiler":type(__import__("tinygrad").Device["NV"].compiler).__name__, "program_name":program.arg.name,
          "source_sha256":hashlib.sha256(source.encode()).hexdigest(), "binary_sha256":hashlib.sha256(binary).hexdigest(),
          "candidate_plan_hash":candidate_hash(plan.candidate), "source_bytes":len(source), "binary_bytes":len(binary)}

def _run_ds4(plan: PrimitivePlan):
  import numpy as np
  from tinygrad import Tensor, dtypes
  from extra.llm_research.prefill.nv_q8_ds4_native_producer import cpu_pack_ds4, emit_ds4_q8_producer, emit_ds4_q8_producer_llama
  from extra.llm_research.prefill.nv_q8_ds4_native_producer_spec import DS4ProducerSpec
  shape=plan.workload["shape"]; spec=DS4ProducerSpec(M=shape["rows"],K=shape["k"]); rng=np.random.default_rng(20260830)
  host=(rng.standard_normal((spec.M,spec.K),dtype=np.float32)*3.0).astype(np.float16)
  emitter=emit_ds4_q8_producer_llama(spec) if plan.parameters["records_per_workgroup"]==4 else emit_ds4_q8_producer(spec)
  x=Tensor(host.reshape(-1),device="NV").realize(); out=Tensor.empty(spec.records*36,dtype=dtypes.uint32,device="NV").realize()
  got=out.uop_program(x,fxn=emitter)[0].realize().numpy().view(np.uint8).reshape(spec.records,144)
  want=cpu_pack_ds4(host.astype(np.float32)); return got,want,x,out,emitter,spec

def check_ds4(payload: Mapping[str, Any], plan: PrimitivePlan) -> Mapping[str, Any]:
  import numpy as np
  try: got,want,*_=_run_ds4(plan)
  except Exception as exc: raise PlanError(f"DS4 execution failed: {type(exc).__name__}: {exc}") from exc
  mismatch=np.flatnonzero(got!=want)
  return {"correct":bool(not mismatch.size), "oracle":"cpu_q8_1_ds4_byte_exact", "bytes":int(got.size),
          "mismatch_bytes":int(mismatch.size), "first_mismatch":int(mismatch[0]) if mismatch.size else None,
          "candidate_plan_hash":candidate_hash(plan.candidate)}

def measure_ds4(payload: Mapping[str, Any], plan: PrimitivePlan) -> Mapping[str, Any]:
  import statistics, time
  from tinygrad import Device
  got,want,x,out,emitter,spec=_run_ds4(plan)
  if got.tobytes()!=want.tobytes(): raise PlanError("incorrect DS4 candidate cannot be measured")
  execution=payload.get("execution",{}); warmups=int(execution.get("warmups",2)); samples=int(execution.get("samples",9))
  def launch(): out.uop_program(x,fxn=emitter)[0].realize()
  for _ in range(warmups): launch()
  Device["NV"].synchronize(); values=[]
  for _ in range(samples):
    Device["NV"].synchronize(); start=time.perf_counter_ns(); launch(); Device["NV"].synchronize(); values.append(time.perf_counter_ns()-start)
  return {"timing_mode":"synchronized_wall_ns", "warmups":warmups, "samples_ns":values,
          "summary_ns":{"min":min(values),"mean":sum(values)/len(values),"median":statistics.median(values),"max":max(values),"range":max(values)-min(values)},
          "work_bytes":{"status":"exact","provenance":"DS4 144-byte output records plus fp16 activation input",
                        "bytes":spec.M*spec.K*2+spec.records*144}}

__all__ = ["PLAN_KIND", "PlanError", "PrimitivePlan", "SCHEMA_VERSION", "candidate_hash", "canonical_json", "check_ds4", "compile_ds4", "measure_ds4", "validate"]
