"""Tinygrad-side admission for BoltBeam primitive-graph candidates."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json
from typing import Any, Mapping

SCHEMA_VERSION = "boltbeam.full_kernel_candidate.v3"
PLAN_KIND = "tinygrad_primitive_graph.v1"
SUPPORTED_OPS = frozenset({"global_load", "global_store", "workgroup_load", "workgroup_store", "workgroup_barrier", "abs", "add", "sub",
  "mul", "div", "precise_div", "select", "round_away", "clamp", "cast", "bitcast", "and", "or", "shift", "subgroup_shuffle",
  "subgroup_reduce_max", "subgroup_reduce_sum", "pack_bytes", "unpack_bits", "dot", "matrix_mma", "online_softmax"})

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
  if not isinstance(workload, dict) or workload.get("operation") not in ("quantize_q8_1_ds4", "quantized_linear"):
    raise PlanError("unsupported primitive workload")
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
  if workload["operation"] == "quantize_q8_1_ds4":
    required = {"records_per_workgroup", "subgroups_per_record", "values_per_lane", "store_width_bytes", "record_order"}
    if not isinstance(params, dict) or set(params) != required: raise PlanError("DS4 plan parameters are incomplete")
    if params["records_per_workgroup"] * params["subgroups_per_record"] != 4: raise PlanError("DS4 workgroup must account for four target subgroups")
    if params["subgroups_per_record"] * params["values_per_lane"] != 4: raise PlanError("DS4 ownership must cover 128 values per record")
    if params["record_order"] != "segment_major": raise PlanError("DS4 output must remain segment-major")
  else:
    shape = workload.get("shape", {})
    if set(shape) != {"m", "n", "k"} or any(not isinstance(shape[x], int) or shape[x] <= 0 for x in shape):
      raise PlanError("quantized linear requires positive named M/N/K axes")
    quant = workload.get("operands", {}).get("packed_weight", {}).get("quantization")
    if quant not in ("Q4_K", "Q6_K") or shape["k"] % 256: raise PlanError("quantized linear requires canonical Q4_K/Q6_K blocks")
    phase = workload.get("phase")
    algorithm = params.get("algorithm") if isinstance(params, dict) else None
    if (phase, algorithm) not in (("decode", "subgroup_dot"), ("prefill", "matrix_mma")):
      raise PlanError("quantized linear phase and algorithm disagree")
    if phase == "prefill" and any(shape[axis] % params[tile] for axis,tile in (("m","tile_m"),("n","tile_n"),("k","tile_k"))):
      raise PlanError("generated prefill lowering currently requires tile-aligned M/N/K")
  return PrimitivePlan(row)

def _quantized_linear_decode_program(plan: PrimitivePlan):
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import Ops
  from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel, q6k_spec_for_role, emit_q6k_gemv_kernel
  shape=plan.workload["shape"]; rows,k=shape["n"],shape["k"]
  quant=plan.workload["operands"]["packed_weight"]["quantization"]
  x=Tensor.empty(k, dtype=dtypes.float16, device="NV")
  if quant == "Q4_K":
    out=Tensor.empty(rows, dtype=dtypes.float32, device="NV")
    style="vector" if plan.parameters.get("load_width")==4 else "scalar"
    if plan.parameters.get("rows_per_workgroup") != 1: raise PlanError("NV Q4_K lowerer does not yet implement multi-row ownership")
    weight=Tensor.empty(rows*(k//256)*36, dtype=dtypes.uint32, device="NV")
    emitter=q4k_g3_lanemap_gemv_kernel(rows,k,load_style=style)
  else:
    if plan.parameters.get("load_width") != 1: raise PlanError("NV Q6_K lowerer does not implement alternate packed-load widths")
    if plan.parameters.get("rows_per_workgroup") != 1: raise PlanError("NV Q6_K lowerer does not yet implement multi-row ownership")
    spec=q6k_spec_for_role(rows,k,parts=1,row_tile=1,use_coop=True,reduction="in_kernel",opts=())
    out=Tensor.empty(rows, dtype=dtypes.float32, device="NV")
    weight=Tensor.empty(rows*(k//256)*105, dtype=dtypes.uint16, device="NV")
    emitter=emit_q6k_gemv_kernel(spec)
  output=out.uop_program(weight,x,fxn=emitter)[0]
  calls=[u for u in output.schedule_linear().src if u.op is Ops.CALL]
  if len(calls) != 1: raise PlanError(f"quantized-linear primitive plan must lower to one call, got {len(calls)}")
  try: program=to_program(calls[0].src[0],Device["NV"].renderer)
  except Exception as exc: raise PlanError(f"compiler lowering failed: {type(exc).__name__}: {exc}") from exc
  source=next((u.arg for u in program.src if u.op is Ops.SOURCE),None)
  binary=next((u.arg for u in program.src if u.op is Ops.BINARY),None)
  if not isinstance(source,str) or not isinstance(binary,bytes): raise PlanError("compiled quantized-linear program lacks source or binary")
  return program,source,binary

def compile_quantized_linear(payload:Mapping[str, Any], plan:PrimitivePlan) -> Mapping[str, Any]:
  if plan.workload["phase"] != "decode":
    from tinygrad import Device, Tensor, dtypes
    from tinygrad.codegen import to_program
    from tinygrad.uop.ops import Ops
    from tinygrad.codegen.opt.postrange import warmstart_candidate_state
    from tinygrad.llm.packed_wmma_prefill import PackedWmmaRoute, generated_warmstart_entry, packed_half_carrier
    shape=plan.workload["shape"]; p=plan.parameters
    if p.get("work_partition") == "stream_k":
      from tinygrad.codegen.opt.stream_k import StreamKSchedule
      from tinygrad.codegen.opt.persistent_accumulator import owner_segments
      from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
      from extra.llm_research.prefill.nv_compiler_q4k_pp512_binding import binding_for as compiler_binding_for
      from extra.llm_research.prefill.nv_compiler_q4k_streamk_transform import transform_compiler_q4k_to_streamk, active_fixup_source
      from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
      if plan.workload["operands"]["packed_weight"]["quantization"] != "Q4_K":
        raise PlanError("NV Stream-K target lowerer currently admits Q4_K only")
      schedule=StreamKSchedule(shape["m"],shape["n"],shape["k"],p["tile_m"],p["tile_n"],p["tile_k"],
                               p["owner_count"],p["boundary_quantum_k_blocks"])
      if (shape["m"],shape["n"],shape["k"],p["tile_m"],p["tile_n"],p["tile_k"],p["owner_count"],p["boundary_quantum_k_blocks"]) != (512,12288,4096,128,128,64,170,8):
        raise PlanError("compiler Q4 Stream-K requires the exact admitted representative geometry")
      binding=compiler_binding_for("NV")
      compiler_source=next((u.arg for u in binding.main_program.toposort() if u.op is Ops.SOURCE and "__global__" in u.arg),None)
      if not isinstance(compiler_source,str): raise PlanError("compiler Q4 binding lacks emitted main source")
      main_source=transform_compiler_q4k_to_streamk(compiler_source,unroll=8); fixup_source=active_fixup_source()
      main_lib=NVRTCCompiler(Device["NV"].arch,ptx=False,cache_key="q4_compiler_streamk_main_unroll8_o170").compile(main_source)
      fixup_lib=NVRTCCompiler(Device["NV"].arch,ptx=False,cache_key="q4_compiler_streamk_active_fixup_v1").compile(fixup_source)
      active_tiles=tuple(sorted({segment.output_tile for owner in range(schedule.owners)
        for segment in owner_segments(schedule,owner) if not segment.direct}))
      main=native_nv_program("q4k_imma_stream",main_lib,global_size=(170,1,1),local_size=(32,2,4),
        globals=tuple(range(5)),outs=(0,1,2),ins=(3,4),vals=(512,12288,4096),shared_mem=0)
      fixup=native_nv_program("q4k_imma_fixup_active",fixup_lib,global_size=(len(active_tiles),1,1),local_size=(256,1,1),
        globals=tuple(range(4)),outs=(0,),ins=(1,2,3),vals=(512,12288))
      main_binary=next((u.arg for u in main.src if u.op is Ops.BINARY),None); fixup_binary=next((u.arg for u in fixup.src if u.op is Ops.BINARY),None)
      if not isinstance(main_binary,bytes) or not isinstance(fixup_binary,bytes): raise PlanError("Stream-K compiler transform lacks main/fixup binaries")
      main_hash,fixup_hash=hashlib.sha256(main_binary).hexdigest(),hashlib.sha256(fixup_binary).hexdigest()
      qualified_hashes=("bdf49e9bc6de56e843f39cc20f0ab087f4febef1b0d3d3c9992bfc01ab6fedbb",
                        "9b3e400e942bf80f4f02184a6b4de0085ff0861ef5eb5bd4197e118a3ceb9acc")
      promotion_eligible=(plan.workload["role"] in ("ffn_gate_up","ffn_gate","ffn_up") and
                          (main_hash,fixup_hash)==qualified_hashes)
      return {"compile_status":"binary_ready","program_kind":"stream_k_main_fixup","compiler":type(Device["NV"].compiler).__name__,
              "main_program":main.arg.name,"fixup_program":fixup.arg.name,"main_source_sha256":hashlib.sha256(main_source.encode()).hexdigest(),
              "fixup_source_sha256":hashlib.sha256(fixup_source.encode()).hexdigest(),"main_binary_sha256":main_hash,
              "fixup_binary_sha256":fixup_hash,"candidate_plan_hash":candidate_hash(plan.candidate),
              "owners":170,"partial_slots":340,"fixup_tiles":len(active_tiles),"active_tiles":active_tiles,"no_external_binary":True,
              "promotion_eligible":promotion_eligible,"qualified_complete_ratio_vs_llama":1.04529}
    geometry=(p["tile_m"],p["tile_n"],p["tile_k"],p["subgroups_m"],p["subgroups_n"],p["buffer_count"])
    row=PackedWmmaRoute(plan.workload["operands"]["packed_weight"]["quantization"],plan.workload["role"],
                        (shape["m"],shape["n"],shape["k"]),geometry,candidate_hash(plan.candidate))
    entry=generated_warmstart_entry(row); transform=entry["transform"]
    packed=Tensor.empty(transform.packed_bytes//transform.storage_dtype.itemsize,dtype=transform.storage_dtype,device="NV")
    activation=Tensor.empty((shape["m"],shape["k"]),dtype=dtypes.float16,device="NV")
    weight=packed_half_carrier(packed,transform,shape["n"],shape["k"])
    with warmstart_candidate_state({entry["key"]:entry["opt"]},{entry["key"]:entry["context"]}):
      linear=(activation@weight.transpose()).contiguous().schedule_linear()
    calls=[u for u in linear.src if u.op is Ops.CALL]
    if len(calls) != 1: raise PlanError(f"prefill quantized-linear plan must lower to one contraction call, got {len(calls)}")
    try: program=to_program(calls[0].src[0],Device["NV"].renderer)
    except Exception as exc: raise PlanError(f"prefill compiler lowering failed: {type(exc).__name__}: {exc}") from exc
    source=next((u.arg for u in program.src if u.op is Ops.SOURCE),None); binary=next((u.arg for u in program.src if u.op is Ops.BINARY),None)
    if not isinstance(source,str) or not isinstance(binary,bytes): raise PlanError("compiled prefill contraction lacks source or binary")
    return {"compile_status":"binary_ready","compiler":type(Device["NV"].compiler).__name__,"program_name":program.arg.name,
            "source_sha256":hashlib.sha256(source.encode()).hexdigest(),"binary_sha256":hashlib.sha256(binary).hexdigest(),
            "source_bytes":len(source),"binary_bytes":len(binary),"candidate_plan_hash":candidate_hash(plan.candidate),
            "canonical_identity":entry["canonical_identity"],"packed_weight_transform":transform.to_json(),
            "geometry":{"tile":list(entry["context"].geometry.tile),"subgroups":list(entry["context"].geometry.waves),
                        "threads":entry["context"].geometry.threads}}
  program,source,binary=_quantized_linear_decode_program(plan)
  return {"compile_status":"binary_ready", "compiler":type(__import__("tinygrad").Device["NV"].compiler).__name__,
          "program_name":program.arg.name, "source_sha256":hashlib.sha256(source.encode()).hexdigest(),
          "binary_sha256":hashlib.sha256(binary).hexdigest(), "candidate_plan_hash":candidate_hash(plan.candidate),
          "source_bytes":len(source), "binary_bytes":len(binary)}

def compile_generated(payload:Mapping[str, Any], plan:PrimitivePlan) -> Mapping[str, Any]:
  return compile_ds4(payload,plan) if plan.workload["operation"] == "quantize_q8_1_ds4" else compile_quantized_linear(payload,plan)

def _run_quantized_linear_decode(plan:PrimitivePlan):
  import numpy as np
  from tinygrad import Tensor, dtypes
  from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel, q6k_spec_for_role, emit_q6k_gemv_kernel
  from extra.llm_research.decode.route_class_numerics import _make_q4k_words, _make_q6k_halfs
  from extra.llm_research.layout import q4_k_reference, q6_k_reference
  shape=plan.workload["shape"]; rows,k=shape["n"],shape["k"]
  quant=plan.workload["operands"]["packed_weight"]["quantization"]
  x_np=np.random.default_rng(20260830).normal(0,0.2,k).astype(np.float16)
  x=Tensor(x_np.copy(),dtype=dtypes.float16,device="NV").contiguous().realize()
  if quant == "Q4_K":
    words,raw=_make_q4k_words(rows,k,20260830)
    weight=Tensor(words.copy(),dtype=dtypes.uint32,device="NV").contiguous().realize()
    out=Tensor.empty(rows,dtype=dtypes.float32,device="NV").realize()
    style="vector" if plan.parameters.get("load_width")==4 else "scalar"
    if plan.parameters.get("rows_per_workgroup") != 1: raise PlanError("NV Q4_K lowerer does not yet implement multi-row ownership")
    emitter=q4k_g3_lanemap_gemv_kernel(rows,k,load_style=style); reference=q4_k_reference
  else:
    if plan.parameters.get("load_width") != 1: raise PlanError("NV Q6_K lowerer does not implement alternate packed-load widths")
    halfs=_make_q6k_halfs(rows,k,20260830); raw=halfs.view(np.uint8)
    weight=Tensor(halfs.copy(),dtype=dtypes.uint16,device="NV").contiguous().realize()
    spec=q6k_spec_for_role(rows,k,parts=1,row_tile=1,use_coop=True,reduction="in_kernel",opts=())
    out=Tensor.empty(rows,dtype=dtypes.float32,device="NV").realize()
    emitter=emit_q6k_gemv_kernel(spec); reference=q6_k_reference
  result=out.uop_program(weight,x,fxn=emitter)[0].realize()
  got=result.numpy().astype(np.float32).reshape(rows)
  weights=reference(Tensor(raw.reshape(-1).copy(),dtype=dtypes.uint8),rows*k).numpy().astype(np.float32).reshape(rows,k)
  want=weights@x_np.astype(np.float32)
  return got,want,result,weight,x,emitter,raw

def _run_quantized_linear_prefill(plan:PrimitivePlan):
  import numpy as np
  from tinygrad import Tensor, TinyJit, dtypes
  from tinygrad.codegen.opt.postrange import warmstart_candidate_state
  from tinygrad.llm.packed_wmma_prefill import PackedWmmaRoute, generated_warmstart_entry, packed_half_carrier
  from extra.llm_research.decode.route_class_numerics import _make_q4k_words, _make_q6k_halfs
  from extra.llm_research.layout import q4_k_reference, q6_k_reference
  shape=plan.workload["shape"]; m,n,k=shape["m"],shape["n"],shape["k"]; p=plan.parameters
  geometry=(p["tile_m"],p["tile_n"],p["tile_k"],p["subgroups_m"],p["subgroups_n"],p["buffer_count"])
  quant=plan.workload["operands"]["packed_weight"]["quantization"]
  row=PackedWmmaRoute(quant,plan.workload["role"],(m,n,k),geometry,candidate_hash(plan.candidate))
  entry=generated_warmstart_entry(row); transform=entry["transform"]
  if quant == "Q4_K": words,raw=_make_q4k_words(n,k,20260830); packed_np=words; dtype=dtypes.uint32; reference=q4_k_reference
  else: packed_np=_make_q6k_halfs(n,k,20260830); raw=packed_np.view(np.uint8); dtype=dtypes.uint16; reference=q6_k_reference
  activation_np=np.random.default_rng(20260831).normal(0,0.2,(m,k)).astype(np.float16)
  packed=Tensor(packed_np.copy(),dtype=dtype,device="NV").contiguous().realize()
  activation=Tensor(activation_np.copy(),dtype=dtypes.float16,device="NV").contiguous().realize()
  @TinyJit
  def run(a:Tensor,w:Tensor): return (a@packed_half_carrier(w,transform,n,k).transpose()).contiguous()
  with warmstart_candidate_state({entry["key"]:entry["opt"]},{entry["key"]:entry["context"]}):
    run(activation,packed).realize(); result=run(activation,packed).realize()
  got=result.numpy().astype(np.float32).reshape(m,n)
  weights=reference(Tensor(raw.reshape(-1).copy(),dtype=dtypes.uint8),n*k).numpy().astype(np.float32).reshape(n,k)
  want=activation_np.astype(np.float32)@weights.T
  return got,want,run,activation,packed,raw

def check_quantized_linear(payload:Mapping[str, Any], plan:PrimitivePlan) -> Mapping[str, Any]:
  import numpy as np
  try: got,want,*_=(_run_quantized_linear_decode(plan) if plan.workload["phase"] == "decode" else _run_quantized_linear_prefill(plan))
  except Exception as exc: raise PlanError(f"quantized-linear execution failed: {type(exc).__name__}: {exc}") from exc
  max_abs=float(np.max(np.abs(got-want))); scale=float(np.max(np.abs(want)))
  atol=max(1e-3,scale*2e-4)
  return {"correct":bool(np.allclose(got,want,rtol=2e-4,atol=atol)),"oracle":"canonical GGUF dequantization plus numpy fp32 matvec",
          "max_abs_error":max_abs,"atol":atol,"rtol":2e-4,"elements":int(got.size),
          "candidate_plan_hash":candidate_hash(plan.candidate)}

def measure_quantized_linear(payload:Mapping[str, Any], plan:PrimitivePlan) -> Mapping[str, Any]:
  import statistics,time
  from tinygrad import Device
  checked=check_quantized_linear(payload,plan)
  if not checked["correct"]: raise PlanError("incorrect quantized-linear candidate cannot be measured")
  if plan.workload["phase"] != "decode":
    got,want,run,activation,packed,raw=_run_quantized_linear_prefill(plan)
    launch=lambda: run(activation,packed).realize()
  else:
    got,want,out,weight,x,emitter,raw=_run_quantized_linear_decode(plan)
    launch=lambda: out.uop_program(weight,x,fxn=emitter)[0].realize()
  execution=payload.get("execution",{}); warmups=int(execution.get("warmups",2)); samples=int(execution.get("samples",9))
  for _ in range(warmups): launch()
  Device["NV"].synchronize(); values=[]
  for _ in range(samples):
    Device["NV"].synchronize(); start=time.perf_counter_ns(); launch(); Device["NV"].synchronize(); values.append(time.perf_counter_ns()-start)
  shape=plan.workload["shape"]
  return {"timing_mode":"synchronized_wall_ns","warmups":warmups,"samples_ns":values,
          "summary_ns":{"min":min(values),"mean":sum(values)/len(values),"median":statistics.median(values),"max":max(values),"range":max(values)-min(values)},
          "work_bytes":{"status":"exact","bytes":int(raw.nbytes+shape["k"]*2+shape["n"]*4)}}

def check_generated(payload:Mapping[str, Any], plan:PrimitivePlan) -> Mapping[str, Any]:
  return check_ds4(payload,plan) if plan.workload["operation"] == "quantize_q8_1_ds4" else check_quantized_linear(payload,plan)

def measure_generated(payload:Mapping[str, Any], plan:PrimitivePlan) -> Mapping[str, Any]:
  return measure_ds4(payload,plan) if plan.workload["operation"] == "quantize_q8_1_ds4" else measure_quantized_linear(payload,plan)

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

__all__ = ["PLAN_KIND", "PlanError", "PrimitivePlan", "SCHEMA_VERSION", "candidate_hash", "canonical_json", "check_ds4", "check_generated",
           "check_quantized_linear", "compile_ds4", "compile_generated", "compile_quantized_linear", "measure_ds4", "measure_generated",
           "measure_quantized_linear", "validate"]
