from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json, pathlib
from collections.abc import Mapping
from typing import Any

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops

_FULL_KERNEL_CANDIDATE_JSON_ENV = "BOLTBEAM_FULL_KERNEL_CANDIDATE_JSON"
_FULL_KERNEL_CANDIDATE_HASH_ENV = "BOLTBEAM_FULL_KERNEL_CANDIDATE_HASH"
_FULL_KERNEL_CANDIDATE_SET_JSON_ENV = "BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_JSON"
_FULL_KERNEL_CANDIDATE_SET_PATH_ENV = "BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_PATH"
_CANDIDATE_ROUTE_CENSUS:ContextVar[dict[str,Any]|None]=ContextVar("candidate_route_census",default=None)

@contextmanager
def candidate_route_census():
  collector={"selected":{},"model_forward":{}}
  token=_CANDIDATE_ROUTE_CENSUS.set(collector)
  try: yield collector
  finally: _CANDIDATE_ROUTE_CENSUS.reset(token)

def _candidate_route_row(admission) -> dict[str,Any]:
  workload=admission.normalized_payload["workload"]; shape=workload["shape"]; target=workload["target"]
  return {"profile":workload["profile"],"role":workload["role"],"shape":{"m":shape["m"],"n":shape["n"],"k":shape["k"]},
          "target":{"backend":target["backend"],"arch":target["arch"],"wave_size":target["wave_size"]},
          "canonical_identity":admission.canonical_identity}

def _structural_route_key(row:dict[str,Any]) -> tuple[Any,...]:
  shape,target=row["shape"],row["target"]
  return (row["role"],shape["m"],shape["n"],shape["k"],target["backend"],target["arch"],target["wave_size"])

def _record_candidate_route(admission) -> None:
  collector=_CANDIDATE_ROUTE_CENSUS.get()
  if collector is None: return
  row=_candidate_route_row(admission); key=_structural_route_key(row)
  prior=collector["selected"].get(key)
  if prior is not None and prior["canonical_identity"] != row["canonical_identity"]:
    raise RuntimeError(f"candidate route census identity drift for {key!r}")
  collector["selected"][key]={**row,"bindings":1 if prior is None else prior["bindings"]+1}

def record_model_forward_candidate(*, role:str, shape:tuple[int,int,int], canonical_identity:str, one_buffer:bool) -> None:
  """Record a separately admitted forward binding without relaxing registry census identity checks."""
  collector=_CANDIDATE_ROUTE_CENSUS.get()
  if collector is None or not one_buffer: return
  if not isinstance(canonical_identity,str) or not canonical_identity or not all(isinstance(x,int) for x in shape): return
  key=(role,*shape,canonical_identity)
  prior=collector["model_forward"].get(key)
  collector["model_forward"][key]={"role":role,"shape":{"m":shape[0],"n":shape[1],"k":shape[2]},
                                    "canonical_identity":canonical_identity,"one_buffer":True,
                                    "bindings":1 if prior is None else prior["bindings"]+1}

def finalize_candidate_route_census(collector:dict[str,Any],registry) -> dict[str,Any]:
  enabled_roles = {admission.normalized_payload["workload"]["role"] for admission in registry.admissions}
  expected={_structural_route_key(_candidate_route_row(admission)):{**_candidate_route_row(admission),"bindings":0}
            for entry,admission in zip(registry.candidate_set.entries,registry.admissions)
            if admission.normalized_payload["workload"]["role"] in enabled_roles}
  selected=dict(collector["selected"]); missing=[expected[k] for k in sorted(expected.keys()-selected.keys())]
  unexpected=[selected[k] for k in sorted(selected.keys()-expected.keys())]
  mismatched=[selected[k] for k in sorted(expected.keys()&selected.keys())
              if selected[k]["canonical_identity"] != expected[k]["canonical_identity"]]
  return {"schema":"prefill-candidate-set-route-census.v1","passed":not (missing or unexpected or mismatched),
          "policy_roles": sorted(enabled_roles),
          "expected_entry_count":len(expected),"selected_entry_count":len(selected),
          "model_forward": [collector["model_forward"][k] for k in sorted(collector["model_forward"])],
          "selected":[selected[k] for k in sorted(selected)],"missing":missing,"unexpected":unexpected,"identity_mismatches":mismatched}

def _candidate_registry_from_env(env:dict[str,Any]):
  """Load an offline candidate registry from an explicit tool configuration.

  Runtime admission uses exact policy attachments and never calls this loader.
  The historical name is retained for research-tool compatibility.
  """
  set_text,set_path=env.get(_FULL_KERNEL_CANDIDATE_SET_JSON_ENV),env.get(_FULL_KERNEL_CANDIDATE_SET_PATH_ENV)
  payload_text,identity=env.get(_FULL_KERNEL_CANDIDATE_JSON_ENV),env.get(_FULL_KERNEL_CANDIDATE_HASH_ENV)
  if set_text is not None and set_path is not None: raise ValueError("candidate set JSON and path are mutually exclusive")
  if (set_text is not None or set_path is not None) and (payload_text is not None or identity is not None):
    raise ValueError("candidate set and legacy candidate environment forms are mutually exclusive")
  from extra.qk.runtime_specs import FullKernelCandidateSet, admit_full_kernel_candidate_set, full_kernel_candidate_set_from_legacy
  if set_path is not None:
    try: set_text=pathlib.Path(str(set_path)).read_text()
    except OSError as exc: raise ValueError(f"candidate set path cannot be read: {exc}") from exc
  if set_text is not None:
    try: row=json.loads(str(set_text))
    except json.JSONDecodeError as exc: raise ValueError(f"candidate set JSON is invalid: {exc}") from exc
    return admit_full_kernel_candidate_set(FullKernelCandidateSet.from_json(row))
  if payload_text is None and identity is None: return None
  if payload_text is None or identity is None:
    raise ValueError(f"{_FULL_KERNEL_CANDIDATE_JSON_ENV} and {_FULL_KERNEL_CANDIDATE_HASH_ENV} must be provided together")
  try: payload=json.loads(str(payload_text))
  except json.JSONDecodeError as exc: raise ValueError(f"{_FULL_KERNEL_CANDIDATE_JSON_ENV} is not valid JSON: {exc}") from exc
  return admit_full_kernel_candidate_set(full_kernel_candidate_set_from_legacy(payload,str(identity)))


def _contiguous_candidate_operand(value:Tensor) -> Tensor:
  """Keep semantic metadata around an already concrete allocation without copying it."""
  return value if value.uop.op is Ops.MEMORY_SEMANTIC and value.uop.src[0].has_buffer_identity() else value.contiguous()

def _install_candidate_matmul(x,w,out_f,in_f,admission,compile_artifact:Mapping[str,Any]|None=None):
  from extra.qk.runtime_specs import candidate_storage_kind
  register_route = candidate_storage_kind(admission.normalized_payload) == "global_register_resident"
  if register_route:
    workload = admission.normalized_payload["workload"]
    from extra.qk.prefill.pure_register_evaluation_gate import runtime_compile_resource_eligibility
    eligibility = runtime_compile_resource_eligibility({"canonical_identity": admission.canonical_identity}, compile_artifact,
      role=workload["role"], shape=(int(workload["shape"]["m"]),out_f,in_f), target=workload["target"])
    if not eligibility["passed"]: return None
  from tinygrad.codegen.opt import Opt, OptOps
  import tinygrad.codegen.opt.postrange as pr
  m = int(x.shape[-2])
  # packed_dtype discriminates same-(m,out_f,in_f) different-quant candidates (e.g. Q4_K vs Q6_K sharing a
  # role's shape) so they land on distinct warmstart-table keys instead of colliding on one (postrange._warmstart_key).
  packed_dtype = admission.context.packed_weight.storage_dtype if admission.context.packed_weight is not None else None
  key=pr.warmstart_key({m,out_f},in_f,packed_dtype)
  existing=(pr._WARMSTART_CANDIDATE_CONTEXTS or {}).get(key)
  if existing is not None and existing.canonical_identity != admission.canonical_identity:
    raise ValueError(f"candidate warmstart key collision for {key!r}")
  pr._WARMSTART_OPTS={**(pr._WARMSTART_OPTS or {}),key:(Opt(OptOps.TC,0,(-1,2,1)),)}
  pr._WARMSTART_CANDIDATE_CONTEXTS={**(pr._WARMSTART_CANDIDATE_CONTEXTS or {}),key:admission.context}
  a=x.reshape(m,in_f).cast(dtypes.float16).contiguous()
  bt=_contiguous_candidate_operand(w.cast(dtypes.float16))
  return (a @ bt.transpose()).reshape(*x.shape[:-1],out_f)

def _attached_candidate_admission(lin, role:str, shape:tuple[int,int,int]):
  """Resolve an exact policy attachment. Model/profile names are provenance and are intentionally not consulted."""
  binding=getattr(lin,"_prefill_graph_gemm_binding",None)
  if not isinstance(binding,dict): return None
  registry,policy,facts=binding.get("candidate_registry"),binding.get("selected_policy"),binding.get("scanned_target_facts")
  inventory_identity,candidate_set_identity=binding.get("inventory_identity"),binding.get("candidate_set_identity")
  if registry is None or not isinstance(policy,dict) or not isinstance(facts,dict): return None
  if not all(isinstance(x,str) and x for x in (inventory_identity,candidate_set_identity)): return None
  target=facts.get("target",facts)
  if not isinstance(target,dict) or not all(k in target for k in ("backend","arch","wave_size")): return None
  target={k:target[k] for k in ("backend","arch","wave_size")}
  expected_shape={"m":shape[0],"n":shape[1],"k":shape[2]}
  if policy.get("role") != role or policy.get("shape") != expected_shape or policy.get("target") != target: return None
  if policy.get("inventory_identity") != inventory_identity or policy.get("candidate_set_identity") != candidate_set_identity: return None
  identity=policy.get("candidate_identity")
  if not isinstance(identity,str) or not identity: return None
  try:
    from extra.qk.route_manifest import canonical_candidate_set_identity
    if canonical_candidate_set_identity(registry.candidate_set.to_json()) != candidate_set_identity: return None
  except (AttributeError,TypeError,ValueError): return None
  matches=[]
  for admission in registry.admissions:
    row=_candidate_route_row(admission)
    if (_structural_route_key(row) == (role,*shape,target["backend"],target["arch"],target["wave_size"]) and
        admission.canonical_identity == identity): matches.append(admission)
  return matches[0] if len(matches) == 1 else None


def route_pf16_graph_gemm(lin, x: Tensor, w: Tensor | None = None) -> Tensor | None:
  # `w` (optional): an explicit fp16 weight to GEMM against. Callers may pass an unstored
  # `lin.weight.cast(fp16).contiguous()` from inside a layer-sized TinyJit, so replay reuses the graph-owned fp16
  # dequant scratch across blocks instead of pinning resident `lin._pf16_w` for every block.
  # Target, memory, and exact workload checks are completed before this JIT
  # dispatch and carried by the attached candidate binding. Device access is
  # deliberately not repeated during capture.
  if w is None: w = getattr(lin, "_pf16_w", None)
  b = getattr(lin, "bias", None)
  if w is None or b is not None or x.ndim < 2: return None
  if not isinstance(x.shape[-2], int) or not isinstance(x.shape[-1], int): return None
  m = x.shape[-2]
  out_f, in_f = w.shape
  if in_f != x.shape[-1]: return None
  role = getattr(lin, "_prefill_graph_role", None)
  # Model/runtime policy owns this attachment after scanning the actual target and selecting an exact inventory row.
  # Environment/profile artifact loaders above remain available to offline tooling, but never establish admission here.
  admission=None if role is None else _attached_candidate_admission(lin,role,(m,out_f,in_f))
  if admission is not None:
    binding=getattr(lin,"_prefill_graph_gemm_binding",{})
    compile_artifact=binding.get("compile_artifact") if isinstance(binding,dict) else None
    result = _install_candidate_matmul(x,w,out_f,in_f,admission,compile_artifact)
    if result is None: return None
    setattr(lin,"_prefill_full_kernel_candidate_identity",admission.canonical_identity)
    _record_candidate_route(admission)
    return result
  # The promoted candidate set is a production allowlist, not a hint. Unsupported roles/shapes return to the ordinary
  # Tensor path; they must never fall through into the slower raw/composed research emitters merely because graph-GEMM
  # is enabled for the promoted exact workloads.
  return None
