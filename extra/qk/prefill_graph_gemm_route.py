from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json, os, pathlib
from typing import Any

from tinygrad import Tensor, dtypes

_FULL_KERNEL_CANDIDATE_JSON_ENV = "BOLTBEAM_FULL_KERNEL_CANDIDATE_JSON"
_FULL_KERNEL_CANDIDATE_HASH_ENV = "BOLTBEAM_FULL_KERNEL_CANDIDATE_HASH"
_FULL_KERNEL_CANDIDATE_SET_JSON_ENV = "BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_JSON"
_FULL_KERNEL_CANDIDATE_SET_PATH_ENV = "BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_PATH"
_PURE_REGISTER_COMPILE_ARTIFACT_JSON_ENV = "BOLTBEAM_PURE_REGISTER_COMPILE_ARTIFACT_JSON"
_CANDIDATE_ROUTE_CENSUS:ContextVar[dict[str,Any]|None]=ContextVar("candidate_route_census",default=None)

def _candidate_env_requested(env: dict[str, Any]) -> bool:
  return any(key in env for key in (_FULL_KERNEL_CANDIDATE_JSON_ENV, _FULL_KERNEL_CANDIDATE_HASH_ENV,
                                     _FULL_KERNEL_CANDIDATE_SET_JSON_ENV, _FULL_KERNEL_CANDIDATE_SET_PATH_ENV))

def _promoted_candidate_policy_selected(env: dict[str, Any]) -> bool:
  raw = str(env.get("PREFILL_GRAPH_GEMM", "1")).strip().lower()
  return raw not in ("", "0", "false", "off", "no") and not _candidate_env_requested(env)

def _candidate_policy_env(env: dict[str, Any]) -> dict[str, Any]:
  if not _promoted_candidate_policy_selected(env): return env
  from extra.qk.route_manifest import promoted_prefill_candidate_policy
  return {**promoted_prefill_candidate_policy()["runtime_env"], **env}

@contextmanager
def candidate_route_census():
  collector={"selected":{}}
  token=_CANDIDATE_ROUTE_CENSUS.set(collector)
  try: yield collector
  finally: _CANDIDATE_ROUTE_CENSUS.reset(token)

def _candidate_route_row(admission) -> dict[str,Any]:
  workload=admission.normalized_payload["workload"]; shape=workload["shape"]; target=workload["target"]
  return {"profile":workload["profile"],"role":workload["role"],"shape":{"m":shape["m"],"n":shape["n"],"k":shape["k"]},
          "target":{"backend":target["backend"],"arch":target["arch"],"wave_size":target["wave_size"]},
          "canonical_identity":admission.canonical_identity}

def _record_candidate_route(admission) -> None:
  collector=_CANDIDATE_ROUTE_CENSUS.get()
  if collector is None: return
  row=_candidate_route_row(admission); shape,target=row["shape"],row["target"]
  key=(row["profile"],row["role"],shape["m"],shape["n"],shape["k"],target["backend"],target["arch"],target["wave_size"])
  prior=collector["selected"].get(key)
  if prior is not None and prior["canonical_identity"] != row["canonical_identity"]:
    raise RuntimeError(f"candidate route census identity drift for {key!r}")
  collector["selected"][key]={**row,"bindings":1 if prior is None else prior["bindings"]+1}

def finalize_candidate_route_census(collector:dict[str,Any],registry) -> dict[str,Any]:
  enabled_roles = {admission.normalized_payload["workload"]["role"] for admission in registry.admissions}
  expected={entry.exact_key:{**_candidate_route_row(admission),"bindings":0}
            for entry,admission in zip(registry.candidate_set.entries,registry.admissions)
            if admission.normalized_payload["workload"]["role"] in enabled_roles}
  selected=dict(collector["selected"]); missing=[expected[k] for k in sorted(expected.keys()-selected.keys())]
  unexpected=[selected[k] for k in sorted(selected.keys()-expected.keys())]
  mismatched=[selected[k] for k in sorted(expected.keys()&selected.keys())
              if selected[k]["canonical_identity"] != expected[k]["canonical_identity"]]
  return {"schema":"prefill-candidate-set-route-census.v1","passed":not (missing or unexpected or mismatched),
          "policy_roles": sorted(enabled_roles),
          "expected_entry_count":len(expected),"selected_entry_count":len(selected),
          "selected":[selected[k] for k in sorted(selected)],"missing":missing,"unexpected":unexpected,"identity_mismatches":mismatched}

def _candidate_registry_from_env(env:dict[str,Any]|None=None):
  env=_candidate_policy_env(os.environ if env is None else env)
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


def _install_candidate_matmul(x,w,out_f,in_f,admission):
  from extra.qk.runtime_specs import candidate_storage_kind
  register_route = candidate_storage_kind(admission.normalized_payload) == "global_register_resident"
  if register_route:
    artifact_text = os.environ.get(_PURE_REGISTER_COMPILE_ARTIFACT_JSON_ENV)
    try: artifact = json.loads(artifact_text) if artifact_text is not None else None
    except json.JSONDecodeError as exc: raise ValueError(f"{_PURE_REGISTER_COMPILE_ARTIFACT_JSON_ENV} is not valid JSON: {exc}") from exc
    workload = admission.normalized_payload["workload"]
    from extra.qk.prefill.pure_register_evaluation_gate import runtime_compile_resource_eligibility
    eligibility = runtime_compile_resource_eligibility({"canonical_identity": admission.canonical_identity}, artifact,
      profile=workload["profile"], role=workload["role"], shape=(512,out_f,in_f), target=workload["target"])
    if not eligibility["passed"]: return None
  from tinygrad.codegen.opt import Opt, OptOps
  import tinygrad.codegen.opt.postrange as pr
  key=(frozenset({512,out_f}),in_f)
  existing=(pr._WARMSTART_CANDIDATE_CONTEXTS or {}).get(key)
  if existing is not None and existing.canonical_identity != admission.canonical_identity:
    raise ValueError(f"candidate warmstart key collision for {key!r}")
  pr._WARMSTART_OPTS={**(pr._WARMSTART_OPTS or {}),key:(Opt(OptOps.TC,0,(-1,2,1)),)}
  pr._WARMSTART_CANDIDATE_CONTEXTS={**(pr._WARMSTART_CANDIDATE_CONTEXTS or {}),key:admission.context}
  a=x.reshape(512,in_f).cast(dtypes.float16).contiguous(); bt=w.cast(dtypes.float16).contiguous()
  return (a @ bt.transpose()).reshape(*x.shape[:-1],out_f)


def route_pf16_graph_gemm(lin, x: Tensor, w: Tensor | None = None) -> Tensor | None:
  # `w` (optional): an explicit fp16 weight to GEMM against. Callers may pass an unstored
  # `lin.weight.cast(fp16).contiguous()` from inside a layer-sized TinyJit, so replay reuses the graph-owned fp16
  # dequant scratch across blocks instead of pinning resident `lin._pf16_w` for every block.
  # NOTE: the gfx1100 arch restriction for default-on lives in model.PREFILL_GRAPH_GEMM (computed once at import);
  # it is NOT checked here because Device[...] access is disallowed during JIT capture (ALLOW_DEVICE_USAGE). The
  # T==512 / tile-divisible / bias / role guards below restrict to the validated dense prefill shapes; everything
  # else silently falls back to the normal PREFILL_V2 matmul.
  if w is None: w = getattr(lin, "_pf16_w", None)
  b = getattr(lin, "bias", None)
  if w is None or b is not None or x.ndim < 2: return None
  if not isinstance(x.shape[-2], int) or not isinstance(x.shape[-1], int): return None
  if x.shape[-2] != 512: return None
  out_f, in_f = w.shape
  if in_f != x.shape[-1]: return None
  role = getattr(lin, "_prefill_graph_role", None)
  registry=_candidate_registry_from_env()
  profile=getattr(lin,"_prefill_model_profile",None) or os.environ.get("BOLTBEAM_MODEL_PROFILE")
  if profile is None and registry is not None:
    # Backward-compatible data-derived default for a single-profile artifact; never bake a model ID into dispatch.
    profiles={entry.exact_key[0] for entry in registry.candidate_set.entries}
    if len(profiles) == 1: profile=next(iter(profiles))
  target={"backend":"AMD","arch":"gfx1100","wave_size":32}
  admission=None if registry is None or role is None or profile is None else registry.get(profile,role,(512,out_f,in_f),target)
  if admission is not None:
    result = _install_candidate_matmul(x,w,out_f,in_f,admission)
    if result is None: return None
    setattr(lin,"_prefill_full_kernel_candidate_identity",admission.canonical_identity)
    _record_candidate_route(admission)
    return result
  # The promoted candidate set is a production allowlist, not a hint. Unsupported roles/shapes return to the ordinary
  # Tensor path; they must never fall through into the slower raw/composed research emitters merely because graph-GEMM
  # is enabled for the promoted exact workloads.
  return None
