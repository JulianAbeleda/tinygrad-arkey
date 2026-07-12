#!/usr/bin/env python3
"""Single-request JSON worker for dynamic full-kernel candidate admission."""
from __future__ import annotations

import json, os, platform, subprocess, sys
from typing import Any

from extra.qk.runtime_specs import (FULL_KERNEL_CANDIDATE_SCHEMA, GFX1100_SINGLE_BUFFER_CAPABILITY,
                                    FullKernelAdmissionError, admit_full_kernel_candidate)

PROTOCOL = "tinygrad.full_kernel_candidate_worker.v1"

class WorkerRequestError(ValueError):
  def __init__(self, code:str, message:str): self.code=code; super().__init__(message)

def _environment() -> dict[str, Any]:
  try:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    dirty = bool(subprocess.check_output(["git","status","--short"],cwd=root,text=True,stderr=subprocess.DEVNULL).strip())
  except (OSError,subprocess.SubprocessError): revision,dirty=None,None
  return {"python":platform.python_version(),"git_revision":revision,"git_dirty":dirty,
          "capability":{"capability_id":GFX1100_SINGLE_BUFFER_CAPABILITY.capability_id,
                        "backend":GFX1100_SINGLE_BUFFER_CAPABILITY.backend,"arch":GFX1100_SINGLE_BUFFER_CAPABILITY.arch,
                        "wave_size":GFX1100_SINGLE_BUFFER_CAPABILITY.wave_size,
                        "max_lds_bytes":GFX1100_SINGLE_BUFFER_CAPABILITY.max_lds_bytes}}

def _request(row:Any) -> tuple[str,dict[str,Any]]:
  if not isinstance(row,dict): raise WorkerRequestError("malformed_request","request must be an object")
  if row.get("protocol") != PROTOCOL: raise WorkerRequestError("protocol_mismatch",f"protocol must be {PROTOCOL}")
  request_id=row.get("request_id")
  if not isinstance(request_id,str) or not request_id: raise WorkerRequestError("invalid_request_id","request_id must be non-empty")
  action=row.get("action")
  if action != "admit": raise WorkerRequestError("unsupported_action",f"unsupported action {action!r}")
  candidate=row.get("candidate" ); workload=row.get("workload")
  if not isinstance(candidate,dict) or set(candidate) != {"payload","canonical_identity"}:
    raise WorkerRequestError("malformed_candidate","candidate requires payload and canonical_identity")
  if not isinstance(workload,dict) or set(workload) != {"profile","role","shape","target"}:
    raise WorkerRequestError("malformed_workload","workload requires profile, role, shape, target")
  if not isinstance(workload["shape"],list) or len(workload["shape"]) != 3:
    raise WorkerRequestError("malformed_workload","workload.shape must be [M,N,K]")
  admission=admit_full_kernel_candidate(candidate["payload"],candidate["canonical_identity"],profile=workload["profile"],
    role=workload["role"],shape=tuple(workload["shape"]),target=workload["target"])
  return request_id,{"status":"admitted","canonical_identity":admission.canonical_identity,
    "capability_id":admission.capability.capability_id,
    "candidate_schema":FULL_KERNEL_CANDIDATE_SCHEMA,"plan":{"tile":list(admission.geometry.tile),"waves":list(admission.geometry.waves),
      "threads":admission.geometry.threads,"active_lds_bytes":admission.active_lds_bytes,
      "subtiles":[admission.plan.subtiles_m,admission.plan.subtiles_n],"k_substeps":admission.plan.k_substeps}}

def process(row:Any) -> dict[str,Any]:
  request_id = row.get("request_id") if isinstance(row,dict) and isinstance(row.get("request_id"),str) else None
  try:
    request_id,result=_request(row)
    return {"protocol":PROTOCOL,"request_id":request_id,"ok":True,"result":result,"environment":_environment()}
  except FullKernelAdmissionError as exc:
    return {"protocol":PROTOCOL,"request_id":request_id,"ok":False,
            "error":{"class":"admission","code":exc.code,"message":str(exc)},"environment":_environment()}
  except WorkerRequestError as exc:
    return {"protocol":PROTOCOL,"request_id":request_id,"ok":False,
            "error":{"class":"request","code":exc.code,"message":str(exc)},"environment":_environment()}
  except (TypeError,ValueError,KeyError) as exc:
    return {"protocol":PROTOCOL,"request_id":request_id,"ok":False,
            "error":{"class":"request","code":"malformed_request","message":str(exc)},"environment":_environment()}

def main() -> int:
  try: row=json.loads(sys.stdin.read())
  except json.JSONDecodeError as exc:
    response={"protocol":PROTOCOL,"request_id":None,"ok":False,
              "error":{"class":"request","code":"malformed_json","message":str(exc)},"environment":_environment()}
  else: response=process(row)
  sys.stdout.write(json.dumps(response,sort_keys=True,separators=(",",":"))+"\n")
  return 0

if __name__ == "__main__": raise SystemExit(main())
