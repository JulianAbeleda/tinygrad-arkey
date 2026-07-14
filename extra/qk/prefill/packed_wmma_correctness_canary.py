"""Deterministic full-output correctness canary for generated packed-weight WMMA prefill.

The activation has one nonzero K element per row, making the independent fp16 reference cheap while still checking
every output element, many K positions, packed metadata/payload decoding, N-row addressing, and the complete WMMA
epilogue. GPU execution is always delegated to the shared spawn-isolated guarded executor.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

from extra.qk.prefill.current_prefill_execution_adapter import (build_current_prefill_bundle,
  prepare_current_prefill_compile)
from extra.qk.prefill.guarded_execution import GuardPolicy
from extra.qk.prefill.host_safety_canary import make_tiny_health_probe
from extra.qk.prefill.isolated_guarded_executor import (ExecutionRequest, make_tinygrad_bundle_builder,
  run_isolated_guarded_execution)
from extra.qk.route_manifest import promoted_prefill_candidate_policy
from extra.qk.runtime_specs import (derive_packed_weight_candidate, full_kernel_workload,
                                    rebind_full_kernel_workload)
from extra.qk.model_profiles import profile_by_id

# Backward-compatible default fixture. Artifact generation and execution below are shape-driven.
M, N, K = 512, 4096, 4096
DEFAULT_PROFILE = "qwen3_8b_q4k_m_gfx1100"
DEFAULT_ROLE = "attn_qo"


def candidate_payload(profile:str=DEFAULT_PROFILE, role:str=DEFAULT_ROLE, candidate_set_path:str|None=None) -> dict:
  """Resolve an exact candidate or legally rebind the same-role schedule template to profile facts."""
  path = candidate_set_path or promoted_prefill_candidate_policy()["candidate_set_path"]
  candidate_set = json.loads(Path(path).read_text())
  payloads = [row["payload"] for row in candidate_set["entries"]]
  if exact := next((p for p in payloads if p["workload"]["profile"] == profile and p["workload"]["role"] == role), None):
    return exact
  role_shape = profile_by_id(profile).role_shape(role)
  template = next((p for p in payloads if p["workload"]["role"] == role), None)
  if template is None: raise ValueError(f"candidate set has no schedule template for role {role!r}")
  return rebind_full_kernel_workload(template, profile=profile, role=role, shape=role_shape.mnk).to_json()["payload"]


def _activation(shape:tuple[int,int,int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  m, _, k = shape
  rows = np.arange(m, dtype=np.int64)
  k_positions = (rows * 251 + 17) % k
  coefficients = (rows % 4 + 1).astype(np.float16)
  activation = np.zeros((m, k), dtype=np.float16)
  activation[rows, k_positions] = coefficients
  return activation, k_positions, coefficients


def _half_bytes(values:np.ndarray) -> np.ndarray:
  return np.ascontiguousarray(values.astype(np.float16)).view(np.uint8).reshape(-1, 2)


def _q4_blocks(n:int, k:int) -> np.ndarray:
  block_count = n * k // 256
  block = np.arange(block_count, dtype=np.int64)
  raw = np.zeros((block_count, 144), dtype=np.uint8)
  raw[:, 0:2], raw[:, 2:4] = _half_bytes((block % 7 + 1) / 64), _half_bytes((block % 3) / 128)
  groups = np.arange(8, dtype=np.int64)[None, :]
  scales = ((block[:, None] + groups) % 8 + 1).astype(np.uint8)
  minima = ((block[:, None] + groups * 2) % 4).astype(np.uint8)
  sb = np.zeros((block_count, 12), dtype=np.uint8)
  for group in range(4):
    sb[:, group] = (scales[:, group] & 63) | ((scales[:, group+4] >> 4) << 6)
    sb[:, 4+group] = (minima[:, group] & 63) | ((minima[:, group+4] >> 4) << 6)
    sb[:, 8+group] = (scales[:, group+4] & 15) | ((minima[:, group+4] & 15) << 4)
  raw[:, 4:16] = sb
  for group in range(8):
    for pos in range(32):
      quant = ((block + group * 3 + pos * 5) % 16).astype(np.uint8)
      raw[:, 16 + (group//2)*32 + pos] |= quant << ((group % 2) * 4)
  return raw


def _q6_blocks(n:int, k:int) -> np.ndarray:
  block_count = n * k // 256
  block = np.arange(block_count, dtype=np.int64)
  raw = np.zeros((block_count, 210), dtype=np.uint8)
  raw[:, 208:210] = _half_bytes((block % 7 + 1) / 64)
  groups = np.arange(16, dtype=np.int64)[None, :]
  scales = ((block[:, None] + groups) % 7 - 3).astype(np.int8)
  scales[scales == 0] = 1
  raw[:, 192:208] = scales.view(np.uint8)
  for group in range(16):
    half, pgroup = group // 8, group % 8
    for pos in range(16):
      quant = ((block + group * 7 + pos * 3) % 64).astype(np.uint8)
      raw[:, half*64 + (pgroup%4)*16 + pos] |= (quant & 15) << (4 if pgroup >= 4 else 0)
      raw[:, 128 + half*32 + (pgroup%2)*16 + pos] |= ((quant >> 4) & 3) << ((pgroup//2)*2)
  return raw


def _decode_selected_q4(raw:np.ndarray, k_position:int, n:int, k:int) -> np.ndarray:
  blocks = np.arange(n) * (k//256) + k_position//256
  within, group, pos = k_position % 256, (k_position % 256)//32, k_position % 32
  d = np.ascontiguousarray(raw[blocks, 0:2]).view(np.float16).reshape(-1).astype(np.float32)
  dmin = np.ascontiguousarray(raw[blocks, 2:4]).view(np.float16).reshape(-1).astype(np.float32)
  if group < 4: scale, minimum = raw[blocks, 4+group] & 63, raw[blocks, 8+group] & 63
  else:
    high = raw[blocks, 12+group-4]
    scale = (high & 15) | ((raw[blocks, 4+group-4] >> 6) << 4)
    minimum = (high >> 4) | ((raw[blocks, 8+group-4] >> 6) << 4)
  quant = (raw[blocks, 16+(group//2)*32+pos] >> ((group%2)*4)) & 15
  return (d * scale.astype(np.float32) * quant.astype(np.float32) - dmin * minimum.astype(np.float32)).astype(np.float16)


def _decode_selected_q6(raw:np.ndarray, k_position:int, n:int, k:int) -> np.ndarray:
  blocks = np.arange(n) * (k//256) + k_position//256
  within, group, pos = k_position % 256, (k_position % 256)//16, k_position % 16
  half, pgroup = group // 8, group % 8
  ql = (raw[blocks, half*64+(pgroup%4)*16+pos] >> (4 if pgroup >= 4 else 0)) & 15
  qh = (raw[blocks, 128+half*32+(pgroup%2)*16+pos] >> ((pgroup//2)*2)) & 3
  quant = (ql | (qh << 4)).astype(np.float32) - 32
  scale = raw[blocks, 192+group].view(np.int8).astype(np.float32)
  d = np.ascontiguousarray(raw[blocks, 208:210]).view(np.float16).reshape(-1).astype(np.float32)
  return (d * quant * scale).astype(np.float16)


def build_artifact(quant_format:str, path:str, shape:tuple[int,int,int]=(M,N,K)) -> dict[str, str | int | list[int]]:
  m, n, k = shape
  activation, k_positions, coefficients = _activation(shape)
  raw = _q4_blocks(n, k) if quant_format == "Q4_K" else _q6_blocks(n, k)
  packed = np.ascontiguousarray(raw).reshape(-1).view(np.uint32 if quant_format == "Q4_K" else np.uint16)
  reference = np.empty((m, n), dtype=np.float16)
  decode = _decode_selected_q4 if quant_format == "Q4_K" else _decode_selected_q6
  for row, (k_position, coefficient) in enumerate(zip(k_positions, coefficients)):
    reference[row] = (decode(raw, int(k_position), n, k).astype(np.float32) * np.float32(coefficient)).astype(np.float16)
  target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
  np.savez(target, a=activation, b=packed, reference=reference)
  return {"quant_format":quant_format, "path":str(target), "packed_bytes":packed.nbytes,
          "reference_elements":reference.size, "shape":list(shape)}


def run_canary(quant_format:str, artifact_path:str, timeout_seconds:float = 30.0, *, base_payload:dict|None=None) -> dict:
  entry = derive_packed_weight_candidate(base_payload or candidate_payload(), quant_format)
  payload = entry.to_json()["payload"]
  workload = full_kernel_workload(payload)
  from extra.qk.prefill.current_prefill_execution_adapter import _arrays
  from extra.qk.prefill.current_prefill_execution_adapter import admit_current_prefill
  admission = admit_current_prefill(payload, entry.canonical_identity)
  inputs, reference = _arrays(artifact_path, workload.shape, admission.context.packed_weight)
  _, evidence = prepare_current_prefill_compile(payload, entry.canonical_identity, device="AMD")
  builder = make_tinygrad_bundle_builder(build=build_current_prefill_bundle, payload=payload,
    canonical_identity=entry.canonical_identity, compile_evidence=evidence, compile_device="AMD", runtime_device="AMD")
  request = ExecutionRequest(inputs, reference,
    GuardPolicy(timeout_seconds=timeout_seconds, check_inputs_unchanged=True, rtol=2e-2, atol=2e-2),
    {"canonical_identity":entry.canonical_identity, "quant_format":quant_format}, np.float16)
  outcome = run_isolated_guarded_execution(builder=builder, request=request,
    health_probe=make_tiny_health_probe(device="AMD"), timeout_seconds=timeout_seconds)
  return outcome.to_dict()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--format", choices=("Q4_K", "Q6_K"), required=True)
  parser.add_argument("--artifact", required=True)
  parser.add_argument("--profile", default=DEFAULT_PROFILE)
  parser.add_argument("--role", default=DEFAULT_ROLE)
  parser.add_argument("--candidate-set", default="")
  parser.add_argument("--build-only", action="store_true")
  parser.add_argument("--timeout", type=float, default=30.0)
  args = parser.parse_args()
  payload = candidate_payload(args.profile, args.role, args.candidate_set or None)
  workload = full_kernel_workload(payload)
  print(json.dumps({"profile":workload.profile, "role":workload.role,
    "artifact":build_artifact(args.format, args.artifact, workload.shape)}, sort_keys=True))
  if not args.build_only:
    print(json.dumps(run_canary(args.format, args.artifact, args.timeout, base_payload=payload), sort_keys=True))


if __name__ == "__main__": main()
