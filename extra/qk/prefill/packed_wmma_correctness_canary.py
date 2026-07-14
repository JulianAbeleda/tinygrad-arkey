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
from extra.qk.runtime_specs import derive_packed_weight_candidate

M, N, K = 512, 4096, 4096


def _base_attn_qo_payload() -> dict:
  candidate_set = json.loads(Path(promoted_prefill_candidate_policy()["candidate_set_path"]).read_text())
  return next(row["payload"] for row in candidate_set["entries"] if row["payload"]["workload"]["role"] == "attn_qo")


def _activation() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  rows = np.arange(M, dtype=np.int64)
  k_positions = (rows * 251 + 17) % K
  coefficients = (rows % 4 + 1).astype(np.float16)
  activation = np.zeros((M, K), dtype=np.float16)
  activation[rows, k_positions] = coefficients
  return activation, k_positions, coefficients


def _half_bytes(values:np.ndarray) -> np.ndarray:
  return np.ascontiguousarray(values.astype(np.float16)).view(np.uint8).reshape(-1, 2)


def _q4_blocks() -> np.ndarray:
  block_count = N * K // 256
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


def _q6_blocks() -> np.ndarray:
  block_count = N * K // 256
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


def _decode_selected_q4(raw:np.ndarray, k_position:int) -> np.ndarray:
  blocks = np.arange(N) * (K//256) + k_position//256
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


def _decode_selected_q6(raw:np.ndarray, k_position:int) -> np.ndarray:
  blocks = np.arange(N) * (K//256) + k_position//256
  within, group, pos = k_position % 256, (k_position % 256)//16, k_position % 16
  half, pgroup = group // 8, group % 8
  ql = (raw[blocks, half*64+(pgroup%4)*16+pos] >> (4 if pgroup >= 4 else 0)) & 15
  qh = (raw[blocks, 128+half*32+(pgroup%2)*16+pos] >> ((pgroup//2)*2)) & 3
  quant = (ql | (qh << 4)).astype(np.float32) - 32
  scale = raw[blocks, 192+group].view(np.int8).astype(np.float32)
  d = np.ascontiguousarray(raw[blocks, 208:210]).view(np.float16).reshape(-1).astype(np.float32)
  return (d * quant * scale).astype(np.float16)


def build_artifact(quant_format:str, path:str) -> dict[str, str | int]:
  activation, k_positions, coefficients = _activation()
  raw = _q4_blocks() if quant_format == "Q4_K" else _q6_blocks()
  packed = np.ascontiguousarray(raw).reshape(-1).view(np.uint32 if quant_format == "Q4_K" else np.uint16)
  reference = np.empty((M, N), dtype=np.float16)
  decode = _decode_selected_q4 if quant_format == "Q4_K" else _decode_selected_q6
  for row, (k_position, coefficient) in enumerate(zip(k_positions, coefficients)):
    reference[row] = (decode(raw, int(k_position)).astype(np.float32) * np.float32(coefficient)).astype(np.float16)
  target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
  np.savez(target, a=activation, b=packed, reference=reference)
  return {"quant_format":quant_format, "path":str(target), "packed_bytes":packed.nbytes,
          "reference_elements":reference.size}


def run_canary(quant_format:str, artifact_path:str, timeout_seconds:float = 30.0) -> dict:
  entry = derive_packed_weight_candidate(_base_attn_qo_payload(), quant_format)
  payload = entry.to_json()["payload"]
  from extra.qk.prefill.current_prefill_execution_adapter import _arrays
  from extra.qk.prefill.current_prefill_execution_adapter import admit_current_prefill
  admission = admit_current_prefill(payload, entry.canonical_identity)
  inputs, reference = _arrays(artifact_path, (M, N, K), admission.context.packed_weight)
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
  parser.add_argument("--build-only", action="store_true")
  parser.add_argument("--timeout", type=float, default=30.0)
  args = parser.parse_args()
  print(json.dumps({"artifact":build_artifact(args.format, args.artifact)}, sort_keys=True))
  if not args.build_only: print(json.dumps(run_canary(args.format, args.artifact, args.timeout), sort_keys=True))


if __name__ == "__main__": main()
