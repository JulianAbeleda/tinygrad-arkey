#!/usr/bin/env python3
"""Deterministic guarded qualification for one exact 14B Q6_K direct-packed fallback row."""
from __future__ import annotations

import argparse, hashlib, json, platform, subprocess, sys, time
from pathlib import Path
from typing import Any

import numpy as np

from extra.qk.layout import Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS
from extra.qk.prefill.current_prefill_execution_adapter import ADAPTER_ID, CurrentPrefillAdapter
from extra.qk.prefill.execution_bridge_contracts import (CorrectnessProtocol, ExecutionRequest, GuardProtocol,
  TimingProtocol, TransportPlan, canonical_digest)
from extra.qk.prefill.operand_path_execution_worker import AdapterRegistry, execute
from extra.qk.prefill.six_row_policy_artifact import DIRECT_PACKED_ROUTE, Q6_EVIDENCE_SCHEMA, TARGET
from extra.qk.runtime_specs import (FullKernelCandidateSet, capability_transport, full_kernel_candidate_capability,
  full_kernel_workload)

DEFAULT_INVENTORY = "bench/prefill-pure-full-kernel/qwen3-14b-mixed-quant-candidate-inventory-v1.json"
ROLE_SHAPES = {"attn_kv": (512, 1024, 5120), "ffn_down": (512, 5120, 17408)}


def _health() -> dict[str, Any]:
  command = ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--showuse", "--showtemp"]
  started = time.monotonic()
  try: completed = subprocess.run(command, text=True, capture_output=True, timeout=15, check=False)
  except (OSError, subprocess.TimeoutExpired) as exc:
    return {"ok":False, "command":command, "error":f"{type(exc).__name__}: {exc}",
            "elapsed_seconds":time.monotonic()-started}
  return {"ok":completed.returncode == 0, "command":command, "returncode":completed.returncode,
          "stdout":completed.stdout, "stderr":completed.stderr, "elapsed_seconds":time.monotonic()-started}


def make_finite_q6k_bytes(n: int, k: int, seed: int) -> np.ndarray:
  if k % Q6_K_BLOCK_ELEMS: raise ValueError("Q6_K K must be block aligned")
  rng = np.random.default_rng(seed)
  raw = rng.integers(0, 256, size=(n, k//Q6_K_BLOCK_ELEMS, Q6_K_BLOCK_BYTES), dtype=np.uint8)
  raw[..., 192:208] = rng.integers(-8, 9, size=raw[..., 192:208].shape, dtype=np.int8).view(np.uint8)
  raw[..., 208:210] = (rng.standard_normal(raw.shape[:2]).astype(np.float32)*0.025).astype("<f2").view(np.uint8).reshape(*raw.shape[:2], 2)
  return raw


def q6k_dequantize_selected_positions(raw: np.ndarray, positions: np.ndarray) -> np.ndarray:
  """Decode selected K positions for every N row from canonical Q6_K bytes."""
  raw, positions = np.asarray(raw), np.asarray(positions, dtype=np.int64)
  if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[2] != Q6_K_BLOCK_BYTES:
    raise ValueError(f"raw must be uint8 [N,K/256,{Q6_K_BLOCK_BYTES}]")
  if positions.ndim != 1 or np.any(positions < 0) or np.any(positions >= raw.shape[1]*Q6_K_BLOCK_ELEMS):
    raise ValueError("positions are outside the packed Q6_K tensor")
  block, within = np.divmod(positions, Q6_K_BLOCK_ELEMS)
  group, pos = np.divmod(within, 16)
  chosen = raw[:, block, :]
  half, pgrp = group//8, group%8
  ql_idx = half*64 + (pgrp%4)*16 + pos
  qh_idx = 128 + half*32 + (pgrp%2)*16 + pos
  cols = np.arange(positions.size)
  ql = (chosen[:, cols, ql_idx] >> np.where(pgrp >= 4, 4, 0)) & 0xf
  qh = ((chosen[:, cols, qh_idx] >> ((pgrp//2)*2)) & 3) << 4
  q = (ql | qh).astype(np.float32) - 32.0
  scale = chosen[:, cols, 192+group].view(np.int8).astype(np.float32)
  d = chosen[:, :, 208:210].copy().view("<f2").reshape(raw.shape[0], -1).astype(np.float32)
  return d*q*scale


def _fixture(path: Path, shape: tuple[int, int, int], seed: int) -> dict[str, Any]:
  m, n, k = shape
  raw = make_finite_q6k_bytes(n, k, seed)
  # M=512 covers each of the 256 physical positions twice, while retaining an inexpensive exact reference.
  positions = np.arange(m, dtype=np.int64) % Q6_K_BLOCK_ELEMS
  positions += ((np.arange(m, dtype=np.int64)//Q6_K_BLOCK_ELEMS) * max(1, k//2//Q6_K_BLOCK_ELEMS))*Q6_K_BLOCK_ELEMS
  coefficients = (((np.arange(m) * 17 + seed * 3) % 15) - 7).astype(np.float32) / 8.0
  a = np.zeros((m, k), dtype=np.float16); a[np.arange(m), positions] = coefficients.astype(np.float16)
  selected = q6k_dequantize_selected_positions(raw, positions)
  reference = (selected.T * coefficients[:, None]).astype(np.float16)
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(path, a=a, b=raw.reshape(-1).copy().view(np.uint16), reference=reference)
  return {"seed":seed, "coverage":"all_256_within_block_positions_twice", "selected_positions_sha256":
          hashlib.sha256(positions.tobytes()).hexdigest(), "coefficients_sha256":hashlib.sha256(coefficients.tobytes()).hexdigest()}


def _select(inventory: dict[str, Any], role: str, shape: tuple[int, int, int]):
  entries = FullKernelCandidateSet.from_json(inventory["candidate_sets"]["Q6_K"]).entries
  found = [entry for entry in entries if (full_kernel_workload(entry.payload).role,
    full_kernel_workload(entry.payload).shape) == (role, shape)]
  bindings = [row for row in inventory["bindings"] if row["inventory_key"]["quant_format"] == "Q6_K" and
              row["inventory_key"]["role"] == role and tuple(row["inventory_key"]["shape"][x] for x in ("m","n","k")) == shape]
  if len(found) != 1 or len(bindings) != 1: raise ValueError(f"expected one Q6_K candidate and binding, got {len(found)}/{len(bindings)}")
  return found[0], bindings[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
  inventory = json.loads(Path(args.inventory).read_text()); shape = ROLE_SHAPES[args.role]
  entry, binding = _select(inventory, args.role, shape)
  before = _health()
  if not before["ok"]: raise RuntimeError("GPU health preflight failed")
  npz = Path(args.workdir)/f"q6-{args.role}-input.npz"; fixture = _fixture(npz, shape, args.seed)
  workload = full_kernel_workload(entry.payload); schedule_digest = canonical_digest(entry.payload["schedule"], "schedule")
  request = ExecutionRequest(experiment_id="q6-direct-packed-qualification", candidate_id=entry.canonical_identity,
    comparator_id="self_reference", workload_digest=canonical_digest({"role":args.role,"shape":list(shape),"quant_format":"Q6_K"}, "workload"),
    schedule_digest=schedule_digest,
    transport_plan=TransportPlan(capability_transport(full_kernel_candidate_capability(entry.payload)), schedule_digest),
    target_context={"session_id":f"q6-fallback-{args.role}-seed{args.seed}", "target_id":"AMD:gfx1100:wave32",
      "system_snapshot_id":platform.node(), "workload":{"role":args.role,"shape":list(shape)}},
    compiler_context={"adapter_id":ADAPTER_ID, "candidate_payload":entry.payload,
      "canonical_identity":entry.canonical_identity, "input_npz":str(npz.resolve())},
    correctness=CorrectnessProtocol("full_output", atol=args.atol, rtol=args.rtol), guard=GuardProtocol(args.timeout_ms),
    timing=TimingProtocol(args.warmups, args.rounds, 0, same_session=True))
  registry = AdapterRegistry(); registry.register(ADAPTER_ID, CurrentPrefillAdapter())
  result = execute(request, registry=registry).to_dict(); after = _health()
  phases = result.get("phases", ())
  qualified = after["ok"] and phases and all(row.get("status") == "passed" for row in phases)
  workload_row = {"phase":"prefill", "role":args.role, "quant_format":"Q6_K",
    "shape":{"m":shape[0],"n":shape[1],"k":shape[2]}, "target":dict(TARGET)}
  evidence = {"schema":Q6_EVIDENCE_SCHEMA, "status":"qualified" if qualified else "failed",
    "route_id":DIRECT_PACKED_ROUTE, "canonical_identity":binding["canonical_identity"], "workload":workload_row,
    "candidate_identity":entry.canonical_identity, "fixture":fixture, "measurement_definition":{
      "scope":"full role output", "warmups":args.warmups, "rounds":args.rounds, "statistic":"median",
      "performance_claim":False}, "health":{"before":before,"after":after}, "result":result}
  evidence["qualification_identity"] = "q6_direct_packed:sha256:" + hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
  return evidence


def main() -> int:
  parser = argparse.ArgumentParser(); parser.add_argument("--inventory", default=DEFAULT_INVENTORY)
  parser.add_argument("--role", choices=tuple(ROLE_SHAPES), required=True); parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--timeout-ms", type=int, default=120000); parser.add_argument("--warmups", type=int, default=1)
  parser.add_argument("--rounds", type=int, default=3); parser.add_argument("--atol", type=float, default=0.02)
  parser.add_argument("--rtol", type=float, default=0.002); parser.add_argument("--workdir", default="/tmp/tinygrad-q6-fallback")
  parser.add_argument("--output", required=True); args = parser.parse_args()
  try: report = run(args)
  except BaseException as exc:
    report = {"schema":Q6_EVIDENCE_SCHEMA, "status":"blocked", "route_id":DIRECT_PACKED_ROUTE,
      "blocker":{"type":type(exc).__name__,"message":str(exc)}, "health_after_blocker":_health()}
  target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
  print(json.dumps(report, sort_keys=True, separators=(",", ":")))
  return 0 if report.get("status") == "qualified" else 1


if __name__ == "__main__": raise SystemExit(main())
