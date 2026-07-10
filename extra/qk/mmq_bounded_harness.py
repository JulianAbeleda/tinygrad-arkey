#!/usr/bin/env python3
"""Bounded Q4_K/Q8_1 MMQ harness for the 14B ffn_gate_up candidate.

This is an opt-in diagnostic harness only. It does not bind the model prefill
route. The reference backend is runnable now; the atom backend is deliberately
fail-loud until the hand MMQ atom lands.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import pathlib
import sys
import time
from typing import Any, Literal

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from tinygrad import Tensor, dtypes

from extra.qk.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS, q8_1_quantize
from extra.qk.mmq_q4k_q8_reference import Q4KQ81MMQTileSpec, describe_q4k_q8_1_mmq_tile, q4k_q8_1_mmq_tile_reference

ROLE = "ffn_gate_up"
M = 512
N = 17408
K = 5120
QUANT = "Q4_K"
ACTIVATION = "Q8_1"
CANDIDATE_ROUTE_ID = "prefill_14b_q4k_q8_1_hybrid_mmq_atom"
PUBLIC_LABEL = "hybrid_machine_search_mmq"
PRIMITIVE_CLASS = "compiler_primitive_spec_owned__hand_mmq_backend_atom"
COMPARATOR_ID = "direct_packed"


class MMQAtomUnavailableError(RuntimeError):
  pass


@dataclass(frozen=True)
class BoundedMMQConfig:
  m_tile: int = 16
  n_tile: int = 16
  k_groups: int = 8
  m_tiles: int = 1
  n_tiles: int = 1
  warmups: int = 0
  rounds: int = 1
  seed: int = 20260710
  backend: Literal["reference", "atom"] = "reference"

  @property
  def bounded_m(self) -> int:
    return self.m_tile * self.m_tiles

  @property
  def bounded_n(self) -> int:
    return self.n_tile * self.n_tiles

  @property
  def bounded_k(self) -> int:
    return self.k_groups * Q8_1_BLOCK_ELEMS

  def validate(self) -> None:
    if self.backend not in ("reference", "atom"): raise ValueError(f"unknown backend={self.backend!r}")
    if min(self.m_tile, self.n_tile, self.k_groups, self.m_tiles, self.n_tiles) <= 0:
      raise ValueError("tile sizes, tile counts, and k_groups must be positive")
    if self.warmups < 0 or self.rounds < 1: raise ValueError("warmups >= 0 and rounds >= 1 are required")
    if self.bounded_m > M or self.bounded_n > N or self.bounded_k > K:
      raise ValueError(f"bounded shape {(self.bounded_m, self.bounded_n, self.bounded_k)} exceeds role shape {(M, N, K)}")
    if self.bounded_k % Q4_K_BLOCK_ELEMS:
      raise ValueError(f"bounded K={self.bounded_k} must be Q4_K block aligned")


def candidate_metadata(config: BoundedMMQConfig | None = None) -> dict[str, Any]:
  cfg = config or BoundedMMQConfig()
  cfg.validate()
  return {
    "role": ROLE,
    "M": M,
    "N": N,
    "K": K,
    "quant": QUANT,
    "activation": ACTIVATION,
    "candidate_route_id": CANDIDATE_ROUTE_ID,
    "public_label": PUBLIC_LABEL,
    "primitive_class": PRIMITIVE_CLASS,
    "comparator_id": COMPARATOR_ID,
    "rollback": COMPARATOR_ID,
    "backend": cfg.backend,
    "bounded_shape": {"M": cfg.bounded_m, "N": cfg.bounded_n, "K": cfg.bounded_k},
    "tile": {"M": cfg.m_tile, "N": cfg.n_tile, "K_groups": cfg.k_groups},
  }


def _finite_q4k_bytes(n:int, k:int, seed:int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  if k % Q4_K_BLOCK_ELEMS: raise ValueError(f"k={k} must be Q4_K block aligned")
  nblocks = n * k // Q4_K_BLOCK_ELEMS
  raw = rng.integers(0, 256, size=(nblocks, Q4_K_BLOCK_BYTES), dtype=np.uint8)
  raw[:, 0:2] = (rng.standard_normal(nblocks).astype(np.float32) * 0.05).astype(np.float16).view(np.uint8).reshape(nblocks, 2)
  raw[:, 2:4] = (rng.standard_normal(nblocks).astype(np.float32) * 0.05).astype(np.float16).view(np.uint8).reshape(nblocks, 2)
  return raw.reshape(n, k // Q4_K_BLOCK_ELEMS, Q4_K_BLOCK_BYTES)


def _q8_inputs(m:int, k:int, seed:int) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  x = Tensor(rng.standard_normal((m, k)).astype(np.float32)).realize()
  xq, xscales = q8_1_quantize(x.cast(dtypes.float32))
  return xq.numpy().reshape(m, k), xscales.numpy().reshape(m, k // Q8_1_BLOCK_ELEMS)


def _source_hash() -> str:
  data = pathlib.Path(__file__).read_bytes()
  return hashlib.sha256(data).hexdigest()[:16]


def _run_reference_tile(q4k_bytes:np.ndarray, xq:np.ndarray, xscales:np.ndarray, spec:Q4KQ81MMQTileSpec) -> np.ndarray:
  return q4k_q8_1_mmq_tile_reference(q4k_bytes, xq, xscales, spec)


def _run_atom_tile(q4k_bytes:np.ndarray, xq:np.ndarray, xscales:np.ndarray, spec:Q4KQ81MMQTileSpec) -> np.ndarray:
  try:
    from extra.qk.mmq_q4k_q8_atom import run_q4k_q8_1_mmq_tile
  except Exception as exc:
    raise MMQAtomUnavailableError(
      f"{CANDIDATE_ROUTE_ID} selected but extra.qk.mmq_q4k_q8_atom.run_q4k_q8_1_mmq_tile is unavailable"
    ) from exc
  return np.asarray(run_q4k_q8_1_mmq_tile(q4k_bytes, xq, xscales, spec), dtype=np.float32)


def run_bounded_harness(config: BoundedMMQConfig) -> dict[str, Any]:
  config.validate()
  q4k_bytes = _finite_q4k_bytes(config.bounded_n, config.bounded_k, config.seed)
  xq, xscales = _q8_inputs(config.bounded_m, config.bounded_k, config.seed + 1)
  runner = _run_reference_tile if config.backend == "reference" else _run_atom_tile

  specs = [
    describe_q4k_q8_1_mmq_tile(role=ROLE, m=config.bounded_m, n=config.bounded_n, k=config.bounded_k,
                               m0=mt * config.m_tile, n0=nt * config.n_tile, m_tile=config.m_tile,
                               n_tile=config.n_tile, k_groups=config.k_groups)
    for mt in range(config.m_tiles) for nt in range(config.n_tiles)
  ]

  reference_tiles = [_run_reference_tile(q4k_bytes, xq, xscales, spec) for spec in specs]
  for _ in range(config.warmups):
    for spec in specs: runner(q4k_bytes, xq, xscales, spec)

  samples_ms: list[float] = []
  last_tiles: list[np.ndarray] = []
  for _ in range(config.rounds):
    t0 = time.perf_counter()
    last_tiles = [runner(q4k_bytes, xq, xscales, spec) for spec in specs]
    samples_ms.append((time.perf_counter() - t0) * 1000.0)

  max_abs = max(float(np.max(np.abs(got - ref))) for got, ref in zip(last_tiles, reference_tiles)) if last_tiles else 0.0
  ok = max_abs <= 2e-5
  return {
    "schema": "q4k-q8-1-mmq-bounded-harness.v1",
    "metadata": candidate_metadata(config),
    "status": "PASS" if ok else "FAIL",
    "correctness": {"max_abs": max_abs, "atol": 2e-5, "tiles": len(specs)},
    "timing": {
      "samples_ms": samples_ms,
      "min_ms": min(samples_ms),
      "median_ms": float(np.median(np.asarray(samples_ms, dtype=np.float64))),
      "comparator_id": COMPARATOR_ID,
      "comparator_status": "named_not_measured",
    },
    "artifacts": {"harness_source_hash": _source_hash(), "emitted_binary_hash": None},
    "blockers": [] if config.backend == "reference" else ["atom backend is diagnostic-only until the hand atom exists"],
  }


def _parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description="Bounded Q4_K/Q8_1 MMQ harness for 14B ffn_gate_up")
  ap.add_argument("--backend", choices=("reference", "atom"), default="reference")
  ap.add_argument("--m-tile", type=int, default=16)
  ap.add_argument("--n-tile", type=int, default=16)
  ap.add_argument("--k-groups", type=int, default=8)
  ap.add_argument("--m-tiles", type=int, default=1)
  ap.add_argument("--n-tiles", type=int, default=1)
  ap.add_argument("--warmups", type=int, default=0)
  ap.add_argument("--rounds", type=int, default=1)
  ap.add_argument("--seed", type=int, default=20260710)
  return ap.parse_args()


def main() -> None:
  args = _parse_args()
  report = run_bounded_harness(BoundedMMQConfig(m_tile=args.m_tile, n_tile=args.n_tile, k_groups=args.k_groups,
                                               m_tiles=args.m_tiles, n_tiles=args.n_tiles, warmups=args.warmups,
                                               rounds=args.rounds, seed=args.seed, backend=args.backend))
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
