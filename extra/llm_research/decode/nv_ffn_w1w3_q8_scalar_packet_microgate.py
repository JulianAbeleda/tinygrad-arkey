#!/usr/bin/env python3
"""Arithmetic gate for the 1024-thread scalar W1/W3 -> Q8_1 producer.

Control is the current promoted pair (fused16 w1w3 -> Q4-down with in-kernel
residual add). Candidate replaces those two kernels with one scalar-packet Q8_1
producer and the four-warp Q4/Q8 DP4A consumer, also with in-kernel residual
add. The producer is compared packet-for-packet against the existing
``pack_q8_1_private`` oracle; the consumer is checked on sampled rows. Included
cost is measured by the full-token qualification/timing harness, not here.
"""
from __future__ import annotations

import argparse, json
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_w1w3_kernel
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from tinygrad.llm.q4k_ffn_down_mmvq import BLOCKS_PER_WARP, emit_ffn_w1w3_q8_scalar_packet, emit_four_warp_direct
from tinygrad.uop.ops import UOp
from extra.llm_research.decode.ffn_q8_cooperative_producer import (
  DOWN_K, DOWN_ROWS, K, ROWS, pack_q8_1_private, q4_q8_ffn_down_row_reference)
from extra.llm_research.decode.route_class_numerics import _make_q4k_words


Q8_WORDS = 3456
PACKETS = ROWS // 32
TOPOLOGY = {"control": ["q4k_g3_lanemap_gemv_w1w3fused16_12288_4096", "q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288"],
  "candidate": ["ffn_w1w3_q8_scalar_packet_12288_4096", "q4k_q8_mmvq_direct_4096_12288_epi_ffnresadd"]}


def _p(name, fn):
  return KernelProgram("research.nv_ffn_w1w3_q8_scalar_packet", name, KernelProgramProvenance.RESEARCH_ONLY, fn)


def _raw_inputs():
  gw, _ = _make_q4k_words(ROWS, K, 202608141)
  uw, _ = _make_q4k_words(ROWS, K, 202608142)
  dw, _ = _make_q4k_words(DOWN_ROWS, DOWN_K, 202608143)
  x = np.random.default_rng(20260814).normal(0, 0.2, K).astype(np.float16)
  h = np.random.default_rng(202608145).normal(0, 0.2, DOWN_ROWS).astype(np.float32)
  return gw, uw, dw, x, h


def _inputs(device, raw=None):
  gw, uw, dw, x, h = _raw_inputs() if raw is None else raw
  return (Tensor(gw, dtype=dtypes.uint32, device=device).contiguous().realize(),
          Tensor(uw, dtype=dtypes.uint32, device=device).contiguous().realize(),
          Tensor(dw, dtype=dtypes.uint32, device=device).contiguous().realize(),
          Tensor(x, dtype=dtypes.float16, device=device).contiguous().realize(),
          Tensor(h, dtype=dtypes.float32, device=device).contiguous().realize())


def correctness():
  dev = Device.DEFAULT
  if not str(dev).startswith("NV"): raise RuntimeError(f"native NV required, got {dev}")
  raw = _raw_inputs(); gw, uw, dw, x, h = _inputs(dev, raw)
  control_w1w3 = _p("control_w1w3", q4k_g3_lanemap_gemv_w1w3_kernel(ROWS, K, "scalar", store_fp16=True))

  # The scalar-packet producer must quantize exactly the fused16 z it replaced.
  z = execute_research_program(Tensor.empty((ROWS,), dtype=dtypes.float16, device=dev), gw, uw, x, program=control_w1w3).realize()
  packed = _candidate_packed(gw, uw, x)
  cand = _candidate_rows(packed, dw, h)
  Device[dev].synchronize()

  z_np, packed_np, cand_np = z.numpy().astype(np.float16), packed.numpy(), cand.numpy()
  expected_packed = pack_q8_1_private(z_np)
  mismatches = []
  for p in range(PACKETS):
    payload = packed_np[p*8:p*8+8]; reference = expected_packed[p*8:p*8+8]
    d_ok = (packed_np[3072+p] & 0xffff) == (expected_packed[3072+p] & 0xffff)
    s_ok = (packed_np[3072+p] >> 16) == (expected_packed[3072+p] >> 16)
    if not (np.array_equal(payload, reference) and d_ok and s_ok):
      mismatches.append({"packet": p, "payload_equal": bool(np.array_equal(payload, reference)),
        "d_equal": bool(d_ok), "s_equal": bool(s_ok)})
  provider_rows = [{"packet": p, "payload_equal": bool(np.array_equal(packed_np[p*8:p*8+8], expected_packed[p*8:p*8+8])),
    "d_equal": bool((packed_np[3072+p] & 0xffff) == (expected_packed[3072+p] & 0xffff)),
    "s_equal": bool((packed_np[3072+p] >> 16) == (expected_packed[3072+p] >> 16))} for p in (0, 191, 383)]
  consumer_rows = [{"row": r, "got": float(cand_np[r]), "oracle": float(q4_q8_ffn_down_row_reference(raw[2], packed_np, r)),
    "abs_error": float(abs(float(cand_np[r] - float(raw[4][r])) - float(q4_q8_ffn_down_row_reference(raw[2], packed_np, r))))}
    for r in (0, 1, 2047, 4095)]
  provider_pass = not mismatches
  consumer_pass = all(r["abs_error"] <= 2e-4 for r in consumer_rows)
  return {"schema": "tinygrad.nv_ffn_w1w3_q8_scalar_packet.microgate.v1", "mode": "correctness", "topology": TOPOLOGY,
    "finite": bool(np.isfinite(cand_np).all()), "provider_oracle": provider_rows, "provider_oracle_pass": provider_pass,
    "provider_all_packets_equal": provider_pass, "provider_packet_mismatches": mismatches,
    "consumer_oracle": consumer_rows, "consumer_oracle_pass": consumer_pass, "gate": "PASS" if provider_pass and consumer_pass else "FAIL"}


def _candidate_packed(gw, uw, x):
  dev = Device.DEFAULT
  provider = _p("candidate_provider", emit_ffn_w1w3_q8_scalar_packet())
  return execute_research_program(Tensor.empty((Q8_WORDS,), dtype=dtypes.uint32, device=dev), gw, uw, x, program=provider).realize()


def _candidate_rows(packed, dw, h):
  dev = Device.DEFAULT
  consumer = _p("candidate_consumer", emit_four_warp_direct(UOp.const(dtypes.weakint, BLOCKS_PER_WARP), resadd=True))
  return execute_research_program(Tensor.empty((DOWN_ROWS,), dtype=dtypes.float32, device=dev), dw, packed, h, program=consumer).realize()


if __name__ == "__main__":
  ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=("correctness",), required=True)
  ap.add_argument("--out")
  ns = ap.parse_args()
  result = correctness()
  print(json.dumps(result, indent=2))
  if ns.out: open(ns.out, "w").write(json.dumps(result, indent=2) + "\n")
