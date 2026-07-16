#!/usr/bin/env python3
"""Fail-closed, compile-only gate for Qwen3-14B Q4_K/Q8_1 WMMA role shapes."""
from __future__ import annotations

import argparse, json, os, subprocess, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

GGUF = Path("/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf")
ROOT = Path(__file__).resolve().parents[2]
RUNTIME_M = 512
ROLE_ORDER = ("attn_kv", "attn_qo", "ffn_down", "ffn_gate_up")
EXPECTED_NK = {"attn_kv": (1024, 5120), "attn_qo": (5120, 5120),
               "ffn_down": (5120, 17408), "ffn_gate_up": (17408, 5120)}
PROTOCOL = "tinygrad.q4k_q8_mmq_uop_role_compile_gate.v1"
PASS = "Q4K_Q8_MMQ_UOP_ROLE_COMPILE_PASS"
BLOCKED = "Q4K_Q8_MMQ_UOP_ROLE_COMPILE_BLOCKED"


@dataclass(frozen=True)
class RoleShape:
  role: str
  m: int
  n: int
  k: int
  quant: str

  @property
  def kernel_name(self) -> str:
    return f"q4k_q8_mmq_uop_wmma_{self.m}x{self.n}x{self.k}"

  @property
  def grid(self) -> dict[str, list[int]]:
    return {"global_size": [self.n // 16, self.m // 16, 1], "local_size": [32, 1, 1]}

  def to_json(self) -> dict[str, Any]:
    return {"role": self.role, "M": self.m, "N": self.n, "K": self.k, "quant": self.quant,
            "kernel_name": self.kernel_name, "grid": self.grid}


def _blocked(reason: str, **evidence: Any) -> dict[str, Any]:
  return {"protocol": PROTOCOL, "passed": False, "verdict": BLOCKED, "first_failure": reason,
          "evidence": evidence}


def derive_role_shapes(path: str | Path = GGUF, *, runtime_m: int = RUNTIME_M,
                       loader: Callable[[str | Path], tuple[dict, dict]] | None = None) -> tuple[RoleShape, ...]:
  """Read only GGUF metadata and derive the four exact loaded role contracts."""
  if runtime_m != RUNTIME_M: raise ValueError(f"runtime M must be exactly {RUNTIME_M}, got {runtime_m}")
  if loader is None:
    from tinygrad.llm.gguf import gguf_load_metadata
    loader = gguf_load_metadata
  from tinygrad.llm.model_facts import model_facts_from_gguf_metadata
  kv, metadata = loader(path)
  facts = model_facts_from_gguf_metadata(kv, metadata)
  out = []
  for role in ROLE_ORDER:
    rows = facts.tensors_for_role(role)
    # Q4_K_M is intentionally mixed quantization.  This gate owns the loaded
    # Q4_K emitter boundary, so Q6_K rows are neither candidates nor errors.
    contracts = {(x.rows, x.cols, x.quant_label) for x in rows if x.quant_label == "Q4_K"}
    if len(contracts) != 1:
      raise ValueError(f"{role}: loaded tensors do not have one N/K/quant contract: {sorted(contracts)}")
    n, k, quant = next(iter(contracts))
    if (n, k) != EXPECTED_NK[role]:
      raise ValueError(f"{role}: expected loaded N/K {EXPECTED_NK[role]}, got {(n, k)}")
    if quant != "Q4_K": raise ValueError(f"{role}: expected Q4_K, got {quant}")
    out.append(RoleShape(role, runtime_m, n, k, quant))
  return tuple(out)


def validate_compile_evidence(shape: RoleShape, row: Mapping[str, Any]) -> dict[str, Any]:
  """Validate final compiler facts only; every missing or extra path fails closed."""
  programs = row.get("programs")
  if not isinstance(programs, list): return _blocked(f"{shape.role}: PROGRAM evidence missing", shape=shape.to_json())
  if len(programs) != 1:
    return _blocked(f"{shape.role}: expected exactly one PROGRAM, got {len(programs)}",
                    shape=shape.to_json(), programs=programs, fallback_used=True)
  program = programs[0]
  if program.get("name") != shape.kernel_name:
    return _blocked(f"{shape.role}: wrong PROGRAM name: {program.get('name')!r}", expected=shape.kernel_name, program=program)
  actual_grid = {"global_size": program.get("global_size"), "local_size": program.get("local_size")}
  if actual_grid != shape.grid:
    return _blocked(f"{shape.role}: wrong PROGRAM grid: {actual_grid!r}", expected=shape.grid, program=program)
  if row.get("fallback_used") is not False:
    return _blocked(f"{shape.role}: fallback evidence is missing or true", raw=dict(row))
  signed = program.get("signed_wmma")
  if (not isinstance(signed, Mapping) or signed.get("source_signed_integer_wmma") is not True or
      not isinstance(signed.get("linear_signed_wmma_count"), int) or signed["linear_signed_wmma_count"] < 1):
    return _blocked(f"{shape.role}: final signed IU8 WMMA evidence missing", program=program)
  return {"protocol": PROTOCOL, "passed": True, "verdict": PASS, "first_failure": None,
          "evidence": {"shape": shape.to_json(), "program": program, "fallback_used": False}}


def _compile_worker(shape: RoleShape) -> dict[str, Any]:
  """Build and compile placeholders; deliberately never realize or dispatch tensors."""
  from tinygrad import dtypes
  from tinygrad.codegen import to_program
  from tinygrad.device import Device
  from tinygrad.uop.ops import Ops, UOp
  from extra.qk.q4k_q8_mmq_uop import describe_q4k_q8_mmq_wmma, emit_q4k_q8_mmq_wmma
  from extra.qk.q4k_q8_mmq_uop_validation import classify_isa

  spec = describe_q4k_q8_mmq_wmma(m=shape.m, n=shape.n, k=shape.k)
  sink = emit_q4k_q8_mmq_wmma(spec)(
    UOp.placeholder((shape.m, shape.n), dtypes.float32, 0),
    UOp.placeholder((shape.n * (shape.k // 256) * 36,), dtypes.uint32, 1),
    UOp.placeholder((shape.m * shape.k,), dtypes.int8, 2),
    UOp.placeholder((shape.m * (shape.k // 32),), dtypes.float32, 3))
  # This is the compile boundary: one authored SINK becomes one final PROGRAM.
  program = to_program(sink, Device["AMD"].renderer)
  if program.op is not Ops.PROGRAM: raise ValueError(f"to_program returned {program.op}, not PROGRAM")
  source = next((u.arg for u in program.src if u.op is Ops.SOURCE and isinstance(u.arg, str)), "")
  linears = [u for u in program.src if u.op is Ops.LINEAR]
  linear_nodes = [u for linear in linears for u in linear.src]
  linear_signed = sum(u.op in (Ops.WMMA, Ops.SHAPED_WMMA) or
                      str(u.arg).lower().startswith("v_wmma_i32_16x16x16_iu8") for u in linear_nodes)
  authored_wmma = sum(u.op in (Ops.WMMA, Ops.SHAPED_WMMA) for u in sink.toposort())
  source_isa = classify_isa(source)
  row = {"name": program.arg.name, "global_size": list(program.arg.global_size),
         "local_size": None if program.arg.local_size is None else list(program.arg.local_size),
         "signed_wmma": {"source_signed_integer_wmma": source_isa["signed_integer_wmma"],
                         "linear_signed_wmma_count": linear_signed,
                         "linear_count": len(linears), "authored_wmma_count": authored_wmma}}
  fallback = authored_wmma != 0 or len(linears) != 1
  return {"programs": [row], "fallback_used": fallback}


def run_gate(path: str | Path = GGUF, *, runtime_m: int = RUNTIME_M, timeout_seconds: float = 120.0,
             python: str = sys.executable, env: Mapping[str, str] | None = None,
             runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
  try: shapes = derive_role_shapes(path, runtime_m=runtime_m)
  except Exception as exc: return _blocked(f"metadata: {type(exc).__name__}: {exc}")
  if timeout_seconds <= 0: return _blocked("timeout_seconds must be positive")
  child_env = dict(os.environ if env is None else env)
  child_env.update({"DEV": child_env.get("DEV", "AMD"), "PYTHONPATH": str(ROOT) + os.pathsep + child_env.get("PYTHONPATH", "")})
  passed = []
  for shape in shapes:
    cmd = [python, "-m", "extra.qk.q4k_q8_mmq_uop_role_compile_gate", "--worker", "--role", shape.role,
           "--m", str(shape.m), "--n", str(shape.n), "--k", str(shape.k), "--quant", shape.quant]
    try:
      proc = runner(cmd, cwd=ROOT, env=child_env, text=True, capture_output=True,
                    timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
      return _blocked(f"{shape.role}: compile timed out after {timeout_seconds:g}s", shape=shape.to_json(), passed=passed)
    except OSError as exc: return _blocked(f"{shape.role}: compile worker could not start: {exc}", passed=passed)
    if proc.returncode != 0:
      try: worker_error = json.loads(proc.stdout.strip().splitlines()[-1]).get("worker_error")
      except (IndexError, json.JSONDecodeError, AttributeError): worker_error = None
      if worker_error:
        return _blocked(f"{shape.role}: compile failed: {worker_error}", shape=shape.to_json(), passed=passed)
      return _blocked(f"{shape.role}: compile worker failed with exit {proc.returncode}", shape=shape.to_json(),
                      stderr=proc.stderr[-4000:], passed=passed)
    try: row = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
      return _blocked(f"{shape.role}: compile worker returned invalid JSON: {exc}", stdout=proc.stdout[-4000:], passed=passed)
    checked = validate_compile_evidence(shape, row)
    if not checked["passed"]: checked["evidence"]["passed_roles"] = passed; return checked
    passed.append(checked["evidence"])
  return {"protocol": PROTOCOL, "passed": True, "verdict": PASS, "first_failure": None,
          "evidence": {"gguf": str(path), "runtime_m": runtime_m, "roles": passed}}


def main(argv: Sequence[str] | None = None) -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--gguf", default=str(GGUF)); p.add_argument("--timeout", type=float, default=120.0)
  p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
  p.add_argument("--role", choices=ROLE_ORDER); p.add_argument("--m", type=int); p.add_argument("--n", type=int)
  p.add_argument("--k", type=int); p.add_argument("--quant")
  args = p.parse_args(argv)
  if args.worker:
    try: row = _compile_worker(RoleShape(args.role, args.m, args.n, args.k, args.quant))
    except BaseException as exc: row = {"worker_error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(row, sort_keys=True, separators=(",", ":"))); return 0 if "worker_error" not in row else 2
  result = run_gate(args.gguf, timeout_seconds=args.timeout)
  print(json.dumps(result, sort_keys=True, indent=2)); return 0 if result["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
