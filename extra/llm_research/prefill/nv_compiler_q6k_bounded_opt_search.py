"""Fail-closed bounded optimizer search for the compiler-owned Q6 main.

Research tool only.  It deliberately replays the pre-capture expression rather than
mutating a retained PROGRAM, so every candidate starts with the packed-fragment TC
contract installed by the binding.
"""
from __future__ import annotations
import argparse, json
from dataclasses import asdict, dataclass
from tinygrad import Tensor, Device
from tinygrad.dtype import dtypes
from tinygrad.codegen import to_program_cache
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import warmstart_candidate_state
from .nv_compiler_q6k_pp512_binding import (CompilerQ6PP512Binding, ROLE_SHAPES, M,
  _build_role_expression)

@dataclass(frozen=True)
class SearchResult:
  role: str
  opts: tuple[str, ...]
  status: str
  programs: int = 0
  error: str | None = None

def _opts(depth: int) -> list[tuple[Opt, ...]]:
  # Axes are intentionally bounded to the post-TC carrier axes.  Invalid axis
  # interpretations are rejected by KernelOptError/type verification.
  one = [Opt(OptOps.LOCAL, a, f) for a in range(3) for f in (2,4,8)]
  one += [Opt(OptOps.UPCAST, a, f) for a in range(3) for f in (2,4,8)]
  one += [Opt(OptOps.UNROLL, a, f) for a in range(2) for f in (0,2,4,8)]
  one += [Opt(OptOps.GROUP, a, f) for a in range(2) for f in (8,16,32)]
  states = [()]
  for _ in range(depth): states += [s+(o,) for s in list(states) if len(s) == _ for o in one]
  return states

def search(role: str, depth: int = 2) -> list[SearchResult]:
  dev = Device["NV"]
  binding = CompilerQ6PP512Binding.compile(dev)
  asset = binding.roles[role]
  n, k = ROLE_SHAPES[role]
  record = Tensor.zeros((M*k + 2*M*(k//32)*4)//4, dtype=dtypes.uint32, device="NV").realize()
  halfs = Tensor.zeros(asset.transform.packed_bytes//2, dtype=dtypes.uint16, device="NV").realize()
  out: list[SearchResult] = []
  for seq in _opts(depth):
    key = asset.warmstart_key
    opts = (Opt(OptOps.TC, 0, (-1, 2, 1)),) + seq
    try:
      before = set(to_program_cache.values())
      with warmstart_candidate_state({key: opts}, {key: asset.context}):
        _build_role_expression(record, halfs, asset.context).realize()
      created = [p for p in to_program_cache.values() if getattr(p, "op", None).name == "PROGRAM" and
                 getattr(getattr(p, "src", (None,))[0], "arg", None) is not None and
                 getattr(getattr(p.src[0].arg, "candidate_context", None), "canonical_identity", None) == asset.context.canonical_identity]
      if len(created) != 1: raise RuntimeError(f"fail-closed: expected one PROGRAM, found {len(created)}")
      out.append(SearchResult(role, tuple(map(repr, opts)), "PASS", 1))
    except Exception as e:
      out.append(SearchResult(role, tuple(map(repr, opts)), "REJECT", error=f"{type(e).__name__}: {e}"))
  return out

if __name__ == "__main__":
  ap = argparse.ArgumentParser(); ap.add_argument("--role", choices=ROLE_SHAPES, default="ffn_down"); ap.add_argument("--depth", type=int, default=2)
  args = ap.parse_args()
  print(json.dumps({"schema":"tinygrad.nv_q6k_bounded_opt_search.v1", "role":args.role,
    "depth":args.depth, "results":[asdict(x) for x in search(args.role, args.depth)]}, indent=2))
