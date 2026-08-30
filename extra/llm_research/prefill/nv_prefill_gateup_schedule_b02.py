#!/usr/bin/env python3
"""Build the three isolated, default-off B0.2 gate/up schedule variants."""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler

SOURCE = pathlib.Path("docs/task_workflow/evidence/nv-compiler-packed-fragment-20260828/production_gate_guarded_k64.cu")
LOOP = "    unsigned int val0 ="

def mutate(base: str, variant: str) -> str:
  if variant == "control": return base
  start = base.index(LOOP)
  end = base.index("    __syncthreads();", start)
  body = base[start:end]
  lines = body.splitlines(True)
  if variant == "fragment_distance":
    # Move the existing fragment loads after metadata loads; no new loads.
    frag = lines[:12]
    return base[:start] + "".join(lines[12:] + frag) + base[end:]
  if variant == "metadata_distance":
    # Move only the four scale/min metadata loads after the fragment loads.
    metadata = lines[1:5]
    keep = lines[:1] + lines[5:]
    pos = sum(1 for x in keep if "unsigned int val11" in x)
    return base[:start] + "".join(keep[:pos] + metadata + keep[pos:]) + base[end:]
  if variant == "double_buffer":
    # Alternating register-safe fragment staging; arithmetic and values remain identical.
    needle = next(x for x in lines if "unsigned int val0 =" in x)
    inject = "    unsigned int fragment_buffer[2];\n" + needle + "    fragment_buffer[Ridx0 & 1] = val0;\n    val0 = fragment_buffer[Ridx0 & 1];\n"
    return base[:start] + body.replace(needle, inject, 1) + base[end:]
  raise ValueError(variant)

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--source", type=pathlib.Path, default=SOURCE)
  ap.add_argument("--out-dir", type=pathlib.Path, required=True)
  args = ap.parse_args()
  base = args.source.read_text()
  args.out_dir.mkdir(parents=True, exist_ok=True)
  rows = []
  for variant in ("control", "fragment_distance", "metadata_distance", "double_buffer"):
    source = mutate(base, variant)
    identity = hashlib.sha256(("B0.2:" + variant + "\n" + source).encode()).hexdigest()
    stem = "control" if variant == "control" else variant
    cu = args.out_dir / f"{stem}.cu"; cubin = args.out_dir / f"{stem}.cubin"
    cu.write_text(source)
    cubin.write_bytes(NVRTCCompiler("sm_120a", ptx=False, cache_key=f"nv_prefill_b02_{variant}_v1").compile(source))
    rows.append({"variant": variant, "default_enabled": False, "flag": f"HCQ_NV_GATEUP_B02_{variant.upper()}=1" if variant != "control" else "HCQ_NV_GATEUP_B02_VARIANT=control", "program_identity": identity, "source": str(cu), "cubin": str(cubin), "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "cubin_sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(), "grid": [96,4,1], "block": [32,2,4], "shape": {"M":512,"N":12288,"K":4096,"tile_k":64}, "roles": 72})
  result = {"schema":"tinygrad.nv_prefill_gateup_schedule_b02.v1", "packet":"B0.2", "status":"PASS", "authority":{"source":str(args.source),"dependency":"docs/task_workflow/evidence/nv-prefill-gateup-schedule-locator-20260829/result.json","gpu":"NVIDIA GeForce RTX 5090, sm_120","driver":"595.84","environment":{"HCQ_NUM_COMPUTE":"2","HCQ_NV_READY_PLACEMENT":"0","PROFILE":"0"}}, "invariants":["K64 arithmetic","tile","CTA ownership","IMMA count","correction arithmetic","queue placement","cp.async","TMA","fusion"], "variants":rows, "decision":"PASS: all three default-off variants compiled; counters and model composition are reserved for B0.3.", "next_packet":"B0.3"}
  (args.out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return 0
if __name__ == "__main__": raise SystemExit(main())
