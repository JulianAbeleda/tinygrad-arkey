"""Single-variant Tensile runner (fault-isolated; one per subprocess — an MMU fault wedges the device).
Uses the PER-VARIANT kernarg captured by the rocBLAS solution-sweep (/tmp/kernargs_all.jsonl from
qk_tensile_solution_sweep + the all-symbols capture shim), so ANY variant launches correctly on tinygrad buffers.
Verifies correctness, and (PMC=1) captures the PMC scoreboard. Prints one ABLATION_ROW JSON line.

Run: DEV=AMD [PMC=1 PROFILE=1 PMC_COUNTERS=...] SYM=<symbol> python3 extra/qk_tensile_variant_ablation.py
"""
import os, re, json, struct
from tinygrad import Tensor, Device, dtypes
from extra.qk_tensile_hcq_launch import unbundle, kd_offset, NamedAMDProgram

SYM = os.environ["SYM"]
KCAP = os.environ.get("KCAP", "/tmp/kernargs_all.jsonl")
M, N, K = 512, 12288, 4096
dev = Device['AMD']
elf = unbundle()
cap = {json.loads(l)["kernel_symbol"]: json.loads(l) for l in open(KCAP)}
assert SYM in cap, f"no captured kernarg for {SYM[-40:]}"
entry = cap[SYM]
raw = bytearray(entry["kernarg_bytes"])
gw, lw = entry["global"], entry["local"]                 # HIP workitems + local
num_wg = tuple(gw[i]//lw[i] for i in range(3))           # tinygrad global_size = workgroups
Tensor.manual_seed(0)
A_t = Tensor.randn(K, M, dtype=dtypes.half).contiguous().realize()
B_t = Tensor.randn(N, K, dtype=dtypes.half).contiguous().realize()
C_t = Tensor.zeros(N, M, dtype=dtypes.half).contiguous().realize()
oracle = (B_t.float() @ A_t.float()).realize(); dev.synchronize()
va = lambda t: t.uop.buffer._buf.va_addr
struct.pack_into("<Q", raw, 16, va(C_t)); struct.pack_into("<Q", raw, 24, va(C_t))   # D, C
struct.pack_into("<Q", raw, 32, va(A_t)); struct.pack_into("<Q", raw, 40, va(B_t))   # A, B
mt = re.search(r'MT(\d+)x(\d+)x(\d+)', SYM)
g = lambda p: (m.group(1) if (m := re.search(p, SYM)) else None)
knobs = {"MT": f"{mt.group(1)}x{mt.group(2)}x{mt.group(3)}", "LDSB": g(r'LDSB(\d)'), "AMAS": g(r'AMAS(\d)'),
         "WGM": g(r'WGM(\d+)'), "PGR": g(r'PGR(\d)'), "PLR": g(r'PLR(\d)')}

kd = kd_offset(elf, SYM)
prg = NamedAMDProgram(dev, "tv", elf, kd, bytes(raw))
prg(global_size=num_wg, local_size=tuple(lw), wait=True, timeout=10000); dev.synchronize()
rel = ((C_t.float() - oracle).abs().max()/(oracle.abs().max()+1e-6)).item()
row = {**knobs, "num_wg": list(num_wg), "local": lw, "lds_B": prg.group_segment_size,
       "vgpr": ((prg.rsrc1 & 0x3F)+1)*8, "rel_err": round(rel, 5), "correct": rel < 2e-2}

if os.environ.get("PMC") and row["correct"]:
  from tinygrad.device import Compiled
  from extra.qk_pmc_capture import decode_pmc
  for _ in range(3): prg(global_size=num_wg, local_size=tuple(lw), wait=True); dev.synchronize()
  base = len([e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"])
  prg(global_size=num_wg, local_size=tuple(lw), wait=True); dev.synchronize(); dev._at_profile_finalize()
  evs = [e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"][base:]
  if evs: row["pmc"] = {k: v for k, v in decode_pmc(evs[-1]).items()}
print("ABLATION_ROW=" + json.dumps(row))
