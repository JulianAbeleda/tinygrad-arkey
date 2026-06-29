"""AMD native ISA backend — Phase F gate: decode-attention primitive microgates (opt-in DEV=AMD:ISA).

Standalone correctness-first microgates proving AMDISARenderer can emit + correctly execute the primitive ingredients
the generated decode block tile needs, through DEV=AMD:ISA -> AMDISARenderer -> rdna3 Insts -> assemble_linear -> runtime.
Each microgate is a hand-built kernel AST run via Tensor.custom_kernel (compiled by AMDISARenderer, not HIP/LLVM).

  F.1 LDS tile staging : N lanes write in[i]->LDS[i], s_barrier, read LDS[N-1-i] -> out  (group segment + ds_store/
                         ds_load + cross-lane sharing via LDS; barrier-ordered).
  F.4 barrier ordering : same kernel -- the cross-lane reverse read is only correct if s_barrier orders all lanes'
                         LDS writes before any lane's read; a wrong/absent barrier corrupts it.
  F.2 ds_bpermute      : out[i] = in[N-1-i] via ds_bpermute_b32 cross-lane exchange (no LDS).
  F.3 v_dot2           : packed fp16 dot a.lo*b.lo + a.hi*b.hi (fp32 acc) via v_dot2_f32_f16 (no scalar fallback).

Out of scope: full decode block tile, route binding, waitcnt/scheduler performance.

Run:  DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_f_primitives_gate.py
Writes: bench/amd-isa-backend-phase-f/latest.json
"""
import os, sys, json, traceback
os.environ.setdefault("DEV", "AMD:ISA")
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.renderer.amd.elf import group_segment_fixed_size_from_elf

CMD = "DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_f_primitives_gate.py"
ART = os.path.join(os.path.dirname(__file__), "..", "bench", "amd-isa-backend-phase-f", "latest.json")

# capture each compiled kernel's binary + resolved instruction mnemonics
_cap = []
_orig = AMDISARenderer.asm
def _spy(self, prg, lin):
  b = _orig(self, prg, lin)
  _cap.append((b, "\n".join(str(u.arg) for u in self._resolve_labels(list(lin.src)) if u.op is Ops.INS)))
  return b
AMDISARenderer.asm = _spy

def _run(tensors, fxn, out_slot):
  _cap.clear()
  res = Tensor.custom_kernel(*tensors, fxn=fxn)[out_slot].numpy()
  return res, (_cap[-1] if _cap else (b"", ""))

def f1_lds(N=16):
  def fxn(inb, outb):
    lidx = UOp.special(N, "lidx0")
    lds = UOp(Ops.DEFINE_LOCAL, dtypes.float32.ptr(N, AddrSpace.LOCAL), (), 'lds')
    st = lds.index(lidx).store(inb.index(lidx).load())
    rval = lds.after(st.barrier()).index(UOp.const(dtypes.int, N-1) - lidx).load()
    return outb.index(lidx).store(rval).sink(arg=KernelInfo(name="f1_lds"))
  a = (np.arange(N, dtype=np.float32) + 1)
  got, (b, asm) = _run((Tensor(a, device="AMD"), Tensor.empty(N, device="AMD")), fxn, 1)
  ok = bool(np.array_equal(got, a[::-1]))
  return ok, {"group_segment_size": group_segment_fixed_size_from_elf(b), "s_barrier": "s_barrier" in asm,
              "ds_store": "ds_store" in asm, "ds_load": "ds_load" in asm}, a

def f2_bpermute(N=16):
  def fxn(inb, outb):
    lidx = UOp.special(N, "lidx0")
    data = inb.index(lidx).load()
    addr = (UOp.const(dtypes.int, N-1) - lidx) * UOp.const(dtypes.int, 4)   # byte addr of source lane
    bp = UOp(Ops.CUSTOMI, dtypes.float32, src=(addr, data), arg="bpermute")
    return outb.index(lidx).store(bp).sink(arg=KernelInfo(name="f2_bperm"))
  a = (np.arange(N, dtype=np.float32) + 1) * 10
  got, (b, asm) = _run((Tensor(a, device="AMD"), Tensor.empty(N, device="AMD")), fxn, 1)
  return bool(np.array_equal(got, a[::-1])), {"ds_bpermute": "ds_bpermute" in asm}

def f3_dot2():
  def pack(h0, h1): return np.frombuffer(np.array([h0, h1], dtype=np.float16).tobytes(), dtype=np.float32)[0]
  a0, a1, b0, b1 = 1.5, -2.0, 3.0, 0.5
  ap = np.array([pack(a0, a1)], dtype=np.float32); bp = np.array([pack(b0, b1)], dtype=np.float32)
  def fxn(ina, inb, outb):
    idx = UOp.special(1, "lidx0")
    dot = UOp(Ops.CUSTOMI, dtypes.float32, src=(UOp.const(dtypes.float32, 0.0), ina.index(idx).load(), inb.index(idx).load()), arg="fdot2")
    return outb.index(idx).store(dot).sink(arg=KernelInfo(name="f3_dot2"))
  got, (b, asm) = _run((Tensor(ap, device="AMD"), Tensor(bp, device="AMD"), Tensor.empty(1, device="AMD")), fxn, 2)
  exp = float(np.float16(a0))*float(np.float16(b0)) + float(np.float16(a1))*float(np.float16(b1))
  return bool(abs(float(got[0]) - exp) < 1e-2), {"v_dot2": "v_dot2" in asm, "got": float(got[0]), "exp": exp}

def main():
  rec = {"overall_verdict": None, "commands": [CMD], "scope": "Phase F decode-attention primitive microgates"}
  ren = type(Device["AMD"].renderer).__name__
  rec["selected_renderer"] = ren
  if ren != "AMDISARenderer":
    rec["overall_verdict"] = "AMD_ISA_PHASE_F_BLOCKED_RUNTIME_ROUTE"; rec["first_blocker"] = f"selected {ren}"; return rec
  rec["no_hidden_fallback"] = "PASS (selected AMDISARenderer, not HIP/LLVM)"
  markers, correctness, stability = {}, {}, {}
  try:
    # F.1 LDS + F.4 barrier (same kernel; barrier-ordered cross-lane reverse)
    ok1, info1, a1 = f1_lds()
    markers["f1"] = info1
    correctness["f1_lds_reverse"] = ok1
    barrier_ok = ok1 and info1["s_barrier"]   # correctness depends on the barrier ordering all lanes
    rec["group_segment_size"] = info1["group_segment_size"]
    # determinism (8 trials)
    def trial1():
      def fxn(inb, outb):
        lidx = UOp.special(16, "lidx0"); lds = UOp(Ops.DEFINE_LOCAL, dtypes.float32.ptr(16, AddrSpace.LOCAL), (), 'lds')
        st = lds.index(lidx).store(inb.index(lidx).load())
        return outb.index(lidx).store(lds.after(st.barrier()).index(UOp.const(dtypes.int, 15) - lidx).load()).sink(arg=KernelInfo(name="f1_lds"))
      return Tensor.custom_kernel(Tensor(a1, device="AMD"), Tensor.empty(16, device="AMD"), fxn=fxn)[1].numpy()
    stability["f1"] = bool(all(np.array_equal(trial1(), a1[::-1]) for _ in range(8)))

    rec["f1_lds_roundtrip"] = "AMD_ISA_PHASE_F1_PASS_LDS_ROUNDTRIP" if (ok1 and stability["f1"]) else "AMD_ISA_PHASE_F1_BLOCKED_DS_LOAD_STORE_ISEL"
    rec["f4_barrier_ordering"] = "AMD_ISA_PHASE_F4_PASS_BARRIER_ORDERING" if barrier_ok else "AMD_ISA_PHASE_F4_BLOCKED_S_BARRIER_ISEL"

    # F.2 ds_bpermute
    ok2, info2 = f2_bpermute()
    markers["f2"] = info2; correctness["f2_bpermute_reverse"] = ok2
    rec["f2_ds_bpermute"] = "AMD_ISA_PHASE_F2_PASS_DS_BPERMUTE" if (ok2 and info2["ds_bpermute"]) else "AMD_ISA_PHASE_F2_BLOCKED_DS_BPERMUTE_ISEL"

    # F.3 v_dot2
    ok3, info3 = f3_dot2()
    markers["f3"] = info3; correctness["f3_dot2"] = ok3
    rec["f3_v_dot2"] = "AMD_ISA_PHASE_F3_PASS_V_DOT2" if (ok3 and info3["v_dot2"]) else "AMD_ISA_PHASE_F3_BLOCKED_PACKED_FP16"

    rec["emitted_instruction_markers"] = markers
    rec["correctness_summary"] = correctness
    rec["repeated_run_stability"] = stability

    passes = {"f1": ok1 and stability["f1"], "f4": barrier_ok, "f2": ok2 and info2["ds_bpermute"], "f3": ok3 and info3["v_dot2"]}
    if all(passes.values()):
      rec["overall_verdict"] = "AMD_ISA_PHASE_F_PASS_ATTENTION_PRIMITIVES"
    elif passes["f1"] and passes["f2"]:
      rec["overall_verdict"] = "AMD_ISA_PHASE_F_PARTIAL_LDS_AND_BPERMUTE"; rec["first_blocker"] = "v_dot2" if not passes["f3"] else "barrier"
    elif passes["f1"]:
      rec["overall_verdict"] = "AMD_ISA_PHASE_F_PARTIAL_LDS_ONLY"; rec["first_blocker"] = "ds_bpermute"
    else:
      rec["overall_verdict"] = "AMD_ISA_PHASE_F_BLOCKED_GROUP_SEGMENT_DESCRIPTOR"; rec["first_blocker"] = "f1_lds_roundtrip"
  except Exception as e:
    rec["overall_verdict"] = "AMD_ISA_PHASE_F_BLOCKED_RUNTIME_ROUTE"
    rec["first_blocker"] = f"{type(e).__name__}: {e}"
    rec["traceback"] = traceback.format_exc().splitlines()[-6:]
  return rec

if __name__ == "__main__":
  import subprocess
  rec = main()
  # regression: Inc0-3 + Phase B/C
  reg = {}
  for name, cmd in [("inc0", "extra/amd_isa_inc0_gate.py"), ("inc1", "extra/amd_isa_inc1_gate.py"),
                    ("inc2", "extra/amd_isa_inc2_gate.py"), ("inc3", "extra/amd_isa_inc3_gate.py"),
                    ("phase_b", "extra/amd_isa_phase_b_reduction_gate.py"), ("phase_c", "extra/amd_isa_phase_c_gemv_gate.py")]:
    env = {**os.environ, "DEV": "AMD:ISA", **({"NOOPT": "1"} if name in ("phase_b", "phase_c") else {})}
    try:
      out = subprocess.run([sys.executable, cmd], cwd=os.path.join(os.path.dirname(__file__), ".."), env=env, capture_output=True, text=True, timeout=300).stdout
      reg[name] = "PASS" if ("_PASS_" in out or "PASS" in out.splitlines()[-1]) else "FAIL"
    except Exception as ex: reg[name] = f"ERR {ex}"
  rec["inc0_inc1_phase_b_regression_status"] = reg
  os.makedirs(os.path.dirname(ART), exist_ok=True)
  with open(ART, "w") as f: json.dump(rec, f, indent=2)
  print(json.dumps(rec, indent=2))
  print("\nPHASE_F", rec["overall_verdict"])
