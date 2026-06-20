"""Prefill boost-state resolution probe (P1/P2). ONE fresh process = one data point.
Measures WMMA (and optionally Tensile) prefill tok/s + REAL GFXCLK (rocm-smi --showgpuclocks sampled from a
SEPARATE subprocess, never inside the timing loop). Supports forcing levers. Multi-run via the bash orchestrator.

Modes:
  wmma                 : measure WMMA-only prefill (default), report tok/s + real sclk + state
  lever=<L1|L2|L3|L4>  : apply a forcing lever, then measure WMMA
  ab                   : interleaved WMMA-vs-Tensile A/B (clock-fair, same boost state)
  generate             : realistic SINGLE prefill via model.generate-style one-shot (P2)
Env: LEVER (lever name), DEV=AMD PREFILL_V2=1 [PREFILL_TENSILE_GEMM=1 for ab]. Clock writes need sudo.
"""
import os, sys, time, statistics, subprocess, re, signal, pathlib
import tinygrad.llm.model as Mod
import tinygrad.codegen.opt.postrange as pr
from tinygrad import Tensor, TinyJit, Device
from tinygrad.llm.model import Transformer, PREFILL_UBATCH
import extra.qk_tensile_inmodel as TI

DEVS="/sys/class/drm/card0/device"
ART=pathlib.Path("bench/qk-prefill-boost"); ART.mkdir(parents=True, exist_ok=True)
PID=os.getpid(); CLKLOG=f"/tmp/boostclk_{PID}.log"

def sudo(cmd): return subprocess.run(["sudo","-n","bash","-c",cmd],capture_output=True,text=True).returncode==0
def set_lever(name):
  if name=="L1": return sudo(f"echo profile_peak > {DEVS}/power_dpm_force_performance_level")
  if name=="L2": return sudo(f"echo manual > {DEVS}/power_dpm_force_performance_level && echo 2 > {DEVS}/pp_dpm_sclk && echo 3 > {DEVS}/pp_dpm_mclk")
  return True  # L3/L4 are in-python
def restore(): sudo(f"echo auto > {DEVS}/power_dpm_force_performance_level")

def start_sampler():
  # separate process: append "epoch sclk_mhz" every 0.2s. cheap, no timing poison.
  f=open(CLKLOG,"w")
  p=subprocess.Popen(["bash","-c",
    "while true; do echo \"$(date +%s.%N) $(rocm-smi --showgpuclocks 2>/dev/null | grep -oP '\\(\\K[0-9]+(?=Mhz)')\"; sleep 0.2; done"],
    stdout=f, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
  return p
def stop_sampler(p):
  try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
  except Exception: pass
def clk_window(t0,t1):
  rows=[]
  for l in open(CLKLOG):
    pr_=l.split()
    if len(pr_)==2 and pr_[1].isdigit(): rows.append((float(pr_[0]),int(pr_[1])))
  win=[c for t,c in rows if t0<=t<=t1]
  busy=[c for c in win if c>800]
  return {"n":len(win),"n_busy":len(busy),
          "sclk_med":statistics.median(busy) if busy else 0,
          "sclk_min":min(busy) if busy else 0,"sclk_max":max(busy) if busy else 0}

# ---- model + jits ----
MODEL='/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf'
Tensor.manual_seed(0)
model,_=Transformer.from_gguf(MODEL,2048)
N=PREFILL_UBATCH; TI.install(Device['AMD'])
t=Tensor([5,6,7,8,9,10]*200+[0]*(2048-1200),dtype='int32').reshape(1,2048); chunk=t[:,0:N].contiguous(); temp=Tensor([0.0])
for b in model.blk: b._use_flash,b._prefill_v2=False,True
saved=pr._WARMSTART_OPTS
def build(flag):
  Mod.PREFILL_TENSILE_GEMM=flag
  return TinyJit(model.forward)
def run(j):
  pr._WARMSTART_OPTS=model._pf16_warmstart
  try: j(chunk,0,temp).realize(); Device['AMD'].synchronize()
  finally: pr._WARMSTART_OPTS=saved
def state(tps): return "BOOSTED" if tps>2400 else ("STUCK" if tps<1800 else "MID")

def measure_window(j,n=25,warm=15):
  for _ in range(warm): run(j)
  sp=start_sampler(); time.sleep(0.3); t0=time.time(); ts=[]
  for _ in range(n): a=time.perf_counter(); run(j); ts.append(time.perf_counter()-a)
  t1=time.time(); time.sleep(0.3); stop_sampler(sp)
  cw=clk_window(t0,t1)
  return round(N/min(ts)), round(N/statistics.median(ts)), cw

def main():
  mode=os.environ.get("MODE","wmma"); lever=os.environ.get("LEVER","")
  if lever in ("L1","L2"): set_lever(lever); time.sleep(1.0)
  try:
    if mode=="ab":
      joff=build(False); run(joff); jon=build(True); run(jon)
      for _ in range(10): run(joff); run(jon)
      sp=start_sampler(); time.sleep(0.3); tw=[]; tt=[]; t0=time.time()
      for _ in range(25):
        a=time.perf_counter(); run(joff); tw.append(time.perf_counter()-a)
        a=time.perf_counter(); run(jon);  tt.append(time.perf_counter()-a)
      t1=time.time(); time.sleep(0.3); stop_sampler(sp); cw=clk_window(t0,t1)
      w=round(N/statistics.median(tw)); te=round(N/statistics.median(tt))
      print(f"AB wmma={w} tensile={te} ratio={statistics.median(tw)/statistics.median(tt):.3f} "
            f"state={state(w)} sclk={cw['sclk_med']}({cw['sclk_min']}-{cw['sclk_max']}) nbusy={cw['n_busy']}/{cw['n']}")
    elif mode=="generate":
      # realistic ONE-SHOT prefill: fresh jit, single timed forward (no tight loop, no pre-warm of THIS shape beyond trace)
      joff=build(False); run(joff)  # trace+compile (a user pays this once; we exclude it)
      sp=start_sampler(); time.sleep(0.3); t0=time.time()
      run(joff); t1=time.time(); time.sleep(0.3); stop_sampler(sp)
      tps=round(N/(t1-t0)); cw=clk_window(t0,t1)
      print(f"GEN single-prefill={tps} state={state(tps)} sclk={cw['sclk_med']}({cw['sclk_min']}-{cw['sclk_max']}) nbusy={cw['n_busy']}/{cw['n']} lever={lever or 'none'}")
    else:  # wmma, with optional in-python levers L3/L4
      joff=build(False); run(joff)
      if lever=="L3":  # boost primer: dense sustained matmul ~3s to latch boost
        A=Tensor.randn(8192,8192,dtype='float16',device='AMD').realize(); B=Tensor.randn(8192,8192,dtype='float16',device='AMD').realize()
        end=time.time()+3.0
        while time.time()<end: (A@B).realize(); Device['AMD'].synchronize()
      if lever=="L4":  # queue saturation: many back-to-back forwards WITHOUT per-call sync before measuring
        for _ in range(40): pr._WARMSTART_OPTS=model._pf16_warmstart; joff(chunk,0,temp).realize()
        Device['AMD'].synchronize(); pr._WARMSTART_OPTS=saved
      b,m,cw=measure_window(joff)
      print(f"WMMA best={b} med={m} state={state(m)} sclk={cw['sclk_med']}({cw['sclk_min']}-{cw['sclk_max']}) nbusy={cw['n_busy']}/{cw['n']} lever={lever or 'none'}")
  finally:
    if lever in ("L1","L2"): restore()
    try: os.remove(CLKLOG)
    except OSError: pass

if __name__=="__main__": main()
