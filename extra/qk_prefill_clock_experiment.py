"""P3 prefill clock-lane experiment (build JITs once, sweep clock lanes, telemetry-verified).

Settles: does any clock lane lift tinygrad WMMA prefill toward the ~2675 'outlier' (catching Tensile),
or does WMMA stay ~1500 at every achievable clock (=> Tensile 1.76x robust)?
Samples sclk/mclk CONTINUOUSLY during each timed window (not gated on the bursty gpu_busy flag).
Restores auto on exit. NO kernel/route/default changes. Scope: docs/prefill-clock-dpm-authority-scope-20260619.md.
"""
import os, time, json, statistics, gc, threading, subprocess, re, pathlib
import numpy as np
import tinygrad.llm.model as Mod
import tinygrad.codegen.opt.postrange as pr
from tinygrad import Tensor, TinyJit, Device, UOp
import extra.qk_tensile_inmodel as TI
from tinygrad.llm.model import Transformer, PREFILL_UBATCH

DEV_SYS="/sys/class/drm/card0/device"
HWMON=next((str(p) for p in pathlib.Path(f"{DEV_SYS}/hwmon").glob("hwmon*")), None)
ART=pathlib.Path("bench/qk-prefill-clock-dpm-authority"); ART.mkdir(parents=True, exist_ok=True)

def _read(p,d=""):
  try: return open(p).read().strip()
  except OSError: return d
def _active(f):
  for ln in _read(f"{DEV_SYS}/{f}").splitlines():
    if "*" in ln:
      m=re.search(r"(\d+)Mhz",ln); return int(m.group(1)) if m else 0
  return 0
def _telem():
  s={"sclk":_active("pp_dpm_sclk"),"mclk":_active("pp_dpm_mclk"),
     "gpu_busy":int(_read(f"{DEV_SYS}/gpu_busy_percent","0") or 0)}
  if HWMON:
    s["power_w"]=round(int(_read(f"{HWMON}/power1_average","0") or 0)/1e6,1)
    s["temp_c"]=round(int(_read(f"{HWMON}/temp1_input","0") or 0)/1e3,1)
  return s

class Sampler(threading.Thread):
  def __init__(self,interval=0.02):
    super().__init__(daemon=True); self.interval=interval; self.rows=[]; self._stopflag=False
  def run(self):
    while not self._stopflag:
      r=_telem(); r["t"]=time.perf_counter(); self.rows.append(r); time.sleep(self.interval)
  def stop(self): self._stopflag=True; self.join(timeout=2); return self.rows

def window_summary(rows,t0,t1):
  w=[r for r in rows if t0<=r["t"]<=t1]
  if not w: return {"n":0}
  sclk=[r["sclk"] for r in w]; mclk=[r["mclk"] for r in w]
  return {"n":len(w),"sclk_med":statistics.median(sclk),"sclk_min":min(sclk),"sclk_max":max(sclk),
          "mclk_med":statistics.median(mclk),
          "power_max":max((r.get("power_w",0) for r in w),default=0),
          "temp_max":max((r.get("temp_c",0) for r in w),default=0)}

def sudo(cmd): return subprocess.run(["sudo","-n","bash","-c",cmd],capture_output=True,text=True).returncode==0
def set_lane(lane):
  base=f"{DEV_SYS}/power_dpm_force_performance_level"
  if lane=="auto": return sudo(f"echo auto > {base}")
  if lane=="high": return sudo(f"echo high > {base}")
  if lane=="profile_peak": return sudo(f"echo profile_peak > {base}")
  if lane=="manual_peak": return sudo(f"echo manual > {base} && echo 2 > {DEV_SYS}/pp_dpm_sclk && echo 3 > {DEV_SYS}/pp_dpm_mclk")
  raise ValueError(lane)
def restore():
  subprocess.run(["sudo","-n","rocm-smi","--resetperfdeterminism"],capture_output=True)
  sudo(f"echo auto > {DEV_SYS}/power_dpm_force_performance_level")

# ---- build model + JITs ONCE ----
MODEL='/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf'
Tensor.manual_seed(0)
model,_=Transformer.from_gguf(MODEL,2048)
N=PREFILL_UBATCH; TI.install(Device['AMD'])
t=Tensor([5,6,7,8,9,10]*200+[0]*(2048-1200),dtype='int32').reshape(1,2048); chunk=t[:,0:N].contiguous()
temp=Tensor([0.0])
for b in model.blk: b._use_flash,b._prefill_v2=False,True
vsp=UOp.variable('start_pos',0,2047); saved=pr._WARMSTART_OPTS; orig=TI.route_pf16

def build(tensile,skip_qo,concrete):
  Mod.PREFILL_TENSILE_GEMM=tensile
  TI.route_pf16=(lambda *a,**k:None) if skip_qo else orig
  sp=0 if concrete else vsp.bind(0)
  j=TinyJit(model.forward); TI.ROUTE_COUNT.clear()
  pr._WARMSTART_OPTS=model._pf16_warmstart
  try: j(chunk,sp,temp).realize(); Device['AMD'].synchronize()
  finally: pr._WARMSTART_OPTS=saved; TI.route_pf16=orig
  return j,sp,dict(TI.ROUTE_COUNT)
def run(j,sp):
  pr._WARMSTART_OPTS=model._pf16_warmstart
  try: r=j(chunk,sp,temp).realize(); Device['AMD'].synchronize(); return r
  finally: pr._WARMSTART_OPTS=saved

print("building JITs ...", flush=True)
jWMMA,spW,rcW=build(False,False,True)              # R2 concrete-KV WMMA
refW=run(jWMMA,spW).float().numpy()
jTEN,spT,rcT=build(True,True,True)                  # R3 concrete + Tensile FFN-only
oT=run(jTEN,spT).float().numpy()
relT=float(np.sqrt(((oT-refW)**2).mean())/(np.sqrt((refW**2).mean())+1e-9))
print(f"built. WMMA route={rcW} Tensile route={rcT} rel_err={relT:.6f}", flush=True)

LANES=os.environ.get("LANES","auto,high,profile_peak,manual_peak").split(",")
NMEAS=int(os.environ.get("NMEAS","30"))
results={"meta":{"N":N,"model":MODEL,"tensile_rel_err":round(relT,6),"nmeas":NMEAS,
                 "wmma_route":rcW,"tensile_route":rcT},"lanes":{}}

def measure(j,sp,sampler):
  for _ in range(15): run(j,sp)            # warmup at this clock
  t0=time.perf_counter(); ts=[]
  for _ in range(NMEAS):
    a=time.perf_counter(); run(j,sp); ts.append(time.perf_counter()-a)
  t1=time.perf_counter()
  return ts,t0,t1

for lane in LANES:
  ok=set_lane(lane); time.sleep(1.5)
  sp_=Sampler(); sp_.start()
  # interleaved clock-fair: alternate WMMA / Tensile in the same window
  for _ in range(15): run(jWMMA,spW); run(jTEN,spT)
  tw=[]; tt=[]; w0=time.perf_counter()
  for _ in range(NMEAS):
    a=time.perf_counter(); run(jWMMA,spW); tw.append(time.perf_counter()-a)
    a=time.perf_counter(); run(jTEN,spT); tt.append(time.perf_counter()-a)
  w1=time.perf_counter()
  rows=sp_.stop()
  summ=window_summary(rows,w0,w1)
  wmma_tokps=round(N/min(tw)); ten_tokps=round(N/min(tt))
  results["lanes"][lane]={"lane_set_ok":ok,
    "wmma_best_tokps":wmma_tokps,"wmma_med_tokps":round(N/statistics.median(tw)),
    "tensile_best_tokps":ten_tokps,"tensile_med_tokps":round(N/statistics.median(tt)),
    "tensile_vs_wmma":round(statistics.median(tw)/statistics.median(tt),4),
    "telemetry":summ,"perf_level":_read(f"{DEV_SYS}/power_dpm_force_performance_level")}
  print(f"LANE {lane:13s} WMMA={wmma_tokps:5d} Tensile={ten_tokps:5d} ratio={results['lanes'][lane]['tensile_vs_wmma']:.3f} "
        f"sclk med/min/max={summ.get('sclk_med')}/{summ.get('sclk_min')}/{summ.get('sclk_max')} "
        f"mclk={summ.get('mclk_med')} P={summ.get('power_max')}W T={summ.get('temp_max')}C", flush=True)
  restore(); time.sleep(1.0)

restore()
(ART/"prefill_clock_matrix.json").write_text(json.dumps(results,indent=2))
print("PREFILL_CLOCK_MATRIX_JSON="+json.dumps(results))
