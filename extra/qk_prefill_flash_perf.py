import numpy as np, time
from tinygrad.device import Device, Buffer
from tinygrad.dtype import dtypes
from extra.qk_prefill_flash import prefill_flash_src
from extra.qk_clock_pin import pinned_perflevel  # GPU perf-state boundary (leak-safe try/finally)
Hd,Hq,Hkv,MAXC=128,32,8,4096
dev=Device["AMD"]
prg=dev.runtime("prefill_flash", dev.compiler.compile(prefill_flash_src(Hd,Hq,Hkv,MAXC)))
def buf(a,dt):
    b=Buffer("AMD",a.size,dt).ensure_allocated(); b.copyin(memoryview(np.ascontiguousarray(a))); return b
rng=np.random.default_rng(0)
q=rng.standard_normal((Hq,512,Hd)).astype(np.float16)
k=rng.standard_normal((Hkv,MAXC,Hd)).astype(np.float16); v=rng.standard_normal((Hkv,MAXC,Hd)).astype(np.float16)
qb,kb,vb=buf(q,dtypes.half),buf(k,dtypes.half),buf(v,dtypes.half)
ob=Buffer("AMD",Hq*512*Hd,dtypes.float32).ensure_allocated()
T=512
with pinned_perflevel("high"):
    for sp in [0,512,1536,3072]:
        for _ in range(3): prg(ob._buf,qb._buf,kb._buf,vb._buf,global_size=(Hq*T,1,1),local_size=(Hd,1,1),vals=(T,sp),wait=True)
        best=min(prg(ob._buf,qb._buf,kb._buf,vb._buf,global_size=(Hq*T,1,1),local_size=(Hd,1,1),vals=(T,sp),wait=True) for _ in range(20))
        print(f"  T=512 start_pos={sp:4d} KV={sp+512:4d}: {best*1e3:.3f} ms/layer  (x36 = {best*1e3*36:.1f} ms/forward)")
