"""Schedule-interleave detector (the `schedule_interleave_gate` prototype) — classify a GEMM kernel as PHASED vs
PIPELINED by measuring how many global loads / LDS ops fall INSIDE the WMMA span of the steady region. This is the
static representation that lets machine search reason about the K-loop software-pipeline primitive (the named prefill
residual). Static-only; no W==D claim. Works on a hand-asm instruction list (build_gemm_lds2) or a compiled .co.

  # our prefill GEMM (instruction list):
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_schedule_interleave_detector.py --builder down
  # a Tensile / any code object:
  PYTHONPATH=. .venv/bin/python extra/qk_schedule_interleave_detector.py --code-object /path/to/kernel.co

See docs/machine-search-representation-expansion-decode-prefill-result-20260623.md."""
from __future__ import annotations
import sys, re, json, subprocess

def _fam(op):
  if 'v_wmma' in op or 'v_dot' in op or 'fdot2' in op: return 'wmma'
  if op.startswith(('global_load', 'buffer_load')): return 'gl'
  if op.startswith(('ds_load', 'ds_read', 'ds_store', 'ds_write')): return 'ds'
  if op.startswith('s_barrier'): return 'barrier'
  if op.startswith('s_waitcnt'): return 'wait'
  return None

def classify(fams):
  """fams: ordered list of family strings. Returns the interleave classification dict."""
  idx = lambda f: [i for i, x in enumerate(fams) if x == f]
  wmma, gl, ds, bar, wait = idx('wmma'), idx('gl'), idx('ds'), idx('barrier'), idx('wait')
  res = {"wmma": len(wmma), "glob_load": len(gl), "ds": len(ds), "s_barrier": len(bar), "s_waitcnt": len(wait)}
  if wmma and (gl or ds):
    w0, w1 = min(wmma), max(wmma)
    gl_in = sum(1 for g in gl if w0 < g < w1); ds_in = sum(1 for d in ds if w0 < d < w1)
    res["glob_loads_inside_wmma_span"] = f"{gl_in}/{len(gl)}"
    res["ds_inside_wmma_span"] = f"{ds_in}/{len(ds)}"
    pipelined = (len(gl) and gl_in / len(gl) >= 0.5) or (len(ds) and ds_in / len(ds) >= 0.5)
    res["classification"] = "PIPELINED" if pipelined else "PHASED"
  else:
    res["classification"] = "UNDETERMINED"
  return res

def from_builder_down():
  import extra.gemm.rdna3_wmma_matmul as ref
  insts = ref.build_gemm_lds2(512, 4096, 12288, 2, 2, 4, 4, 32, 16, 0, PLRA=1)
  fams = []
  for i in insts:
    s = str(i).strip(); op = s.split('(')[0].split()[0] if '(' in s else (s.split()[0] if s else '')
    fams.append(_fam(op))
  return [f for f in fams if f]

def from_code_object(path):
  from extra.qk_amdgpu_isa_primitive_audit import unbundle, OBJDUMP
  elf = unbundle(path)
  dis = subprocess.run([OBJDUMP, "-d", elf], capture_output=True, text=True).stdout
  lines = dis.splitlines(); funcs = {}; name = None
  for l in lines:
    m = re.match(r'^[0-9a-f]+ <(\S+)>:', l)
    if m: name = m.group(1); funcs[name] = []
    elif name: funcs[name].append(l)
  if not funcs: return []
  _, body = max(funcs.items(), key=lambda kv: sum('v_wmma' in x for x in kv[1]))
  fams = []
  for l in body:
    t = l.strip().split('\t')[-1] if '\t' in l else l.strip()
    op = t.split()[0] if t else ''
    fams.append(_fam(op))
  return [f for f in fams if f]

if __name__ == "__main__":
  if "--code-object" in sys.argv:
    fams = from_code_object(sys.argv[sys.argv.index("--code-object") + 1]); src = "code_object"
  else:
    fams = from_builder_down(); src = "build_gemm_lds2(down)"
  r = classify(fams); r["source"] = src
  print("INTERLEAVE " + json.dumps(r))
