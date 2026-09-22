#!/usr/bin/env python3
"""Standalone CUDA/ncu counters for current S8 versus depth-bounded S4 Flash."""
from __future__ import annotations

import argparse, contextlib, csv, hashlib, io, json, os, pathlib, re, statistics, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.flash_decode_attention import flash_vec_llama_score_pv_kernel
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp

Hq, Hkv, Hd, MAXC, Tc, W = 32, 8, 128, 1024, 512, 130
NVCC, NCU = "/usr/local/cuda-13.2/bin/nvcc", "/usr/local/bin/ncu"
METRICS = ",".join((
  "dram__bytes.sum", "dram__bytes_op_read.sum", "dram__bytes_op_write.sum",
  "dram__throughput.avg.pct_of_peak_sustained_elapsed", "gpu__time_duration.sum", "lts__t_bytes.sum",
  "lts__t_sector_op_read_hit_rate.pct", "l1tex__t_bytes.sum", "sm__inst_executed.sum",
  "sm__throughput.avg.pct_of_peak_sustained_elapsed", "launch__registers_per_thread",
  "smsp__warps_active.avg.pct_of_peak_sustained_active",
  "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
  "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
  "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_ld.sum",
  "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
  "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
  "smsp__inst_executed_op_global_ld.sum", "smsp__inst_executed_op_shared_ld.sum",
  "smsp__inst_executed_op_shared_st.sum",
  "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
  "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
  "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
  "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
  "smsp__warp_issue_stalled_wait_per_warp_active.pct",
  "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
  "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
  "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"))


def _render(splits:int, token_bound:int|None, *, transpose_pv_smem:bool=False,
            v_pipeline_tail:int=0, v_dimension_major:bool=False, shared_probability_ownership:bool=False,
            packed_pv_f16:bool=False, warp_probability_ownership:bool=False) -> tuple[str, str]:
  out = UOp.placeholder((Hq*splits*W,), dtypes.float32, 0)
  q = UOp.placeholder((Hq*Hd,), dtypes.float32, 1)
  cache = UOp.placeholder((2*Hkv*MAXC*Hd//2,), dtypes.uint32, 2)
  sink = flash_vec_llama_score_pv_kernel(Hd, Hq, Hkv, MAXC, splits, UOp.const(dtypes.int, Tc),
    wide_kv=True, wide_q=False, token_bound=token_bound, transpose_pv_smem=transpose_pv_smem)(out, q, cache)
  if v_pipeline_tail or v_dimension_major or shared_probability_ownership or packed_pv_f16 or warp_probability_ownership:
    sink = flash_vec_llama_score_pv_kernel(Hd, Hq, Hkv, MAXC, splits, UOp.const(dtypes.int, Tc),
      wide_kv=True, wide_q=False, token_bound=token_bound, transpose_pv_smem=transpose_pv_smem,
      v_pipeline_tail=v_pipeline_tail, v_dimension_major=v_dimension_major,
      shared_probability_ownership=shared_probability_ownership, packed_pv_f16=packed_pv_f16,
      warp_probability_ownership=warp_probability_ownership)(out, q, cache)
  program = to_program(sink, CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False))
  source = next(x.arg for x in program.src if x.op is Ops.SOURCE)
  return program.arg.name, source


def _readonly_cache_source(source:str, symbol:str) -> tuple[str, str, int]:
  """Research spelling: route immutable wide K/V vectors through CUDA's read-only load intrinsic."""
  readonly_symbol = symbol + "_readonlykv"
  source = source.replace(symbol + "(", readonly_symbol + "(", 1)
  out, replaced = [], 0
  for line in source.splitlines():
    if " = (*" in line and "data2_" in line and "uint4" in line and line.endswith(";"):
      lhs, rhs = line.rsplit(" = ", 1)
      expr = rhs[:-1]
      if expr.startswith("(*") and expr.endswith(")"):
        line = f"{lhs} = __ldg({expr[2:-1]});"
        replaced += 1
    out.append(line)
  if replaced != 4: raise RuntimeError(f"expected four wide K/V loads, rewrote {replaced}")
  return readonly_symbol, "\n".join(out), replaced


def _unroll_column_source(source:str, symbol:str) -> tuple[str, str, int]:
  """Force the score/max/PV eight-column loops open to test load-level parallelism."""
  unrolled_symbol = symbol + "_unrollcols"
  source = source.replace(symbol + "(", unrolled_symbol + "(", 1)
  out, inserted = [], 0
  for line in source.splitlines():
    if any(f"for (int Ridx{x} = 0; Ridx{x} < 8; Ridx{x}++)" in line for x in (5, 7, 9)):
      out.append("  #pragma unroll")
      inserted += 1
    out.append(line)
  if inserted != 3: raise RuntimeError(f"expected three eight-column loops, marked {inserted}")
  return unrolled_symbol, "\n".join(out), inserted


def _wide_q_f32_source(source:str, symbol:str) -> tuple[str, str, int]:
  """Replace sixteen scalar float Q loads with four aligned float4 loads."""
  wide_symbol = symbol + "_wideqf32"
  source = source.replace(symbol + "(", wide_symbol + "(", 1)
  data = re.search(r"float\* (data1_\d+)", source)
  if data is None: raise RuntimeError("could not identify float32 Q argument")
  marker = "  for (int Lidx40_0 = 0; Lidx40_0 < 2; Lidx40_0++) {\n"
  start = source.index(marker)
  end = source.index("  int gidx0 = blockIdx.x;", start)
  block = source[start:end]
  if "float val0 =" not in block: raise RuntimeError("could not isolate scalar Q loads")
  comps = ("x", "y", "z", "w")
  replacement = ["  int wide_q_base = (alu3+(gidx1<<7));"]
  for i, off in enumerate((0, 4, 64, 68)):
    replacement.append(f"  float4 wide_q_{i} = (*((float4*)({data.group(1)}+wide_q_base+{off})));" )
  for i in range(16): replacement.append(f"  (*(buf1+{i})) = ((half)(wide_q_{i//4}.{comps[i%4]}));")
  return wide_symbol, source[:start] + "\n".join(replacement) + "\n" + source[end:], 1


def _inflight_column_source(source:str, symbol:str, width:int, force_ptx:bool=False) -> tuple[str, str, int]:
  """Batch K and V loads ahead of their consumers without changing column order."""
  if width not in (2, 4, 8): raise ValueError(f"unsupported in-flight width {width}")
  inflight_symbol = symbol + f"_inflight{width}"
  source = source.replace(symbol + "(", inflight_symbol + "(", 1)

  def batch_loop(src:str, ridx:str, end_marker:str, loads:tuple[tuple[str, str], ...]) -> tuple[str, int]:
    marker = f"  for (int {ridx} = 0; {ridx} < 8; {ridx}++) {{\n"
    start = src.index(marker)
    end = src.index(end_marker, start)
    original = src[start:end]
    if not original.endswith("  }\n"): raise RuntimeError(f"could not isolate {ridx} loop")
    body = original[len(marker):-4]
    arrays, preload = [], []
    for pos, (decl, ptr) in enumerate(loads):
      arrays.append(f"    uint4 inflight_{ridx}_{pos}[{width}];")
      expr = f"forced_ldg128((const uint4*)({ptr}))" if force_ptx else f"(*((uint4*)({ptr})))"
      preload.append(f"      inflight_{ridx}_{pos}[Pidx_{ridx}] = {expr};")
      body, count = re.subn(rf"    uint4 {decl} = .*?;\n", f"      uint4 {decl} = inflight_{ridx}_{pos}[Pidx_{ridx}];\n", body, count=1)
      if count != 1: raise RuntimeError(f"could not replace {ridx}/{decl} load")
    # The original body is indented for one loop. It now lives under batch and consume loops.
    body = "\n".join(("    "+line if line else line) for line in body.splitlines()) + "\n"
    alu = "alu12" if ridx == "Ridx5" else "alu40"
    base = "alu9+(Cidx_{r}+Pidx_{r})*64+alu10+alu11".format(r=ridx)
    staged = [f"  for (int Cidx_{ridx} = 0; Cidx_{ridx} < 8; Cidx_{ridx} += {width}) {{", *arrays,
      "    #pragma unroll", f"    for (int Pidx_{ridx} = 0; Pidx_{ridx} < {width}; Pidx_{ridx}++) {{",
      f"      int {alu}_pre = ({base});"]
    for line in preload: staged.append(line.replace(alu, alu+"_pre"))
    staged += ["    }", "    #pragma unroll", f"    for (int Pidx_{ridx} = 0; Pidx_{ridx} < {width}; Pidx_{ridx}++) {{",
      f"      int {ridx} = Cidx_{ridx}+Pidx_{ridx};", body.rstrip("\n"), "    }", "  }", ""]
    return src[:start] + "\n".join(staged) + src[end:], 1

  source, k = batch_loop(source, "Ridx5", "  float buf5[1];", (
    ("val1", "data2_1048576+(alu12+32)"),
    ("val2", "data2_1048576+alu12")))
  source, v = batch_loop(source, "Ridx9", "  float buf8[1];", (
    ("val3", "data2_1048576+(alu40+524288)"),
    ("val4", "data2_1048576+(alu40+524320)")))
  if force_ptx:
    helper = ('__device__ __forceinline__ uint4 forced_ldg128(const uint4 *p) { uint4 o; '
      'asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : '
      '"=r"(o.x), "=r"(o.y), "=r"(o.z), "=r"(o.w) : "l"(p) : "memory"); return o; }\n')
    source = source.replace('extern "C"', helper+'extern "C"', 1)
  return inflight_symbol, source, k+v


def _unroll_pv_smem_source(source:str, symbol:str) -> tuple[str, str, int]:
  """Keep the transposed final-PV shared stage compile-time expanded."""
  unrolled_symbol = symbol + "_unrollpv"
  source = source.replace(symbol + "(", unrolled_symbol + "(", 1)
  out, inserted = [], 0
  for line in source.splitlines():
    if ("for (int Lidx13 = 0; Lidx13 < 16; Lidx13++)" in line or
        "for (int Lidx17 = 0; Lidx17 < 16; Lidx17++)" in line):
      out.append("  #pragma unroll")
      inserted += 1
    out.append(line)
  if inserted != 2: raise RuntimeError(f"expected two transposed-PV loops, marked {inserted}")
  return unrolled_symbol, "\n".join(out), inserted


def _sass_load_grammar(binary:pathlib.Path, symbol:str) -> dict:
  nvdisasm = ROOT / ".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm"
  env = dict(os.environ, NVDISASM_PATH=str(nvdisasm), PATH=f"{nvdisasm.parent}:{os.environ.get('PATH','')}")
  cp = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(binary)],
                      capture_output=True, text=True, check=True, env=env)
  marker=f"Function : {symbol}\n"; start=cp.stdout.index(marker); tail=cp.stdout[start+len(marker):]
  if "Function :" in tail: tail=tail[:tail.index("Function :")]
  code_words=re.findall(r"/\* (0x[0-9a-f]+) \*/", tail)
  return {"ldg_e_128_constant":tail.count("LDG.E.128.CONSTANT"),
          "ldg_e_128_ordinary":sum(1 for x in tail.splitlines() if "LDG.E.128 " in x),
          "ldg_total":sum(1 for x in tail.splitlines() if " LDG." in x),
          "machine_word_count":len(code_words),
          "machine_words_sha256":hashlib.sha256("\n".join(code_words).encode()).hexdigest()}


HARNESS = r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
__CONTROL_SOURCE__
__CANDIDATE_SOURCE__
static void ck(cudaError_t e,const char* w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));exit(2);}}
static double run_control(float* o,float* q,unsigned int* c,int n){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int i=0;i<n;i++)__CONTROL__<<<dim3(__CONTROL_SPLITS__,32,1),dim3(32,4,1)>>>(o,q,c);cudaEventRecord(b);ck(cudaEventSynchronize(b),"control");float ms;cudaEventElapsedTime(&ms,a,b);return 1000.0*ms/n;}
static double run_candidate(float* o,float* q,unsigned int* c,int n){cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);for(int i=0;i<n;i++)__CANDIDATE__<<<dim3(__CAND_SPLITS__,32,1),dim3(32,4,1)>>>(o,q,c);cudaEventRecord(b);ck(cudaEventSynchronize(b),"candidate");float ms;cudaEventElapsedTime(&ms,a,b);return 1000.0*ms/n;}
int main(int ac,char**av){int n=ac>1?atoi(av[1]):400,r=ac>2?atoi(av[2]):9;float *oc,*on,*q;unsigned int*c;ck(cudaMalloc(&oc,33280*4),"oc");ck(cudaMalloc(&on,__CAND_OUT__*4),"on");ck(cudaMalloc(&q,4096*4),"q");ck(cudaMalloc(&c,1048576*4),"c");float*hq=(float*)malloc(4096*4);unsigned int*hc=(unsigned int*)malloc(1048576*4);for(int i=0;i<4096;i++)hq[i]=((i*17+3)%127-63)/256.0f;for(int i=0;i<1048576;i++)hc[i]=(i*2654435761u)^0x3c003c00u;cudaMemcpy(q,hq,4096*4,cudaMemcpyHostToDevice);cudaMemcpy(c,hc,1048576*4,cudaMemcpyHostToDevice);free(hq);free(hc);__CONTROL__<<<dim3(__CONTROL_SPLITS__,32,1),dim3(32,4,1)>>>(oc,q,c);__CANDIDATE__<<<dim3(__CAND_SPLITS__,32,1),dim3(32,4,1)>>>(on,q,c);ck(cudaDeviceSynchronize(),"warm");if(__COMPARE__){float *a=(float*)malloc(__CAND_OUT__*4),*b=(float*)malloc(__CAND_OUT__*4);cudaMemcpy(a,oc,__CAND_OUT__*4,cudaMemcpyDeviceToHost);cudaMemcpy(b,on,__CAND_OUT__*4,cudaMemcpyDeviceToHost);int ne=0;float md=0;for(int i=0;i<__CAND_OUT__;i++){unsigned int ua=*((unsigned int*)(a+i)),ub=*((unsigned int*)(b+i));if(ua!=ub)ne++;float d=fabsf(a[i]-b[i]);if(d>md)md=d;}printf("exact_mismatches=%d max_abs=%.9g\n",ne,md);free(a);free(b);}for(int i=0;i<r;i++)printf("rep=%d control=%.6f candidate=%.6f\n",i,run_control(oc,q,c,n),run_candidate(on,q,c,n));}
'''


def _ncu(binary:pathlib.Path, symbol:str, cache_control:str) -> list[dict[str, str]]:
  cp = subprocess.run(["sudo", "-n", NCU, "-k", symbol, "--launch-skip", "1", "--launch-count", "1",
    "--cache-control", cache_control, "--metrics", METRICS, "--csv", str(binary), "2", "1"],
    capture_output=True, text=True)
  if cp.returncode: raise RuntimeError(f"ncu {symbol}/{cache_control} failed: {cp.stderr[-5000:]}")
  rows, header = [], None
  for cols in csv.reader(io.StringIO(cp.stdout)):
    if cols and cols[0] == "ID": header = cols; continue
    if header is not None and len(cols) == len(header):
      row = dict(zip(header, cols)); rows.append({"metric":row["Metric Name"], "unit":row["Metric Unit"],
                                                  "value":row["Metric Value"]})
  return rows


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--passes", type=int, default=400)
  ap.add_argument("--reps", type=int, default=9); ap.add_argument("--ncu", action="store_true")
  ap.add_argument("--readonly-candidate", action="store_true")
  ap.add_argument("--unroll-columns-candidate", action="store_true")
  ap.add_argument("--inflight-columns-candidate", type=int, choices=(2, 4, 8))
  ap.add_argument("--force-inflight-ptx", action="store_true")
  ap.add_argument("--transpose-pv-smem-candidate", action="store_true")
  ap.add_argument("--wide-q-f32-candidate", action="store_true")
  ap.add_argument("--unroll-pv-smem-candidate", action="store_true")
  ap.add_argument("--matched-control", action="store_true")
  ap.add_argument("--use-fast-math", action="store_true")
  ap.add_argument("--artifacts-dir", type=pathlib.Path,
                  help="preserve probe.cu and probe for follow-up disassembly/source-counter attribution")
  ap.add_argument("--candidate-splits", type=int, default=4); ap.add_argument("--token-bound", type=int, default=512)
  ap.add_argument("--out", type=pathlib.Path, required=True); args = ap.parse_args()
  if args.token_bound != args.candidate_splits*128 or args.token_bound > MAXC:
    raise ValueError("token-bound must equal candidate-splits*128 and fit MAXC")
  control_splits, control_bound = ((args.candidate_splits, args.token_bound) if args.matched_control else (8, None))
  control, control_src = _render(control_splits, control_bound)
  candidate, candidate_src = _render(args.candidate_splits, args.token_bound,
                                     transpose_pv_smem=args.transpose_pv_smem_candidate)
  wide_q_rewrites = 0
  if args.wide_q_f32_candidate:
    candidate, candidate_src, wide_q_rewrites = _wide_q_f32_source(candidate_src, candidate)
  readonly_rewrites = 0
  if args.readonly_candidate:
    candidate, candidate_src, readonly_rewrites = _readonly_cache_source(candidate_src, candidate)
  unroll_pragmas = 0
  if args.unroll_columns_candidate:
    candidate, candidate_src, unroll_pragmas = _unroll_column_source(candidate_src, candidate)
  inflight_rewrites = 0
  if args.inflight_columns_candidate is not None:
    candidate, candidate_src, inflight_rewrites = _inflight_column_source(candidate_src, candidate, args.inflight_columns_candidate,
                                                                           args.force_inflight_ptx)
  elif args.force_inflight_ptx: raise ValueError("forced in-flight PTX requires --inflight-columns-candidate")
  pv_smem_unroll_pragmas = 0
  if args.unroll_pv_smem_candidate:
    if not args.transpose_pv_smem_candidate: raise ValueError("PV-smem unroll requires the transposed candidate")
    candidate, candidate_src, pv_smem_unroll_pragmas = _unroll_pv_smem_source(candidate_src, candidate)
  candidate_start = candidate_src.index("__device__ __forceinline__ uint4 forced_ldg128") if args.force_inflight_ptx else \
    candidate_src.index('extern "C"')
  candidate_src = candidate_src[candidate_start:]
  source = HARNESS.replace("__CONTROL_SOURCE__", control_src).replace("__CANDIDATE_SOURCE__", candidate_src)
  source = source.replace("__CONTROL__", control).replace("__CANDIDATE__", candidate)
  source = source.replace("__CONTROL_SPLITS__", str(control_splits))
  source = source.replace("__CAND_SPLITS__", str(args.candidate_splits)).replace("__CAND_OUT__", str(Hq*args.candidate_splits*W))
  source = source.replace("__COMPARE__", "1" if args.matched_control else "0")
  if args.artifacts_dir is not None:
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
  workdir = contextlib.nullcontext(str(args.artifacts_dir)) if args.artifacts_dir is not None else \
    tempfile.TemporaryDirectory(prefix="nv_flash_bounded_")
  with workdir as td:
    cu, binary = pathlib.Path(td)/"probe.cu", pathlib.Path(td)/"probe"
    cu.write_text(source)
    compile_cmd=[NVCC, "-arch=sm_120a", "-O3", "-lineinfo", "-std=c++17", "--ptxas-options=-v"]
    if args.use_fast_math: compile_cmd.append("-use_fast_math")
    build = subprocess.run([*compile_cmd, str(cu), "-o", str(binary)],
                           capture_output=True, text=True)
    if build.returncode: raise RuntimeError(build.stderr[-10000:])
    run = subprocess.run([str(binary), str(args.passes), str(args.reps)], capture_output=True, text=True, check=True)
    sass = {"control":_sass_load_grammar(binary, control), "candidate":_sass_load_grammar(binary, candidate)}
    counters = {arm:{state:_ncu(binary, symbol, "none" if state == "hot" else "all") for state in ("hot", "cold")}
                for arm,symbol in (("control",control),("candidate",candidate))} if args.ncu else None
  cv, nv = [], []
  exactness = None
  for line in run.stdout.splitlines():
    if m := re.match(r"rep=\d+ control=([0-9.]+) candidate=([0-9.]+)", line): cv.append(float(m.group(1))); nv.append(float(m.group(2)))
    if m := re.match(r"exact_mismatches=(\d+) max_abs=([0-9.eE+-]+)", line):
      exactness = {"bit_mismatches":int(m.group(1)), "max_abs":float(m.group(2))}
  payload = {"schema":"tinygrad.nv_flash_bounded_counter_probe.v1", "shape":{"Hq":Hq,"Hkv":Hkv,"Hd":Hd,"MAXC":MAXC,"Tc":Tc},
    "control":{"symbol":control,"splits":control_splits,"token_bound":control_bound,"samples_us":cv,"median_us":statistics.median(cv)},
    "candidate":{"symbol":candidate,"splits":args.candidate_splits,"token_bound":args.token_bound,
                 "samples_us":nv,"median_us":statistics.median(nv)},
    "ratio":statistics.median(nv)/statistics.median(cv), "readonly_candidate":args.readonly_candidate,
    "readonly_rewrites":readonly_rewrites, "unroll_columns_candidate":args.unroll_columns_candidate,
    "unroll_pragmas":unroll_pragmas, "inflight_columns_candidate":args.inflight_columns_candidate,
    "inflight_rewrites":inflight_rewrites, "force_inflight_ptx":args.force_inflight_ptx,
    "transpose_pv_smem_candidate":args.transpose_pv_smem_candidate,
    "wide_q_f32_candidate":args.wide_q_f32_candidate,
    "wide_q_rewrites":wide_q_rewrites,
    "unroll_pv_smem_candidate":args.unroll_pv_smem_candidate, "pv_smem_unroll_pragmas":pv_smem_unroll_pragmas,
    "matched_control":args.matched_control,
    "exactness":exactness,
    "use_fast_math":args.use_fast_math, "sass_load_grammar":sass, "ptxas":build.stderr.splitlines()}
  if counters is not None: payload["ncu"] = {"method":"one post-warmup launch; cache-control none=hot, all=cold", "arms":counters}
  args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
  print(json.dumps(payload, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
