#!/usr/bin/env python3
"""Final causal split of the Q6 Region A producer.

Arm A retains canonical Q6 loads and the complete low/high-bit decode expressions,
but has no shared-memory publication. Arm B publishes 80 already-decoded bytes
through shared memory with two barriers, but has no canonical Q6/Q8 path.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, re, statistics, subprocess
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.runtime.ops_nv import NVProgram
from extra.llm_research.layout import GGML_Q6_K, packed_u16_slice, read_metadata

ROOT = pathlib.Path(__file__).resolve().parents[3]
M, N, K = 512, 4096, 12288
CTAS, THREADS, ITERATIONS = 128, 256, 192


def _decode_source(base: str) -> tuple[str, dict[str, int]]:
  loop_start = base.find("  for (int Ridx0 = 0; Ridx0 < 192; Ridx0++) {")
  barrier0 = base.find("    __syncthreads();", loop_start)
  barrier1 = base.find("    __syncthreads();", barrier0+1)
  if min(loop_start, barrier0, barrier1) < 0: raise ValueError("compiler producer boundaries not found")
  load_region, decode_region = base[loop_start:barrier0], base[barrier0:barrier1]
  loads = []
  for line in load_region.splitlines()[1:]:
    text = line.strip()
    if re.match(r"int alu(?:77|78|79|80|81) =", text) or re.match(r"unsigned short val(?:[0-9]|[12][0-9]|3[0-5]) =", text):
      loads.append(line.replace("data2_20643840", "canonical_q6"))
  prelude = []
  for line in decode_region.splitlines():
    if re.match(r"\s*(?:int alu85|unsigned short cast1|int alu86) =", line): prelude.append(line)
  decoded = [line for line in decode_region.splitlines() if "*(buf1+" in line and "+-32" in line]
  # One generated K64 producer phase emits 32 Q6 values per thread. Across
  # the two lane/phase selections these expressions cover both Q6 low/high
  # halves; requiring 64 here would incorrectly include another loop phase.
  if len(loads) != 41 or len(decoded) != 32 or len(prelude) != 3:
    raise ValueError(f"unexpected decode extraction: loads={len(loads)} decoded={len(decoded)} prelude={len(prelude)}")
  sinks = []
  for i, line in enumerate(decoded):
    rhs = line.split(" = ", 1)[1].rstrip(";")
    sinks.append(f"    sink{i&7} = sink{i&7} * 16777619u ^ (unsigned int)(unsigned char)({rhs});")
  src = f'''extern "C" __global__ void __launch_bounds__(256) nv_q6_region_a_decode_only(unsigned int *out, const unsigned short *canonical_q6) {{
  int gidx0 = blockIdx.x;
  int gidx1 = blockIdx.y;
  int lidx0 = threadIdx.x;
  int lidx1 = threadIdx.y;
  int lidx2 = threadIdx.z;
  int alu0 = (lidx0>>2);
  int alu3 = (lidx0&3);
  int alu12 = ((lidx0>>1)&1);
  unsigned int sink0=2166136261u, sink1=2166136260u, sink2=2166136259u, sink3=2166136258u;
  unsigned int sink4=2166136257u, sink5=2166136256u, sink6=2166136255u, sink7=2166136254u;
  for (int Ridx0 = 0; Ridx0 < 192; Ridx0++) {{
{chr(10).join(loads)}
{chr(10).join(prelude)}
{chr(10).join(sinks)}
  }}
  int tid = lidx0 + 32*lidx1 + 64*lidx2;
  int cta = gidx0 + 32*gidx1;
  out[cta*256+tid] = sink0^sink1^sink2^sink3^sink4^sink5^sink6^sink7;
}}
'''
  return src, {"canonical_q6_load_statements": 36, "decode_output_expressions": len(decoded), "register_sinks": 8}


def _publication_source() -> str:
  stores = []
  for slot in range(80):
    stores.append(f"    published[{slot}*256+tid] = (unsigned char)((tid*17 + {slot}*29) & 255);")
  return f'''extern "C" __global__ void __launch_bounds__(256) nv_q6_region_a_publication_only(unsigned int *out, const unsigned char *predecoded_selector) {{
  __shared__ unsigned char shared_bytes[20480];
  volatile unsigned char *published = shared_bytes;
  int tid = threadIdx.x + 32*threadIdx.y + 64*threadIdx.z;
  int cta = blockIdx.x + 32*blockIdx.y;
  unsigned int selected = (unsigned int)(predecoded_selector[tid] % 80);
  unsigned int sink = 2166136261u ^ (unsigned int)(cta*256+tid);
  #pragma unroll 1
  for (int Ridx0 = 0; Ridx0 < 192; Ridx0++) {{
    __syncthreads();
{chr(10).join(stores)}
    __syncthreads();
    sink = sink * 16777619u ^ (unsigned int)published[selected*256+tid];
  }}
  out[cta*256+tid] = sink;
}}
'''


def _sass(binary: bytes, stem: pathlib.Path) -> tuple[dict[str, object], str]:
  cubin, sass_path = stem.with_suffix(".cubin"), stem.with_suffix(".sass")
  cubin.write_bytes(binary)
  tool = ROOT/".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm"
  env = dict(os.environ, NVDISASM_PATH=str(tool), PATH=f"{tool.parent}:{os.environ.get('PATH','')}")
  cp = subprocess.run(["/usr/local/cuda-13.2/bin/cuobjdump", "--dump-resource-usage", "--dump-sass", str(cubin)], capture_output=True, text=True, env=env)
  text = cp.stdout+cp.stderr; sass_path.write_text(text)
  match = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", text)
  count = lambda op: len(re.findall(rf"\b{op}(?:\.|\s)", text))
  info = {"sha256": hashlib.sha256(binary).hexdigest(), "returncode": cp.returncode,
          "counts": {op.lower(): count(op) for op in ("LDG", "LDS", "STS", "BAR", "IMMA", "LDL", "STL", "PRMT", "LOP3")},
          "resources": dict(zip(("registers", "stack_bytes", "shared_bytes", "local_bytes"), map(int, match.groups()))) if match else None,
          "cubin": str(cubin), "sass": str(sass_path)}
  return info, text


def _stats(samples: list[float]) -> dict[str, object]:
  kept = samples[3:]
  return {"samples_us": samples, "warmup_discarded": 3, "min_us": min(kept), "median_us": statistics.median(kept), "max_us": max(kept)}


def _launch(name: str, source: str, buffers: tuple, rounds: int, stem: pathlib.Path) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
  binary = Device["NV"].compiler.compile(source); sass, _ = _sass(binary, stem)
  p = NVProgram(Device["NV"], name, binary)
  out = Tensor.full((CTAS*THREADS,), 0xFFFFFFFF, dtype=dtypes.uint32, device="NV").contiguous().realize()
  args = (out.uop.buffer.get_buf("NV"),)+buffers
  p(*args, global_size=(32,4,1), local_size=(32,2,4), wait=True, timeout=10)
  first = out.numpy().copy()
  out.assign(0xFFFFFFFF).realize(); p(*args, global_size=(32,4,1), local_size=(32,2,4), wait=True, timeout=10)
  second = out.numpy().copy()
  samples = [p(*args, global_size=(32,4,1), local_size=(32,2,4), wait=True, timeout=10)*1e6 for _ in range(rounds)]
  return second, sass, {"timing": _stats(samples), "repeat_exact": bool(np.array_equal(first, second)), "nonzero": int(np.count_nonzero(second)), "sample": second[:8].tolist()}


def _publication_reference(selector: np.ndarray) -> np.ndarray:
  result = np.empty((CTAS, THREADS), dtype=np.uint32)
  for cta in range(CTAS):
    for tid in range(THREADS):
      sink = np.uint32(2166136261 ^ (cta*THREADS+tid))
      value = np.uint32((tid*17 + int(selector[tid] % 80)*29) & 255)
      for _ in range(ITERATIONS): sink = np.uint32(np.uint64(sink)*16777619) ^ value
      result[cta, tid] = sink
  return result.ravel()


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--base-source", default=str(ROOT/"docs/task_workflow/evidence/nv-q6-region-a-20260831/full-artifacts/region_a_base.cu"))
  ap.add_argument("--rounds", type=int, default=9); ap.add_argument("--out", required=True); ap.add_argument("--artifacts", required=True)
  args = ap.parse_args()
  if args.rounds < 9: raise ValueError("R9 or higher is required")
  art = pathlib.Path(args.artifacts); art.mkdir(parents=True, exist_ok=True)
  base = pathlib.Path(args.base_source).read_text(); decode_src, decode_logical = _decode_source(base)
  publication_src = _publication_source()
  (art/"decode_only.cu").write_text(decode_src); (art/"publication_only.cu").write_text(publication_src)

  model = pathlib.Path(args.model); meta = read_metadata(model); info = next(i for i in meta.infos if i.name == "blk.0.ffn_down.weight")
  if info.typ != GGML_Q6_K: raise RuntimeError("fixture is not canonical Q6_K")
  q6 = packed_u16_slice(model, meta, info, device="NV").contiguous().realize()
  decoded, decoded_sass, decoded_run = _launch("nv_q6_region_a_decode_only", decode_src, (q6.uop.buffer.get_buf("NV"),), args.rounds, art/"decode_only")

  selector_np = ((np.arange(THREADS, dtype=np.uint16)*37+11)%80).astype(np.uint8)
  selector = Tensor(selector_np, device="NV").contiguous().realize()
  published, published_sass, published_run = _launch("nv_q6_region_a_publication_only", publication_src, (selector.uop.buffer.get_buf("NV"),), args.rounds, art/"publication_only")
  publication_ref = _publication_reference(selector_np)
  published_run["cpu_exact"] = bool(np.array_equal(published, publication_ref))

  dc, dr = decoded_sass["counts"], decoded_sass["resources"]
  pc, pr = published_sass["counts"], published_sass["resources"]
  decode_gates = {"zero_shared_allocation": dr is not None and dr["shared_bytes"] == 0, "zero_sts": dc["sts"] == 0,
                  "zero_bar": dc["bar"] == 0, "zero_imma": dc["imma"] == 0, "zero_q8_source": "data1_" not in decode_src,
                  "canonical_q6_source": "canonical_q6" in decode_src, "no_local_spills": dc["ldl"] == dc["stl"] == 0}
  publication_gates = {"logical_shared_stores_80": publication_src.count("published[")-1 == 80,
                       "sass_sts_80": pc["sts"] == 80, "sass_bar_2": pc["bar"] == 2, "zero_imma": pc["imma"] == 0,
                       "zero_canonical_source": all(x not in publication_src for x in ("canonical_q6", "data1_", "data2_", "+-32")),
                       "shared_allocation_covers_20480": pr is not None and pr["shared_bytes"] >= 20480, "no_local_spills": pc["ldl"] == pc["stl"] == 0}
  result = {"schema": "tinygrad.nv_q6_region_a_final_split.v1", "shape": {"M":M,"N":N,"K":K},
            "launch": {"ctas":CTAS,"threads":THREADS,"local_size":[32,2,4],"iterations":ITERATIONS,"rounds":args.rounds},
            "decode_only": {"logical":decode_logical,"sass":decoded_sass,"run":decoded_run,"gates":decode_gates},
            "publication_only": {"logical":{"shared_store_statements":80,"barriers":2},"sass":published_sass,"run":published_run,"gates":publication_gates}}
  result["passed"] = all(decode_gates.values()) and all(publication_gates.values()) and decoded_run["repeat_exact"] and decoded_run["nonzero"] == CTAS*THREADS and published_run["repeat_exact"] and published_run["cpu_exact"] and published_run["nonzero"] == CTAS*THREADS
  pathlib.Path(args.out).write_text(json.dumps(result, indent=2)+"\n"); print(json.dumps(result, sort_keys=True))
  return 0 if result["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
