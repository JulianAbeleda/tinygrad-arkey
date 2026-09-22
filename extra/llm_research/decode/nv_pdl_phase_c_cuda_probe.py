#!/usr/bin/env python3
"""Phase C CUDA-PDL half of the same-grid semantic discriminator.

Compiles one CUDA program whose producer/consumer bodies, grid geometry, spin
budget, and checksum match ``nv_pdl_trigger_probe.py`` exactly, then runs one
arm per process:

  no_pdl          normal same-stream serialization, no in-kernel PDL
  pdl_end         CUDA PDL attribute on the producer, trigger at body end
  pdl_start       CUDA PDL attribute on the producer, trigger at body start
  pdl_prologue    trigger at end, consumer waits after index math before
                  its first dependent load (llama's wait placement)

Every arm records producer grid start, launch-complete trigger, producer grid
end, consumer grid start, data-wait exit, and consumer end from
``%globaltimer``.  Correctness is a full-coverage checksum.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCHEMA = "tinygrad.nv_pdl_phase_c_cuda.v1"
LS, GS = 256, 4096
NPROD = 1 << 23
NCONS = 513 * 1024
SPIN_NS = 100_000

ARMS = ("no_pdl", "pdl_end", "pdl_start", "pdl_prologue")

CPP = r"""
#include <cuda_runtime.h>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define LS 256u
#define GS 4096u
#define NPROD (1u << 23)
#define NCONS (513u * 1024u)
#define SPIN_NS 100000ull

__device__ __forceinline__ unsigned long long gt() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void pdl_producer(unsigned int* out, unsigned long long* t) {
  unsigned long long now;
#if ARM_PDL == 2
  // The trigger itself provides no memory-visibility guarantee, so make the
  // t[6] timestamp visible before the dependent grid can observe the trigger.
  cudaTriggerProgrammaticLaunchCompletion();
  now = gt();
  if (threadIdx.x == 0u && blockIdx.x == GS - 1u) t[6] = now;
  __threadfence();
#else
  now = gt();
#endif
  if (threadIdx.x == 0u && blockIdx.x == 0u) t[0] = now;
  atomicMin((unsigned long long*)&t[7], now);
  unsigned int i = blockIdx.x * LS + threadIdx.x;
  for (; i < NPROD; i += GS * LS) out[i] = i * 2u + 1u;
  if (threadIdx.x == 0u && blockIdx.x == GS - 1u) {
    unsigned long long end = now + SPIN_NS;
    do { now = gt(); } while (now < end);
  }
  if (threadIdx.x == 0u && blockIdx.x == 0u) t[4] = now;
#if ARM_PDL == 1
  now = gt();
  if (threadIdx.x == 0u && blockIdx.x == GS - 1u) t[6] = now;
  __threadfence();
  cudaTriggerProgrammaticLaunchCompletion();
#endif
  now = gt();
  if (threadIdx.x == 0u) atomicMax((unsigned long long*)&t[1], now);
}

__global__ void pdl_consumer(const unsigned int* in, unsigned int* chk, unsigned long long* t) {
  unsigned long long now = gt();
  if (threadIdx.x == 0u && blockIdx.x == 0u) t[2] = now;
#if ARM_WAIT == 2
  // Match llama's MMVQ placement: prologue index math precedes the dependency
  // wait, and no dependent load occurs before the synchronization.
  unsigned int i = threadIdx.x;
  unsigned int s0 = 0u, s1 = 0u, s2 = 0u, s3 = 0u;
  cudaGridDependencySynchronize();
  now = gt();
  if (threadIdx.x == 0u && blockIdx.x == 0u) t[5] = now;
#elif ARM_WAIT == 1
  cudaGridDependencySynchronize();
  now = gt();
  if (threadIdx.x == 0u && blockIdx.x == 0u) t[5] = now;
  unsigned int i = threadIdx.x;
  unsigned int s0 = 0u, s1 = 0u, s2 = 0u, s3 = 0u;
#else
  now = gt();
  if (threadIdx.x == 0u && blockIdx.x == 0u) t[5] = now;
  unsigned int i = threadIdx.x;
  unsigned int s0 = 0u, s1 = 0u, s2 = 0u, s3 = 0u;
#endif
  for (; i < NCONS; i += LS * 4u) {
    s0 += in[i];
    s1 += in[i + LS];
    s2 += in[i + LS * 2u];
    s3 += in[i + LS * 3u];
  }
  atomicAdd(chk, s0 + s1 + s2 + s3);
  now = gt();
  if (threadIdx.x == 0u && blockIdx.x == 0u) t[3] = now;
}

static const char* cuda_error_name(cudaError_t e) { return cudaGetErrorString(e); }

int main(int argc, char** argv) {
  if (argc != 4) { fprintf(stderr, "usage: probe <arm> <reps> <json>\n"); return 2; }
  const char* arm_name = argv[1];
  int use_pdl = strcmp(arm_name, "no_pdl") != 0;
  int reps = atoi(argv[2]);
  unsigned int* out = nullptr;
  unsigned int* chk = nullptr;
  unsigned long long* t = nullptr;
  cudaError_t err = cudaMallocManaged(&out, NPROD * 4u);
  if (err != cudaSuccess) { fprintf(stderr, "out alloc: %s\n", cuda_error_name(err)); return 3; }
  err = cudaMallocManaged(&chk, 4u);
  if (err != cudaSuccess) return 4;
  err = cudaMallocManaged(&t, 8u * 8u);
  if (err != cudaSuccess) return 5;
  cudaStream_t stream;
  err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) return 6;
  cudaMemset(out, 0, NPROD * 4u);
  cudaDeviceSynchronize();

  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute attrs[1] = {};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.gridDim = dim3(GS, 1, 1);
  cfg.blockDim = dim3(LS, 1, 1);
  cfg.stream = stream;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;

  FILE* fp = fopen(argv[3], "wb");
  fprintf(fp, "{\"arm\":\"%s\",\"rows\":[", arm_name);
  for (int rep = 0; rep < reps; rep++) {
    // Reset the timestamp and checksum buffers on the same stream as the
    // launches.  Previous versions used the default stream and were not
    // ordered against this stream, which corrupted slots after the first rep.
    cudaMemsetAsync(t, 0, 64u, stream);
    unsigned long long max_u64 = ~0ull;
    cudaMemcpyAsync(&t[7], &max_u64, 8u, cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(chk, 0, 4u, stream);
    // Zero out every rep so the checksum actually certifies the current
    // producer body, not data left over from the first rep.
    cudaMemsetAsync(out, 0, NPROD * 4u, stream);
    if (use_pdl) {
      // The programmatic stream serialization attribute must be present on the
      // dependent launch for the driver to start it before the primary grid
      // completes. llama.cpp sets it on every kernel; this probe does the same.
      // Restore the producer grid every rep; the consumer launch below rewrites
      // cfg.gridDim, and without this the next producer silently becomes a
      // one-block grid.
      cfg.gridDim = dim3(GS, 1, 1);
      err = cudaLaunchKernelEx(&cfg, pdl_producer, out, t);
      if (err != cudaSuccess) { fprintf(stderr, "producer launch: %s\n", cuda_error_name(err)); return 7; }
#ifndef ARM_NO_CONSUMER
      cfg.gridDim = dim3(1, 1, 1);
      err = cudaLaunchKernelEx(&cfg, pdl_consumer, out, chk, t);
      if (err != cudaSuccess) { fprintf(stderr, "consumer launch: %s\n", cuda_error_name(err)); return 9; }
#endif
    } else {
      pdl_producer<<<GS, LS, 0, stream>>>(out, t);
#ifndef ARM_NO_CONSUMER
      pdl_consumer<<<1, LS, 0, stream>>>(out, chk, t);
#endif
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) { fprintf(stderr, "sync: %s\n", cuda_error_name(err)); return 8; }
    // Managed-memory host reads are only reliable after a device-wide sync.
    cudaDeviceSynchronize();
    unsigned int checksum = 0;
    cudaMemcpy(&checksum, chk, 4u, cudaMemcpyDeviceToHost);
    unsigned long long slot[8] = {0};
    cudaMemcpy(slot, t, 64u, cudaMemcpyDeviceToHost);
    unsigned int tail_value = 0;
    cudaMemcpy(&tail_value, &out[NPROD - 1], 4u, cudaMemcpyDeviceToHost);
    if (rep) fprintf(fp, ",");
    fprintf(fp, "{\"block0_start_ns\":%llu,\"grid_start_ns\":%llu,\"trigger_ns\":%llu,\"prod_end_ns\":%llu,"
      "\"cons_grid_start_ns\":%llu,\"wait_exit_ns\":%llu,\"cons_end_ns\":%llu,\"checksum\":%u,\"tail_value\":%u,",
      slot[0], slot[7], slot[6], slot[1], slot[2], slot[5], slot[3], checksum, tail_value);
    fprintf(fp, "\"checksum_correct\":%s,\"tail_correct\":%s}",
      checksum == (unsigned int)((uint64_t)NCONS * NCONS % (1ull << 32)) ? "true" : "false",
      tail_value == (NPROD - 1) * 2u + 1u ? "true" : "false");
  }
  fprintf(fp, "]}\n");
  fclose(fp);
  cudaFree(out); cudaFree(chk); cudaFree(t);
  cudaStreamDestroy(stream);
  return 0;
}
"""


def _atomic_json(path: pathlib.Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(f".{path.name}.tmp")
  tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  tmp.replace(path)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--arm", choices=ARMS)
  ap.add_argument("--reps", type=int, default=12)
  ap.add_argument("--warmup", type=int, default=1, help="initial reps excluded from medians")
  ap.add_argument("--producer-only", action="store_true",
                  help="debug mode: launch the producer without the consumer")
  ap.add_argument("--compile-only", action="store_true",
                  help="compile all four arm binaries without touching the GPU")
  ap.add_argument("--workdir", type=pathlib.Path, default=pathlib.Path("/tmp/nv-pdl-phase-c"))
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()
  if not args.compile_only and args.arm is None:
    ap.error("--arm is required unless --compile-only is set")

  def compile_arm(arm: str, workdir: pathlib.Path) -> pathlib.Path:
    armdir = workdir / arm
    armdir.mkdir(parents=True, exist_ok=True)
    src = armdir / "probe.cu"
    exe = armdir / "probe"
    arm_pdl = {"no_pdl": 0, "pdl_end": 1, "pdl_start": 2, "pdl_prologue": 1}[arm]
    arm_wait = {"no_pdl": 0, "pdl_end": 1, "pdl_start": 1, "pdl_prologue": 2}[arm]
    extra = "#define ARM_NO_CONSUMER 1\n" if args.producer_only else ""
    src.write_text(f"#define ARM_PDL {arm_pdl}\n#define ARM_WAIT {arm_wait}\n{extra}" + CPP)
    nvcc = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
    compile_cmd = [nvcc, "-arch=sm_120", "-O3", "-std=c++17", "-o", str(exe), str(src)]
    run = subprocess.run(compile_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if run.returncode:
      raise RuntimeError(f"nvcc failed for {arm}: {run.stderr[-6000:]}")
    return exe

  if args.compile_only:
    built = {arm: str(compile_arm(arm, args.workdir)) for arm in ARMS}
    _atomic_json(args.out, {"schema": SCHEMA, "compile_only": True, "binaries": built})
    print(json.dumps(built, indent=2))
    return 0

  workdir = args.workdir / args.arm
  workdir.mkdir(parents=True, exist_ok=True)
  exe = compile_arm(args.arm, args.workdir)
  src = args.workdir / args.arm / "probe.cu"

  raw = workdir / "raw.json"
  run = subprocess.run([str(exe), args.arm, str(args.reps), str(raw)],
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if run.returncode:
    raise RuntimeError(f"probe failed rc={run.returncode}: {run.stderr[-6000:]}")
  raw_rows = json.loads(raw.read_text())["rows"]
  rows = raw_rows[args.warmup:]
  if not rows:
    raise RuntimeError("no measured rows remain after warmup")

  def us(row, a, b):
    return (row[b] - row[a]) / 1000.0

  for row in rows:
    row.update({
      "producer_us": us(row, "grid_start_ns", "prod_end_ns"),
      "trigger_shadow_us": None if row["trigger_ns"] == 0 else us(row, "grid_start_ns", "trigger_ns"),
      "launch_shadow_us": us(row, "grid_start_ns", "cons_grid_start_ns"),
      "wait_us": us(row, "cons_grid_start_ns", "wait_exit_ns"),
      "overlap_us": us(row, "cons_grid_start_ns", "prod_end_ns"),
      "wall_us": us(row, "grid_start_ns", "cons_end_ns"),
    })
  def med(key):
    values = sorted(r[key] for r in rows if r[key] is not None)
    return round(values[len(values) // 2], 3) if values else None
  payload = {
    "schema": SCHEMA,
    "arm": args.arm,
    "producer_only": args.producer_only,
    "reps": len(rows),
    "total_rows": len(raw_rows),
    "warmup": args.warmup,
    "grid": {"GS": GS, "LS": LS, "spin_ns": SPIN_NS},
    "median": {k: med(k) for k in
               ("producer_us", "trigger_shadow_us", "launch_shadow_us", "wait_us",
                "overlap_us", "wall_us")},
    "checksum_correct_all": all(r["checksum_correct"] for r in rows),
    "checksum_correct_all_rows": all(r["checksum_correct"] for r in raw_rows),
    "rows": rows,
    "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
    "binary_path": str(exe),
  }
  _atomic_json(args.out, payload)
  print(json.dumps(payload["median"], indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
