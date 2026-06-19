#!/usr/bin/env python3
from __future__ import annotations

import array, json, os, shutil, struct, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTDIR = ROOT / "bench/amd-scheduler-tooling-backend"
WORKDIR = OUTDIR / "r1p2_hcq_replay_work"
OUT = OUTDIR / "r1p2_hcq_replay.json"
PREDISPATCH = OUTDIR / "r1p2_hcq_replay_predispatch.json"
ROCM = Path(os.environ.get("ROCM_PATH", "/opt/rocm-7.2.4"))

HELPER_SRC = r"""
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

typedef struct { uint64_t handle; } aqlprofile_handle_t;
typedef union {
  uint32_t raw;
  struct {
    uint32_t device_access : 1;
    uint32_t host_access : 1;
    uint32_t memory_hint : 6;
    uint32_t _reserved : 24;
  };
} aqlprofile_buffer_desc_flags_t;
typedef struct {
  hsa_ven_amd_aqlprofile_parameter_name_t parameter_name;
  union {
    uint32_t value;
    struct { uint32_t counter_id : 28; uint32_t simd_mask : 4; };
  };
} aqlprofile_att_parameter_t;
typedef struct {
  hsa_agent_t agent;
  const aqlprofile_att_parameter_t* parameters;
  uint32_t parameter_count;
} aqlprofile_att_profile_t;
typedef struct {
  hsa_ext_amd_aql_pm4_packet_t start_packet;
  hsa_ext_amd_aql_pm4_packet_t stop_packet;
} aqlprofile_att_control_aql_packets_t;
typedef hsa_status_t (*alloc_cb_t)(void**, uint64_t, aqlprofile_buffer_desc_flags_t, void*);
typedef void (*free_cb_t)(void*, void*);
typedef hsa_status_t (*copy_cb_t)(void*, const void*, size_t, void*);
typedef hsa_status_t (*att_create_packets_t)(aqlprofile_handle_t*, aqlprofile_att_control_aql_packets_t*, aqlprofile_att_profile_t, alloc_cb_t, free_cb_t, copy_cb_t, void*);

struct AllocRec {
  void* base;
  void* ptr;
  uint64_t size;
  uint32_t raw;
  uint32_t device_access;
  uint32_t host_access;
  uint32_t memory_hint;
};
static std::vector<AllocRec> allocs;
static hsa_agent_t gpu_agent{};
static bool found_gpu = false;

static hsa_status_t find_gpu(hsa_agent_t agent, void*) {
  hsa_device_type_t type{};
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) == HSA_STATUS_SUCCESS && type == HSA_DEVICE_TYPE_GPU) {
    gpu_agent = agent;
    found_gpu = true;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t cb_alloc(void** ptr, uint64_t size, aqlprofile_buffer_desc_flags_t flags, void*) {
  void* base = nullptr;
  size_t alloc_size = (size_t)size + 0x3000;
  int rc = posix_memalign(&base, 0x1000, alloc_size);
  if (rc != 0 || base == nullptr) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  memset(base, 0, alloc_size);
  uintptr_t aligned = ((uintptr_t)base + 0xfff) & ~((uintptr_t)0xfff);
  *ptr = (void*)aligned;
  allocs.push_back({base, *ptr, size, flags.raw, flags.device_access, flags.host_access, flags.memory_hint});
  return HSA_STATUS_SUCCESS;
}

static void cb_free(void* ptr, void*) {
  for (auto &r : allocs) {
    if (r.ptr == ptr && r.base != nullptr) {
      free(r.base);
      r.base = nullptr;
      return;
    }
  }
}

static hsa_status_t cb_copy(void* dst, const void* src, size_t size, void*) {
  if (size) memcpy(dst, src, size);
  return HSA_STATUS_SUCCESS;
}

static void print_hex_bytes(const void* p, size_t n) {
  const unsigned char* b = (const unsigned char*)p;
  for (size_t i = 0; i < n; i++) printf("%02x", b[i]);
}

static int packet_nonzero(const hsa_ext_amd_aql_pm4_packet_t* p) {
  const uint32_t* w = (const uint32_t*)p;
  int n = 0;
  for (size_t i = 0; i < sizeof(*p) / sizeof(uint32_t); i++) if (w[i]) n++;
  return n;
}

static void print_packet(const char* name, hsa_ext_amd_aql_pm4_packet_t* p) {
  p->header = 0x1500;  // vendor-specific type 0 + barrier + system acquire/release fences
  p->completion_signal.handle = 0;
  printf("\"%s\":{\"bytes\":%zu,\"nonzero_words\":%d,\"hex\":\"", name, sizeof(*p), packet_nonzero(p));
  print_hex_bytes(p, sizeof(*p));
  printf("\"}");
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  hsa_status_t hsa_init_st = hsa_init();
  if (hsa_init_st != HSA_STATUS_SUCCESS) {
    printf("{\"ok\":false,\"stage\":\"hsa_init\",\"hsa_init_status\":%u}\n", (unsigned)hsa_init_st);
    return 2;
  }
  hsa_status_t find_st = hsa_iterate_agents(find_gpu, nullptr);
  if (!found_gpu) {
    printf("{\"ok\":false,\"stage\":\"find_gpu\",\"find_status\":%u}\n", (unsigned)find_st);
    return 3;
  }
  void* lib = dlopen("libhsa-amd-aqlprofile64.so", RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    printf("{\"ok\":false,\"stage\":\"dlopen\",\"error\":\"%s\"}\n", dlerror());
    return 4;
  }
  auto create = (att_create_packets_t)dlsym(lib, "aqlprofile_att_create_packets");
  if (!create) {
    printf("{\"ok\":false,\"stage\":\"dlsym\"}\n");
    return 5;
  }

  aqlprofile_att_parameter_t params[4]{};
  params[0].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET; params[0].value = 1;
  params[1].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK; params[1].value = 1;
  params[2].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION; params[2].value = 1;
  params[3].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE; params[3].value = 8u << 20;

  aqlprofile_att_profile_t profile{};
  profile.agent = gpu_agent;
  profile.parameters = params;
  profile.parameter_count = 4;
  aqlprofile_handle_t handle{};
  aqlprofile_att_control_aql_packets_t packets{};
  hsa_status_t create_st = create(&handle, &packets, profile, cb_alloc, cb_free, cb_copy, nullptr);

  printf("{\"ok\":%s,\"hsa_init_status\":%u,\"find_status\":%u,\"create_status\":%u,\"handle\":%lu,",
         create_st == HSA_STATUS_SUCCESS ? "true" : "false", (unsigned)hsa_init_st, (unsigned)find_st,
         (unsigned)create_st, (unsigned long)handle.handle);
  printf("\"allocations\":[");
  for (size_t i = 0; i < allocs.size(); i++) {
    if (i) printf(",");
    auto &r = allocs[i];
    printf("{\"idx\":%zu,\"ptr\":%lu,\"size\":%lu,\"raw\":%u,\"device_access\":%u,\"host_access\":%u,\"memory_hint\":%u",
           i, (unsigned long)(uintptr_t)r.ptr, (unsigned long)r.size, r.raw, r.device_access, r.host_access, r.memory_hint);
    if (r.size <= 4096) {
      printf(",\"hex\":\"");
      print_hex_bytes(r.ptr, r.size);
      printf("\"");
    }
    printf("}");
  }
  printf("],");
  print_packet("start_packet", &packets.start_packet);
  printf(",");
  print_packet("stop_packet", &packets.stop_packet);
  printf("}\n");
  return create_st == HSA_STATUS_SUCCESS ? 0 : 6;
}
"""

BODY_SRC = r"""
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1, 64)))
void r1p2_body(float *out) {
  int gid = __builtin_amdgcn_workitem_id_x() + __builtin_amdgcn_workgroup_id_x() * 64;
  float x = (float)gid;
  #pragma unroll
  for (int i = 0; i < 4096; i++) x = __builtin_fmaf(x, 1.000113f, 0.25f);
  out[gid & 255] = x;
}
"""

def run(cmd: list[str], *, env: dict[str, str] | None = None, timeout: int = 120) -> dict[str, Any]:
  t0 = time.perf_counter()
  try:
    cp = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return {"cmd": cmd, "returncode": cp.returncode, "elapsed_s": round(time.perf_counter() - t0, 3),
            "stdout_tail": cp.stdout.splitlines()[-20:], "stderr_tail": cp.stderr.splitlines()[-40:]}
  except subprocess.TimeoutExpired as e:
    return {"cmd": cmd, "timeout": True, "elapsed_s": round(time.perf_counter() - t0, 3),
            "stdout_tail": (e.stdout or "").splitlines()[-20:] if isinstance(e.stdout, str) else [],
            "stderr_tail": (e.stderr or "").splitlines()[-40:] if isinstance(e.stderr, str) else []}

def build_and_run_helper() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
  shutil.rmtree(WORKDIR, ignore_errors=True)
  WORKDIR.mkdir(parents=True, exist_ok=True)
  src = WORKDIR / "aqlprofile_export_helper.cpp"
  exe = WORKDIR / "aqlprofile_export_helper"
  src.write_text(HELPER_SRC)
  build = run(["g++", "-O2", "-std=c++17", "-I", str(ROCM / "include"), str(src),
               "-L", str(ROCM / "lib"), "-Wl,-rpath," + str(ROCM / "lib"),
               "-ldl", "-lhsa-runtime64", "-o", str(exe)])
  if build.get("returncode") != 0: return build, {"ok": False, "reason": "build_failed"}, None
  env = os.environ.copy()
  env["LD_LIBRARY_PATH"] = f"{ROCM / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
  res = run([str(exe)], env=env, timeout=120)
  parsed = None
  for line in res.get("stdout_tail", [])[::-1]:
    if line.startswith("{") and line.endswith("}"):
      parsed = json.loads(line)
      break
  return build, res, parsed

def patch_ptr_ranges(blob: bytes, ranges: list[tuple[int, int, int]]) -> tuple[bytes, int]:
  out = bytearray(blob)
  patches = 0
  for off in range(0, max(0, len(out) - 7), 4):
    val = struct.unpack_from("<Q", out, off)[0]
    for old, new, size in ranges:
      if old <= val < old + size:
        struct.pack_into("<Q", out, off, new + (val - old))
        patches += 1
        break
  return bytes(out), patches

def patch_pm4_page_ranges(blob: bytes, ranges: list[tuple[int, int, int]]) -> tuple[bytes, int]:
  out = bytearray(blob)
  patches = 0
  page_ranges = []
  for old, new, size in ranges:
    page_ranges.append((old >> 12, new >> 12, (size + 0xfff) >> 12))
  for off in range(0, max(0, len(out) - 3), 4):
    val = struct.unpack_from("<I", out, off)[0]
    for old_page, new_page, pages in page_ranges:
      old_low = old_page & 0xffffffff
      if old_low <= val < old_low + pages:
        struct.pack_into("<I", out, off, (new_page + (val - old_low)) & 0xffffffff)
        patches += 1
        break
  return bytes(out), patches

def packet_words(pkt: bytes) -> list[int]:
  return list(struct.unpack("<16I", pkt))

def refs_in_blob(blob: bytes, ranges: list[tuple[int, int, int]]) -> list[dict[str, int]]:
  refs: list[dict[str, int]] = []
  for off in range(0, max(0, len(blob) - 7), 4):
    val = struct.unpack_from("<Q", blob, off)[0]
    for idx, (old, _new, size) in enumerate(ranges):
      if old <= val < old + size:
        refs.append({"offset": off, "allocation_idx": idx, "value": val, "delta": val - old})
        break
  return refs

def classify_sqtt_blob(blob: bytes) -> dict[str, Any]:
  nz = sum(1 for b in blob if b)
  first = next((i for i, b in enumerate(blob) if b), None)
  top: dict[str, int] = {}
  decode_error = None
  body_count = 0
  try:
    from tinygrad.renderer.amd.sqtt import decode, INST, INST_RDNA4, VALUINST, IMMEDIATE, IMMEDIATE_MASK, VMEMEXEC, ALUEXEC
    for i, p in enumerate(decode(blob)):
      name = type(p).__name__
      top[name] = top.get(name, 0) + 1
      if isinstance(p, (INST, INST_RDNA4, VALUINST, IMMEDIATE, IMMEDIATE_MASK, VMEMEXEC, ALUEXEC)): body_count += 1
      if i > 200000: break
  except Exception as e:
    decode_error = repr(e)
  return {"bytes": len(blob), "nonzero_bytes": nz, "first_nonzero_offset": first, "packet_top": top,
          "body_like_packet_count": body_count, "decode_error": decode_error}

def replay_with_tinygrad(export: dict[str, Any], mode: str) -> dict[str, Any]:
  from tinygrad.device import BufferSpec
  from tinygrad.dtype import dtypes
  from tinygrad.device import Device, Buffer
  from tinygrad.runtime.ops_amd import AMDComputeAQLQueue
  from tinygrad.runtime.autogen import hsa
  from tinygrad.helpers import data64_le

  dev = Device["AMD"]
  if not getattr(dev, "is_aql", 0): raise RuntimeError("R1-P2 replay requires AMD_AQL=1")

  class RawAQLQueue(AMDComputeAQLQueue):
    def vendor_packet(self, pkt: bytes):
      assert len(pkt) == 64
      self._q.append(pkt)
      return self
    def _prep_aql(self, q: list[Any], pm4_buf) -> list[bytes | hsa.hsa_kernel_dispatch_packet_t]:
      int_count = sum(1 for c in q if isinstance(c, int))
      if int_count: pm4_buf.cpu_view().view(fmt='I')[:int_count] = array.array('I', [c for c in q if isinstance(c, int)])
      aql_cmds: list[bytes | hsa.hsa_kernel_dispatch_packet_t] = []
      cursor = 0
      pm4_off = 0
      while cursor < len(q):
        if isinstance(q[cursor], int):
          start = cursor
          while cursor < len(q) and isinstance(q[cursor], int): cursor += 1
          cnt = cursor - start
          aql_cmds.append(self._pm4_pkt(pm4_buf.va_addr + pm4_off * 4, cnt))
          pm4_off += cnt
        else:
          aql_cmds.append(q[cursor])
          cursor += 1
      return aql_cmds
    def _submit(self, dev):
      cq = dev.compute_queue_desc(self.queue_idx)
      pm4_count = sum(1 for c in self._q if isinstance(c, int))
      pm4_buf = dev.pm4_ibs.offset(dev.pm4_ib_alloc.alloc(max(pm4_count, 1) * 4, 16))
      cmds = self._prep_aql(self._q, pm4_buf)
      aql_bytes = b''.join(bytes(c) if isinstance(c, hsa.hsa_kernel_dispatch_packet_t) else c for c in cmds)
      assert len(aql_bytes) % 64 == 0
      assert len(aql_bytes) < cq.ring.nbytes, "submit is too large for the queue"
      cp_bytes = min(len(aql_bytes), (cq.ring.nbytes - (cq.put_value * 64) % cq.ring.nbytes))
      cq.ring.view(offset=(cq.put_value * 64) % cq.ring.nbytes, fmt='B')[:cp_bytes] = aql_bytes[:cp_bytes]
      if (tail_bytes := len(aql_bytes) - cp_bytes) > 0: cq.ring.view(offset=0, fmt='B')[:tail_bytes] = aql_bytes[cp_bytes:]
      cq.put_value += len(aql_bytes) // 64
      cq.signal_doorbell(dev, doorbell_value=cq.put_value - 1)

  ranges: list[tuple[int, int, int]] = []
  old_ranges_requested: list[tuple[int, int, int]] = []
  replay_allocs: list[dict[str, Any]] = []
  for alloc in export["allocations"]:
    size = int(alloc["size"])
    guard = size if int(alloc.get("device_access", 0)) else 0
    alloc_size = size + guard
    buf = dev.allocator.alloc(alloc_size, BufferSpec(cpu_access=True, nolru=True, uncached=True))
    ranges.append((int(alloc["ptr"]), int(buf.va_addr), alloc_size))
    old_ranges_requested.append((int(alloc["ptr"]), int(buf.va_addr), size))
    if "hex" in alloc:
      content, patches = patch_ptr_ranges(bytes.fromhex(alloc["hex"]), ranges)
      dev.allocator._copyin(buf, memoryview(content))
    else:
      dev.allocator._copyin(buf, memoryview(bytearray(alloc_size)))
      patches = 0
    replay_allocs.append({**alloc, "replay_va": int(buf.va_addr), "replay_alloc_size": alloc_size, "guard_bytes": guard, "patches": patches, "_buf": buf})

  # Patch again after all replacement VAs are known.
  for row in replay_allocs:
    if "hex" in row:
      content, patches = patch_ptr_ranges(bytes.fromhex(row["hex"]), ranges)
      content, page_patches = patch_pm4_page_ranges(content, ranges)
      dev.allocator._copyin(row["_buf"], memoryview(content))
      row["patches"] = patches
      row["page_patches"] = page_patches

  start_pkt, start_patches = patch_ptr_ranges(bytes.fromhex(export["start_packet"]["hex"]), ranges)
  stop_pkt, stop_patches = patch_ptr_ranges(bytes.fromhex(export["stop_packet"]["hex"]), ranges)
  command_row = next((x for x in replay_allocs if int(x["size"]) == 1536), None)
  command_bytes = bytearray(int(command_row["size"])) if command_row is not None else bytearray()
  command_refs: list[dict[str, int]] = []
  if command_row is not None:
    dev.allocator._copyout(memoryview(command_bytes), command_row["_buf"].offset(0, len(command_bytes)))
    command_refs = refs_in_blob(bytes(command_bytes), ranges)
  body_buf = Buffer("AMD", 256, dtypes.float32).ensure_allocated()._buf
  prg = dev.runtime("r1p2_body", dev.compiler.compile(BODY_SRC))
  args_state = prg.fill_kernargs((body_buf,), ())

  predispatch = {
    "device": {"arch": dev.arch, "is_aql": int(dev.is_aql), "timeline_value": dev.timeline_value},
    "mode": mode,
    "patching": {"start_packet_patches": start_patches, "stop_packet_patches": stop_patches,
                 "replacement_count": len(ranges), "start_words": packet_words(start_pkt), "stop_words": packet_words(stop_pkt)},
    "allocations": [{k: v for k, v in row.items() if k != "_buf"} for row in replay_allocs],
    "old_ranges_requested": [{"old": old, "new": new, "size": size} for old, new, size in old_ranges_requested],
    "command_refs_after_patch": command_refs,
    "command_hex_after_patch_prefix": bytes(command_bytes[:512]).hex(),
  }
  PREDISPATCH.write_text(json.dumps(predispatch, indent=2, sort_keys=True) + "\n")

  q = RawAQLQueue(dev).wait(dev.timeline_signal, dev.timeline_value - 1).memory_barrier()
  if mode in {"start_only", "start_stop", "start_body_stop"}: q.vendor_packet(start_pkt)
  if mode in {"body_only", "start_body_stop"}: q.exec(prg, args_state, (4096, 1, 1), (64, 1, 1))
  if mode in {"start_stop", "start_body_stop"}: q.vendor_packet(stop_pkt)
  q.signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
  replay_status = {"submitted": True, "sync_ok": False, "error": None}
  try:
    dev.synchronize(timeout=10000)
    replay_status["sync_ok"] = True
  except Exception as e:
    replay_status["error"] = repr(e)

  output_row = max(replay_allocs, key=lambda x: int(x["size"]))
  out_bytes = bytearray(min(int(output_row["size"]), 16 << 20))
  if replay_status["sync_ok"]:
    dev.allocator._copyout(memoryview(out_bytes), output_row["_buf"].offset(0, len(out_bytes)))
  cls = classify_sqtt_blob(bytes(out_bytes))
  clean_allocs = [{k: v for k, v in row.items() if k != "_buf"} for row in replay_allocs]
  return {
    "device": {"arch": dev.arch, "is_aql": int(dev.is_aql), "timeline_value": dev.timeline_value},
    "patching": {"start_packet_patches": start_patches, "stop_packet_patches": stop_patches,
                 "replacement_count": len(ranges), "start_words": packet_words(start_pkt), "stop_words": packet_words(stop_pkt)},
    "allocations": clean_allocs,
    "body_kernel": {"name": prg.name, "global_size": [4096, 1, 1], "local_size": [64, 1, 1]},
    "mode": mode,
    "replay": replay_status,
    "trace_output": cls,
  }

def main() -> int:
  mode = os.environ.get("R1P2_REPLAY_MODE", "start_body_stop")
  if mode not in {"body_only", "start_only", "start_stop", "start_body_stop"}:
    raise RuntimeError(f"bad R1P2_REPLAY_MODE={mode!r}")
  OUTDIR.mkdir(parents=True, exist_ok=True)
  build, helper_run, export = build_and_run_helper()
  result: dict[str, Any] = {
    "date": "2026-06-19",
    "phase": "R1-P2 P1/P2 two-process HCQ replay",
    "mode": mode,
    "helper": {"build": build, "run": helper_run, "export": export},
    "tinygrad_replay": None,
    "gates": {},
    "verdict": "NOT_RUN",
  }
  if not isinstance(export, dict) or not export.get("ok"):
    result["verdict"] = "HELPER_EXPORT_FAIL"
  else:
    try:
      replay = replay_with_tinygrad(export, mode)
      result["tinygrad_replay"] = replay
      sync_ok = bool(replay["replay"]["sync_ok"])
      nonzero = int(replay["trace_output"]["nonzero_bytes"]) > 0
      body = int(replay["trace_output"]["body_like_packet_count"]) > 0
      result["gates"] = {
        "helper_packet_export": "PASS",
        "p1_patch_to_tinygrad_buffers": "PASS" if replay["patching"]["replacement_count"] >= 3 and replay["patching"]["start_packet_patches"] else "FAIL",
        "p2_hcq_submit_sync": "PASS" if sync_ok else "FAIL",
        "p2_trace_nonzero": "PASS" if nonzero else "FAIL",
        "p2_body_packets": "PASS" if body else "FAIL",
      }
      if sync_ok and body: result["verdict"] = "PASS_BODY_ATTRIBUTION"
      elif sync_ok and nonzero: result["verdict"] = "PARTIAL_LIFECYCLE_OR_UNMAPPED_TRACE"
      elif sync_ok: result["verdict"] = "P2_SYNC_BUT_EMPTY_TRACE"
      else: result["verdict"] = "P2_REPLAY_FAIL"
    except Exception as e:
      result["tinygrad_replay"] = {"error": repr(e)}
      result["gates"] = {"helper_packet_export": "PASS", "p1_p2_exception": "FAIL"}
      result["verdict"] = "P1_P2_EXCEPTION"

  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"out": str(OUT.relative_to(ROOT)), "verdict": result["verdict"], "gates": result["gates"]}, indent=2))
  return 0 if result["verdict"] in {"PASS_BODY_ATTRIBUTION", "PARTIAL_LIFECYCLE_OR_UNMAPPED_TRACE", "P2_SYNC_BUT_EMPTY_TRACE"} else 1

if __name__ == "__main__":
  raise SystemExit(main())
