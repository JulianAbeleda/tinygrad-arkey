#!/usr/bin/env python3
from __future__ import annotations

import json, os, shutil, subprocess, textwrap, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "bench/amd-scheduler-tooling-backend"
WORKDIR = OUTDIR / "r1p2_v2_exporter_work"
OUT = OUTDIR / "r1p2_v2_exporter.json"
ROCM = Path(os.environ.get("ROCM_PATH", "/opt/rocm-7.2.4"))

SRC = r"""
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

typedef struct { uint64_t handle; } aqlprofile_handle_t;
typedef struct { uint64_t handle; } aqlprofile_agent_handle_t;
typedef enum { AQLPROFILE_AGENT_VERSION_NONE = 0, AQLPROFILE_AGENT_VERSION_V0 = 1, AQLPROFILE_AGENT_VERSION_V1 = 2 } aqlprofile_agent_version_t;
typedef struct {
  const char* agent_gfxip;
  uint32_t xcc_num;
  uint32_t se_num;
  uint32_t cu_num;
  uint32_t shader_arrays_per_se;
  uint32_t domain;
  uint32_t location_id;
} aqlprofile_agent_info_v1_t;
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
    struct {
      uint32_t counter_id : 28;
      uint32_t simd_mask : 4;
    };
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
typedef hsa_status_t (*register_agent_info_t)(aqlprofile_agent_handle_t*, const void*, aqlprofile_agent_version_t);
typedef hsa_status_t (*att_create_packets_t)(aqlprofile_handle_t*, aqlprofile_att_control_aql_packets_t*, aqlprofile_att_profile_t, alloc_cb_t, free_cb_t, copy_cb_t, void*);
typedef void (*att_delete_packets_t)(aqlprofile_handle_t);
typedef hsa_status_t (*att_iterate_data_t)(aqlprofile_handle_t, hsa_status_t (*)(uint32_t, void*, uint64_t, void*), void*);

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

static void reset_allocs() {
  for (auto &r : allocs) {
    if (r.base != nullptr) {
      free(r.base);
      r.base = nullptr;
    }
  }
  allocs.clear();
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

static hsa_status_t iter_cb(uint32_t shader, void*, uint64_t size, void*) {
  printf("{\"iter_cb_shader\":%u,\"iter_cb_size\":%lu}", shader, (unsigned long)size);
  return HSA_STATUS_SUCCESS;
}

static int packet_nonzero(const hsa_ext_amd_aql_pm4_packet_t* p) {
  const uint32_t* w = (const uint32_t*)p;
  int n = 0;
  for (size_t i = 0; i < sizeof(*p) / sizeof(uint32_t); i++) if (w[i]) n++;
  return n;
}

static void print_packet(const char* name, const hsa_ext_amd_aql_pm4_packet_t* p) {
  const uint32_t* w = (const uint32_t*)p;
  printf("\"%s\":{\"bytes\":%zu,\"nonzero_words\":%d,\"words\":[", name, sizeof(*p), packet_nonzero(p));
  for (size_t i = 0; i < sizeof(*p) / sizeof(uint32_t); i++) {
    if (i) printf(",");
    printf("%u", w[i]);
  }
  printf("]}");
}

static void print_allocs() {
  printf("\"allocations\":[");
  for (size_t i = 0; i < allocs.size(); i++) {
    if (i) printf(",");
    auto &r = allocs[i];
    printf("{\"idx\":%zu,\"ptr\":\"0x%lx\",\"base\":\"0x%lx\",\"size\":%lu,\"raw\":%u,\"device_access\":%u,\"host_access\":%u,\"memory_hint\":%u}",
           i, (unsigned long)(uintptr_t)r.ptr, (unsigned long)(uintptr_t)r.base, (unsigned long)r.size,
           r.raw, r.device_access, r.host_access, r.memory_hint);
  }
  printf("]");
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  hsa_status_t hsa_init_st = hsa_init();
  if (hsa_init_st != HSA_STATUS_SUCCESS) {
    printf("{\"ok\":false,\"stage\":\"hsa_init\",\"hsa_init_status\":%u}\n", (unsigned)hsa_init_st);
    return 5;
  }
  hsa_status_t find_st = hsa_iterate_agents(find_gpu, nullptr);
  if (!found_gpu) {
    printf("{\"ok\":false,\"stage\":\"find_gpu\",\"hsa_init_status\":%u,\"find_status\":%u}\n",
           (unsigned)hsa_init_st, (unsigned)find_st);
    return 6;
  }
  void* lib = dlopen("libhsa-amd-aqlprofile64.so", RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    printf("{\"ok\":false,\"stage\":\"dlopen\",\"error\":\"%s\"}\n", dlerror());
    return 2;
  }
  auto reg = (register_agent_info_t)dlsym(lib, "aqlprofile_register_agent_info");
  auto create = (att_create_packets_t)dlsym(lib, "aqlprofile_att_create_packets");
  auto del = (att_delete_packets_t)dlsym(lib, "aqlprofile_att_delete_packets");
  auto iter = (att_iterate_data_t)dlsym(lib, "aqlprofile_att_iterate_data");
  if (!reg || !create || !del || !iter) {
    printf("{\"ok\":false,\"stage\":\"dlsym\",\"reg\":%d,\"create\":%d,\"delete\":%d,\"iter\":%d}\n", !!reg, !!create, !!del, !!iter);
    return 3;
  }

  aqlprofile_agent_info_v1_t info{};
  info.agent_gfxip = "gfx1100";
  info.xcc_num = 1;
  info.se_num = 6;
  info.cu_num = 96;
  info.shader_arrays_per_se = 2;
  info.domain = 0;
  info.location_id = 0;
  aqlprofile_agent_handle_t agent{};
  hsa_status_t reg_st = reg(&agent, &info, AQLPROFILE_AGENT_VERSION_V1);

  struct Attempt { const char* name; aqlprofile_att_parameter_t params[4]; uint32_t count; };
  Attempt attempts[5]{};
  attempts[0].name = "cu_se_simd_buf";
  attempts[0].params[0].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET; attempts[0].params[0].value = 1;
  attempts[0].params[1].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK; attempts[0].params[1].value = 1;
  attempts[0].params[2].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION; attempts[0].params[2].value = 1;
  attempts[0].params[3].parameter_name = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE; attempts[0].params[3].value = 64u << 20;
  attempts[0].count = 4;
  attempts[1].name = "cu_se_simd";
  attempts[1].params[0] = attempts[0].params[0]; attempts[1].params[1] = attempts[0].params[1]; attempts[1].params[2] = attempts[0].params[2];
  attempts[1].count = 3;
  attempts[2].name = "cu_se_only";
  attempts[2].params[0] = attempts[0].params[0]; attempts[2].params[1] = attempts[0].params[1];
  attempts[2].count = 2;
  attempts[3].name = "cu_only";
  attempts[3].params[0] = attempts[0].params[0];
  attempts[3].count = 1;
  attempts[4].name = "no_params";
  attempts[4].count = 0;

  printf("{\"hsa_init_status\":%u,\"find_status\":%u,\"register_status\":%u,\"registered_agent_handle\":%lu,\"hsa_gpu_agent_handle\":%lu,\"attempts\":[",
         (unsigned)hsa_init_st, (unsigned)find_st, (unsigned)reg_st, (unsigned long)agent.handle, (unsigned long)gpu_agent.handle);
  bool any_ok = false;
  for (size_t i = 0; i < 5; i++) {
    reset_allocs();
    aqlprofile_att_profile_t profile{};
    profile.agent = gpu_agent;
    profile.parameters = attempts[i].count ? attempts[i].params : nullptr;
    profile.parameter_count = attempts[i].count;
    aqlprofile_handle_t handle{};
    aqlprofile_att_control_aql_packets_t packets{};
    hsa_status_t create_st = HSA_STATUS_ERROR;
    if (reg_st == HSA_STATUS_SUCCESS) create_st = create(&handle, &packets, profile, cb_alloc, cb_free, cb_copy, nullptr);
    uint16_t vendor_barrier_header = (uint16_t)((7 << 8) | (1 << 0));
    if (create_st == HSA_STATUS_SUCCESS) {
      any_ok = true;
      packets.start_packet.header = vendor_barrier_header;
      packets.stop_packet.header = vendor_barrier_header;
      packets.start_packet.completion_signal.handle = 0;
      packets.stop_packet.completion_signal.handle = 0;
    }
    if (i) printf(",");
    printf("{\"name\":\"%s\",\"ok\":%s,\"create_status\":%u,\"handle\":%lu,", attempts[i].name,
           create_st == HSA_STATUS_SUCCESS ? "true" : "false", (unsigned)create_st, (unsigned long)handle.handle);
    print_allocs();
    printf(",");
    print_packet("start_packet", &packets.start_packet);
    printf(",");
    print_packet("stop_packet", &packets.stop_packet);
    printf(",\"pre_iterate_status\":null");
    printf("}");
  }
  printf("],\"ok\":%s}\n", any_ok ? "true" : "false");
  // Smoke only: avoid destructor/deallocator paths so a crash there does not hide packet creation status.
  (void)del;
  (void)lib;
  return any_ok ? 0 : 4;
}
"""

def run(cmd: list[str], *, timeout: int = 120) -> dict:
  t0 = time.perf_counter()
  try:
    cp = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return {"cmd": cmd, "returncode": cp.returncode, "elapsed_s": round(time.perf_counter() - t0, 3),
            "stdout_tail": cp.stdout.splitlines()[-20:], "stderr_tail": cp.stderr.splitlines()[-40:]}
  except subprocess.TimeoutExpired as e:
    return {"cmd": cmd, "timeout": True, "elapsed_s": round(time.perf_counter() - t0, 3),
            "stdout_tail": (e.stdout or "").splitlines()[-20:] if isinstance(e.stdout, str) else [],
            "stderr_tail": (e.stderr or "").splitlines()[-40:] if isinstance(e.stderr, str) else []}

def main() -> int:
  OUTDIR.mkdir(parents=True, exist_ok=True)
  shutil.rmtree(WORKDIR, ignore_errors=True)
  WORKDIR.mkdir(parents=True, exist_ok=True)
  src = WORKDIR / "v2_exporter_smoke.cpp"
  exe = WORKDIR / "v2_exporter_smoke"
  src.write_text(SRC)
  build = run(["g++", "-O2", "-std=c++17", "-I", str(ROCM / "include"), str(src),
               "-L", str(ROCM / "lib"), "-Wl,-rpath," + str(ROCM / "lib"),
               "-ldl", "-lhsa-runtime64", "-o", str(exe)])
  parsed = None
  run_res = {"ok": False, "reason": "build_failed"}
  if build.get("returncode") == 0:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{ROCM / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    t0 = time.perf_counter()
    cp = subprocess.run([str(exe)], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    run_res = {"cmd": [str(exe)], "returncode": cp.returncode, "elapsed_s": round(time.perf_counter() - t0, 3),
               "stdout_tail": cp.stdout.splitlines()[-20:], "stderr_tail": cp.stderr.splitlines()[-40:]}
    for line in cp.stdout.splitlines()[::-1]:
      if line.startswith("{") and line.endswith("}"):
        try:
          parsed = json.loads(line)
          break
        except json.JSONDecodeError:
          pass

  attempts = [] if not isinstance(parsed, dict) else parsed.get("attempts", [])
  working = [a for a in attempts if a.get("ok")]
  allocations = []
  start_nz = 0
  stop_nz = 0
  for attempt in working:
    allocations.extend(attempt.get("allocations", []))
    start_nz = max(start_nz, attempt.get("start_packet", {}).get("nonzero_words", 0))
    stop_nz = max(stop_nz, attempt.get("stop_packet", {}).get("nonzero_words", 0))
  p0_pass = bool(isinstance(parsed, dict) and parsed.get("ok") and allocations and start_nz and stop_nz)
  has_device_alloc = any(a.get("device_access") for a in allocations)
  has_host_alloc = any(a.get("host_access") for a in allocations)
  result = {
    "date": "2026-06-19",
    "phase": "R1-P2 v2 AQLprofile exporter P0/P1",
    "source": str(src.relative_to(ROOT)),
    "exe": str(exe.relative_to(ROOT)),
    "build": build,
    "run": run_res,
    "parsed": parsed,
    "gates": {
      "p0_v2_create_packets": "PASS" if p0_pass else "FAIL",
      "p0_allocation_table": "PASS" if allocations else "FAIL",
      "p0_start_stop_packets": "PASS" if start_nz and stop_nz else "FAIL",
      "p1_mappable_strategy": "BLOCKED_PENDING_HCQ_VA_BINDING" if p0_pass and has_device_alloc else "FAIL",
    },
    "classification": {
      "attempt_count": len(attempts),
      "working_attempts": [a.get("name") for a in working],
      "allocation_count": len(allocations),
      "has_device_access_alloc": has_device_alloc,
      "has_host_access_alloc": has_host_alloc,
      "start_nonzero_words": start_nz,
      "stop_nonzero_words": stop_nz,
    },
    "verdict": "P0_PASS_P1_NEEDS_HCQ_VA_BINDING" if p0_pass else "P0_FAIL",
    "next": "Bind AQLprofile device/host allocations to tinygrad-owned or tinygrad-submittable GPU VAs, then run P2 one-dispatch replay." if p0_pass else "Stop; v2 API cannot create usable packet material.",
  }
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"out": str(OUT.relative_to(ROOT)), "verdict": result["verdict"], "gates": result["gates"]}, indent=2))
  return 0 if p0_pass else 1

if __name__ == "__main__":
  raise SystemExit(main())
