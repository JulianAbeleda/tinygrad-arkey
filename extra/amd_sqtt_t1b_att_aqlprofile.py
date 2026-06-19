#!/usr/bin/env python3
from __future__ import annotations

import json, os, pathlib, shutil, subprocess, textwrap, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "bench/amd-scheduler-tooling-backend"
WORKDIR = OUTDIR / "t1b_att_aqlprofile_work"
OUT = OUTDIR / "t1b_att_aqlprofile.json"
ROCM = pathlib.Path(os.environ.get("ROCM_PATH", "/opt/rocm-7.2.4"))

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); return 2; } } while (0)

__global__ void body_kernel(float *out, const float *a, const float *b, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  float x = a[gid], y = b[gid];
  #pragma unroll 64
  for (int i = 0; i < 64; i++) {
    x = fmaf(x, 1.000113f, y);
    y = fmaf(y, 0.999887f, x);
  }
  out[gid] = x + y;
}

int main() {
  int n = 1 << 22;
  float *a = nullptr, *b = nullptr, *c = nullptr;
  CHECK(hipSetDevice(0));
  CHECK(hipMalloc(&a, n * sizeof(float)));
  CHECK(hipMalloc(&b, n * sizeof(float)));
  CHECK(hipMalloc(&c, n * sizeof(float)));
  CHECK(hipMemset(a, 1, n * sizeof(float)));
  CHECK(hipMemset(b, 2, n * sizeof(float)));
  CHECK(hipMemset(c, 0, n * sizeof(float)));
  dim3 block(256), grid((n + block.x - 1) / block.x);
  for (int i = 0; i < 2; i++) {
    body_kernel<<<grid, block>>>(c, a, b, n);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
  }
  CHECK(hipFree(a));
  CHECK(hipFree(b));
  CHECK(hipFree(c));
  return 0;
}
"""

AQL_SRC = r"""
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static hsa_agent_t gpu_agent{};
static bool found_gpu = false;

static const char* status_name(hsa_status_t s) {
  const char *x = nullptr;
  if (hsa_status_string(s, &x) == HSA_STATUS_SUCCESS && x) return x;
  return "unknown";
}

static hsa_status_t find_gpu(hsa_agent_t agent, void*) {
  hsa_device_type_t type{};
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) == HSA_STATUS_SUCCESS && type == HSA_DEVICE_TYPE_GPU) {
    gpu_agent = agent;
    found_gpu = true;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

static void dump_pm4(const char *name, const hsa_ext_amd_aql_pm4_packet_t &pkt) {
  uint32_t words[HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE / sizeof(uint32_t)]{};
  hsa_status_t st = hsa_ven_amd_aqlprofile_legacy_get_pm4(&pkt, words);
  printf("\"%s\":{\"status\":%u,\"words\":[", name, (unsigned)st);
  for (size_t i = 0; i < sizeof(words) / sizeof(words[0]); i++) {
    if (i) printf(",");
    printf("%u", words[i]);
  }
  printf("]}");
}

static void dump_words(const char *name, const void *ptr, size_t bytes) {
  const uint32_t *words = reinterpret_cast<const uint32_t*>(ptr);
  size_t n = bytes / sizeof(uint32_t);
  if (n > 256) n = 256;
  printf("\"%s\":{\"bytes\":%zu,\"words\":[", name, bytes);
  for (size_t i = 0; i < n; i++) {
    if (i) printf(",");
    printf("%u", words[i]);
  }
  printf("]}");
}

int main() {
  hsa_status_t st = hsa_init();
  if (st != HSA_STATUS_SUCCESS) {
    printf("{\"ok\":false,\"stage\":\"hsa_init\",\"status\":%u,\"status_name\":\"%s\"}\n", (unsigned)st, status_name(st));
    return 2;
  }
  st = hsa_iterate_agents(find_gpu, nullptr);
  if (!found_gpu) {
    printf("{\"ok\":false,\"stage\":\"find_gpu\",\"status\":%u,\"status_name\":\"%s\"}\n", (unsigned)st, status_name(st));
    hsa_shut_down();
    return 3;
  }

  char name[128]{};
  hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, name);

  const uint32_t MASK = 0xffffffffu;
  struct ParamSet { const char *name; std::vector<hsa_ven_amd_aqlprofile_parameter_t> params; };
  std::vector<ParamSet> sets = {
    {"cu_se_token_token2_kconcurrent", {
      {HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET, 1},
      {HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK, 1},
      {HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK, MASK},
      {HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2, MASK},
      {HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_K_CONCURRENT, 1},
    }},
    {"cu_se_token_token2", {
      {HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET, 1},
      {HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK, 1},
      {HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK, MASK},
      {HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2, MASK},
    }},
    {"cu_se_mask_token", {
      {HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET, 1},
      {HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK, 1},
      {HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK, MASK},
      {HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK, MASK},
    }},
    {"cu_se_only", {
      {HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET, 1},
      {HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK, 1},
    }},
    {"cu_only", {
      {HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET, 1},
    }},
    {"no_params", {}},
  };

  printf("{\"agent\":\"%s\",\"attempts\":[", name);
  bool any_ok = false;
  for (size_t i = 0; i < sets.size(); i++) {
    auto &set = sets[i];
    hsa_ven_amd_aqlprofile_profile_t profile{};
    profile.agent = gpu_agent;
    profile.type = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE;
    profile.parameters = set.params.empty() ? nullptr : set.params.data();
    profile.parameter_count = (uint32_t)set.params.size();
    uint32_t command_buffer_size = 0;
    hsa_status_t info_st = hsa_ven_amd_aqlprofile_get_info(&profile, HSA_VEN_AMD_AQLPROFILE_INFO_COMMAND_BUFFER_SIZE, &command_buffer_size);
    if (info_st != HSA_STATUS_SUCCESS || command_buffer_size == 0) command_buffer_size = 8192;
    std::vector<uint8_t> command_buffer(command_buffer_size);
    std::vector<uint8_t> output_buffer(16 << 20);
    profile.command_buffer = {command_buffer.data(), (uint32_t)command_buffer.size()};
    profile.output_buffer = {output_buffer.data(), (uint32_t)output_buffer.size()};

    hsa_ext_amd_aql_pm4_packet_t start_pkt{}, stop_pkt{}, read_pkt{};
    hsa_status_t start_st = hsa_ven_amd_aqlprofile_start(&profile, &start_pkt);
    hsa_ven_amd_aqlprofile_descriptor_t enable_cmd{};
    hsa_ven_amd_aqlprofile_descriptor_t disable_cmd{};
    hsa_status_t enable_info_st = hsa_ven_amd_aqlprofile_get_info(&profile, HSA_VEN_AMD_AQLPROFILE_INFO_ENABLE_CMD, &enable_cmd);
    hsa_status_t disable_info_st = hsa_ven_amd_aqlprofile_get_info(&profile, HSA_VEN_AMD_AQLPROFILE_INFO_DISABLE_CMD, &disable_cmd);
    hsa_status_t stop_st = hsa_ven_amd_aqlprofile_stop(&profile, &stop_pkt);
    hsa_status_t read_st = hsa_ven_amd_aqlprofile_read(&profile, &read_pkt);
    any_ok = any_ok || (start_st == HSA_STATUS_SUCCESS && stop_st == HSA_STATUS_SUCCESS);

    if (i) printf(",");
    printf("{\"name\":\"%s\",\"ok\":%s,\"info_status\":%u,\"command_buffer_size\":%u,\"output_buffer_size\":%u,",
           set.name, (start_st == HSA_STATUS_SUCCESS && stop_st == HSA_STATUS_SUCCESS) ? "true" : "false",
           (unsigned)info_st, command_buffer_size, (unsigned)output_buffer.size());
    printf("\"start_status\":%u,\"stop_status\":%u,\"read_status\":%u,\"enable_info_status\":%u,\"disable_info_status\":%u,",
           (unsigned)start_st, (unsigned)stop_st, (unsigned)read_st, (unsigned)enable_info_st, (unsigned)disable_info_st);
    dump_words("command_buffer_words", command_buffer.data(), command_buffer.size());
    printf(",");
    if (enable_info_st == HSA_STATUS_SUCCESS && enable_cmd.ptr && enable_cmd.size) dump_words("enable_cmd_words", enable_cmd.ptr, enable_cmd.size);
    else printf("\"enable_cmd_words\":{\"bytes\":0,\"words\":[]}");
    printf(",");
    if (disable_info_st == HSA_STATUS_SUCCESS && disable_cmd.ptr && disable_cmd.size) dump_words("disable_cmd_words", disable_cmd.ptr, disable_cmd.size);
    else printf("\"disable_cmd_words\":{\"bytes\":0,\"words\":[]}");
    printf(",");
    dump_pm4("start_pm4", start_pkt);
    printf(",");
    dump_pm4("stop_pm4", stop_pkt);
    printf(",");
    dump_pm4("read_pm4", read_pkt);
    printf("}");
  }
  printf("],\"ok\":%s}\n", any_ok ? "true" : "false");
  hsa_shut_down();
  return any_ok ? 0 : 4;
}
"""

def run(cmd: list[str], *, cwd: pathlib.Path = ROOT, env: dict[str, str] | None = None, timeout: int = 120) -> dict[str, Any]:
  t0 = time.perf_counter()
  try:
    cp = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return {
      "cmd": cmd, "returncode": cp.returncode, "elapsed_s": round(time.perf_counter() - t0, 3),
      "stdout_tail": cp.stdout.splitlines()[-80:], "stderr_tail": cp.stderr.splitlines()[-80:],
    }
  except subprocess.TimeoutExpired as e:
    return {
      "cmd": cmd, "timeout": True, "elapsed_s": round(time.perf_counter() - t0, 3),
      "stdout_tail": (e.stdout or "").splitlines()[-80:] if isinstance(e.stdout, str) else [],
      "stderr_tail": (e.stderr or "").splitlines()[-80:] if isinstance(e.stderr, str) else [],
    }

def find_paths() -> dict[str, Any]:
  paths = {
    "rocprofv3": shutil.which("rocprofv3") or str(ROCM / "bin/rocprofv3"),
    "hipcc": shutil.which("hipcc") or str(ROCM / "bin/hipcc"),
    "rocm": str(ROCM),
  }
  checks = {
    "rocprofiler_sdk_cmake": sorted(str(p) for p in ROCM.glob("lib/cmake/rocprofiler-sdk/*config*.cmake")),
    "rocprofiler_sdk_libs": sorted(str(p) for p in ROCM.glob("lib/librocprofiler-sdk.so*")),
    "aqlprofile_libs": sorted(str(p) for p in ROCM.glob("lib/libhsa-amd-aqlprofile64.so*")),
    "trace_decoder_libs": sorted(str(p) for p in ROCM.glob("lib/librocprof-trace-decoder*")),
    "thread_trace_sample": str(ROCM / "share/rocprofiler-sdk/samples/thread_trace"),
  }
  return {"paths": paths, "checks": checks}

def build_hip_control() -> dict[str, Any]:
  src = WORKDIR / "att_body_control.cpp"
  exe = WORKDIR / "att_body_control"
  src.write_text(HIP_SRC)
  cmd = [shutil.which("hipcc") or str(ROCM / "bin/hipcc"), "--offload-arch=gfx1100", "-O3", str(src), "-o", str(exe)]
  res = run(cmd, cwd=WORKDIR, timeout=180)
  return {"source": str(src.relative_to(ROOT)), "exe": str(exe.relative_to(ROOT)), "build": res, "ok": res.get("returncode") == 0 and exe.exists()}

def run_att(exe_rel: str) -> dict[str, Any]:
  outdir = WORKDIR / "rocprof_att"
  shutil.rmtree(outdir, ignore_errors=True)
  outdir.mkdir(parents=True, exist_ok=True)
  decoder_alias_dir = WORKDIR / "decoder_alias"
  decoder_alias_dir.mkdir(parents=True, exist_ok=True)
  decoder_alias = decoder_alias_dir / "librocprof-trace-decoder.so"
  try:
    if decoder_alias.exists() or decoder_alias.is_symlink(): decoder_alias.unlink()
    decoder_alias.symlink_to(ROCM / "lib/librocprofiler-sdk.so")
  except OSError:
    pass
  rocprofv3 = shutil.which("rocprofv3") or str(ROCM / "bin/rocprofv3")
  exe = ROOT / exe_rel
  cmd = [
    rocprofv3, "--att", "--kernel-trace", "--att-buffer-size", "67108864",
    "--att-shader-engine-mask", "1", "--att-target-cu", "1", "--att-simd-select", "1",
    "--att-serialize-all", "--att-library-path", str(decoder_alias_dir), str(ROCM / "lib"),
    "-d", str(outdir), "-o", "att_control", "-f", "json", "--", str(exe),
  ]
  env = os.environ.copy()
  env["LD_LIBRARY_PATH"] = f"{ROCM / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
  res = run(cmd, cwd=WORKDIR, env=env, timeout=240)
  files = []
  text_hits = []
  for p in sorted(outdir.rglob("*")):
    if p.is_file():
      rel = str(p.relative_to(ROOT))
      files.append({"path": rel, "bytes": p.stat().st_size})
      if p.stat().st_size < 2_000_000 and p.suffix.lower() in {".txt", ".log", ".json", ".csv"}:
        txt = p.read_text(errors="replace")
        if any(x in txt.lower() for x in ("thread", "trace", "wave", "kernel", "body_kernel", "hotspot")):
          text_hits.append({"path": rel, "preview": txt[:4000]})
  return {
    "command_result": res,
    "output_dir": str(outdir.relative_to(ROOT)),
    "decoder_alias": str(decoder_alias.relative_to(ROOT)) if decoder_alias.exists() else None,
    "files": files,
    "text_hits": text_hits[:8],
    "ok": res.get("returncode") == 0 and len(files) > 0,
    "has_att_payload": any(("att" in f["path"].lower() or "thread" in f["path"].lower() or "trace" in f["path"].lower()) and f["bytes"] > 0 for f in files),
  }

def build_aql_pm4() -> dict[str, Any]:
  src = WORKDIR / "aqlprofile_pm4_dump.cpp"
  exe = WORKDIR / "aqlprofile_pm4_dump"
  src.write_text(AQL_SRC)
  cmd = [
    "g++", "-O2", "-std=c++17", "-I", str(ROCM / "include"), "-I", "/usr/include",
    str(src), "-L", str(ROCM / "lib"), "-Wl,-rpath," + str(ROCM / "lib"),
    "-lhsa-runtime64", "-lhsa-amd-aqlprofile64", "-o", str(exe),
  ]
  res = run(cmd, cwd=WORKDIR, timeout=120)
  return {"source": str(src.relative_to(ROOT)), "exe": str(exe.relative_to(ROOT)), "build": res, "ok": res.get("returncode") == 0 and exe.exists()}

def run_aql_pm4(exe_rel: str) -> dict[str, Any]:
  exe = ROOT / exe_rel
  env = os.environ.copy()
  env["LD_LIBRARY_PATH"] = f"{ROCM / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
  res = run([str(exe)], cwd=WORKDIR, env=env, timeout=120)
  parsed = None
  if res.get("stdout_tail"):
    try:
      parsed = json.loads(res["stdout_tail"][-1])
    except Exception as exc:
      parsed = {"parse_error": repr(exc)}
  pm4_nonzero = {}
  if isinstance(parsed, dict):
    for attempt in parsed.get("attempts", []):
      for k in ("command_buffer_words", "enable_cmd_words", "disable_cmd_words", "start_pm4", "stop_pm4", "read_pm4"):
        words = attempt.get(k, {}).get("words", []) if isinstance(attempt.get(k), dict) else []
        pm4_nonzero[f"{attempt.get('name')}:{k}"] = sum(1 for w in words if w)
  return {"command_result": res, "parsed": parsed, "pm4_nonzero_words": pm4_nonzero, "ok": res.get("returncode") == 0 and isinstance(parsed, dict) and parsed.get("ok") is True}

def main() -> int:
  OUTDIR.mkdir(parents=True, exist_ok=True)
  shutil.rmtree(WORKDIR, ignore_errors=True)
  WORKDIR.mkdir(parents=True, exist_ok=True)

  result: dict[str, Any] = {
    "date": "2026-06-19",
    "phase": "T1b_ATT_AQLprofile_oracle_and_PM4_recovery",
    "purpose": "Use ROCm's mature ATT path as an oracle and recover the AQLprofile PM4 setup needed to fix tinygrad HCQ SQTT body mapping.",
    "sdk_layout": find_paths(),
  }

  hip = build_hip_control()
  result["hip_control"] = hip
  result["rocprofv3_att"] = run_att(hip["exe"]) if hip["ok"] else {"ok": False, "reason": "hip_control_build_failed"}

  aql = build_aql_pm4()
  result["aqlprofile_pm4_build"] = aql
  result["aqlprofile_pm4_run"] = run_aql_pm4(aql["exe"]) if aql["ok"] else {"ok": False, "reason": "aqlprofile_build_failed"}

  att_ok = bool(result["rocprofv3_att"].get("ok") and result["rocprofv3_att"].get("has_att_payload"))
  pm4_ok = bool(result["aqlprofile_pm4_run"].get("ok") and any(v > 0 for v in result["aqlprofile_pm4_run"].get("pm4_nonzero_words", {}).values()))
  result["gates"] = {
    "external_att_oracle": "PASS" if att_ok else "BLOCKED",
    "aqlprofile_pm4_recovery": "PASS" if pm4_ok else "BLOCKED",
  }
  result["verdict"] = (
    "T1B_BOTH_PATHS_PASS" if att_ok and pm4_ok else
    "T1B_PARTIAL_PASS" if att_ok or pm4_ok else
    "T1B_BLOCKED"
  )
  if att_ok and pm4_ok:
    result["next"] = "Diff AQLprofile PM4/register writes against tinygrad SQTT setup, implement the missing sequence behind env, then rerun t1_body_mapping_proof."
  elif att_ok:
    result["next"] = "Use rocprofv3 ATT as the body-trace oracle; PM4 recovery still needs a working AQLprofile build/run path."
  elif pm4_ok:
    result["next"] = "Use recovered PM4 to patch tinygrad SQTT setup; external decoded ATT oracle is still missing."
  else:
    result["next"] = "Fix ROCm ATT/AQLprofile environment before more tinygrad SQTT decoder work."

  OUT.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"out": str(OUT.relative_to(ROOT)), "verdict": result["verdict"], "gates": result["gates"], "next": result["next"]}, indent=2))
  return 0 if result["verdict"] != "T1B_BLOCKED" else 1

if __name__ == "__main__":
  raise SystemExit(main())
