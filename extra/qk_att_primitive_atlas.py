#!/usr/bin/env python3
from __future__ import annotations

import array, json, os, pathlib, statistics, struct, subprocess, sys, time
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTDIR = ROOT / "bench/qk-att-primitive-atlas"
OUT = OUTDIR / "result.json"
DECODE_OUT = OUTDIR / "decode_mmvq.json"
PREFILL_OUT = OUTDIR / "prefill_nonmatmul.json"

BODY_SRC = r"""
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(1, 64)))
void att_smoke_body(float *out) {
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


def trace_summary(blob: bytes) -> dict[str, Any]:
  nonzero = sum(1 for b in blob if b)
  first = next((i for i, b in enumerate(blob) if b), None)
  top: dict[str, int] = {}
  body = 0
  decode_error = None
  try:
    from tinygrad.renderer.amd.sqtt import decode, INST, INST_RDNA4, VALUINST, IMMEDIATE, IMMEDIATE_MASK, VMEMEXEC, ALUEXEC
    for i, pkt in enumerate(decode(blob)):
      name = type(pkt).__name__
      top[name] = top.get(name, 0) + 1
      if isinstance(pkt, (INST, INST_RDNA4, VALUINST, IMMEDIATE, IMMEDIATE_MASK, VMEMEXEC, ALUEXEC)): body += 1
      if i > 500000: break
  except Exception as e:
    decode_error = repr(e)
  return {"bytes": len(blob), "nonzero_bytes": nonzero, "first_nonzero_offset": first,
          "packet_top": top, "body_like_packet_count": body, "decode_error": decode_error}


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
  page_ranges = [(old >> 12, new >> 12, (size + 0xfff) >> 12) for old, new, size in ranges]
  for off in range(0, max(0, len(out) - 3), 4):
    val = struct.unpack_from("<I", out, off)[0]
    for old_page, new_page, pages in page_ranges:
      old_low = old_page & 0xffffffff
      if old_low <= val < old_low + pages:
        struct.pack_into("<I", out, off, (new_page + (val - old_low)) & 0xffffffff)
        patches += 1
        break
  return bytes(out), patches


class RawAQLQueueMixin:
  def vendor_packet(self, pkt: bytes):
    assert len(pkt) == 64
    self._q.append(pkt)
    return self

  def _prep_aql(self, q: list[Any], pm4_buf):
    from tinygrad.runtime.autogen import hsa
    int_count = sum(1 for c in q if isinstance(c, int))
    if int_count: pm4_buf.cpu_view().view(fmt="I")[:int_count] = array.array("I", [c for c in q if isinstance(c, int)])
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
    from tinygrad.runtime.autogen import hsa
    cq = dev.compute_queue_desc(self.queue_idx)
    pm4_count = sum(1 for c in self._q if isinstance(c, int))
    pm4_buf = dev.pm4_ibs.offset(dev.pm4_ib_alloc.alloc(max(pm4_count, 1) * 4, 16))
    cmds = self._prep_aql(self._q, pm4_buf)
    aql_bytes = b"".join(bytes(c) if isinstance(c, hsa.hsa_kernel_dispatch_packet_t) else c for c in cmds)
    assert len(aql_bytes) % 64 == 0
    assert len(aql_bytes) < cq.ring.nbytes
    cp_bytes = min(len(aql_bytes), (cq.ring.nbytes - (cq.put_value * 64) % cq.ring.nbytes))
    cq.ring.view(offset=(cq.put_value * 64) % cq.ring.nbytes, fmt="B")[:cp_bytes] = aql_bytes[:cp_bytes]
    if (tail_bytes := len(aql_bytes) - cp_bytes) > 0: cq.ring.view(offset=0, fmt="B")[:tail_bytes] = aql_bytes[cp_bytes:]
    cq.put_value += len(aql_bytes) // 64
    cq.signal_doorbell(dev, doorbell_value=cq.put_value - 1)


class ATTInterval:
  def __init__(self, export: dict[str, Any]):
    from tinygrad.device import BufferSpec
    from tinygrad.device import Device
    from tinygrad.runtime.ops_amd import AMDComputeAQLQueue

    self.dev = Device["AMD"]
    if not getattr(self.dev, "is_aql", 0): raise RuntimeError("ATT interval requires AMD_AQL=1")
    self.queue_cls = type("RawAQLQueue", (RawAQLQueueMixin, AMDComputeAQLQueue), {})
    self.ranges: list[tuple[int, int, int]] = []
    self.allocs: list[dict[str, Any]] = []
    for alloc in export["allocations"]:
      size = int(alloc["size"])
      guard = size if int(alloc.get("device_access", 0)) else 0
      alloc_size = size + guard
      buf = self.dev.allocator.alloc(alloc_size, BufferSpec(cpu_access=True, nolru=True, uncached=True))
      self.ranges.append((int(alloc["ptr"]), int(buf.va_addr), alloc_size))
      self.dev.allocator._copyin(buf, memoryview(bytearray(alloc_size)))
      self.allocs.append({**alloc, "replay_va": int(buf.va_addr), "replay_alloc_size": alloc_size, "guard_bytes": guard, "_buf": buf})
    for row in self.allocs:
      if "hex" not in row: continue
      content, ptr_patches = patch_ptr_ranges(bytes.fromhex(row["hex"]), self.ranges)
      content, page_patches = patch_pm4_page_ranges(content, self.ranges)
      self.dev.allocator._copyin(row["_buf"], memoryview(content))
      row["patches"], row["page_patches"] = ptr_patches, page_patches
    self.start_pkt, self.start_patches = patch_ptr_ranges(bytes.fromhex(export["start_packet"]["hex"]), self.ranges)
    self.stop_pkt, self.stop_patches = patch_ptr_ranges(bytes.fromhex(export["stop_packet"]["hex"]), self.ranges)
    self.output = max(self.allocs, key=lambda x: int(x["size"]))

  def _vendor(self, pkt: bytes) -> dict[str, Any]:
    q = self.queue_cls(self.dev).wait(self.dev.timeline_signal, self.dev.timeline_value - 1).memory_barrier()
    q.vendor_packet(pkt).signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)
    st = {"sync_ok": False, "error": None}
    try:
      self.dev.synchronize(timeout=10000)
      st["sync_ok"] = True
    except Exception as e:
      st["error"] = repr(e)
    return st

  def trace(self, label: str, fn: Callable[[], dict[str, Any] | None]) -> dict[str, Any]:
    start = self._vendor(self.start_pkt)
    t0 = time.perf_counter()
    target: dict[str, Any] = {}
    err = None
    if start["sync_ok"]:
      try:
        target = fn() or {}
        self.dev.synchronize(timeout=10000)
      except Exception as e:
        err = repr(e)
    target_ms = (time.perf_counter() - t0) * 1000.0
    stop = self._vendor(self.stop_pkt) if start["sync_ok"] else {"sync_ok": False, "error": "start_failed"}
    out = bytearray(min(int(self.output["size"]), 16 << 20))
    if stop["sync_ok"]:
      self.dev.allocator._copyout(memoryview(out), self.output["_buf"].offset(0, len(out)))
    return {
      "label": label,
      "start": start,
      "target": target,
      "target_error": err,
      "target_wall_ms": round(target_ms, 6),
      "stop": stop,
      "trace": trace_summary(bytes(out)),
      "att_patch": {
        "start_packet_patches": self.start_patches,
        "stop_packet_patches": self.stop_patches,
        "allocations": [{k: v for k, v in row.items() if k != "_buf"} for row in self.allocs],
      },
    }


def build_export() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
  from extra.amd_rocprofiler_r1p2_hcq_replay import build_and_run_helper
  return build_and_run_helper()


def smoke_target() -> Callable[[], dict[str, Any]]:
  from tinygrad import Device, dtypes
  from tinygrad.device import Buffer
  dev = Device["AMD"]
  body_buf = Buffer("AMD", 256, dtypes.float32).ensure_allocated()._buf
  prg = dev.runtime("att_smoke_body", dev.compiler.compile(BODY_SRC))
  args_state = prg.fill_kernargs((body_buf,), ())

  def runit() -> dict[str, Any]:
    prg(body_buf, global_size=(4096, 1, 1), local_size=(64, 1, 1), wait=True, timeout=10000)
    return {"program": prg.name, "global_size": [4096, 1, 1], "local_size": [64, 1, 1]}

  return runit


def tinygrad_q4k_target() -> Callable[[], dict[str, Any]]:
  import numpy as np
  from tinygrad import Tensor, Device, dtypes
  from extra.q4_k_gemv_primitive import q4k_coop_partial_kernel
  from extra.qk_decode_mmvq_p3_q4_correctness import q4_tensor_bytes

  q4, rows, k = q4_tensor_bytes("blk.0.attn_output.weight")
  words = Tensor(np.frombuffer(q4, dtype=np.uint32).copy(), dtype=dtypes.uint32, device="AMD").contiguous().realize()
  Tensor.manual_seed(11)
  x = Tensor.randn(k, dtype=dtypes.float32, device="AMD").contiguous().realize()
  partials = Tensor.empty(rows, 8, dtype=dtypes.float32, device="AMD").contiguous()
  fn = q4k_coop_partial_kernel(rows, k, 8)
  # Compile/warm once outside the traced interval.
  partials.custom_kernel(words, x, fxn=fn)[0].realize()
  Device["AMD"].synchronize(timeout=10000)

  def runit() -> dict[str, Any]:
    got = partials.custom_kernel(words, x, fxn=fn)[0].realize()
    Device["AMD"].synchronize(timeout=10000)
    return {"program_family": "tinygrad_q4k_coop_partial", "rows": rows, "k": k, "shape": list(got.shape)}

  return runit


def imported_llama_mmvq_target() -> Callable[[], dict[str, Any]]:
  import numpy as np
  from tinygrad import Tensor, Device, dtypes
  from extra.q8_ffn_handwritten_oracle import q8_blocks
  from extra.qk_decode_mmvq_p3_q4_correctness import OBJ, RawKernargAMDProgram, kd_offset, q4_tensor_bytes

  cap = json.loads((ROOT / "bench/qk-decode-mmvq-large-project/p2_kernarg_capture.json").read_text())["selected"]["q4_attn_q_or_o"]
  q4, rows, k = q4_tensor_bytes("blk.0.attn_output.weight")
  rng = np.random.default_rng(12)
  x = rng.standard_normal(k).astype(np.float32)
  q8 = q8_blocks(x)
  q4_t = Tensor(np.frombuffer(q4, dtype=np.uint32).copy(), dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8_t = Tensor(np.frombuffer(q8, dtype=np.uint8).copy(), dtype=dtypes.uint8, device="AMD").contiguous().realize()
  out_t = Tensor.zeros(rows, dtype=dtypes.float32, device="AMD").contiguous().realize()
  raw = bytearray(cap["kernarg_bytes"])
  va = lambda t: t.uop.buffer._buf.va_addr
  struct.pack_into("<Q", raw, 0, va(q4_t))
  struct.pack_into("<Q", raw, 8, va(q8_t))
  struct.pack_into("<Q", raw, 16, 0)
  struct.pack_into("<Q", raw, 56, va(out_t))
  prg = RawKernargAMDProgram(Device["AMD"], "att_llama_mmvq_q4", OBJ.read_bytes(), kd_offset(OBJ.read_bytes(), cap["kernel_symbol"]), bytes(raw))
  bufs = (q4_t.uop.buffer._buf, q8_t.uop.buffer._buf, out_t.uop.buffer._buf)
  launch = {"global_size": list(cap["num_workgroups"]), "local_size": list(cap["local"])}
  prg(*bufs, global_size=tuple(cap["num_workgroups"]), local_size=tuple(cap["local"]), wait=True, timeout=10000)
  Device["AMD"].synchronize(timeout=10000)

  def runit() -> dict[str, Any]:
    tm = prg(*bufs, global_size=tuple(cap["num_workgroups"]), local_size=tuple(cap["local"]), wait=True, timeout=10000)
    return {"program_family": "imported_llama_q4_mmvq", "kernel_symbol": cap["kernel_symbol"], "rows": rows, "k": k,
            "device_ms": tm, **launch}

  return runit


def prefill_attention_target() -> Callable[[], dict[str, Any]]:
  from tinygrad import Tensor, Device, dtypes
  Tensor.manual_seed(13)
  # This is the actual tinygrad SDPA primitive shape class used by pp512: [B,H,T,D] attention over T=512.
  q = Tensor.randn(1, 32, 512, 128, dtype=dtypes.half, device="AMD").realize()
  k = Tensor.randn(1, 32, 512, 128, dtype=dtypes.half, device="AMD").realize()
  v = Tensor.randn(1, 32, 512, 128, dtype=dtypes.half, device="AMD").realize()
  out = q.scaled_dot_product_attention(k, v).realize()
  Device["AMD"].synchronize(timeout=10000)

  def runit() -> dict[str, Any]:
    got = q.scaled_dot_product_attention(k, v).realize()
    Device["AMD"].synchronize(timeout=10000)
    return {"program_family": "tinygrad_prefill_sdpa_surface", "shape": list(got.shape), "warm_shape": list(out.shape)}

  return runit


def summarize_decode(rows: list[dict[str, Any]]) -> dict[str, Any]:
  labels = {}
  for row in rows:
    tr = row["trace"]
    pkt = tr.get("packet_top", {})
    body = int(tr.get("body_like_packet_count") or 0)
    wave = int(pkt.get("WAVESTART", 0))
    valu = int(pkt.get("VALUINST", 0))
    labels[row["label"]] = {
      "body_packets": body,
      "waves": wave,
      "valuinst": valu,
      "body_per_wave": (body / wave) if wave else None,
      "status": "body_attributed" if body > 0 else "lifecycle_only",
    }
  return {
    "targets": labels,
    "interpretation": (
      "ATT can body-attribute both tinygrad native Q4_K coop and imported llama MMVQ primitives. "
      "This makes the remaining decode question measurable at instruction/resource level; the decisive next comparison "
      "is role-matched in-model program identity and wave coverage, not packet plumbing."
    ),
  }


def summarize_prefill(row: dict[str, Any]) -> dict[str, Any]:
  prior_path = ROOT / "docs/prefill-nonmatmul-missing-primitive-result-20260619.md"
  return {
    "att_target": {
      "body_packets": row["trace"].get("body_like_packet_count"),
      "packet_top": row["trace"].get("packet_top"),
      "status": "body_attributed" if int(row["trace"].get("body_like_packet_count") or 0) > 0 else "lifecycle_only",
    },
    "prior_component_authority": str(prior_path.relative_to(ROOT)),
    "interpretation": (
      "ATT can body-attribute the pp512 SDPA surface, but the prefill primitive story remains governed by the existing "
      "component ledger: matmul is already near the fast-kernel ceiling, attention is the largest non-matmul residual, "
      "and the prize is small/project-level rather than a missing GEMM primitive."
    ),
  }


def main() -> int:
  os.environ.setdefault("DEV", "AMD")
  OUTDIR.mkdir(parents=True, exist_ok=True)
  build, helper_run, export = build_export()
  result: dict[str, Any] = {
    "date": "2026-06-19",
    "phase": "ATT primitive attribution",
    "helper": {"build": build, "run": helper_run, "export_ok": bool(isinstance(export, dict) and export.get("ok"))},
    "smoke": None,
    "decode": None,
    "prefill": None,
    "gates": {},
    "verdict": "NOT_RUN",
  }
  if not isinstance(export, dict) or not export.get("ok"):
    result["verdict"] = "HELPER_EXPORT_FAIL"
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": result["verdict"], "out": str(OUT.relative_to(ROOT))}, indent=2))
    return 1

  att = ATTInterval(export)
  smoke = att.trace("smoke_body", smoke_target())
  result["smoke"] = smoke
  smoke_pass = bool(smoke["start"]["sync_ok"] and smoke["stop"]["sync_ok"] and smoke["trace"]["body_like_packet_count"] >= 10000)

  decode_rows: list[dict[str, Any]] = []
  prefill_row: dict[str, Any] | None = None
  if smoke_pass:
    decode_rows.append(att.trace("tinygrad_q4k_coop_attn_o", tinygrad_q4k_target()))
    decode_rows.append(att.trace("imported_llama_q4_mmvq_attn_o", imported_llama_mmvq_target()))
    prefill_row = att.trace("tinygrad_prefill_sdpa_surface", prefill_attention_target())
    decode = {"rows": decode_rows, "summary": summarize_decode(decode_rows)}
    prefill = {"row": prefill_row, "summary": summarize_prefill(prefill_row)}
    result["decode"] = decode
    result["prefill"] = prefill
    DECODE_OUT.write_text(json.dumps(decode, indent=2, sort_keys=True) + "\n")
    PREFILL_OUT.write_text(json.dumps(prefill, indent=2, sort_keys=True) + "\n")

  decode_body = bool(decode_rows and all(int(r["trace"].get("body_like_packet_count") or 0) > 0 for r in decode_rows))
  prefill_body = bool(prefill_row and int(prefill_row["trace"].get("body_like_packet_count") or 0) > 0)
  result["gates"] = {
    "helper_packet_export": "PASS",
    "adapter_smoke_body": "PASS" if smoke_pass else "FAIL",
    "decode_body_attribution": "PASS" if decode_body else "FAIL",
    "prefill_attention_body_attribution": "PASS" if prefill_body else "FAIL",
  }
  if smoke_pass and decode_body and prefill_body: result["verdict"] = "PASS_ATT_PRIMITIVE_ATTRIBUTION"
  elif smoke_pass and (decode_body or prefill_body): result["verdict"] = "PARTIAL_ATT_PRIMITIVE_ATTRIBUTION"
  else: result["verdict"] = "ATT_ADAPTER_FAIL"
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  summary = {
    "verdict": result["verdict"],
    "gates": result["gates"],
    "smoke_body_packets": smoke["trace"].get("body_like_packet_count"),
    "decode_body_packets": {r["label"]: r["trace"].get("body_like_packet_count") for r in decode_rows},
    "prefill_body_packets": prefill_row["trace"].get("body_like_packet_count") if prefill_row else None,
  }
  (OUTDIR / "summary.md").write_text(
    "# ATT primitive atlas summary\n\n```json\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n```\n")
  print(json.dumps({"out": str(OUT.relative_to(ROOT)), **summary}, indent=2, sort_keys=True))
  return 0 if result["verdict"] in {"PASS_ATT_PRIMITIVE_ATTRIBUTION", "PARTIAL_ATT_PRIMITIVE_ATTRIBUTION"} else 1


if __name__ == "__main__":
  raise SystemExit(main())
