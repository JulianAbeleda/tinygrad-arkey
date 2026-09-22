#!/usr/bin/env python3
"""Reverse-bracket wall for the fused Q/K norm+rope admission.

Control and candidate differ only by the Q/K head norm+rope fusion: control
runs the installed chain (``reduce_output_rmsnorm_{32,8}_128`` followed by the
4096/1024-thread ``apply_rope`` elementwise kernel), candidate folds the rotary
rotation into the reduce-output epilogue (144 kernels -> 72 kernels across 36
blocks).  Both arms keep every other promoted route, so the delta isolates the
fusion.  Each arm is a fresh model load under its own
``flock -w 600 /tmp/gpu-bench.lock``.

The harness toggles the closed-default semantic REDUCE_OUTPUT epilogue on the
admitted blocks.  Disabled arms retain the installed graph unchanged.
"""
from __future__ import annotations

import argparse, contextvars, json, os, pathlib, statistics, subprocess, sys

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
LOCK = "/tmp/gpu-bench.lock"

_CTX_BLOCK = contextvars.ContextVar("qk_norm_rope_block", default=None)


def _gpu_state() -> dict:
  fields = ("name", "pstate", "clocks.sm", "clocks.mem", "persistence_mode", "temperature.gpu")
  raw = subprocess.check_output(["nvidia-smi", f"--query-gpu={','.join(fields)}",
                                 "--format=csv,noheader,nounits"], text=True).strip()
  return dict(zip(fields, (x.strip() for x in raw.split(","))))


def _install_fusion_hook(enabled: bool):
  """Install (or clear) the harness-only fused norm+rope hook."""
  import tinygrad.llm.model as model_module

  orig_attention = model_module.TransformerBlock._attention
  orig_dror = model_module._decode_reduce_output_rmsnorm
  orig_apply_rope = model_module.apply_rope

  if not enabled:
    # Restore is best-effort; each timing child is a fresh process so this is
    # primarily for reuse within one process during smoke tests.
    model_module.TransformerBlock._attention = orig_attention
    model_module._decode_reduce_output_rmsnorm = orig_dror
    model_module.apply_rope = orig_apply_rope
    return

  from tinygrad.llm.qk_norm_rope_mmvq import qk_norm_rope_call

  def attention_wrapper(self, x, start_pos, ring_freqs=None, residual_for_output=None):
    self._qk_norm_rope_pending = {}
    token = _CTX_BLOCK.set(self)
    try:
      return orig_attention(self, x, start_pos, ring_freqs, residual_for_output)
    finally:
      _CTX_BLOCK.reset(token)

  def dror_wrapper(norm, x, promoted):
    block = _CTX_BLOCK.get()
    if (promoted and block is not None and
        (norm is getattr(block, "attn_q_norm", None) or norm is getattr(block, "attn_k_norm", None))):
      admission = getattr(block, "_qk_norm_rope_admission", None)
      if admission is not None:
        key = "q" if norm is block.attn_q_norm else "k"
        block._qk_norm_rope_pending[key] = (x, norm)
        return x
    return orig_dror(norm, x, promoted)

  def apply_rope_wrapper(x, freqs_cis):
    block = _CTX_BLOCK.get()
    if block is not None:
      pending = getattr(block, "_qk_norm_rope_pending", {})
      for key in ("q", "k"):
        if key in pending:
          pre_norm, norm = pending.pop(key)
          admission = getattr(block, "_qk_norm_rope_admission", None)
          fused = qk_norm_rope_call(admission, block, key, pre_norm, norm, freqs_cis)
          if fused is not None:
            return fused
    return orig_apply_rope(x, freqs_cis)

  model_module.TransformerBlock._attention = attention_wrapper
  model_module._decode_reduce_output_rmsnorm = dror_wrapper
  model_module.apply_rope = apply_rope_wrapper


def _set_admission(model, enabled: bool) -> list[int]:
  from tinygrad.llm.qk_norm_rope_mmvq import QKNormRopeAdmission
  out = []
  for idx, block in enumerate(model.blk):
    if not hasattr(block, "attn_q_norm") or not hasattr(block, "attn_k_norm"):
      continue
    if enabled:
      block._qk_norm_rope_admission = QKNormRopeAdmission(idx)
      block._decode_qk_norm_rope_promoted = True
    else:
      if hasattr(block, "_qk_norm_rope_admission"): delattr(block, "_qk_norm_rope_admission")
      block._decode_qk_norm_rope_promoted = False
    out.append(idx)
  return out


def timing_child(depth: int, count: int, max_context: int, reps: int, enabled: bool|None, composed: bool,
                 out: pathlib.Path, adaptive_s64:bool=False, static_s64_horizon:bool=False) -> dict:
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  model = _load(MODEL, max_context)
  model._flash_decode_adaptive_s64_lease = adaptive_s64
  if enabled is None:
    admitted = [idx for idx, block in enumerate(model.blk)
                if getattr(block, "_decode_qk_norm_rope_promoted", False)]
    if not getattr(model, "_decode_qk_norm_rope_promoted", False) or len(admitted) != len(model.blk):
      raise RuntimeError(f"production Q/K norm+RoPE policy did not promote every block: model="
                         f"{getattr(model, '_decode_qk_norm_rope_promoted', False)} blocks={admitted}")
  else:
    admitted = _set_admission(model, enabled)
  model._decode_direct_greedy_promoted = composed
  model._decode_feedback_pingpong_promoted = composed
  if adaptive_s64:
    # Capture the second geometry before prompt execution so the first token
    # crossing context 768 never pays compilation/capture latency.  The dummy
    # KV slot is overwritten before it becomes live in the real generation.
    from tinygrad import Tensor,UOp
    tok,temp=Tensor([[1]],dtype="int32").contiguous(),Tensor([0.0]);pos=UOp.variable("start_pos",0,max_context-1)
    if composed:
      for slot in (0,1):
        for _ in range(3):model(tok,pos.bind(min(800,max_context-1)),temp,use_flash=True,greedy=True,feedback_slot=slot,flash_split_count=64).realize()
    else:
      for _ in range(3):model(tok,pos.bind(min(800,max_context-1)),temp,use_flash=True,flash_split_count=64).realize()
  gen = model.generate(_prompt(MODEL, depth), chunk_size=32, temperature=0.0,
                       expected_output_tokens=(count*reps+7 if static_s64_horizon else None))
  try:
    settled = _settled_continuous_windows(gen, Device[Device.DEFAULT], count, reps)
  finally:
    gen.close()
  result = {"schema": "tinygrad.qk_norm_rope_wall_bracket.v1",
    "enabled": enabled, "route_source": "production-policy" if enabled is None else "research-override",
    "composed": composed, "adaptive_s64":adaptive_s64,"static_s64_horizon":static_s64_horizon,"qk_norm_rope_blocks": admitted,
    "gpu_state": _gpu_state(),
    "depth": depth, "count": count, "reps": reps,
    "included_cost": True, "settled_continuous": True, **settled}
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return result


def bracket(depth: int, count: int, max_context: int, reps: int, composed: bool, out: pathlib.Path) -> dict:
  root = pathlib.Path(str(out).removesuffix(".json"))
  root.mkdir(parents=True, exist_ok=True)

  def child(enabled: bool, label: str):
    o = root / f"{label}.json"
    cmd = ["timeout", "1800", "flock", "-w", "600", LOCK, sys.executable,
      str(pathlib.Path(__file__).resolve()), "--mode", "timing-child",
      "--depth", str(depth), "--count", str(count), "--max-context", str(max_context),
      "--reps", str(reps), "--out", str(o)]
    if enabled:
      cmd.append("--enabled")
    if composed:
      cmd.append("--composed")
    run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      env={**os.environ, "PYTHONPATH": "/home/ubuntu/tinygrad-arkey"})
    if run.returncode:
      raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-4000:]}")
    return json.loads(o.read_text())

  rows = [child(False, "control_a"), child(True, "candidate"), child(False, "control_c")]
  control_mid = statistics.median((rows[0]["median_ms_per_token"], rows[2]["median_ms_per_token"]))
  candidate = rows[1]["median_ms_per_token"]
  hashes = {r["token_stream_hash"] for r in rows}
  result = {"schema": "tinygrad.qk_norm_rope_wall_bracket.v1", "mode": "reverse-bracket",
    "depth": depth, "count": count, "reps": reps, "composed": composed,
    "control_a_ms_per_token": rows[0]["median_ms_per_token"],
    "candidate_ms_per_token": candidate,
    "control_c_ms_per_token": rows[2]["median_ms_per_token"],
    "control_midpoint_ms_per_token": control_mid,
    "candidate_minus_control_ms_per_token": candidate - control_mid,
    "candidate_speedup_pct": (control_mid / candidate - 1) * 100,
    "all_token_hashes_equal": len(hashes) == 1,
    "token_stream_hash": sorted(hashes)[0] if len(hashes) == 1 else sorted(hashes),
    "verdict": "WALL_PASS" if len(hashes) == 1 and candidate < control_mid else "NO_GO_WALL",
    "arms": rows}
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return result


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=("timing-child", "timing"), default="timing")
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--count", type=int, default=32)
  ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--enabled", action="store_true")
  ap.add_argument("--production-default", action="store_true")
  ap.add_argument("--composed", action="store_true")
  ap.add_argument("--adaptive-s64", action="store_true")
  ap.add_argument("--static-s64-horizon", action="store_true")
  ap.add_argument("--out", default="/tmp/qk_norm_rope_wall_bracket.json")
  args = ap.parse_args()
  if args.mode == "timing-child":
    if args.enabled and args.production_default: ap.error("--enabled and --production-default are mutually exclusive")
    enabled = None if args.production_default else args.enabled
    row = timing_child(args.depth, args.count, args.max_context, args.reps, enabled, args.composed,
      pathlib.Path(args.out),args.adaptive_s64,args.static_s64_horizon)
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0
  result = bracket(args.depth, args.count, args.max_context, args.reps, args.composed, pathlib.Path(args.out))
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
