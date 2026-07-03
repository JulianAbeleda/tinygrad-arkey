from __future__ import annotations
import subprocess
from tinygrad import getenv

AUTO_MAX_CONTEXT = "auto"
MIN_USABLE_CTX = getenv("MIN_USABLE_CTX", 2048)
VRAM_ADMIT_FRACTION = float(getenv("VRAM_ADMIT_FRACTION", "0.8"))

def detect_total_vram_bytes() -> int|None:
  # cheap one-shot total-VRAM probe via rocm-smi; None on any failure.
  try:
    out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=10).stdout
    for ln in out.splitlines():
      if "VRAM Total Memory" in ln: return int(ln.split(":")[-1].strip())
  except Exception: return None
  return None

def detect_free_vram_bytes() -> int|None:
  # free = total - used, parsed from rocm-smi. Probe before weights/KV are resident.
  try:
    out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=10).stdout
    total = used = None
    for ln in out.splitlines():
      if "VRAM Total Used Memory" in ln: used = int(ln.split(":")[-1].strip())
      elif "VRAM Total Memory" in ln: total = int(ln.split(":")[-1].strip())
    if total is not None and used is not None: return total - used
  except Exception: return None
  return None

def resolve_max_context_admission(requested, trained_ctx:int, free_bytes:int|None, weights_bytes:int,
                                  kv_per_tok:int, prefill_per_tok:int, flash_scratch_bytes:int,
                                  model_label:str, kv_quant_supported:bool=False, scale_per_tok:int=0
                                  ) -> tuple[int, bool, dict]:
  """Resolve max_context against available VRAM via a tier ladder.

  Tier 1 is fp16 KV. Tier 2 is int8 KV plus per-token fp16 scales, available only when the caller has a route that
  can consume quantized KV correctly. The arithmetic is model-name agnostic: weights + flash scratch +
  (KV + prefill peak) * context must fit under the configured VRAM fraction.
  """
  is_auto = requested is None or requested == AUTO_MAX_CONTEXT
  if free_bytes is None:
    if is_auto:
      raise RuntimeError("--max_context auto needs a VRAM free-probe but rocm-smi is unavailable. "
                         "Pass an explicit --max_context (it will be used as-is, unchecked).")
    mc = min(int(requested), trained_ctx)
    return mc, False, {"mode": "explicit_no_probe", "max_context": mc, "trained_ctx": trained_ctx}

  budget = free_bytes * VRAM_ADMIT_FRACTION
  fixed = weights_bytes + flash_scratch_bytes
  kv_q8_per_tok = kv_per_tok // 2 + scale_per_tok
  def _mc(kv_pt:int) -> int:
    pt = kv_pt + prefill_per_tok
    return int((budget - fixed) // pt) if pt > 0 else trained_ctx

  mc_fp16 = _mc(kv_per_tok)
  mc_q8 = _mc(kv_q8_per_tok) if kv_quant_supported else -1
  target = trained_ctx if is_auto else int(requested)
  need = MIN_USABLE_CTX if is_auto else target
  report = {"free_gb": free_bytes/1e9, "budget_gb": budget/1e9, "weights_gb": weights_bytes/1e9,
            "flash_scratch_gb": flash_scratch_bytes/1e9, "kv_gb_per_1k": kv_per_tok*1000/1e9,
            "prefill_gb_per_1k": prefill_per_tok*1000/1e9, "mc_fp16": max(mc_fp16, 0),
            "mc_q8": max(mc_q8, 0), "trained_ctx": trained_ctx}

  if mc_fp16 >= need:
    mc = min(target, mc_fp16, trained_ctx)
    report.update(mode=("auto" if is_auto else "explicit"), kv_quant=False, max_context=mc, mc_mem=max(mc_fp16, 0))
    return mc, False, report

  if kv_quant_supported and mc_q8 >= need:
    mc = min(target, mc_q8, trained_ctx)
    report.update(mode=("auto+q8" if is_auto else "explicit+q8"), kv_quant=True, max_context=mc,
                  kv_gb_per_1k=kv_q8_per_tok*1000/1e9, mc_mem=max(mc_q8, 0))
    return mc, True, report

  q8note = (f" int8 KV-quant admits {max(mc_q8,0)} (still < {need})." if kv_quant_supported
            else " KV-quant unsupported for this shape.")
  raise RuntimeError(
    f"{model_label}: {'requested --max_context '+str(requested) if not is_auto else 'auto-scan'} needs {need} tokens; "
    f"fp16 KV admits {max(mc_fp16,0)},{q8note} (free {free_bytes/1e9:.1f}GB, weights {weights_bytes/1e9:.1f}GB, "
    f"budget {budget/1e9:.1f}GB @{VRAM_ADMIT_FRACTION}). Reduce --max_context, free VRAM, or await Q4-KV / eviction tiers.")
