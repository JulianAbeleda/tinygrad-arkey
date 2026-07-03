from __future__ import annotations
import subprocess
from tinygrad import getenv

AUTO_MAX_CONTEXT = "auto"
MIN_USABLE_CTX = getenv("MIN_USABLE_CTX", 2048)
MIN_RING_WINDOW = getenv("MIN_RING_WINDOW", 1024)   # below this a StreamingLLM window is a toy -> don't offer the ring
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
                                  model_label:str, kv_quant_supported:bool=False, scale_per_tok:int=0,
                                  stream:str="auto", ring_supported:bool=False) -> tuple[int, bool, dict]:
  """Resolve max_context against available VRAM via a tier ladder. Returns (max_context/window N, kv_quant, report);
  report['ring'] flags the lossy StreamingLLM streaming tier and report['mode'] names the tier.

  Ladder (model-name agnostic arithmetic): Tier 1 fp16 KV (exact) -> Tier 2 int8 Q8 KV (near-lossless) -> Tier 3 RING
  (fp16, lossy: unbounded LOGICAL context in an N-token physical window, content older than N + 4 sinks forgotten).
  stream='on' forces the ring even when a lossless tier fits (for deliberately-unbounded generation); 'off' stops the
  ladder at Q8 (exact-context semantics, today's refuse-loud); 'auto' uses the ring only as the last rung. The ring
  window N = the fp16 cap (ring K is fp16 un-roped; Q8+ring not composed), floored at MIN_RING_WINDOW, capped at
  trained_ctx (positions re-base into [0,N) and must stay in-distribution)."""
  is_auto = requested is None or requested == AUTO_MAX_CONTEXT
  ring_ok = ring_supported and stream != "off"
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
  ring_N = min(mc_fp16, trained_ctx) if is_auto else min(int(requested), mc_fp16, trained_ctx)  # fp16 window, in-distribution
  target = trained_ctx if is_auto else int(requested)
  need = MIN_USABLE_CTX if is_auto else target
  report = {"free_gb": free_bytes/1e9, "budget_gb": budget/1e9, "weights_gb": weights_bytes/1e9,
            "flash_scratch_gb": flash_scratch_bytes/1e9, "kv_gb_per_1k": kv_per_tok*1000/1e9,
            "prefill_gb_per_1k": prefill_per_tok*1000/1e9, "mc_fp16": max(mc_fp16, 0),
            "mc_q8": max(mc_q8, 0), "mc_ring": max(ring_N, 0), "trained_ctx": trained_ctx, "ring": False}

  def _ring_banner(N:int) -> str:
    return (f"STREAMING MODE: fp16 KV window N={N} -- generation is unbounded, but content older than the last ~{N} "
            f"tokens (plus 4 sinks) is evicted and forgotten (StreamingLLM). --stream=off to disable.")

  # stream='on': deliberately unbounded generation -- use the ring even when a lossless tier fits.
  if stream == "on":
    if not ring_supported:
      raise RuntimeError(f"{model_label}: --stream=on requires the ring-capable decode route (structural shape class).")
    if ring_N < MIN_RING_WINDOW:
      raise RuntimeError(f"{model_label}: --stream=on needs a >={MIN_RING_WINDOW}-token fp16 window but only "
                         f"{max(ring_N,0)} fits (free {free_bytes/1e9:.1f}GB, weights {weights_bytes/1e9:.1f}GB).")
    report.update(mode="ring", kv_quant=False, ring=True, max_context=ring_N, mc_mem=max(ring_N, 0), banner=_ring_banner(ring_N))
    return ring_N, False, report

  # Tier 1: fp16 exact.
  if mc_fp16 >= need:
    mc = min(target, mc_fp16, trained_ctx)
    report.update(mode=("auto" if is_auto else "explicit"), kv_quant=False, max_context=mc, mc_mem=max(mc_fp16, 0))
    return mc, False, report

  # Tier 2: int8 Q8 (near-lossless).
  if kv_quant_supported and mc_q8 >= need:
    mc = min(target, mc_q8, trained_ctx)
    report.update(mode=("auto+q8" if is_auto else "explicit+q8"), kv_quant=True, max_context=mc,
                  kv_gb_per_1k=kv_q8_per_tok*1000/1e9, mc_mem=max(mc_q8, 0))
    return mc, True, report

  q8note = (f" int8 KV-quant admits {max(mc_q8,0)} (still < {need})." if kv_quant_supported
            else " KV-quant unsupported for this shape.")
  # Explicit --max_context is STRICT: the ring never rescues a requested int (N is physical; 8000 fp16 doesn't fit).
  if not is_auto:
    _hint = (f" For unbounded generation with a smaller lossy window, pass --stream (window would be N={max(ring_N,0)})."
             if ring_supported and ring_N >= MIN_RING_WINDOW else "")
    raise RuntimeError(
      f"{model_label}: requested --max_context {requested} needs {need} tokens; fp16 KV admits {max(mc_fp16,0)},{q8note} "
      f"(free {free_bytes/1e9:.1f}GB, weights {weights_bytes/1e9:.1f}GB, budget {budget/1e9:.1f}GB @{VRAM_ADMIT_FRACTION})."
      f" Largest admissible is {max(max(mc_q8,0), max(mc_fp16,0))}. Reduce --max_context or free VRAM.{_hint}")

  # Tier 3 (auto, stream!=off): the RING -- lossy unbounded context, only when nothing lossless is usable.
  if ring_ok and ring_N >= MIN_RING_WINDOW:
    report.update(mode="ring", kv_quant=False, ring=True, max_context=ring_N, mc_mem=max(ring_N, 0), banner=_ring_banner(ring_N))
    return ring_N, False, report

  # Final refusal: not even the streaming window is usable.
  _ringnote = (f" even the streaming window would be under {MIN_RING_WINDOW} tokens." if ring_supported
               else " (streaming unsupported for this shape).")
  raise RuntimeError(
    f"{model_label}: auto-scan needs {need} tokens; fp16 KV admits {max(mc_fp16,0)},{q8note}{_ringnote} "
    f"(free {free_bytes/1e9:.1f}GB, weights {weights_bytes/1e9:.1f}GB, budget {budget/1e9:.1f}GB @{VRAM_ADMIT_FRACTION}). "
    f"Refusing: free VRAM or use a smaller model. (Q8+ring composition would ~double the streaming window; not yet implemented.)")
