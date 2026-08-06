"""M4 residual_add rangeify substrate schedule gate (CPU hermetic, no GPU).

Scope: `docs/task_workflow/input/m4-resadd-rangeify-substrate-scope-20260806.md` section 4,
S2.  This is the regression gate for the two scheduler deltas (D1
`remove_movement_op_after_rangeify` REDUCE arm, D2 `fix_assign` WAR AFTER skip): with the
PRODUCTION residual fold ACTIVE, the open-resadd flash-decode graph must schedule end to end
on CPU (1620 kernels on this HEAD), the fold must fire on the real block-output chains
(blocks 1+) and fail closed at layer 0, and the open census families must match the
documented substrate baseline.

The NV sm_120 device facts are FAKE: tensors stay on CPU, no GPU is touched, and the crash
being locked is at schedule time, so `create_linear_with_vars` reproduces it without any
execution.  The census is the EXECUTION census (occurrences in `linear.src`), not the
toposort-unique census: the precompiled block bodies' PARAM-form kernels inflate the
toposort count to 71 epi calls, but the schedule executes exactly 36 epi_resadd GEMVs (one
per block).  The closed arm (default records) must stay at its 953-kernel baseline with the
same copy class.
"""
import argparse, collections, json, subprocess, sys
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad import Tensor, UOp, dtypes
from tinygrad.llm.device_facts import DeviceCapabilities, DeviceFacts, ProbeRecord
from tinygrad.llm.flash_decode_attention import FlashDecodeCapability
from tinygrad.llm.model import Transformer
from tinygrad.schedule import create_linear_with_vars
from tinygrad.uop.ops import Ops
import tinygrad.llm.decode_routes as dr
import tinygrad.llm.kernel_program as kp
import tinygrad.llm.model as tgm
import tinygrad.llm.model_route_plan as mrp
import tinygrad.llm.qk_primitives as qkp


MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
NOW = "2026-08-06T00:00:00+00:00"

EXPECTED_KERNELS = 1620
EXPECTED_EPI_RESADD = 36
EXPECTED_LEGACY_4096_4096 = 36  # the Q projections; the o-proj is the epi_resadd GEMV
EXPECTED_COPY_CLASS = 150
CLOSED_KERNELS = 953
CLOSED_LEGACY_4096_4096 = 72  # 36 Q projections + 36 legacy o-proj


def fake_facts(*a, **k):
  return DeviceFacts("CPU", "NV", "sm_120", 32 << 30, 24 << 30,
                     DeviceCapabilities(wave_size=32, supports_warp_shfl_xor=True, supports_tensor_cores=True,
                                        supports_fp16=True, max_workgroup_threads=1024,
                                        max_workgroup_dimensions=(1024, 1024, 1024), lds_bytes=227 << 10,
                                        lds_allocation_granularity=128, global_allocation_granularity=4096),
                     ProbeRecord("fake", NOW, "ok"), ProbeRecord("fake", NOW, "ok"))


def _census(open_mode: bool) -> dict:
  import tinygrad.llm.decode_routes as dr
  import tinygrad.llm.kernel_program as kp
  import tinygrad.llm.model as tgm
  import tinygrad.llm.model_route_plan as mrp
  import tinygrad.llm.qk_primitives as qkp
  saved = {}
  tgm.scan_device_facts = fake_facts
  if open_mode:
    for mod in (mrp, tgm, qkp):
      if hasattr(mod, "decode_q4k_epilogue_resadd_promoted"):
        saved[mod] = mod.decode_q4k_epilogue_resadd_promoted
        mod.decode_q4k_epilogue_resadd_promoted = lambda target: True
  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()
  dr._flash_decode_capability_and_target_for_device = lambda device: (
    FlashDecodeCapability(supports_warp_shfl_xor=True, supports_fdot2=True), ("NV", "sm_120"))

  # Fold-fire census: every residual-slot validation during the live forward is recorded.
  verdicts: list[tuple[str, bool]] = []
  orig_fold = kp._validated_residual_view
  def patched(uop, request, program):
    view, reason = orig_fold(uop, request, program)
    base = uop.base
    verdicts.append((base.op.name, view is not None))
    return view, reason
  kp._validated_residual_view = patched

  model, kv = Transformer.from_gguf(MODEL, 4608)
  for block in model.blk:
    block._use_flash, block._is_prefill = True, False
  for lin in model._q4k_linears.linears:
    lin.decode_enabled = True

  v_start_pos = UOp.variable("start_pos", 0, 4607)
  tokens = Tensor([[1]], dtype=dtypes.int32).contiguous()
  temp = Tensor([0.0], dtype=dtypes.float32).contiguous()
  sp = v_start_pos.bind(513)
  out = model.forward(tokens, sp, temp)
  from tinygrad.callify import transform_to_call
  from tinygrad.tensor import _apply_map_to_tensors
  big_sink, becomes = transform_to_call(UOp.sink(out.uop))
  _apply_map_to_tensors(becomes, name="buffers")
  linear, var_vals = create_linear_with_vars(big_sink)
  for mod, orig in saved.items():
    mod.decode_q4k_epilogue_resadd_promoted = orig
  counts = collections.Counter(getattr(getattr(u.src[0], "arg", None), "name", None) or "<anon>"
                               for u in linear.src)
  copy_class = sum(1 for u in linear.src if _is_copy_call(u))
  epi_resadd = sum(c for n, c in counts.items() if "epi_resadd" in n)
  legacy_4096 = counts.get("q4k_g3_lanemap_gemv_4096_4096", 0)

  accepted = [b for b, ok in verdicts if ok]
  rejected = [b for b, ok in verdicts if not ok]
  return {"kernels": len(linear.src), "counts": counts, "copy_class": copy_class,
          "epi_resadd": epi_resadd, "legacy_4096": legacy_4096,
          "verdicts": (accepted, rejected)}


def run_arm(open_mode: bool) -> int:
  census = _census(open_mode=open_mode)
  if not open_mode:
    print(json.dumps({"kernels": census["kernels"], "copy_class": census["copy_class"],
                      "legacy_4096": census["legacy_4096"]}, indent=1))
    return 0
  print(f"OPEN  SCHEDULE OK {census['kernels']} kernels, copy_class {census['copy_class']}, "
        f"epi_resadd {census['epi_resadd']}, legacy 4096_4096 {census['legacy_4096']}")
  accepted, rejected = census["verdicts"]
  print(f"  fold verdicts: accepted {len(accepted)} (bases "
        f"{collections.Counter(accepted).most_common()}), rejected {len(rejected)} (bases "
        f"{collections.Counter(rejected).most_common()})")
  print("  census:")
  for name, c in census["counts"].most_common():
    print(f"    {c:5d} {name}")
  print(json.dumps({"kernels": census["kernels"], "copy_class": census["copy_class"],
                    "epi_resadd": census["epi_resadd"], "legacy_4096": census["legacy_4096"],
                    "accepted": len(accepted), "rejected": len(rejected)}, indent=1))
  failures = []
  if census["kernels"] != EXPECTED_KERNELS:
    failures.append(f"open kernel count {census['kernels']} != {EXPECTED_KERNELS}")
  if census["epi_resadd"] != EXPECTED_EPI_RESADD:
    failures.append(f"open epi_resadd {census['epi_resadd']} != {EXPECTED_EPI_RESADD}")
  if census["legacy_4096"] != EXPECTED_LEGACY_4096_4096:
    failures.append(f"open legacy 4096_4096 {census['legacy_4096']} != {EXPECTED_LEGACY_4096_4096}")
  if census["copy_class"] != EXPECTED_COPY_CLASS:
    failures.append(f"open copy_class {census['copy_class']} != {EXPECTED_COPY_CLASS}")
  if not accepted:
    failures.append("fold did not fire on any real residual chain (blocks 1+)")
  if not any(b == "CAST" or b == "REDUCE" for b in rejected):
    failures.append("layer-0 fail-closed rejection missing from fold verdicts")
  if failures:
    print("GATE FAIL")
    for f in failures: print(f"  - {f}")
    return 1
  print("GATE PASS")
  return 0


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--arm", choices=("open", "closed", "both"), default="both")
  args = parser.parse_args()
  if args.arm != "both":
    return run_arm(args.arm == "open")
  results = {}
  for arm, open_mode in (("open", True), ("closed", False)):
    proc = subprocess.run([sys.executable, __file__, "--arm", arm],
                          capture_output=True, text=True)
    txt = proc.stdout
    if txt.find("{") != -1:
      results[arm] = json.JSONDecoder().raw_decode(txt[txt.find("{"):])[0]
    else:
      results[arm] = {"error": proc.stderr[-2000:], "stdout_tail": txt[-1000:]}
  open_, closed = results.get("open", {}), results.get("closed", {})
  print(f"OPEN  kernels {open_.get('kernels')}, copy_class {open_.get('copy_class')}, "
        f"epi_resadd {open_.get('epi_resadd')}")
  print(f"CLOSED kernels {closed.get('kernels')}, copy_class {closed.get('copy_class')}, "
        f"legacy 4096_4096 {closed.get('legacy_4096')}")
  failures = []
  if open_.get("kernels") != EXPECTED_KERNELS:
    failures.append(f"open kernel count {open_.get('kernels')} != {EXPECTED_KERNELS}")
  if open_.get("epi_resadd") != EXPECTED_EPI_RESADD:
    failures.append(f"open epi_resadd {open_.get('epi_resadd')} != {EXPECTED_EPI_RESADD}")
  if closed.get("kernels") != CLOSED_KERNELS:
    failures.append(f"closed kernel count {closed.get('kernels')} != {CLOSED_KERNELS}")
  if closed.get("legacy_4096") != CLOSED_LEGACY_4096_4096:
    failures.append(f"closed legacy 4096_4096 {closed.get('legacy_4096')} != {CLOSED_LEGACY_4096_4096}")
  if open_.get("copy_class") != closed.get("copy_class"):
    failures.append(f"copy class differs (open {open_.get('copy_class')}, closed {closed.get('copy_class')})")
  if failures:
    print("GATE FAIL")
    for f in failures: print(f"  - {f}")
    return 1
  print("GATE PASS")
  return 0


def _is_copy_call(u: UOp) -> bool:
  bufs = [s.buf_uop for s in u.src[1:]]
  return len(bufs) == 2 and bufs[0].shape == bufs[1].shape and bufs[0].dtype == bufs[1].dtype


if __name__ == "__main__":
  sys.exit(main())
