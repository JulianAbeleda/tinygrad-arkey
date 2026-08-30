#!/usr/bin/env python3
"""Close the cooperative-Q production-conditioning identity with live data.

This is measurement tooling only.  It monkeypatches ``NVProgram.__call__``
during one real decode capture, retains the exact programs and buffers for the
cross-layer chain

  FFN gate/up -> FFN down -> RMSNorm/Q8 provider -> cooperative Q projection,

then replays data-linked clones while keeping the immutable production weight
allocations and their virtual addresses.  It also reads the normal installed
Q command interval P from HCQ graph-profile timestamps emitted by the same
process.  No production, renderer, scheduler, runtime, model, or route file is
changed.

Each invocation is one fresh-process bracket session.  Run both forward and
reverse ``--order`` sequences and aggregate them outside the child process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from nv_r_residual_cache_dispatch_probe import _alloc, _make_queue  # noqa: E402

SCHEMA = "tinygrad.nv_r_predecessor_conditioned_exact.v1"
PROVIDER = "rmsnorm_q8_1_llama_provider_4096"
TARGET = "q4k_warp_coop_q8_dp4a_partial_4096_4096"
GATEUP = "q4k_g3_lanemap_gemv_w1w3fused16_12288_4096"
FFNDOWN = "q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd"
SELECTED = (GATEUP, FFNDOWN, PROVIDER, TARGET)
DECODE_GROUP_SIZES = (32, 64, 128, 256, 116)
WEIGHT_INDICES = {GATEUP: {1, 2}, FFNDOWN: {1}, PROVIDER: {2}, TARGET: {1}}

# Captured production layouts.  Every ordinal is asserted against both size
# and live-VA dependency before it is used.
G_OUT, G_W1, G_W3, G_IN = 0, 1, 2, 3
F_OUT, F_W, F_IN, F_RESID = 0, 1, 2, 3
# Fused RMSNorm/Q8 provider layout is [packed_q8_out, fp32_hidden_in,
# bf16_norm_weight], unlike the projection [out, weight, input] layout.
P_OUT, P_IN, P_W = 0, 1, 2
T_OUT, T_W, T_IN = 0, 1, 2
WEIGHT_BYTES_MIN = 1 << 20


def _sha(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _va(buf) -> int:
  va = getattr(buf, "va_addr", None)
  if va is None: va = getattr(getattr(buf, "_buf", None), "va_addr", None)
  return int(va)


def _size(buf) -> int:
  sz = getattr(buf, "size", None)
  if sz is None: sz = getattr(getattr(buf, "_buf", None), "size", None)
  return int(sz)


def _copy(dev, buf) -> bytes:
  host = memoryview(bytearray(_size(buf)))
  dev.allocator._copyout(host, buf)
  return bytes(host)


def _clone(dev, buf):
  data = _copy(dev, buf)
  out = _alloc(dev, len(data))
  dev.allocator._copyin(out, memoryview(bytearray(data)))
  return out


def _zero(dev, size: int):
  out = _alloc(dev, size)
  dev.allocator._copyin(out, memoryview(bytearray(size)))
  return out


def _checksum(dev, buf) -> str:
  return _sha(_copy(dev, buf))


def _gpu_state() -> dict[str, str]:
  fields = ("name", "driver_version", "persistence_mode", "pstate", "clocks.sm", "clocks.mem",
            "power.draw", "temperature.gpu")
  raw = subprocess.check_output(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                                text=True).strip()
  vals = [x.strip() for x in raw.split(",")]
  return dict(zip(fields, vals))


def install_hook(cap: dict[str, Any], occurrence: int, prefix_count: int):
  import tinygrad.runtime.ops_nv as ops_nv
  orig_call = ops_nv.NVProgram.__call__
  seen: dict[str, int] = {}
  seq = 0
  recent: list[dict[str, Any]] = []

  def patched_call(self, *bufs, global_size=(1, 1, 1), local_size=(1, 1, 1),
                   vals=(), wait=False, timeout=None):
    nonlocal seq
    my_seq, seq = seq, seq + 1
    name = self.name
    family_occurrence = None
    selected = False
    if name in SELECTED:
      family_occurrence = seen.get(name, 0)
      seen[name] = family_occurrence + 1
      selected = family_occurrence == occurrence

    # Waiting only affects first-pass capture.  Normal installed P is read
    # later from steady HCQ graph replays, not from this call.
    ret = orig_call(self, *bufs, global_size=global_size, local_size=local_size,
                    vals=vals, wait=(wait or selected), timeout=timeout)
    if selected:
      live = tuple(bufs)
      # Snapshot every mutable/small allocation at the exact call boundary;
      # retain large immutable production weights at their original VAs.
      snapshots = tuple(_clone(self.dev, b) if _size(b) < WEIGHT_BYTES_MIN else None for b in live)
      cap[name] = {
        "name": name, "program": self, "live": live, "snapshots": snapshots,
        "cubin": bytes(self.lib), "cubin_sha": _sha(bytes(self.lib)),
        "buf_va": [_va(b) for b in live], "buf_size": [_size(b) for b in live],
        "vals": list(vals), "global_size": list(global_size), "local_size": list(local_size),
        "seq": my_seq, "family_occurrence": family_occurrence,
      }
      if name == GATEUP:
        cap["_prefix"] = list(recent[-prefix_count:])
    # Retain a bounded exact launch ring.  These records are replayed only
    # after generation has ended, so mutating their ordinary output buffers
    # cannot affect a token gate.
    recent.append({"name": name, "program": self, "live": tuple(bufs), "buf_va": [_va(b) for b in bufs],
                   "buf_size": [_size(b) for b in bufs], "vals": list(vals),
                   "global_size": list(global_size), "local_size": list(local_size),
                   "cubin_sha": _sha(bytes(self.lib)), "seq": my_seq})
    if len(recent) > max(64, prefix_count + 8): del recent[0]
    return ret

  ops_nv.NVProgram.__call__ = patched_call
  return orig_call


def restore_hook(orig_call) -> None:
  import tinygrad.runtime.ops_nv as ops_nv
  ops_nv.NVProgram.__call__ = orig_call


def run_decode(cap: dict[str, Any], occurrence: int, prefix_count: int, model: str, decode_tokens: int) -> list[int]:
  from tinygrad import Device
  from tinygrad.llm.generate import load_model_and_tokenizer

  orig = install_hook(cap, occurrence, prefix_count)
  tokens: list[int] = []
  try:
    dev = Device[Device.DEFAULT]
    mdl, tok = load_model_and_tokenizer(model, 4608, seed=20260617)
    base = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
    gen = mdl.generate(base[:512].copy(), chunk_size=32, temperature=0.0)
    try:
      for _ in range(decode_tokens): tokens.append(int(next(gen)))
      dev.synchronize()
    finally:
      gen.close()
  finally:
    restore_hook(orig)
  return tokens


def _complete_profile_replays(path: pathlib.Path) -> list[list[dict[str, Any]]]:
  lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
  sizes = [len(x.get("entries", [])) for x in lines]
  out: list[list[dict[str, Any]]] = []
  i = 0
  while i + len(DECODE_GROUP_SIZES) <= len(lines):
    if tuple(sizes[i:i+len(DECODE_GROUP_SIZES)]) == DECODE_GROUP_SIZES:
      out.append([e for row in lines[i:i+len(DECODE_GROUP_SIZES)] for e in row.get("entries", [])])
      i += len(DECODE_GROUP_SIZES)
    else:
      i += 1
  return out


def _installed_p(path: pathlib.Path, occurrence: int, warmup: int) -> dict[str, Any]:
  replays = _complete_profile_replays(path)
  rows: list[float] = []
  target_counts: list[int] = []
  for replay in replays:
    targets = [float(e["duration"]) for e in replay if e.get("name") == TARGET]
    target_counts.append(len(targets))
    if len(targets) > occurrence: rows.append(targets[occurrence])
  steady = rows[warmup:]
  if not steady:
    raise RuntimeError(f"no steady installed P rows: complete={len(replays)} counts={target_counts} warmup={warmup}")
  return {
    "complete_replay_count": len(replays), "target_counts": target_counts,
    "all_us": [round(x, 3) for x in rows], "warmup_dropped": warmup,
    "steady_us": [round(x, 3) for x in steady],
    "median_us": round(statistics.median(steady), 3),
  }


def _buf(rec: dict[str, Any], idx: int):
  # Immutable weights stay at their production VA regardless of allocation
  # size (the fused provider's norm weight is only 8 KiB).
  if idx in WEIGHT_INDICES[rec["name"]]: return rec["live"][idx]
  return rec["snapshots"][idx] if rec["snapshots"][idx] is not None else rec["live"][idx]


def _assert_capture(cap: dict[str, Any], occurrence: int, prefix_count: int) -> dict[str, Any]:
  missing = [name for name in SELECTED if name not in cap]
  if missing: raise RuntimeError(f"capture missing {missing}; have={sorted(cap)}")
  g, f, p, t = (cap[x] for x in SELECTED)
  expected_sizes = {
    GATEUP: [24576, 28311552, 28311552, 8192],
    FFNDOWN: [16384, 41287680, 24576, 16384],
    PROVIDER: [8192, 16384, 8192],
    TARGET: [65536, 9437184, 8192],
  }
  for name in SELECTED:
    rec = cap[name]
    if rec["family_occurrence"] != occurrence:
      raise RuntimeError(f"occurrence mismatch {name}: {rec['family_occurrence']} != {occurrence}")
    if rec["buf_size"] != expected_sizes[name]:
      raise RuntimeError(f"layout mismatch {name}: {rec['buf_size']} != {expected_sizes[name]}")
  deps = {
    "gateup_out_to_down_in": g["buf_va"][G_OUT] == f["buf_va"][F_IN],
    "down_out_to_provider_in": f["buf_va"][F_OUT] == p["buf_va"][P_IN],
    "provider_out_to_target_in": p["buf_va"][P_OUT] == t["buf_va"][T_IN],
  }
  seq = [g["seq"], f["seq"], p["seq"], t["seq"]]
  prefix = cap.get("_prefix", [])
  prefix_seqs = [x["seq"] for x in prefix]
  prefix_contiguous = len(prefix) == prefix_count and prefix_seqs == list(range(g["seq"] - prefix_count, g["seq"]))
  if not all(deps.values()) or seq != sorted(seq) or len(set(seq)) != 4 or not prefix_contiguous:
    raise RuntimeError(f"not an exact ordered live chain: deps={deps} seq={seq}")
  return {"dependencies": deps, "sequence": dict(zip(SELECTED, seq)),
          "family_occurrence": occurrence, "prefix_count": len(prefix),
          "prefix_contiguous": prefix_contiguous, "prefix_names": [x["name"] for x in prefix],
          "prefix_sequence": prefix_seqs}


def _restore_snapshot(dev, rec: dict[str, Any], idx: int) -> None:
  snap = rec["snapshots"][idx]
  if snap is None: raise RuntimeError(f"no mutable snapshot for {rec['name']}[{idx}]")
  data = _copy(dev, snap)
  dev.allocator._copyin(rec["live"][idx], memoryview(bytearray(data)))


def _build_arms(dev, cap: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
  g, f, p, t = (cap[x] for x in SELECTED)
  prod_checksum = _checksum(dev, t["snapshots"][T_OUT])

  # C0: exact target input/weight, isolated.
  c0_out = _zero(dev, t["buf_size"][T_OUT])
  c0_tbufs = (c0_out, _buf(t, T_W), _buf(t, T_IN))

  # C2: exact provider input/weight -> target, with a private linked output.
  c2_pout = _zero(dev, p["buf_size"][P_OUT])
  c2_pbufs = [_buf(p, i) for i in range(len(p["live"]))]
  c2_pbufs[P_OUT] = c2_pout
  c2_out = _zero(dev, t["buf_size"][T_OUT])
  c2_tbufs = (c2_out, _buf(t, T_W), c2_pout)

  # C3: exact data-linked cross-layer chain.  Outputs are private; immutable
  # production weights retain their live VAs and cache-set mappings.
  c3_gout = _zero(dev, g["buf_size"][G_OUT])
  c3_gbufs = [_buf(g, i) for i in range(len(g["live"]))]
  c3_gbufs[G_OUT] = c3_gout
  c3_fout = _zero(dev, f["buf_size"][F_OUT])
  c3_fbufs = [_buf(f, i) for i in range(len(f["live"]))]
  c3_fbufs[F_OUT], c3_fbufs[F_IN] = c3_fout, c3_gout
  c3_pout = _zero(dev, p["buf_size"][P_OUT])
  c3_pbufs = [_buf(p, i) for i in range(len(p["live"]))]
  c3_pbufs[P_OUT], c3_pbufs[P_IN] = c3_pout, c3_fout
  c3_out = _zero(dev, t["buf_size"][T_OUT])
  c3_tbufs = (c3_out, _buf(t, T_W), c3_pout)

  # C4: the identical chain on every original production VA.  Restore only
  # the two external read-only sources captured at this call boundary; every
  # linked intermediate is regenerated by the chain itself.
  def restore_live_sources() -> None:
    _restore_snapshot(dev, g, G_IN)
    _restore_snapshot(dev, f, F_RESID)
  restore_live_sources()
  c4_gbufs, c4_fbufs, c4_pbufs, c4_tbufs = g["live"], f["live"], p["live"], t["live"]

  def launch(rec, bufs):
    return (rec["program"], tuple(bufs), tuple(rec["vals"]), tuple(rec["global_size"]), tuple(rec["local_size"]))

  prefix_launches = [launch(rec, rec["live"]) for rec in cap["_prefix"]]
  # Same QMD depth as the 19-command production prefix, but only the tiny
  # provider footprint.  Gate/down/provider still immediately precede Q, so
  # C6 isolates command position/front-end depth from the real prefix state.
  position_pad_launches = [launch(p, c2_pbufs) for _ in cap["_prefix"]]

  arms = {
    "C0": {"preds": [], "target_bufs": c0_tbufs, "output": c0_out},
    "C2": {"preds": [launch(p, c2_pbufs)], "target_bufs": c2_tbufs, "output": c2_out},
    "C3": {"preds": [launch(g, c3_gbufs), launch(f, c3_fbufs), launch(p, c3_pbufs)],
           "target_bufs": c3_tbufs, "output": c3_out},
    "C4": {"preds": [launch(g, c4_gbufs), launch(f, c4_fbufs), launch(p, c4_pbufs)],
           "target_bufs": c4_tbufs, "output": c4_tbufs[T_OUT], "prepare": restore_live_sources},
    "C5": {"preds": prefix_launches + [launch(g, c3_gbufs), launch(f, c3_fbufs), launch(p, c3_pbufs)],
           "target_bufs": c3_tbufs, "output": c3_out},
    "C6": {"preds": position_pad_launches + [launch(g, c3_gbufs), launch(f, c3_fbufs), launch(p, c3_pbufs)],
           "target_bufs": c3_tbufs, "output": c3_out},
  }
  return arms, prod_checksum


def _run_once(dev, target: dict[str, Any], arm: dict[str, Any], timeout_s: float) -> float:
  q = _make_queue(dev)
  for prg, bufs, vals, grid, block in arm["preds"]:
    q.exec(prg, prg.fill_kernargs(bufs, vals=vals), grid, block)
  st, en = dev.new_signal(), dev.new_signal()
  q.timestamp(st)
  targs = target["program"].fill_kernargs(tuple(arm["target_bufs"]), vals=tuple(target["vals"]))
  q.exec(target["program"], targs, tuple(target["global_size"]), tuple(target["local_size"]))
  q.timestamp(en)
  q.signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
  dev.synchronize(timeout=int(timeout_s * 1000))
  return float(en.timestamp - st.timestamp)


def _validate_arm(dev, target: dict[str, Any], arm: dict[str, Any], expected: str, timeout_s: float) -> str:
  if prepare := arm.get("prepare"): prepare()
  _run_once(dev, target, arm, timeout_s)
  got = _checksum(dev, arm["output"])
  if got != expected: raise RuntimeError(f"arm output mismatch: got={got} expected={expected}")
  return got


def _measure_arm(dev, target: dict[str, Any], arm: dict[str, Any], n: int, timeout_s: float) -> list[float]:
  if prepare := arm.get("prepare"): prepare()
  return [_run_once(dev, target, arm, timeout_s) for _ in range(n)]


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--occurrence", type=int, default=0)
  ap.add_argument("--n", type=int, default=24)
  ap.add_argument("--warmup", type=int, default=4)
  ap.add_argument("--profile-warmup", type=int, default=2)
  ap.add_argument("--decode-tokens", type=int, default=12)
  ap.add_argument("--prefix-count", type=int, default=19)
  ap.add_argument("--order", default="C0,C2,C3,C4,C5,C6,C5,C4,C3,C2,C0")
  ap.add_argument("--timeout-s", type=float, default=180.0)
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--profile-jsonl", type=pathlib.Path, required=True)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  if os.environ.get("PROFILE") != "1": raise SystemExit("PROFILE=1 is required for same-session installed P")
  env_profile = os.environ.get("HCQ_GRAPH_PROFILE_JSON")
  if not env_profile or pathlib.Path(env_profile).resolve() != args.profile_jsonl.resolve():
    raise SystemExit("HCQ_GRAPH_PROFILE_JSON must equal --profile-jsonl before process start")
  if args.profile_jsonl.exists(): raise SystemExit(f"refusing stale profile path: {args.profile_jsonl}")
  order = [x.strip() for x in args.order.split(",") if x.strip()]
  if not order or any(x not in ("C0", "C2", "C3", "C4", "C5", "C6") for x in order): raise SystemExit(f"invalid order {order}")

  from tinygrad import Device
  gpu_state_before = _gpu_state()
  cap: dict[str, Any] = {}
  tokens = run_decode(cap, args.occurrence, args.prefix_count, args.model, args.decode_tokens)
  dev = Device["NV"]
  capture_identity = _assert_capture(cap, args.occurrence, args.prefix_count)
  installed_p = _installed_p(args.profile_jsonl, args.occurrence, args.profile_warmup)
  arms, production_checksum = _build_arms(dev, cap)

  arm_checksums = {name: _validate_arm(dev, cap[TARGET], arm, production_checksum, args.timeout_s)
                   for name, arm in arms.items()}
  runs: list[dict[str, Any]] = []
  by_arm: dict[str, list[float]] = {name: [] for name in arms}
  for name in order:
    vals = _measure_arm(dev, cap[TARGET], arms[name], args.n, args.timeout_s)[args.warmup:]
    by_arm[name].extend(vals)
    runs.append({"arm": name, "durations_us": [round(x, 3) for x in vals],
                 "median_us": round(statistics.median(vals), 3)})

  med = {name: round(statistics.median(vals), 3) for name, vals in by_arm.items() if vals}
  required = {"C0", "C2", "C3", "C4", "C5", "C6"}
  if set(med) != required: raise RuntimeError(f"order must sample all arms, got {sorted(med)}")
  p = float(installed_p["median_us"])
  terms = {
    "P_minus_C0_us": round(p - med["C0"], 3),
    "C2_minus_C0_us": round(med["C2"] - med["C0"], 3),
    "C3_minus_C2_us": round(med["C3"] - med["C2"], 3),
    "C4_minus_C3_us": round(med["C4"] - med["C3"], 3),
    "C5_minus_C4_us": round(med["C5"] - med["C4"], 3),
    "P_minus_C5_us": round(p - med["C5"], 3),
  }
  terms["identity_residual_us"] = round(terms["P_minus_C0_us"] -
      (terms["C2_minus_C0_us"] + terms["C3_minus_C2_us"] + terms["C4_minus_C3_us"] +
       terms["C5_minus_C4_us"] + terms["P_minus_C5_us"]), 6)
  adjudication = {
    "position_only_C6_minus_C3_us": round(med["C6"] - med["C3"], 3),
    "real_prefix_minus_position_C5_minus_C6_us": round(med["C5"] - med["C6"], 3),
  }
  gpu_state_after = _gpu_state()

  result = {
    "schema": SCHEMA,
    "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "gpu_state_before": gpu_state_before, "gpu_state_after": gpu_state_after,
    "occurrence": args.occurrence, "order": order, "n_per_block": args.n,
    "warmup_per_block": args.warmup, "capture_identity": capture_identity,
    "token_ids": tokens, "token_sha256": _sha(",".join(map(str, tokens)).encode()),
    "production_output_checksum": production_checksum, "arm_output_checksums": arm_checksums,
    "cubin_sha256": {name: cap[name]["cubin_sha"] for name in SELECTED},
    "buffer_sizes": {name: cap[name]["buf_size"] for name in SELECTED},
    "buffer_vas": {name: cap[name]["buf_va"] for name in SELECTED},
    "installed_P": installed_p, "arm_runs": runs, "arm_medians_us": med,
    "identity": terms, "adjudication": adjudication,
    "verdict": "MEASURED_IDENTITY" if terms["identity_residual_us"] == 0 else "IDENTITY_FAIL",
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
