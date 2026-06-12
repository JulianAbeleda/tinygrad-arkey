#!/usr/bin/env python3
import argparse, json, math, pathlib, re, statistics
from collections import Counter
from dataclasses import dataclass, field

TOKEN_RE = re.compile(r"^\s*(?P<ms>[0-9.]+) ms,\s+(?P<tps>[0-9.]+) tok/s,\s+(?P<gbs>[0-9.]+) GB/s,")
AMD_RE = re.compile(r"^\*\*\* AMD\s+\d+\s+(?P<name>.*?)\s+arg\s+\d+\s+mem\s+.*?\btm\s+(?P<tm>[0-9.]+)(?P<unit>us|ms|s)/")

BUCKETS = (
  "q4k_primitive_gemv",
  "q4k_primitive_reduction",
  "fallback_q4k_fused",
  "attention_misc",
  "norm_sampling_misc",
  "copy",
  "other_amd",
  "residual_overhead",
)

@dataclass
class Kernel:
  name: str
  ms: float

@dataclass
class Token:
  wall_ms: float
  tok_s: float
  gb_s: float
  kernels: list[Kernel] = field(default_factory=list)

def _time_to_ms(value:float, unit:str) -> float:
  if unit == "us": return value / 1000.0
  if unit == "ms": return value
  if unit == "s": return value * 1000.0
  raise ValueError(f"unknown unit {unit}")

def _label(path:pathlib.Path) -> tuple[str, str]:
  stem = path.name
  model = "14B" if "14b" in stem.lower() else "8B" if "8b" in stem.lower() else "unknown"
  mode = "Q4K_PRIMITIVE=1" if "primitive" in stem.lower() or "q4k" in stem.lower() else "baseline"
  if "baseline" in stem.lower(): mode = "baseline"
  if "batched" in stem.lower(): mode += " batched"
  if "jitbs1" in stem.lower(): mode += " named"
  return model, mode

def parse_log(path:pathlib.Path) -> list[Token]:
  tokens: list[Token] = []
  pending: list[Kernel] = []
  for line in path.read_text(errors="replace").splitlines():
    if (m:=AMD_RE.match(line)):
      pending.append(Kernel(m.group("name").strip(), _time_to_ms(float(m.group("tm")), m.group("unit"))))
      continue
    if (m:=TOKEN_RE.match(line)):
      tokens.append(Token(float(m.group("ms")), float(m.group("tps")), float(m.group("gbs")), pending))
      pending = []
  return tokens

def _is_attention_kernel(name:str) -> bool:
  # Decode attention kernels carry start_pos/toks dimensions and softmax-ish small reductions.
  return "start_pos" in name or "toks" in name

def _is_norm_sampling_kernel(name:str) -> bool:
  if name.startswith("E_"): return True
  if re.search(r"^r_32_4_\d+", name) is not None: return True
  return name in ("r_16_8", "r_16_8n1", "r_16_256", "r_16_256n1", "r_16_256n2", "r_1024_16_4_2_32")

def _is_fallback_q4k_kernel(name:str) -> bool:
  if not name.startswith("r_"): return False
  # Baseline fused Q4_K matvec/dequant kernels are large anonymous reductions.
  # These patterns intentionally favor the known dense decode matvec shapes over
  # attention kernels, which are handled before this bucket.
  dense_patterns = (
    r"^r_128_32_3_16_4_2_32",
    r"^r_\d+_32_4_\d+_(2_2_2_32|4_2_32)",
    r"^r_\d+_16_4_2_32",
    r"^r_\d+_256_16_3_16_4_2_32",
    r"^r_\d+_64_16_4_48_2_2_2_32",
    r"^r_1187_32_4_\d+_2_2_2_32",
  )
  return any(re.search(pat, name) is not None for pat in dense_patterns)

def classify_token(kernels:list[Kernel]) -> list[tuple[Kernel, str]]:
  out: list[tuple[Kernel, str]] = []
  primitive_reduce_followups = 0
  for k in kernels:
    name = k.name
    bucket = "other_amd"
    if name.startswith("q4k_gemv_partial_"):
      bucket = "q4k_primitive_gemv"
      parts = name.rsplit("_", 1)[-1]
      primitive_reduce_followups = max(primitive_reduce_followups, 3 if parts != "1" else 0)
    elif primitive_reduce_followups and name.startswith("r_"):
      bucket = "q4k_primitive_reduction"
      primitive_reduce_followups -= 1
    elif name.startswith("copy"):
      bucket = "copy"
    elif _is_fallback_q4k_kernel(name):
      bucket = "fallback_q4k_fused"
    elif _is_attention_kernel(name):
      bucket = "attention_misc"
    elif _is_norm_sampling_kernel(name):
      bucket = "norm_sampling_misc"
    elif name.startswith("E_"):
      bucket = "norm_sampling_misc"
    out.append((k, bucket))
  return out

def summarize(tokens:list[Token], steady_drop:int) -> dict:
  used = tokens[steady_drop:] if len(tokens) > steady_drop else tokens
  walls = [t.wall_ms for t in used]
  tok_s = [t.tok_s for t in used]
  bucket_ms: Counter[str] = Counter()
  top: Counter[str] = Counter()
  kernel_total = 0.0
  for token in used:
    for kernel, bucket in classify_token(token.kernels):
      bucket_ms[bucket] += kernel.ms
      top[kernel.name] += kernel.ms
      kernel_total += kernel.ms
  wall_total = sum(walls)
  residual = max(0.0, wall_total - kernel_total)
  bucket_ms["residual_overhead"] += residual
  median_ms = statistics.median(walls) if walls else 0.0
  outliers = [x for x in walls if median_ms and x > 1.5 * median_ms]
  count = len(used)
  return {
    "samples": count,
    "wall_ms_tok": statistics.mean(walls) if walls else 0.0,
    "median_ms_tok": median_ms,
    "tok_s": statistics.mean(tok_s) if tok_s else 0.0,
    "amd_kernel_ms_tok": kernel_total / count if count else 0.0,
    "residual_ms_tok": residual / count if count else 0.0,
    "residual_pct": (100.0 * residual / wall_total) if wall_total else 0.0,
    "outlier_count": len(outliers),
    "outlier_max_ms": max(outliers) if outliers else 0.0,
    "bucket_ms_tok": {b: bucket_ms[b] / count if count else 0.0 for b in BUCKETS},
    "bucket_pct_wall": {b: (100.0 * bucket_ms[b] / wall_total) if wall_total else 0.0 for b in BUCKETS},
    "bucket_pct_amd": {b: (100.0 * bucket_ms[b] / kernel_total) if kernel_total and b != "residual_overhead" else 0.0 for b in BUCKETS},
    "top_kernels": top.most_common(20),
  }

def fmt(x:float) -> str:
  if math.isnan(x) or math.isinf(x): return "nan"
  return f"{x:.2f}"

def md_table(headers:list[str], rows:list[list[str]]) -> str:
  return "\n".join(["| " + " | ".join(headers) + " |",
                    "| " + " | ".join("---" for _ in headers) + " |"] +
                   ["| " + " | ".join(row) + " |" for row in rows])

def make_report(results:list[dict], steady_drop:int) -> str:
  lines = [
    "# Q4_K Residual Decode Profile",
    "",
    f"Steady-state rows drop the first {steady_drop} benchmark token(s). `batched` logs use normal graph batching and are the real runtime profile. `named` logs set `JIT_BATCH_SIZE=1`; they keep the rollout JIT but avoid graph batching so DEBUG=2 exposes kernel names for attribution.",
    "",
    "## Summary",
    "",
  ]
  rows = []
  for r in results:
    s = r["summary"]
    rows.append([r["model"], r["mode"], str(s["samples"]), fmt(s["tok_s"]), fmt(s["wall_ms_tok"]),
                 fmt(s["amd_kernel_ms_tok"]), fmt(s["residual_ms_tok"]), fmt(s["residual_pct"])])
  lines.append(md_table(["model", "mode", "samples", "tok/s", "wall ms/tok", "AMD kernel ms/tok", "residual ms/tok", "residual %"], rows))
  lines += ["", "## Buckets", ""]
  rows = []
  for r in results:
    s = r["summary"]
    for bucket in BUCKETS:
      rows.append([r["model"], r["mode"], bucket, fmt(s["bucket_ms_tok"][bucket]), fmt(s["bucket_pct_wall"][bucket]),
                   fmt(s["bucket_pct_amd"][bucket]),
                   ", ".join(name for name,_ in [x for x in s["top_kernels"] if classify_token([Kernel(x[0], x[1])])[0][1] == bucket][:3])])
  lines.append(md_table(["model", "mode", "bucket", "ms/tok", "% wall", "% AMD kernel", "top kernels"], rows))
  lines += ["", "## Outliers", ""]
  rows = []
  for r in results:
    s = r["summary"]
    rows.append([r["model"], r["mode"], str(s["outlier_count"]), fmt(s["median_ms_tok"]), fmt(s["outlier_max_ms"])])
  lines.append(md_table(["model", "mode", "outliers", "median ms", "max outlier ms"], rows))
  lines += ["", "## Top Kernels", ""]
  for r in results:
    lines += [f"### {r['model']} {r['mode']}", ""]
    rows = [[name, fmt(ms / r["summary"]["samples"] if r["summary"]["samples"] else 0.0), fmt(ms)] for name, ms in r["summary"]["top_kernels"]]
    lines.append(md_table(["kernel", "ms/tok", "total ms"], rows))
    lines.append("")
  lines += ["## Decision Gates", ""]
  for r in results:
    s = r["summary"]
    is_named = "named" in r["mode"]
    is_batched = "batched" in r["mode"]
    pct = s["bucket_pct_amd"] if is_named else s["bucket_pct_wall"]
    decisions = []
    if is_batched:
      if pct["residual_overhead"] > 20 or s["outlier_count"] > 0:
        decisions.append("real graph-batched runtime has residual/outliers: inspect runtime/allocator/dispatch/memory pressure")
      else:
        decisions.append("real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership")
    else:
      if pct["q4k_primitive_gemv"] > 50: decisions.append("primitive GEMV >50% of profile basis: build primitive v2")
      if pct["q4k_primitive_reduction"] > 15: decisions.append("primitive reductions >15% of profile basis: fuse/avoid partial reduction")
      if pct["fallback_q4k_fused"] > 20: decisions.append("fallback/generic Q4_K remains large: extend primitive coverage or revise policy")
      non_q4 = pct["attention_misc"] + pct["norm_sampling_misc"] + pct["other_amd"] + pct["copy"]
      if non_q4 > 35: decisions.append("non-Q4 kernels >35% of profile basis: pivot to the new dominant bucket")
      if s["residual_pct"] > 20:
        decisions.append("named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions")
    if not decisions: decisions.append("no single pre-set gate fired: use top-kernel contributors")
    basis = "AMD-kernel basis" if is_named else "wall-time basis"
    lines += [f"- **{r['model']} {r['mode']}** ({basis}): " + "; ".join(decisions)]
  lines.append("")
  return "\n".join(lines)

def main():
  parser = argparse.ArgumentParser(description="Parse tinygrad DEBUG=2 LLM decode logs into residual Q4_K profile buckets")
  parser.add_argument("logs", nargs="+", type=pathlib.Path)
  parser.add_argument("--out", type=pathlib.Path)
  parser.add_argument("--json", type=pathlib.Path)
  parser.add_argument("--steady-drop", type=int, default=1)
  args = parser.parse_args()

  results = []
  for path in args.logs:
    model, mode = _label(path)
    tokens = parse_log(path)
    results.append({"path": str(path), "model": model, "mode": mode, "tokens": len(tokens),
                    "summary": summarize(tokens, args.steady_drop)})

  report = make_report(results, args.steady_drop)
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
  else:
    print(report)
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
  main()
