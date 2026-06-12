#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, pathlib, re, statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

TOKEN_RE = re.compile(r"^\s*(?P<ms>[0-9.]+) ms,\s+(?P<tps>[0-9.]+) tok/s,\s+(?P<gbs>[0-9.]+) GB/s,")
AMD_RE = re.compile(r"^\*\*\* AMD\s+\d+\s+(?P<name>.*?)\s+arg\s+\d+\s+mem\s+.*?\btm\s+(?P<tm>[0-9.]+)(?P<unit>us|ms|s)/")
MODEL_RE = re.compile(r"(?:^|[-_])(8b|14b)(?:[-_]|$)", re.IGNORECASE)

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

FALLBACK_Q4K_PATTERNS = (
  # Baseline fused Q4_K matvec/dequant kernels are large anonymous reductions.
  # These patterns intentionally describe the known dense decode matvec shapes;
  # attention and norm/sampling signatures are kept disjoint below.
  r"^r_128_32_3_16_4_2_32",
  r"^r_\d+_32_4_\d+_(2_2_2_32|4_2_32)",
  r"^r_\d+_16_4_2_32",
  r"^r_\d+_256_16_3_16_4_2_32",
  r"^r_\d+_64_16_4_48_2_2_2_32",
  r"^r_1187_32_4_\d+_2_2_2_32",
)
NORM_SAMPLING_EXACT = ("r_16_8", "r_16_8n1", "r_16_256", "r_16_256n1", "r_16_256n2")
MODEL_ORDER = {"8B": 0, "14B": 1}
MODE_ORDER = {"baseline batched": 0, "Q4K_PRIMITIVE=1 batched": 1, "baseline named": 2, "Q4K_PRIMITIVE=1 named": 3}

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

@dataclass(frozen=True)
class ParseStats:
  lines: int = 0
  amd_lines: int = 0
  token_lines: int = 0
  ignored_lines: int = 0
  non_amd_debug_lines: int = 0
  trailing_amd_lines: int = 0

@dataclass
class ParsedLog:
  tokens: list[Token]
  stats: ParseStats

@dataclass(frozen=True)
class KernelRule:
  bucket: str
  name: str
  predicate: Callable[[str], bool]

def _time_to_ms(value:float, unit:str) -> float:
  if unit == "us": return value / 1000.0
  if unit == "ms": return value
  if unit == "s": return value * 1000.0
  raise ValueError(f"unknown unit {unit}")

def _label(path:pathlib.Path) -> tuple[str, str]:
  stem = path.name.lower()
  model_match = MODEL_RE.search(stem)
  if model_match is None: raise ValueError(f"{path}: filename must include 8b or 14b")
  model = model_match.group(1).upper()
  is_baseline = "baseline" in stem
  is_primitive = "primitive" in stem
  if is_baseline == is_primitive:
    raise ValueError(f"{path}: filename must include exactly one of baseline or primitive")
  is_batched = "batched" in stem
  is_named = "jitbs1" in stem or "named" in stem
  if is_batched == is_named:
    raise ValueError(f"{path}: filename must include exactly one of batched or jitbs1/named")
  mode = "Q4K_PRIMITIVE=1" if is_primitive else "baseline"
  mode += " named" if is_named else " batched"
  return model, mode

def parse_log(path:pathlib.Path) -> ParsedLog:
  tokens: list[Token] = []
  pending: list[Kernel] = []
  lines = amd_lines = token_lines = ignored_lines = non_amd_debug_lines = 0
  try:
    text = path.read_text()
  except UnicodeDecodeError as e:
    raise ValueError(f"{path}: invalid UTF-8 in profile log") from e
  for lineno, line in enumerate(text.splitlines(), 1):
    lines += 1
    if (m:=AMD_RE.match(line)):
      pending.append(Kernel(m.group("name").strip(), _time_to_ms(float(m.group("tm")), m.group("unit"))))
      amd_lines += 1
      continue
    if line.startswith("*** AMD"):
      raise ValueError(f"{path}:{lineno}: malformed AMD DEBUG line: {line[:160]}")
    if line.startswith("*** "):
      non_amd_debug_lines += 1
      continue
    if (m:=TOKEN_RE.match(line)):
      tokens.append(Token(float(m.group("ms")), float(m.group("tps")), float(m.group("gbs")), pending))
      pending = []
      token_lines += 1
      continue
    if "tok/s" in line and "GB/s" in line:
      raise ValueError(f"{path}:{lineno}: malformed token summary line: {line[:160]}")
    ignored_lines += 1
  if not tokens:
    raise ValueError(f"{path}: parsed zero token summaries from {lines} lines")
  if amd_lines == 0:
    raise ValueError(f"{path}: parsed zero AMD DEBUG lines; expected DEBUG=2 log")
  return ParsedLog(tokens, ParseStats(lines, amd_lines, token_lines, ignored_lines, non_amd_debug_lines, len(pending)))

def _is_attention_kernel(name:str) -> bool:
  # Decode attention kernels carry start_pos/toks dimensions and softmax-ish small reductions.
  return "start_pos" in name or "toks" in name

def _is_norm_sampling_kernel(name:str) -> bool:
  if "start_pos" in name or "toks" in name: return False
  if name.startswith("E_"): return True
  if re.search(r"^r_32_4_\d+", name) is not None: return True
  return name in NORM_SAMPLING_EXACT

def _is_fallback_q4k_kernel(name:str) -> bool:
  if not name.startswith("r_"): return False
  return any(re.search(pat, name) is not None for pat in FALLBACK_Q4K_PATTERNS)

KERNEL_RULES = (
  KernelRule("copy", "copy kernels", lambda name: name.startswith("copy")),
  KernelRule("fallback_q4k_fused", "known fused Q4_K decode matvec signatures", _is_fallback_q4k_kernel),
  KernelRule("attention_misc", "decode attention signatures", _is_attention_kernel),
  KernelRule("norm_sampling_misc", "norm/sampling signatures", _is_norm_sampling_kernel),
)

def _bucket_for_kernel(name:str) -> str:
  matches = [rule for rule in KERNEL_RULES if rule.predicate(name)]
  if len(matches) > 1:
    detail = ", ".join(f"{rule.bucket}:{rule.name}" for rule in matches)
    raise ValueError(f"ambiguous kernel bucket for {name!r}: {detail}")
  return matches[0].bucket if matches else "other_amd"

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
    else:
      bucket = _bucket_for_kernel(name)
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

def result_sort_key(result:dict) -> tuple[int, int, str]:
  return (MODEL_ORDER[result["model"]], MODE_ORDER[result["mode"]], result["path"])

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
  lines += ["", "## Parse Health", ""]
  rows = []
  for r in results:
    st = r["parse_stats"]
    rows.append([r["model"], r["mode"], str(st["lines"]), str(st["tokens"]), str(st["amd_lines"]),
                 str(st["ignored_lines"]), str(st["non_amd_debug_lines"]), str(st["trailing_amd_lines"])])
  lines.append(md_table(["model", "mode", "lines", "tokens", "AMD lines", "ignored lines", "non-AMD DEBUG lines", "trailing AMD lines"], rows))
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
    parsed = parse_log(path)
    results.append({"path": str(path), "model": model, "mode": mode, "tokens": len(parsed.tokens),
                    "parse_stats": {**parsed.stats.__dict__, "tokens": len(parsed.tokens)},
                    "summary": summarize(parsed.tokens, args.steady_drop)})
  results.sort(key=result_sort_key)

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
