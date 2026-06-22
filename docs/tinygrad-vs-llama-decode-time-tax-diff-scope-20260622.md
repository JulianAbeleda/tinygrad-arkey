# Tinygrad-vs-Llama Decode Time-Tax Diff — Scope (2026-06-22)

**Status:** scope / methodology. **Authority:** none yet (this doc defines the method;
numbers land in the `-result-` doc). **Default behavior changed:** no (audit/tooling/docs only).

## Goal
Build an apples-to-apples tinygrad-vs-llama **per-primitive decode time-tax diff** for
Qwen3-8B-Q4_K_M on gfx1100, so the remaining decode gap after `Q4K_GEMV_WARP` is explained
per bucket and the next primitive is ranked by **`gap_ms`**, not by tinygrad-internal share.

Tinygrad-internal audits already name the FFN Q4_K GEMV as tinygrad's largest *share*
(`docs/decode-time-tax-audit-result-20260622.md`). But share ≠ gap: a bucket can dominate
tinygrad's own time yet already be at llama parity. This diff measures the **gap** vs the
llama oracle, bucket by bucket.

## Data sources

| source | role | ctx | provenance | per-kernel durations |
|---|---|---|---|---|
| `bench/qk-decode-time-tax-audit/latest.json` (→ copied to `bench/qk-tinygrad-vs-llama-time-tax/tinygrad_default.json`) | tinygrad **default-route** per-bucket | 512/1024/2048/4096 | this repo HEAD, `qk_decode_time_tax_audit.py`, gfx1100 | yes (ProfileGraphEvent GPU-busy, median-of-5) + wall token_ms (median-of-40, `.item()`) |
| `bench/qk-tinygrad-vs-llama-time-tax/tinygrad_warp.json` | tinygrad **`Q4K_GEMV_WARP=1 _DOWN=1`** per-bucket | 512/1024/2048/4096 | same tool, warp flags (L5) | same |
| `bench/qk-llama-decode-primitive-audit/decode_kernel_trace.json` | llama **per-family** µs/token + `decode_ms_per_tok` + `llama_tok_s_gpu` | 512/1024/4096 | llama.cpp `ac4cddeb` b9592, rocprofv3 `--kernel-trace`, llama-bench `-p 0 -n 32 -d <ctx> -r 1`, gfx1100 | yes (family sums; already normalized so families sum to `decode_ms_per_tok`) |
| `bench/qk-llama-decode-primitive-audit/llama_decode_kernel_trace_ctx1024.csv` (10.9 MB) | llama **per-dispatch raw trace** → per-ROLE reconstruction | 1024 only | same | yes (per-dispatch Start/End timestamps + `Grid_Size_X`) |
| `bench/qk-tinygrad-vs-llama-time-tax/llama_capture/*.csv` (L4, new) | llama per-dispatch raw → per-ROLE | 512/2048/4096 | fresh rocprofv3 capture, same build/model/flags | yes (if capture succeeds) |
| `bench/llama-kernel-residual-primitive-audit-20260619/rocprof_decode_d0/trace_kernel_stats.csv` | llama d0 per-kernel type/share (Q4_K type 12 vs Q6_K type 14) cross-reference | d0 | same | yes (TotalDurationNs, Calls, Pct) |

llama clean tok/s (llama-bench, no rocprof overhead): d512 = 97.71, d1024 = 97.39,
d4096 = 92.37 tok/s.

## Required buckets (both sides mapped to these 9)
FFN gate/up · FFN down · FFN activation · attention qk/softmax/pv ·
attention q/o/k/v projections · norm/rope/small ops · lm_head · graph/runtime/host ·
unknown/unmapped.

## Mapping rules

### Tinygrad kernel → bucket (reuses `classify()` in `extra/qk_decode_time_tax_audit.py:20-31`)
The audit already buckets by dim-signature substrings (Qwen3-8B: hidden 4096, FFN 12288,
vocab 151936). The diff regroups its 10 buckets into the 9 required ones:

| audit bucket | dim signature | → required bucket |
|---|---|---|
| `ffn_gate_up` | `12288_4096` (incl. `q4k_gemv_warp_12288_4096`) | FFN gate/up |
| `ffn_down` | `4096_12288` (incl. `q4k_gemv_warp_4096_12288`) | FFN down |
| `ffn_activation` | `E_49152` / `E_1536` | FFN activation |
| `attention_compute` | `flash_*` / `start_pos` | attention qk/softmax/pv |
| `attn_qo_proj` + `attn_kv_proj` | `4096_4096`, `1024_4096` | attention q/o/k/v projections |
| `norm_rope_small_ops` + `q8_route` | `E_*` / `r_*`, `q8` | norm/rope/small ops |
| `lm_head` | `151936` | lm_head |
| `host_graph_overhead_ms` (field) | wall − gpu_busy | graph/runtime/host |
| `unknown` | else | unknown/unmapped |

The warp kernel keeps its dim signature (`q4k_gemv_warp_<out>_<in>`), so the warp-on run
buckets identically — verified before the L5 run.

### Llama kernel → bucket
Family-level (from `decode_kernel_trace.json`), all **HIGH** confidence:

| llama family | → required bucket |
|---|---|
| `ffn_silu` | FFN activation |
| `attention_tile` + `attention_combine` + `attention_streamk_fixup` | attention qk/softmax/pv |
| `rmsnorm` + `rope` + `q8_1_activation_quant` + `copy_cast_kv` + `residual_add` | norm/rope/small ops |
| `other` | unknown/unmapped |
| `mmvq_weight_gemv` | **split per-role** (below) |

Per-role split of `mmvq_weight_gemv` (the one combined weight-GEMV family) from the raw
per-dispatch CSV, by `(ggml_type, Grid_Size_X)`. Grid = `out_features × 32`:

| `Grid_Size_X` | out_features | tensors | quant | → required bucket | confidence |
|---|---|---|---|---|---|
| 393216 | 12288 | gate, up | Q4_K (12) | FFN gate/up | HIGH |
| 4861952 | 151936 | output | Q6_K (14) | lm_head | HIGH |
| 131072 | 4096 | q, o, **down** | Q4_K + Q6_K(down layers) | FFN down **and** attn proj | MEDIUM |
| 32768 | 1024 | k, v | Q4_K + Q6_K(v layers) | attention q/o/k/v projections | HIGH |

The `out=4096` group (grid 131072) conflates `q_proj`, `o_proj`, and `ffn_down`. Split rule:
the **Q6_K** dispatches at grid 131072 are `ffn_down` (only out-4096 tensor bumped to Q6_K in
Q4_K_M); the **Q4_K** dispatches at grid 131072 are `q + o + (Q4_K-down layers)`, split by
per-layer call-count (equal grid ⇒ equal per-call time) into `ffn_down` (Q4_K-down layers) and
`attn proj` (q + o). Labeled **MEDIUM** confidence; the result doc states the assumption.

Cross-check: the reconstructed per-role weight-GEMV buckets must sum to the family
`mmvq_weight_gemv` µs/token at the same ctx (validates the split before trusting it). The
diff scales the per-role split to that family total, so the reconciliation holds by
construction at ctx1024 (and at 512/2048/4096 if L4 captures the family totals too).

## Normalization (units → ms/token)
Both sides are reduced to ms/token. Two views are emitted:
- **Raw GPU-time view**: per-bucket GPU-time/token as measured. Tinygrad's per-dispatch
  GPU-busy *sum* exceeds its wall token_ms (HCQ graph overlap is not removed by the audit);
  llama's family sum already equals its `decode_ms_per_tok` (serial stream, ~no overlap).
- **Wall-normalized view (headline for `gap_ms`)**: each side's buckets are scaled so they
  sum to that side's measured wall token_ms (tinygrad × `token_ms/gpu_busy`; llama × 1.0).
  This makes per-bucket `gap_ms` sum to the real wall token_ms gap. **Assumption:** overlap
  is distributed uniformly across tinygrad buckets — flagged as an assumption; the raw view
  is provided alongside so the reader can see the un-normalized picture.

Per bucket the diff reports: `tinygrad_ms, llama_ms, gap_ms (=tg−llama), ratio (=tg/llama),
tinygrad_share, llama_share, confidence, notes` — ranked by `gap_ms`.

## Limitations
1. **llama per-role split is full only where a per-dispatch raw CSV exists** — ctx1024 today,
   ctx 512/2048/4096 only if L4 capture succeeds. Without it, those ctx keep one collapsed
   "weight-GEMV (all roles)" row (HIGH-but-coarse) and FFN-gate/up/down/proj/lm_head are not
   individually resolved there.
2. **ctx2048 has no llama trace at all** (family or raw) until L4 captures it.
3. **down-vs-q/o ambiguity** within the out-4096 group → MEDIUM confidence (see split rule).
4. **Overlap not removed** on the tinygrad side; the wall-normalized headline assumes uniform
   distribution (raw view provided).
5. **rocprofv3 is blind to tinygrad's HCQ queue** (`docs/tinygrad-hcq-profiling-visibility-result-20260621.md`),
   so each side uses its own profiler (llama: rocprofv3 HW timestamps; tinygrad:
   ProfileGraphEvent). Both are HW-timestamp GPU time, but cross-stack absolute comparison
   carries this caveat.
6. llama family numbers at ctx512/4096 attention are measured; the older oracle's ctx512/4096
   attention constants were *derived* — this diff uses the measured family JSON, not those
   constants.

## Expected artifacts
- `bench/qk-tinygrad-vs-llama-time-tax/latest.json` — the diff (both views, ranked table, per-ctx).
- `bench/qk-tinygrad-vs-llama-time-tax/{tinygrad_default,tinygrad_warp}.json` — tinygrad inputs.
- `bench/qk-tinygrad-vs-llama-time-tax/llama_capture/*` — L4 raw CSVs + derived per-role JSON (if captured).
- `docs/tinygrad-vs-llama-decode-time-tax-diff-result-20260622.md` — result doc + README bullet.

## Stop conditions
- **L4 capture instability** (rocprofv3/llama-bench crashes, inconsistent with the ctx1024
  oracle's family totals beyond a stated tolerance, or the capture balloons broad): STOP the
  capture, keep the reuse-only diff (per-role @ctx1024, collapsed weight-GEMV @512/4096, no
  2048), and record verdict `LLAMA_TRACE_INSUFFICIENT_CAPTURE_SCOPE_READY` with the exact
  missing fields + a tightened capture scope.
- **Hard env block** (GPU/driver/profiler unavailable): `LLAMA_DIFF_BLOCKED_BY_ENV`.
- **Trace exists but unusable** (corrupt timestamps, unparseable): `LLAMA_DIFF_BLOCKED_BY_TRACE_QUALITY`.
- Otherwise, with the diff built: `LLAMA_DIFF_AUDIT_READY`.
