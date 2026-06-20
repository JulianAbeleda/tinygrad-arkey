# Prefill Graph GEMM OOM + Policy Audit Result (Gate 4) - 2026-06-20

Verdict: `PASS_PREFILL_GRAPH_GEMM_OOM_ENGINEERING_READY_POLICY_PENDING`

Run:

```bash
DEV=AMD PYTHONPATH=. python3 extra/qk_prefill_graph_gemm_oom_policy_audit.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
```

## OOM checks

| check | expected | result |
|---|---|---|
| `PREFILL_V2=1` model load + forward succeeds | pass | ✓ |
| `PREFILL_V2=1 PREFILL_GRAPH_GEMM=1` model load + forward succeeds | pass | ✓ |
| graph route run has no additional OOM beyond baseline | pass | ✓ |
| route kernel cache realizes no extra model-sized buffer | pass | ✓ |
| unsupported fallback allocates no output before returning `None` | pass | ✓ |

Notes:
- Each load runs a real forward in a fresh subprocess, so both the load and run paths are exercised; no
  `MemoryError`/allocation failure on either side.
- `route._kernel()` returns only an instruction list + ints (no `Tensor`); the route's sole allocation is the
  per-call output `Tensor.empty(512, out_f)` = 512×out_f×2 bytes (≤ ~12.6 MB for gate/up), bounded far below
  model size (~4.5 GB). No model-sized buffer is added by enabling the flag.
- Unsupported-shape `None`-returns all precede the first output allocation in `route_pf16_graph_gemm` (also
  confirmed in Gate 3), so a fallback costs no VRAM.
- The previously-observed OOM was the **full-window (512×vocab) NLL quality harness**, not the route; the
  route itself introduces no new memory failure class. The VRAM-safe quality gate (sampled/chunked NLL) is the
  correct promotion check and is used elsewhere.

## Policy

| question | decision |
|---|---|
| Is `max_abs_dNLL > 0.01` a blocker? | **policy** (report-only; worst row is favorable, `dNLL = -0.0176`) |
| Is `max_positive_dNLL ≤ 0.01` + generation exactness sufficient? | **policy** |
| Restrict default-on to gfx1100 + Qwen3-8B-like dense shapes first? | **yes** |
| **policy field** | **`DEFAULT_ON_POLICY_PENDING`** |

Engineering OOM gate passes; the absolute-parity (`max_abs_dNLL = 0.017593`) acceptance is a policy decision,
carried as report-only. Default-on remains pending that policy call.
