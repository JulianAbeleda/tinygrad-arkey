# 8B Prefill Generated Route Closeout - 2026-07-05

## Scope

This closes the 8B graph-GEMM prefill route, not the 14B/32B Q4_K direct-packed/MMQ work.

Solved means:

- 8B prefill runs through the canonical prefill authority harness.
- The runtime graph-GEMM prefill emitter is spec-generated only.
- The legacy fixed emit rollback selected by `PREFILL_GENERATED_SCHEDULE=0` is removed.
- Profile trace shows generated graph-GEMM kernels and no legacy graph-GEMM kernels.

## Code State

Runtime graph-GEMM prefill now has one emitter:

- `extra/qk/prefill_graph_gemm_route.py::route_pf16_graph_gemm`
- `extra/qk/prefill_schedule_spec.py::describe_prefill_schedule`
- `extra/qk/prefill_schedule_spec.py::emit_prefill_gemm_from_spec`

Removed/closed:

- `_kernel(out_f, in_f)` legacy fixed emit.
- Runtime branch on `PREFILL_GENERATED_SCHEDULE`.
- `prefill_pipe_role_selective_default` route-manifest entry.
- Pure-search rollback mapping from prefill generated route to the legacy oracle.
- Route-policy support for prefill generated-schedule params.

Generated route identity is now `prefill_gen_sched_gemm_*`.

## Host Gates

Command:

```bash
PYTHONPATH=. .venv/bin/python extra/qk/prefill_generated_schedule_gate.py
```

Result:

```text
verdict: TG_P4_PASS_PREFILL_GENERATED_SCHEDULE
all_generated_builds_present: true
role_policy_preserved: true
generated_names_preserved: true
no_legacy_rollback: true
```

Command:

```bash
PYTHONPATH=. .venv/bin/python extra/qk/pure_search_guard.py
```

Result:

```text
prefill_gemm -> prefill_pipe_role_selective_generated, pure
violations: []
```

## 8B Authority

Command:

```bash
PURE_MACHINE_SEARCH_ONLY=1 ALLOW_DEVICE_USAGE=1 .venv/bin/python extra/qk/bench.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --prefill
```

Result:

```text
PREFILL AUTHORITY (synced, K=8, warmups=4, rounds=3) model=Qwen3-8B-Q4_K_M.gguf GRAPH_GEMM=True
chunk@start_pos=0:    100.1ms (5117 tok/s)
chunk@start_pos=512:  108.1ms (4735 tok/s)
chunk@start_pos=1024: 118.5ms (4322 tok/s)
chunk@start_pos=2048: 150.9ms (3394 tok/s)
chunk@start_pos=3584: 174.4ms (2935 tok/s)
WHOLE-PREFILL@512: 5117 tok/s
WHOLE-PREFILL@1024: 4918 tok/s
WHOLE-PREFILL@2048: 4439 tok/s
WHOLE-PREFILL@4096: 3684 tok/s
```

The pure-search route report for that run selected:

```text
prefill_gemm -> prefill_pipe_role_selective_generated
```

## Profile Trace

Command:

```bash
PREFILL_CHUNKED=0 PURE_MACHINE_SEARCH_ONLY=1 PROFILE=1 ALLOW_DEVICE_USAGE=1 PYTHONPATH=. \
  .venv/bin/python extra/qk/prefill_boltbeam_trace.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --context 512 --mode profile \
  --out /tmp/8b-prefill-generated-trace-profile.json
```

Extract:

```text
rows 31
kernel_rows 30
profile_raw_total_us 114614.88000000002
profile_empty_chunks []
gen_sched 4
legacy_graph 0
direct_packed 0
prefill_gen_sched_gemm_512_12288_4096
prefill_gen_sched_gemm_512_4096_12288
prefill_gen_sched_gemm_512_4096_4096
prefill_gen_sched_gemm_512_1024_4096
```

## Remaining Work

The strict global default-purity report still has explicit transitional debt:

- `prefill_q4k_direct_tile4x4_default`

That is the 14B/32B memory-safe Q4_K direct-packed path. It is not part of this 8B graph-GEMM prefill closure. The next solve for that track is a generated quantized MMQ substrate that fuses dequant and matmul.
