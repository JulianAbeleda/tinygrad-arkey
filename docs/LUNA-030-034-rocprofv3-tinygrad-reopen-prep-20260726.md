# LUNA-030--034 rocprofv3 / tinygrad reopen preparation

Date: 2026-07-26

Status: CPU/static preparation only. No profile in this document is evidence of
tinygrad route attribution. Do not run a GPU command from this worktree until
the LUNA-030 llama ctx128 gate has passed.

## Prior constraint and verdict vocabulary

tinygrad's AMD runtime submits AQL directly through `/dev/kfd`. The prior
`DEBUG=2` investigation recorded that `rocprofv3` cannot observe those
submissions because it hooks the HSA/HIP path. Consequently, `rocprofv3` is
allowed to establish its own positive control with llama, but an empty tinygrad
trace is **`TOOL_FAILURE`**, not an absent-kernel, route, performance, or
correctness result. A populated tinygrad trace is only `TRACE_OBSERVED` until
the process, command, and in-worker positive control are preserved.

Use only these verdicts for LUNA-030--034: `PASS`, `NOT_RUN`, `TOOL_FAILURE`,
`GPU_FAULT`, `INCONCLUSIVE`, and `TRACE_OBSERVED`. None authorizes a route
claim. In particular, do not infer tinygrad route identity from a profiler
kernel name or from a matching duration.

## Shared ownership and isolation contract

* The outer `flock -n /tmp/gpu-bench.lock` owns the GPU for the complete
  `rocprofv3` child lifecycle, including process startup and artifact writing.
  Do not acquire the same lock inside a worker.
* One matrix row is one fresh shell process and one fresh profiled child. Do
  not combine contexts, models, or llama and tinygrad in one profiler session.
* Tinygrad model harnesses may create their own `run_isolated` subprocesses.
  Route and UOp logging must be installed in that worker, not in the parent;
  parent monkeypatches are known to observe zero kernels.
* Preserve `power_dpm_force_performance_level=auto` before and after every
  actual GPU row. A non-`auto` final state stops the row.
* Each output directory is created once, is never reused, and contains
  `command.txt`, `stdout.log`, `stderr.log`, `meta.json`, profiler output, and
  the worker's route/UOp log. Record worktree path, branch, commit, PID, boot
  ID, argv, model SHA/path, and profiler version in `meta.json`.

The commands below intentionally use variables. Resolve them in LUNA-030 and
write their exact expanded argv into `command.txt`; do not silently substitute
a different llama binary, model, or tinygrad harness.

```bash
WT=/home/ubuntu/worktrees/luna-tinygrad-profiler-prep
ROOT="$WT/bench/luna-rocprofv3/20260726"
LOCK=/tmp/gpu-bench.lock
ROCPROF=rocprofv3
LLAMA_BENCH=/absolute/path/to/llama-bench
LLAMA_8B=/absolute/path/to/8b.gguf
LLAMA_14B=/absolute/path/to/14b.gguf
TG_RUNNER=/absolute/path/to/the-LUNA-tinygrad-fixed-depth-runner
TG_8B=/absolute/path/to/8b.gguf
TG_14B=/absolute/path/to/14b.gguf
```

## Exact rocprofv3 command matrix

The command shape is fixed: `rocprofv3 --kernel-trace --output-directory
"$OUT/rocprof" -- <child>`. LUNA-030 must first verify this installed version's
`--help` spelling without a GPU workload and record it. If the installed
rocprofv3 rejects this argv, record `TOOL_FAILURE`; do not fall back to
`rocprof` or alter flags while calling the row comparable.

All model rows use a fresh `$OUT` and the outer lock:

```bash
mkdir -p "$OUT"
printf '%q ' "$ROCPROF" --kernel-trace --output-directory "$OUT/rocprof" -- "$@" >"$OUT/command.txt"
printf '\n' >>"$OUT/command.txt"
flock -n "$LOCK" sh -c 'exec "$@"' sh \
  "$ROCPROF" --kernel-trace --output-directory "$OUT/rocprof" -- "$@" \
  >"$OUT/stdout.log" 2>"$OUT/stderr.log"
```

| Task | Child | ctx | Output directory | Required positive control | Expected disposition |
|---|---|---:|---|---|---|
| LUNA-030 | llama 8B | 128 | `$ROOT/LUNA-030/llama-8b-ctx128` | At least one bounded known llama decode kernel in `rocprof/` | `PASS` only with nonempty trace and match |
| LUNA-030 | llama 14B | 128 | `$ROOT/LUNA-030/llama-14b-ctx128` | Same, model-specific known llama kernel | `PASS` only with nonempty trace and match |
| LUNA-031 | tinygrad 8B | 128 | `$ROOT/LUNA-031/tinygrad-8b-ctx128` | Worker log matches both expected flash-tile and fused-combine identities | Normally `TOOL_FAILURE` if profiler trace is empty; never route proof |
| LUNA-031 | tinygrad 14B | 128 | `$ROOT/LUNA-031/tinygrad-14b-ctx128` | Same; expect G5 KV_BOTH tile identity after the route split lands | Same |
| LUNA-032 | tinygrad 8B | 512 | `$ROOT/LUNA-032/tinygrad-8b-ctx512` | Worker positive control plus finite/nonzero output | Same |
| LUNA-032 | tinygrad 14B | 512 | `$ROOT/LUNA-032/tinygrad-14b-ctx512` | Worker positive control plus finite/nonzero output | Same |
| LUNA-033 | tinygrad 8B | 4096 | `$ROOT/LUNA-033/tinygrad-8b-ctx4096` | Worker positive control plus finite/nonzero output | Same |
| LUNA-033 | tinygrad 14B | 4096 | `$ROOT/LUNA-033/tinygrad-14b-ctx4096` | Worker positive control plus finite/nonzero output | Same |
| LUNA-034 | tinygrad 8B then 14B | 128, 512, 4096 | `$ROOT/LUNA-034/<model>-ctx<ctx>` | Exact route decision record and UOp/program identity from the isolated worker | `INCONCLUSIVE` unless a later supported trace bridge binds dispatches to the worker record |

For llama use the project authority arguments, one child per depth:

```bash
"$LLAMA_BENCH" -m "$LLAMA_MODEL" -p 0 -n 128 -d "$CTX" -ngl 99 -fa 1 -r 3
```

For tinygrad, LUNA-031--034 must call the owned fixed-depth runner, not a
generic generation loop. The runner argv must include the GGUF path, exact
fixed context, an explicit decode token count, deterministic seed, and output
paths for `--route-log` and `--uop-log`; those flags are a required LUNA-034
harness addition if absent. Its actual command is:

```bash
"$TG_RUNNER" --model "$TG_MODEL" --fixed-context "$CTX" --decode-tokens 128 --seed 0 \
  --route-log "$OUT/worker-route.jsonl" --uop-log "$OUT/worker-uops.jsonl"
```

The supplied variables deliberately prevent inventing a runner path or flag
contract before LUNA-030 resolves the owned harness. Do not replace this with
`extra/llm/model_e2e_bench.py`: that harness is whole-model measurement, not
the fixed-depth decode authority.

## Kernel, route, and UOp requirements

The tinygrad positive control is not rocprof output. In the isolated worker,
record a JSONL row per realized linear with: model identity, ctx, phase,
selected decode candidate ID, geometry (`Hq`, `Hkv`, `Hd`, split, KV mode),
linear ordinal, generated program/source/code-object hashes, launch grid and
block, and all matched kernel names. Require exactly the expected flash-tile
and fused-combine identities for the relevant geometry; empty records, zero
matches, non-finite output, a reset/fault, or a missing UOp record is a stop.

For UOps, log a stable serialization/hash of the pre-render lowered UOp sink
and the rendered program identity, tagged with the same worker row. Do not use
`DEBUG` console text as the sole artifact: it is neither structured nor a
dispatch-to-route binding. The log establishes only a same-worker correlation
candidate, not attribution.

## Semantics-preserving logging seams

These are candidate locations for a later LUNA-034 patch. All callbacks must
be opt-in, context-local, and no-op when unset; they must not alter candidate
selection, graph construction, scheduling, compilation, or dispatch order.

| Location | What it can log | Limitation |
|---|---|---|
| `tinygrad/llm/decode_routes.py` at candidate bind/selection | admitted candidate ID and G4/G5 geometry before graph construction | Decision only; no realized-program identity |
| `tinygrad/llm/model.py` at the decode attention call boundary | model/phase/context tied to the selected decode route | Must execute in the isolated worker |
| `tinygrad/llm/prefill_route_observer.py` pattern | reusable `ContextVar` + context-manager design for opt-in observers | It covers prefill today, not decode; do not repurpose it to mislabel decode |
| `tinygrad/engine/realize.py` at linear realization | linear ordinal, UOp sink hash, runner/program identity and launch metadata | Generic logging must be filtered to the target worker/phase to avoid excessive artifacts |
| `tinygrad/uop/spec.py` / UOp rendering boundary | stable lowered-UOp text/hash when explicitly requested | Compiler representation evidence, not execution evidence |

The existing prefill observer calls in `tinygrad/llm/model.py` and
`tinygrad/llm/prefill_routes.py` demonstrate a semantics-neutral attachment
pattern. Decode must receive its own name and schema because prefill route
events cannot prove decode route choice.

## Handoff after llama ctx128 succeeds

1. Freeze LUNA-030 artifacts: retain exact profiler version/argv and the known
   llama kernel identity. Do not overwrite them.
2. Resolve and freeze the owned tinygrad fixed-depth runner path and its worker
   entrypoint. Add the opt-in worker route/UOp logger only if its no-op path is
   byte/behavior preserving by review; do not launch it yet.
3. Run one fresh locked tinygrad 8B ctx128 row (LUNA-031). Require the worker
   tile+combine positive controls and structured logs. Classify profiler
   emptiness as `TOOL_FAILURE`.
4. Run the matching fresh locked tinygrad 14B ctx128 row. Require the G5
   identity, not the stale G4 label. If the G4/G5 split is not landed, stop as
   `INCONCLUSIVE` rather than attributing it.
5. Only after both ctx128 worker records are complete, progress separately to
   ctx512 (LUNA-032) and ctx4096 (LUNA-033), then assemble the correlation-only
   LUNA-034 bundle. No row upgrades profiler correlation to route attribution.

## Blockers

* The direct-KFD submission path is a known rocprofv3 visibility blocker.
* The exact owned fixed-depth runner and its route/UOp flag contract must be
  resolved before any tinygrad command is executable.
* G5 must have an explicit runtime route identity before 14B evidence can be
  named honestly; the prior singleton G4 label admits G5 geometry.
* A supported dispatch trace bridge is required before profiler records can be
  bound to tinygrad worker route decisions.
