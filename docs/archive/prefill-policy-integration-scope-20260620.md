# Prefill Policy Integration Scope

Date: 2026-06-20

Executor: Claude

## Objective

Turn the solved prefill fast paths into a practical policy:

1. enable `PREFILL_V2` automatically only when VRAM/load budget makes it safe;
2. enable `PREFILL_CONCRETE_KV` only for server/long-prompt regimes where precompile cost amortizes;
3. avoid routing real prompts through the slow 32-token symbolic fallback when a prefill-v2 route is available;
4. keep low-VRAM universal default safe.

This is **not** a kernel performance scope. Do not build new prefill kernels.

## Starting Evidence

Read first:

- `docs/prefill-default-policy-evaluation-result-20260620.md`
- `docs/prefill-increment0-shipped-result-20260620.md`
- `docs/prefill-concrete-kv-increment0-result-20260620.md`
- `docs/prefill-branch-b-tc-attention-result-20260620.md`
- `docs/prefill-flash-increment2-result-20260620.md`
- `tinygrad/llm/model.py` around:
  - `PREFILL_V2`;
  - `PREFILL_CONCRETE_KV`;
  - `PREFILL_TC_ATTN`;
  - `Transformer.from_gguf`;
  - `realize_prefill_v2_weights`;
  - `precompile_concrete_prefill_jits`;
  - `generate`.

Current policy facts:

| fact | implication |
|---|---|
| true default `PREFILL_V2=0` routes long prompts through many 32-token symbolic calls and is broken-slow | default is safe but not useful for real long prompts |
| `PREFILL_V2=1` costs about `+14GB` VRAM on Qwen3-8B because fp16 realized weights coexist with Q4 | safe on 24GB, likely OOM on 16GB |
| `PREFILL_CONCRETE_KV=1` gives best warm/server prefill but adds load-time precompile | server/long-prompt, not one-shot short-prompt default |
| Branch B concrete first chunk is default-on under `PREFILL_V2`/gfx1100 and is byte-identical | keep it |
| flash prefill v1 is correct but too slow | do not reopen |

## Desired Policy

The final policy should have three advertised modes:

| mode | intended user | behavior |
|---|---|---|
| universal default | any card, including 16GB | safe; may keep `PREFILL_V2=0` unless auto policy proves safe |
| 24GB recommended | RX 7900 XTX class | auto-enable or strongly recommend `PREFILL_V2=1` |
| server/long-prompt profile | repeated prompts / long prompts | `PREFILL_V2=1 PREFILL_CONCRETE_KV=1` with precompile amortized |

## Phase 1: VRAM-Aware `PREFILL_V2` Auto Policy

### Goal

Add a policy layer that can choose `PREFILL_V2` based on available VRAM and model requirements, without making
16GB cards OOM.

### Design Requirements

Implement a small decision function, not scattered env checks.

Suggested location:

- `tinygrad/llm/model.py`, or a new small helper if cleaner.

Suggested API:

```python
def prefill_v2_auto_enabled(model_path: pathlib.Path, max_context: int, gpu_vram_bytes: int | None) -> tuple[bool, str]:
  ...
```

or a similar internal helper.

Inputs to consider:

- explicit env override:
  - `PREFILL_V2=0` forces off;
  - `PREFILL_V2=1` forces on;
  - new `PREFILL_V2=auto` or `PREFILL_V2_AUTO=1` enables policy;
- detected total/free VRAM if cheaply available;
- estimated fp16 realized-weight cost;
- model size / Qwen3-8B known cost;
- safety margin for KV cache and decode buffers.

Initial conservative rule:

```text
enable PREFILL_V2 auto only when total VRAM >= 24GB and estimated headroom passes margin
otherwise keep off and print/recommend opt-in when appropriate
```

Do not try to perfectly solve all GPUs in the first pass. Conservative > clever.

### Artifacts

Create:

- `extra/qk_prefill_v2_auto_policy_probe.py`
- `bench/qk-prefill-policy-integration/prefill_v2_auto_policy.json`
- `docs/prefill-v2-auto-policy-result-20260620.md`

Probe should report:

| field |
|---|
| detected GPU name / VRAM |
| estimated Q4 model memory |
| estimated fp16 realized-weight memory |
| selected policy |
| reason string |
| env override behavior |
| whether model load succeeds |
| peak VRAM after load |

### Gate

- `PREFILL_V2=auto` or equivalent enables on 24GB host.
- Explicit `PREFILL_V2=0/1` still works.
- No behavior change when policy env is not enabled, unless owner explicitly approves defaulting to auto.
- OOM case is handled as a safe off or clean error, not a crash after half-loading if avoidable.

## Phase 2: Server / Long-Prompt `PREFILL_CONCRETE_KV` Policy

### Goal

Enable concrete-KV precompile only when it pays off.

### Policy Inputs

- explicit env:
  - `PREFILL_CONCRETE_KV=0/1`;
  - optional new `PREFILL_CONCRETE_KV=auto`;
- max context;
- expected prompt length if known;
- server mode flag;
- precompile count/cost estimate:

```text
ceil(max_context / PREFILL_UBATCH)
```

Suggested new mode flag:

```text
TINYGRAD_LLM_SERVER=1
```

or:

```text
PREFILL_SERVER_PROFILE=1
```

Policy recommendation:

| condition | decision |
|---|---|
| explicit `PREFILL_CONCRETE_KV=1` | on |
| server profile and `PREFILL_V2` active | on |
| prompt length / max_context above threshold and load-time precompile accepted | on |
| cold one-shot short prompt | off |

### Artifacts

Create:

- `extra/qk_prefill_concrete_kv_policy_probe.py`
- `bench/qk-prefill-policy-integration/concrete_kv_policy.json`
- `docs/prefill-concrete-kv-policy-result-20260620.md`

Probe matrix:

| mode | max_context | prompt_len | server flag | selected concrete_kv | load_s | first_prefill_s | warm_prefill_s | tok0 |
|---|---:|---:|---|---|---:|---:|---:|---:|

At minimum test prompt lengths:

- `512`;
- `1024`;
- `2048`;
- `4096` if time/VRAM allow.

### Gate

- Concrete-KV auto does not turn on for short one-shot case unless explicitly forced.
- Server mode turns it on and reproduces warm-prefill win.
- Same tok0 / correctness status as baseline for tested prompts.
- Load-time precompile cost is reported.

## Phase 3: Avoid 32-Token Symbolic Fallback Trap

### Goal

The policy result says true default and some prefix-cache/remainder cases route prompts through many 32-token symbolic
calls. That is the bad path. Fix routing so eligible prompt work uses prefill-v2 512 chunks whenever possible.

### Read First

- `tinygrad/llm/model.py`, `generate`;
- existing chunking logic around `PREFILL_UBATCH`;
- prefix-cache logic and `cache_start_pos` behavior;
- `extra/qk_prefill_concrete_kv_a2_verify.py`, which logs call schedules.

### Required Instrumentation

Create or extend:

- `extra/qk_prefill_route_schedule_probe.py`
- `bench/qk-prefill-policy-integration/route_schedule_probe.json`
- `docs/prefill-route-schedule-result-20260620.md`

For each mode/prompt, log:

- number of forward calls;
- call schedule:
  - `int512`;
  - `UOp512`;
  - `UOp32`;
  - remainder sizes;
- whether `PREFILL_V2` path engaged;
- whether `PREFILL_TC_ATTN` concrete path engaged;
- prefill wall time.

### Fix Candidate

Route as much prompt work as possible through:

```text
PREFILL_UBATCH = 512
```

instead of falling to `chunk_size=32`, when:

- `PREFILL_V2` is active;
- enough prompt tokens remain for a full or useful prefill chunk;
- cache/prefix state does not make it semantically unsafe.

For remainder `<512`:

- do not force a bad path blindly;
- measure whether a smaller prefill-v2 chunk is safe or whether current 32-token path is necessary.

### Gate

- For fresh 1024+ prompt with `PREFILL_V2=1`, schedule must be `int512 + UOp512` or `int512 + int512` under concrete-KV,
  not `32 x UOp32`.
- Prefix-cache scenario must avoid pathological `32 x UOp32` when a 512 prefill chunk is valid.
- Tok0 matches baseline.
- No decode regression.

## Phase 4: User-Facing CLI / Docs Policy

### Goal

Make the fast path discoverable and safe.

Update docs/CLI messaging:

- true default row is universal but slow;
- recommended 24GB row is `PREFILL_V2=1`;
- server/long-prompt row is `PREFILL_V2=1 PREFILL_CONCRETE_KV=1`;
- decode throughput is unaffected by prefill policy.

Potential CLI warning:

```text
Detected 24GB AMD GPU and long prompt. Set PREFILL_V2=1 for faster prefill, or PREFILL_V2=auto if enabled.
```

If auto policy is owner-approved:

```text
PREFILL_V2=auto selected on 24GB GPU; use PREFILL_V2=0 to disable.
```

Artifacts:

- update `bench/README.md`;
- update `docs/README.md`;
- create `docs/prefill-policy-integration-result-20260620.md`.

## Phase 5: Optional VRAM Reduction Design

Only scope/design unless explicitly asked to implement.

Goal:

- reduce `PREFILL_V2` +14GB VRAM cost so 16GB cards can use it.

Possible approaches:

1. per-layer fp16 realization / streaming realization;
2. lazy realize only covered layers;
3. discard Q4 source for prefill-covered tensors only if decode fallback/quality remains safe;
4. compress or reuse fp16 buffers.

Deliverable if scoped:

- `docs/prefill-v2-vram-reduction-scope-20260620.md`

Do not implement in this policy pass unless owner explicitly asks. This is higher risk.

## Final Report

Create:

- `docs/prefill-policy-integration-result-20260620.md`

It must include:

1. VRAM auto-policy decision and evidence.
2. Concrete-KV server/long-prompt policy decision and evidence.
3. Route-schedule probe results.
4. Any routing fix and before/after schedules.
5. TTFT/load/VRAM table.
6. Correctness/tok0 status.
7. Default behavior changed: yes/no.
8. Exact commands.
9. Artifact paths.
10. Recommended user-facing policy.

## Success Criteria

Minimum success:

- policy doc and probes clearly decide when to recommend `PREFILL_V2` and `PREFILL_CONCRETE_KV`.

Strong success:

- `PREFILL_V2=auto` safely enables on 24GB and stays off on insufficient VRAM / forced-off override.

Best success:

- server/long-prompt policy plus route-schedule fix gives fast prefill automatically for eligible prompts without
  breaking low-VRAM default safety.

## Do Not Do

- Do not build new prefill kernels.
- Do not reopen flash prefill v2.
- Do not route external Tensile as a default.
- Do not change decode.
- Do not make global default-on changes without owner approval.

