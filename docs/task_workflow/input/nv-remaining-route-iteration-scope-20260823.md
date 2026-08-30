# NV remaining-route iteration scope (2026-08-23)

Status: measurement and research-only iteration driver. No production model,
renderer, scheduler, runtime, or route policy may be changed by this scope.
Landing any route is a separate, explicitly authorized follow-up.

## 1. Frozen authority

Repository `/home/ubuntu/tinygrad-arkey`, branch `nvidia-bringup-20260731`,
HEAD `6570abc025514273faa100c66b979e531585a1e1`. GPU RTX 5090 sm_120, model
`/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`, depth 512, `DEV=NV`.

Measured and not to be regenerated:

- Q occurrence-0 closed: `P-C5 = +0.008 us`, prefix lever `C5-C6 = +0.664 us`.
- K/V occurrence-0 closed: `P-C5 = +0.520 us/call`, count-weighted
  `+13.520 us/token`.
- O occurrence-0 closed: `P-C5 = +1.032 us/call`, count-weighted
  `+37.152 us/token`.
- Q6 FFN-down four-warp fp16 route: single-bracket `WALL_PASS`,
  `-52.873 us/token`, token SHA identical across control/candidate/control.
  Authority `docs/task_workflow/output/nv-ffn-next-q6-wall-result-20260823.md`.

Ledger snapshot remains unchanged:

```text
node_sum          = 4677.920 us (tinygrad) / 3878.254 us (llama)
union             = 4671.500 us (tinygrad) / 3878.254 us (llama PDL-off)
overlap           = 6.420 us (tinygrad) / 0 us (llama PDL-off)
wall              = 4771.423 us (fresh control)
booked_recovery   = 0.000 us
remaining_to_240  = 604.756 us
```

The Q6 win is a measured research candidate, not booked recovery and not
parity.

## 2. Reusable per-candidate pipeline

Every route candidate follows the same five gates. Reuse the Q6 artifacts as
the working template.

1. **Admission module.** Add a closed-default module under `tinygrad/llm/`
   whose call returns `None` unless a research harness installs an explicit
   admission object. Copy the shape of `q6k_ffn_down_mmvq.py`.
2. **Census / exact-live gate.** Prove the swap changes only the intended
   program families, no weight materialization, and matching output SHA.
3. **Microgate.** If a kernel-body question survives, isolate it first with
   the existing `*_microgate.py` harness before spending a wall bracket.
4. **Reverse wall bracket.** Copy `q6k_ffn_down_four_warp_wall_bracket.py`:
   control -> candidate -> control, fresh processes, locked clocks, settled
   continuous windows, identical token SHA. Pass requires candidate below
   both controls and a delta of at least `+50 us/token`.
5. **Confirmation and landing.** A single bracket is a signal, not recovery.
   Book only after a second independent reverse bracket and a non-regression
   check at one other depth. Landing requires its own authorized scope that
   updates the route-policy record.

Do not skip gates. A microgate or wall bracket that fails a gate terminates
that candidate with a named verdict; it does not justify weakening the gate.

## 3. Candidate queue

Ordered by non-overlapping wall leverage from
`nv-installed-islands-phase11-handoff-result-20260822.md`.

### Tier 0: R partition (blocks the 240 verdict)

Partition the K/V and O `P-C5` remainders into admission, wait, and body with
the Stage 3 `%globaltimer` machinery. Full protocol:
`docs/task_workflow/input/nv-r-pc5-partition-handoff-scope-20260823.md`.
This stays first because `R ~ 404 us` is the named prerequisite.

### Tier 1, rank 1: FFN GEMV DRAM streaming (163.5 us ceiling)

- Q6 down: single-bracket `WALL_PASS +52.873 us`. Next action is only the
  confirmation gate above, then ledger entry.
- gate/up four-warp: `+90.97 us` ceiling, the largest remaining GEMV lever.
  No admission module exists yet. Build a four-warps-per-row w1w3 gate/up
  consumer (`[12288,1,1] / [32,4,1]`) and its wall bracket from the Q6
  template. Reuse `q4k_g3_lanemap_gemv_w1w3_kernel` in
  `tinygrad/llm/decode_kernels.py`.
- Q4 down: `+36.00 us` ceiling but recorded `CLOSED_BLOCKED` on 2026-08-21.
  Re-audit only the un-landed fp16-fma spelling if the Q6 result shows it is
  material; do not re-litigate the precision-blocked DP4A variants.

### Tier 1, ranks 2-5

Each already has a scope and tooling; run them in rank order and stop at the
first gate failure per candidate.

| rank | candidate | ceiling | scope |
| ---: | --- | ---: | --- |
| 2 | Q/K head norm + completion launch elimination | 96.1 us | `nv-qk-norm-completion-launch-elimination-scope-20260822.md` |
| 3 | vocab tail reduction topology | 58.3 us | `nv-vocab-tail-reduction-topology-scope-20260822.md` |
| 4 | flash combine body topology | 46.1 us | `nv-flash-combine-body-topology-scope-20260822.md` |
| 5 | attn norm body (provider-preserving) | 40.1 us | `nv-attn-norm-body-provider-scope-20260822.md` |

All ceilings are attribution, not booked recovery.

## 4. Gates and verdicts

- `WALL_PASS`: candidate below both controls, token SHA identical,
  `delta >= +50 us/token`.
- `NO_GO_WALL`: correct but no wall win, or win below the bar.
- `PRECISION_BLOCKED`: wall win but precision/logit gate fails.
- `CLOSED_BLOCKED`: every legal path measured and blocked.
- `UNMEASURED`: the required observable cannot be obtained; stop and report.

Overall verdict stays `240_UNMEASURED` until booked recovery is demonstrated
by token-SHA reverse wall brackets totaling at least the missing
`604.756 us`. Never sum ceilings or single-bracket results into that total.

## 5. Efficiency rules

- One candidate at a time; no parallel GPU arms.
- Reuse the Q6 bracket and admission files as templates instead of writing new
  timing code.
- Each arm is one fresh process under `/tmp/gpu-bench.lock`, locked clocks,
  reverse arm order, discard warmup, retain raw samples.
- The exact-live hook can capture Q, K/V, and O in one model load when a row
  needs triage.
- Reset GPU clocks after every run and at the end of each session.

## 6. Artifacts

- One research module or probe per candidate under `tinygrad/llm/` or
  `extra/llm_research/decode/`.
- One findings report under `docs/task_workflow/output/`.
- One evidence directory with raw arm JSON, token/output SHAs, and a
  `sha256.txt` manifest.
- Labels `[MEASURED]`, `[INFERRED]`, `[UNMEASURED]`, `[INVALIDATED]`.
- `py_compile` and `git diff --check` pass; no production file changes.
