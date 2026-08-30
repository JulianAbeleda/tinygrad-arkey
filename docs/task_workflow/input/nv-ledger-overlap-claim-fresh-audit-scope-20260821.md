# NV ledger overlap-claim fresh audit scope

Date: 2026-08-21

Status: **measurement-first audit**. This packet authorizes one fresh,
locked, same-session re-test of the claim that the tinygrad-vs-llama decode
gap is exposed device timeline, not kernel body speed. It authorizes no
production change, no promotion, and no new architectural conclusion.

Repository: `nvidia-bringup-20260731` at `6570abc02`.

## 1. The claim under test

Before this result is reused as a reference, test these three statements:

1. **Timeline claim.** At current HEAD, llama overlaps a large share of its
   kernel residence time while tinygrad's steady decode token is near-serial:
   at most a handful of short overlaps, not a llama-scale pipeline.
2. **Attribution claim.** The measured wall gap decomposes through
   the exact interval identity
   `delta_union = delta_node_sum - delta_overlap`, and the decisive segment
   is S1 (Q end to O start), not the GEMV anchor bodies.
3. **Roofline claim.** Both routes are memory-bound; tinygrad moves more
   accounting bytes per token and still keeps the DRAM idle during its
   serial support chain.

These are claims about kernel intervals and their arithmetic. They are not a
claim that every overlapping interval contains simultaneous useful traffic:
without per-kernel wait-exit timestamps, useful-body concurrency stays
`unmeasured`.

## 2. Belief-flip gates

Write the gate verdicts before running:

| gate | observable | supports claim when | refutes claim when |
| --- | --- | --- | --- |
| G1 llama overlap real | raw CUPTI replay kernel intervals | at least one overlapping pair and a negative minimum inter-kernel gap | no overlapping pairs |
| G2 tinygrad near-serial | raw HCQ profile token intervals | fewer than 20 overlapping pairs and overlap mass below 20 us | overlap mass comparable to llama's share |
| G3 identity closes | fresh ledger arithmetic | delta_union = delta_node_sum - delta_overlap and wall = union + host residual | nonzero unassigned residual |
| G4 location is S1 | fresh segment table | S1 delta is the largest segment and anchors are not the deficit | largest segment elsewhere |
| G5 anchor bodies | fresh anchor union | tinygrad anchor union is at or below llama's | tinygrad anchor union exceeds llama by more than the S1 delta |
| G6 roofline | fresh byte estimate over fresh wall | tinygrad effective GB/s is below llama while its byte total is higher | tinygrad is already at or above llama effective bandwidth |

## 3. Method

Same RTX 5090, same session discipline, Qwen3-8B-Q4_K_M, d512 decode,
`flock /tmp/gpu-bench.lock`, one fresh process per arm.

### 3.1 Tinygrad route

Fresh current-HEAD control:

```bash
flock -w 600 /tmp/gpu-bench.lock \
  env DEV=NV PROFILE=1 HCQ_GRAPH_PROFILE_JSON=/tmp/nv-ledger-overlap-audit-20260821.profile.jsonl \
  .venv/bin/python extra/llm_research/decode/nv_rmsnorm_current_head_topology.py \
  --arm control --sites ffn \
  --out docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/tinygrad-capture.json
```

The capture itself is profiled topology evidence only. The paired unprofiled
wall comes from one control/candidate/control A bracket:

```bash
for arm in control candidate control; do
  timeout 900 flock -w 600 /tmp/gpu-bench.lock \
    env DEV=NV .venv/bin/python extra/llm_research/decode/nv_norm_native_wall_ab.py \
    --arm "$arm" --sites ffn --count 24 --reps 4 \
    --out "docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/tinygrad-wall-${arm}.json"
done
```

The three rows are joined with the retained
`validate_timing_bracket` contract. Token SHA is recorded by the harness and
must be identical across the control/candidate/control rows.

### 3.2 Llama route

Unprofiled authority row first:

```bash
flock -w 600 /tmp/gpu-bench.lock \
  /home/ubuntu/env/llama.cpp/build-cuda/bin/llama-bench \
  -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf -ngl 99 -fa 1 \
  -p 512 -n 20 -d 512 -r 5 -o json \
  > docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/llama-unprofiled.json
```

Then one CUPTI node trace with the real graph edge dump:

```bash
flock -w 600 /tmp/gpu-bench.lock \
  env GGML_CUDA_GRAPH_DUMP=/tmp/llama_ledger_overlap_dump_20260821.txt \
  /usr/local/bin/nsys profile --cuda-graph-trace=node --resolve-symbols=false \
  --force-overwrite=true --output=/tmp/llama_ledger_overlap_20260821.nsys-rep \
  /home/ubuntu/env/llama.cpp/build-cuda/bin/llama-bench \
  -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf -ngl 99 -fa 1 \
  -p 512 -n 10 -d 512 -r 3 -o json
/usr/local/bin/nsys export --type=sqlite \
  --output=/tmp/llama_ledger_overlap_20260821.sqlite \
  /tmp/llama_ledger_overlap_20260821.nsys-rep
```

Choose the steady graph id by replay census, then emit the weighted real-edge
DAG with the existing tool:

```bash
.venv/bin/python extra/llm_research/decode/llama_weighted_dag.py \
  --trace /tmp/llama_ledger_overlap_20260821.sqlite \
  --dump /tmp/llama_ledger_overlap_dump_20260821.txt \
  --graph-id <authority_graph_id> --warmup 2 \
  --out docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/llama-dag.json
```

### 3.3 Reconciliation and overlap census

Run the retained inter-anchor analyzer on the fresh inputs, then the small
audit tool that computes the raw overlap-pair census, serialization
counterfactual, and effective bandwidth from the two fresh artifacts:

```bash
.venv/bin/python extra/llm_research/decode/nv_inter_anchor_analysis.py \
  --tinygrad docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/tinygrad-capture.json \
  --llama docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/llama-dag.json \
  --wall-bracket docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/tinygrad-wall-bracket.json \
  --llama-unprofiled docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/llama-unprofiled.json \
  --out-control docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/tinygrad-canonical.json \
  --out-ledger docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/ledger.json \
  --out-sensitivity docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/sensitivity.json

.venv/bin/python extra/llm_research/decode/nv_ledger_overlap_claim_audit.py \
  --tinygrad docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/tinygrad-capture.json \
  --llama docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/llama-dag.json \
  --ledger docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/ledger.json \
  --llama-unprofiled docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/llama-unprofiled.json \
  --out docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/overlap-claim-audit.json
```

## 4. Acceptance criteria

- all GPU rows run under the lock in fresh processes;
- token SHA is retained and identical inside every bracket;
- G1-G6 are written as observed/refuted, never inferred from an older run;
- the useful-body portion of llama overlap is explicitly `unmeasured`;
- byte totals are labeled accounting estimates, not DRAM counters;
- every retained raw trace is named and SHA-256 is recorded where possible.

## 5. Outputs

Evidence:
`docs/task_workflow/evidence/nv-ledger-overlap-audit-20260821/**`

Result:
`docs/task_workflow/output/nv-ledger-overlap-claim-fresh-audit-result-20260821.md`

Allowed tooling path for the new audit script:
`extra/llm_research/decode/nv_ledger_overlap_claim_audit.py`
