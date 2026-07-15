# Measurement-regime audit: llama prefill comparator

Date: 2026-07-15  
Scope: fresh llama.cpp inputs/command, tinygrad `prefill_whole_synced`, collector behavior, and roofline wall override.  
Runtime code changed: no.

## Findings

The retained llama wrapper is `extra/llm/llama_bench.py:16-18`; callers use
`extra/llm/llama_cpp_bench.py:29` and `extra/llm/model_authority_bench.py:36`.
For the normal pp512 comparison the generated command is:

```text
/home/ubuntu/env/llama.cpp/build/bin/llama-bench \
  -m MODEL -ngl 99 -p 512 -n 128 -r 5 -o json
```

The installed binary (`llama-bench --help`, inspected 2026-07-15) reports:

```text
-p/--n-prompt 512     -n/--n-gen 128
-b/--batch-size 2048  -ub/--ubatch-size 512
```

Thus `n_prompt=512` is explicit, while `n_batch=2048` and `n_ubatch=512`
are currently implicit defaults.  The parser correctly selects the prompt
row with `n_prompt` set and `n_gen` absent (`extra/llm/llama_bench.py:28-31`).
The `-n 128` value does not contaminate the selected pp row, but it causes a
decode row to be run and should be made explicit in the record for reproducibility.

## Tinygrad comparison

`extra/qk/prefill_harness.py:19-29,108-124` resolves the authority profile to:

```text
chunk_n       = 512
K             = 8 bursts (minimum selected by each burst)
warmups       = 4 per start position
rounds        = 3 per start position
start_pos     = 0,512,1024,2048,3584
whole_lengths = 512,1024,2048,4096
max_context   = 4608
```

The exact generated 8B invocation is:

```bash
PYTHONPATH=/home/ubuntu/tinygrad-arkey \
PREFILL_V2=1 BOLTBEAM_MODEL_PROFILE=qwen3_8b_q4k_m_gfx1100 \
PREFILL_GRAPH_GEMM=1 \
python3 extra/qk/prefill_whole_synced.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --mode authority --model-profile qwen3_8b_q4k_m_gfx1100 \
  -K 8 --warmups 4 --rounds 3 \
  --start-positions 0,512,1024,2048,3584 \
  --whole-lengths 512,1024,2048,4096 --max-context 4608
```

`prefill_whole_synced.py:252-294` times the real synchronized
`model.__call__(chunk, concrete_start_pos, ..., use_flash=True)` path, with
`Device.synchronize()` before and after each timed burst.  Its 512-token chunk
is analogous to llama `n_prompt=512`, but there is no llama-style `n_batch` /
`n_ubatch` setting in this path: the graph shape is one `[1,512]` chunk and
the model/compiler owns internal tiling.  The collector at lines 293-294 is a
`candidate_route_census` around the same bursts; it records route/candidate
binding and is not an independent timing stream.

The `extra/qk/bench.py` entry point delegates to this authority through
`prefill_authority_argv`; it does not invoke llama.cpp and does not establish
an n_batch/ubatch equivalence.  `model_e2e_bench.py` is not a valid substitute:
its TTFT includes host/generation behavior, while the whole-sync authority
explicitly excludes that (`prefill_whole_synced.py:1-10`).

## Corrected commands

Make llama defaults explicit when collecting a fresh comparator artifact:

```bash
/home/ubuntu/env/llama.cpp/build/bin/llama-bench \
  -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf -ngl 99 \
  -p 512 -n 128 -b 2048 -ub 512 -r 5 -o json \
  > bench/llama-qwen3-8b-pp512-explicit.json
```

For a prefill-only llama measurement, avoid paying for an unneeded decode row
and make the intended zero-generation case explicit (verify the installed
llama-bench accepts `-n 0` before using its result as the stored comparator):

```bash
/home/ubuntu/env/llama.cpp/build/bin/llama-bench \
  -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf -ngl 99 \
  -p 512 -n 0 -b 2048 -ub 512 -r 5 -o json \
  > bench/llama-qwen3-8b-pp512-prefill-only.json
```

Use the first command for continuity with the existing parser/artifacts; use
the second only after confirming that the output still contains a prompt row
with `n_prompt=512, n_gen=0`.

## Roofline wall override assessment

The external `/home/ubuntu/boltbeam-runs/qwen3-8b-current-20260713/prefill/roofline_input.json`
records `measured=3881.0`, `raw_ceiling=8650.351`, and
`practical_ceiling=4450.505`.  Its practical basis says it rescales the
current PROFILE trace to a clean 131.925 ms authority wall, then lifts
`ffn_down/attn_qo/attn_kv` to the current generated `ffn_gate_up` oracle rate
while holding non-GEMM time fixed.

That is methodologically sound only as a labeled *counterfactual modeled
ceiling*: it answers “what if those roles matched this measured role while
everything else stayed fixed?” It is not a measured roofline, a fresh llama
comparison, or evidence that the lifted roles can reach that rate. The note is
also internally sensitive to regime: `prefill_whole_synced.py` distinguishes
generated-pure, spec-owned-hybrid, and external-handwritten provenance and
marks only generated-pure authoritative for promotion.

Recommended policy:

1. Keep the override, but label it `counterfactual_practical_ceiling` and
   retain the source wall, profile timestamp/artifact, route provenance, and
   per-role measured-vs-lifted flags.
2. Do not compare it directly with llama’s pp512 tok/s or call the difference
   “roofline headroom” unless all roles are measured in the same synchronized
   whole-prefill regime.
3. For a measured ceiling, use the clean authority wall and measured role
   device/profile ranges only; report hypothetical role lifts in a separate
   sensitivity table.

## Audit verdict

The llama prompt input is correct (`n_prompt=512`), and the tinygrad chunk is
also 512 tokens.  Batch semantics are not currently proven equivalent because
llama’s `2048/512` defaults are implicit and tinygrad has no corresponding
public knobs in this authority path.  Explicit llama flags plus recorded
binary/build/GPU metadata are required for a reproducible fresh comparator.
The roofline wall override is useful scenario analysis, but must remain
counterfactual rather than being treated as measured performance evidence.
