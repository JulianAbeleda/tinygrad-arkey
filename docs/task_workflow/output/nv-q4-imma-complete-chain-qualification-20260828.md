# NV packed-Q4 IMMA complete-chain qualification gate

## Status

The independent gate is executable and fail-closed.  It builds a legal real
Qwen3-8B gate fixture, extracts canonical packed Q4_K words directly from the
GGUF, executes and qualifies the full production-shaped Q8 producer, and
requires an explicit provider manifest before a packed main/fixup can enter the
measurement bracket.

The packed-byte-to-IMMA provider ABI is not stable yet, so the main+fixup arm is
correctly `BLOCKED`, not substituted with expanded int8 weights.  No production
route was edited.

## Frozen ABI

- shape `(M,N,K)=(512,12288,4096)`;
- Q4 words: canonical GGML Q4_K row-major
  `[N,K/256,36]` `uint32`;
- Q8 values: row-major `[M,K]` `int8`;
- Q8 scales and raw pre-quantization sums: `[M,K/32]` `float32`; sums
  preserve llama's FP32 reduction -> FP16 metadata round -> FP32 ABI widening;
- main output: FP32 row-major final output or FP32 stream-K partials;
- stream-K providers must declare and supply a separate fixup symbol;
- complete timing always charges producer + main + fixup/epilogue.

The manifest validator additionally requires symbol, artifact, exact argument
order, grid, block, shared bytes, output dtype, partial-write declaration, and
fixup symbol.  Missing fields, expanded-weight artifacts, or final-output-only
stream-K declarations fail closed.

## Executed fixture and producer qualification

The fixture uses `blk.0.ffn_gate.weight` from
`Qwen3-8B-Q4_K_M.gguf` and a deterministic real model normalization boundary.
All size guards pass.  The Q4 word SHA is
`1380e2e8...a937ad8`; activation SHA is `a256c7e8...08da7bd`.

| gate | hot R9 | fresh-process R7 | result |
| --- | ---: | ---: | --- |
| Q8 producer complete synchronized wall | 104.156 us | 105.609 us | stable |
| Q8 values | | | exact to canonical reference |
| Q8 scales | | | exact to canonical reference |
| raw FP16-rounded sums | | | exact, max abs `0` |
| finite and shape guards | | | PASS |

This producer is correct but not yet roofline-quality.  Its Q8 value/scale
surface launches one warp per 32-value group (65,536 groups), creating a large
scheduling surface; the raw-sum output is a separate legal reduction.  Its
rough packet service is only about 100--120 GB/s.  A shared gate/up producer
pays this once per layer, but it still costs about 3.75 ms over 36 layers.
Producer geometry must therefore be included in the
optimization campaign rather than treated as free.

## Wall and roofline translation

The corrected FP16 gate/up population costs 41.020 profiled ms for 72
projections.  With the current shared producer and a llama-like packed
main+fixup of approximately 175 us/projection, a pair costs about 454 us and
36 layers cost about 16.4 ms.  That would recover roughly 24.6 profiled ms, or
about 21.6 ms after the audit's `84.025/95.810` wall scaling: an estimated
62.4 ms pp512 before other roles.  This is planning math, not a measured
candidate claim.

If the producer reaches llama-like single-digit microseconds per shared input,
the same packed body approaches 12.7 ms for the population and recovers about
25 ms of authority wall, matching the earlier ~59 ms planning endpoint.

The packed Q4 analytic bandwidth roof remains 407.7 TMAC/s.  Promotion should
report both useful MAC rate and complete lifecycle rate.  A fast main that is
hidden by producer/fixup overhead does not qualify.

## Provider execution gates

Once the primary agent supplies the stable artifact, this harness must be
extended from `ABI_ACCEPTED` to actual launch and require:

1. producer + main + fixup R9, fresh R7, and reverse-order R9;
2. finite output and declared Q8 semantic tolerance against a CPU packed-Q4
   reference, plus guard canaries and input read-only hashes;
3. no global expanded Q4-to-int8 weight buffer;
4. cubin/SASS proof of `IMMA.16832.S8.S8`, with HMMA/DP4A census;
5. registers, shared memory, stack/spills, occupancy, DRAM/L2 bytes, and
   tensor-pipe counters for both main and fixup;
6. complete lifecycle below the corrected 569.7 us/projection control and a
   paired/shared-producer wall below the FP16 gate/up pair.

## Artifacts

- harness: `extra/llm_research/prefill/nv_q4_imma_complete_chain_gate.py`;
- hot R9: `docs/task_workflow/evidence/nv-q4-imma-complete-chain-20260828/hot-r9.json`;
- fresh R7: `docs/task_workflow/evidence/nv-q4-imma-complete-chain-20260828/cold-r7.json`.
