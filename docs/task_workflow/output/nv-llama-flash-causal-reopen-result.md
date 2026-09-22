# Llama flash causal reopen

## Outcome

Llama's flash path is genuinely faster in production, and the reason is now
specific. It uses the vector decode kernel, not the tensor-core tile path. At
the captured depth-512 graph it operates on a padded 768-token KV extent and
launches six KV partitions for each of 32 query heads:

```text
flash_attn_ext_vec<128,1,F16,F16,false>
grid  (1, 6, 32), block (32, 4, 1)

flash_attn_combine_results<128>
grid  (1, 32, 1), block (128, 1, 1)
```

The old campaign statement that llama used four partitions at depth 512 is
not the production-graph truth. Four is optimal when the physical extent is
actually 512. The retained production graph pads the extent to 768 and uses
six partitions. The corrected isolated sweep reproduces both choices.

## Causal tests

The unmodified llama vector body was timed with its partition count varied.
Each arm used 5,000 back-to-back launches and was repeated in reverse order.

| partitions | median body at physical extent 768 |
|---:|---:|
| 1 | 8.477 us |
| 2 | 5.660 us |
| 3 | 4.690 us |
| 4 | 4.661 us |
| 5 | 4.723 us |
| 6 | **4.325 us** |

At physical extent 512, the same body instead reaches its minimum at four
partitions, 3.388 us. This proves that the production split is a service-rate
choice derived from the padded work extent rather than an arbitrary constant.

A corrected NCU pass on the actual six-part geometry measured about 0.895 M
thread instructions, 12.79 MB of L2 traffic, 0.418 MB of DRAM reads, 8.31% SM
throughput and 9.52% achieved occupancy. DRAM was only 3.78% of peak. Llama is
therefore not winning by reducing KV bytes from memory. It deliberately
duplicates GQA KV reads that are cache-served, then uses the simpler
head-parallel/token-parallel instruction grammar to finish sooner.

## Why it beats the current tinygrad topology

Tinygrad's score grid is 48 splits x 8 KV heads = 384 CTAs. Each four-warp CTA
shares one K/V tile across four GQA query heads and processes a short token
tile serially. That is byte-efficient, but it produces 48 partials per query
head and makes the combine consume 48 terms.

Llama takes the opposite trade:

- one query head per CTA, accepting cached K/V duplication across GQA heads;
- four warps expose 128 token rows concurrently using 8-lane dot groups;
- only six partials per head;
- a 128-thread combine assigns one output dimension to each thread.

Thus llama spends more L1/L2 bytes but performs less instruction-heavy
staging/reduction and creates one eighth as many combine partials. This is the
transferable principle: at short decode contexts, cached bytes are cheap and
parallel service rate is the scarce resource.

The current production ledger reflects that distinction:

| flash region | tinygrad | llama PDL-off |
|---|---:|---:|
| main/score role | 230.112 us/token | 164.130 us/token raw |
| combine role | 102.192 us/token | 37.056 us/token raw |
| score-start to combine-end island span | about 332.3 us serialized | 207.7 us |

The raw island-span difference is about 124.6 us/token. The ledger's adjusted
role comparison is 132.3 us/token. These are ceilings, not yet booked wall
recovery.

## PDL is not the flash explanation

Llama's programmatic dependent launch is valuable over the complete token
lifecycle, but it is not why this flash island is faster. In the retained
matched traces the sum of per-layer score-to-combine spans is 207.746 us with
PDL off and 212.678 us with PDL on. With PDL on, the combine interval starts
early and contains dependency wait; the useful island completion barely
changes.

The flash lever is therefore physical kernel topology. Copying PDL around the
existing tinygrad score/combine pair will not create llama's service rate.

## Endpoint clarification and translation

The observed promoted tinygrad endpoint remains approximately 240 tok/s. The
prior post-landing observation was 4.166708 ms/token, or 239.998 tok/s; the
fresh same-path reconfirmation is 4.172757 ms/token, or 239.650 tok/s. The
often-mentioned 241.2 tok/s was a causal projection obtained by applying a
matched bracket to an earlier endpoint. It was never a new observed endpoint.

If all 132.299 us of the current flash role debt translated to wall, the prior
4.166708 ms observation would become 4.034409 ms/token, or about 247.87 tok/s.
That is a gain of about 7.87 tok/s and would leave less than one tok/s to the
retained llama endpoint. This is why flash is worth reopening, but the number
must remain a ceiling until a native candidate passes the token-wall bracket.

## Next admissible construction

Do not further tune the current 48-split staging grammar. Build a true
head-parallel/token-parallel vector candidate with:

1. a physical-extent-derived partition count (six at the current padded d512
   graph, not a model-specific literal);
2. a dynamic/symbolic chunk loop so generated code does not unroll to MAXC;
3. the existing partial ABI first, followed by the six-part 128-thread combine;
4. semantic qualification under the campaign's accepted numerical contract;
5. isolated body/resource tests, then a reps>=9 reverse token-wall bracket.

The existing research `flash_vec_llama_score_pv_kernel` proves that UOps can
express the algorithm, but it is not a performance substrate yet: its old
bounded body was still much slower than llama and its old S=4 test targeted
the wrong production extent. That construction should be repaired or replaced
before any production investment.

## Evidence

- `docs/task_workflow/evidence/nv-flash-causal-reopen/llama-flash-causal-summary.json`
- `docs/task_workflow/evidence/nv-flash-causal-reopen/current-canonical-r9.json`
- `docs/task_workflow/evidence/nv-token-lifecycle-vs-llama-20260825/llama-dag.json`
- `docs/task_workflow/evidence/nv-lifecycle-recovery-tests-20260826/llama-pdl-ab/pdl-off-dag.json`
- `extra/llm_research/microbench/llama_fattn_vec_iso.cu`
