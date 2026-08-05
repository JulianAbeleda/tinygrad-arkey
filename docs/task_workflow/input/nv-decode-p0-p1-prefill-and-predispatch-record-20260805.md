# NV decode P0/P1 prefill correctness and predispatch measurement record

Date: 2026-08-05. Target: native `DEV=NV`, RTX 5090 / sm_120, Qwen3-8B
Q4_K_M, d512, `max_context=1024` for the diagnostic harness.

## P0: causal correction and admission

The apparent persistent-KV failure was not a STORE, `AFTER`, JIT-held-buffer,
or memory-planner failure. A raw persistent STORE and ordinary prefill both
persisted correctly. The first invalid value was the NV prefill-v2 fp16 Q
projection: its d512 output was entirely NaN before the KV store. Therefore
the repair is deliberately narrow: `prefill_v2_target_admitted` closes
concrete fp16 prefill-v2 only on `NV/sm_120`; ordinary prefill remains the
fallback. The gate is removable once an independent finite-output fp16
projection qualification passes. Other targets and callers with prefill-v2
already disabled are unchanged.

With the gate enabled, the real d512 census observed all **36/36** block KV
prefixes finite, each with zero NaNs. The first decode full-logit oracle was
finite and in vocabulary range.

The full-logit diagnostic initially exposed a stale model-specific output
lifetime: sampled tokens differed from the returned logits' argmax after
replay. Sampling itself was correct. The diagnostic-only return now clones
the logits; production sampling and its graph remain unchanged. Each replay
now asserts `sample.item() == returned_logits.argmax()` and rejects unchanged
snapshots while position advances.

P0 count-8 pins (same in every P1 logit arm):

| field | value |
|---|---:|
| full-logit shape | `[8, 1, 151936]` fp32 |
| full-logit SHA-256 | `71c0a2b092cbc2e40c22b42cd4f6f3c84fe56fd40f2bfd008efc5b76be0ae0f0` |
| sampled / returned argmax | `[64461, 4710, 64461, 1837, 64461, 1837, 64461, 64461]` |
| all sample/argmax matches | true |
| snapshots change with position | true |

## P1: reverse A/B timing

The two default-on host-side protections are `JIT_INPUT_DESCRIPTOR_CACHE` and
`JIT_REUSE_WRITTEN_INPUT_SHADOWS`. OFF disables both. ON enables both. Every
arm used a fresh bounded subprocess (`timeout` + `flock`), valid greedy
production tokens, d512/count16, and three repetitions.

| arm | median ms/token | output hash | within-arm identical |
|---|---:|---|---:|
| OFF-A | 5.607174000 | `cc523cc67a8622f6a0efc130c1ec7497faa0620448f7bc5ea59385273ac92fb6` | true |
| ON | 5.548225875 | same | true |
| OFF-B | 5.622601938 | same | true |

The control midpoint is **5.614887969 ms/token**. ON recovers
**66.662094 us/token**, exceeding the 40-us P1 gate. OFF-A/OFF-B span is
15.427938 us/token, while ON is below both controls.

The full-logit OFF-A/ON/OFF-B runs also have exactly the P0 SHA-256 and
sample/argmax pins above. The switches operate only in `CapturedJit` replay:
the descriptor cache memoizes input metadata, and the shadow cache reuses a
private defensive-copy destination. They do not mutate the captured `linear`,
add device programs, change graph topology, or remove a copy/dependency edge.
Thus the device-window/kernel topology is structurally invariant; this record
does not claim a new device-window measurement.

## Verdict and rollback

P0 and P1 pass. The caches remain default-on; set either corresponding
environment variable to `0` for immediate per-process rollback. The NV
prefill-v2 gate remains closed by default until its producer-level numerical
gate is supplied. No broader route promotion is authorized by this record.

## Reproduction

```bash
PYTHONPATH=. DEV=NV .venv/bin/python \
  extra/llm_research/decode/nv_predispatch_full_logits_qualification.py \
  --mode logits --depth 512 --count 8 --max-context 1024 --out /tmp/logits.json
PYTHONPATH=. DEV=NV JIT_INPUT_DESCRIPTOR_CACHE=0 JIT_REUSE_WRITTEN_INPUT_SHADOWS=0 \
  .venv/bin/python extra/llm_research/decode/nv_predispatch_full_logits_qualification.py \
  --mode timing --depth 512 --count 16 --max-context 1024 --reps 3 --out /tmp/timing.json
```
