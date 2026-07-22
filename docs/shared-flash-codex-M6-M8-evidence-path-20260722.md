# Shared flash attention M6/M8 geometry and evidence path

**Status:** evidence infrastructure complete; production promotion remains
fail-closed until dual-WMMA and real hardware measurements exist.

## Machine-checkable promotion record

`extra/qk/shared_attention_promotion.py` is the single validator for the final
record. It requires both model profiles, at least 200 warmed samples, generated
source and ISA artifacts, an allocation census, and positive timing/FLOP/byte
accounting. Missing evidence always produces `promotion_eligible=false`.

Keep `qk_wmma=false` and `pv_wmma=false` until generated source and AMD ISA
prove both contractions in the same fused call.

## One shared search domain

`extra/qk/shared_attention_evidence.py` is the single workload and candidate
manifest for both routes. It derives all rows from `MODEL_PROFILES` and produces
the same tile labels for fp16 Q/K/V attention:

| Profile | B | Hq | Hkv | G | Hd | Contexts |
|---|---:|---:|---:|---:|---:|---|
| `qwen3_8b_q4k_m_gfx1100` | 1 | 32 | 8 | 4 | 128 | 512, 2048, 4096 |
| `qwen3_14b_q4k_m_gfx1100` | 1 | 40 | 8 | 5 | 128 | 512, 2048, 4096 |

The initial labels are `Bq/Bkv/waves/stages`; they are not rankings and cannot
be route selectors. Existing BubbleBeam code is packed-linear-specific, so it
must consume measured attention records before it can rank these labels.

## Required measurement sequence

Run one GPU process at a time, with a clean health observation before and after
every run. Do not run these commands until the fused candidate passes scheduler
correctness and code generation.

```bash
# Baseline, 8B
PYTHONPATH=. DEV=AMD NOOPT=0 .venv/bin/python extra/qk/bench.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --model-profile 8b \
  --prefill --prefill-mode authority --pin-clock \
  --prefill-artifact bench/shared-flash/8b-baseline.json

# Baseline, 14B
PYTHONPATH=. DEV=AMD NOOPT=0 .venv/bin/python extra/qk/bench.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf --model-profile 14b \
  --prefill --prefill-mode authority --pin-clock \
  --prefill-artifact bench/shared-flash/14b-baseline.json
```

After a real fused candidate exists, rerun the exact same commands with its
proof-bound policy/artifact active, writing `*-candidate.json`. Do not use a
manual override: the policy rejects override-based admission. Compare only
same-machine, same-driver, same-clock-policy, same model bytes, same context,
and same route census runs.

For each geometry/context, collect 200 warmed attention dispatches, GPU `tm`,
raw samples, median and dispersion, QK/PV generated WMMA provenance under
`NOOPT=0`, score/probability allocation census, and fp32-reference error.
Whole-prefill authority uses its existing pp512/1024/2048/4096 profile; pp512,
pp2048, and pp4096 are mandatory model gates.

## Missing hardware evidence

No current artifact contains all of the following for a selected fused path:

- generated source or ISA tying two WMMA calls to QK and PV;
- allocation evidence excluding global `T*KV` score/probability buffers;
- warmed 200-dispatch GPU timing samples for each baseline/candidate pair;
- roofline ceilings with device, driver, clocks, commands, raw samples, and
  consistent FLOP/byte accounting;
- real 8B and 14B route census, output parity, peak memory, and decode
  non-regression.

The old phase-0 timing table is baseline context only, not promotion evidence.
