# Decode Oracle Semantic Map Result - 2026-06-20

Verdict: `PASS_DECODE_ORACLE_SEMANTIC_MAP_STATIC_COMPLETE_OES5_REQUIRED`

The extracted `q8_mmvq_gateup` oracle disassembly is now stage-labeled into a concrete decode body. This completes OES-4 at static level and makes the next blocker explicit: we need PC/timeline or stall attribution before reopening native scheduling.

## Oracle Body Stages

| stage | instructions | key counts | semantic role |
| --- | ---: | --- | --- |
| S0 setup/bounds/addresses | 38 | 2 branch, 1 waitcnt | Load kernargs, derive row/lane ids, select gate/up base pointers, handle edge lanes. |
| S1 scale/min byte select | 34 | 5 global loads, 5 waitcnt | Load q4 scale/min bytes and unpack lane-local scale/min selectors. |
| S2 q4 vector load prefetch | 22 | 6 global loads, 2 waitcnt | Compute q4 block addresses and issue four `b128` data loads plus scale/min payload loads. |
| S3 interleaved unpack/dot4/scale | 61 | 16 dot4, 4 convert, 4 waitcnt | Interleave q4 nibble select, waitcnt ladder, 16 dot4 ops, q8/q4 scaling, and final fma. |
| S4 cross-lane partial reduce | 48 | 5 `ds_bpermute`, 5 waitcnt, 1 branch | Five-step cross-lane reduction and lane-0 partial LDS store. |
| S5 final writeback | 27 | 1 barrier, 1 ds load, 1 global store | Barrier, lane-0 `ds_load_b128`, four-float final reduce, one global store. |

Trailing `s_code_end` padding is outside the semantic body but remains covered by the artifact-level instruction contract.

## Static Diff Against Native

The static count table is useful for orientation but still not a promotion objective:

- Oracle has 11 grouped global loads; native contract has 22.
- Oracle has 7 DS ops; native contract has 10.
- Oracle has 16 dot4, same as native.
- Oracle uses a compact `waitcnt` ladder across S1/S2/S3/S4/S5; native timing cannot be inferred from count parity.

The central body objective is S3: preserve oracle wait/load/dot/scale interleaving. But OES-4 does not prove whether S3 is the time gap; S4/S5 reduction handoff or launch/runtime can still dominate.

## Decision

Do not resume native decode scheduling from this static map alone.

Next phase: OES-5 PC timeline and stall attribution.

The first dynamic question is:

`Is the native gap dominated by S3 wait/load/dot/scale issue serialization, S4/S5 reduction handoff, or launch/runtime outside the body?`

Probe: `extra/qk_decode_oracle_semantic_map.py`

