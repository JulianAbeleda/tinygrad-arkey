# Decode Native Renderer DNR-3C2 Dataflow Emitter Result - 2026-06-20

## Verdict

`PASS_DNR3C2_B128_PRELOAD_CORRECT_LOAD_BUDGET_CLOSED_BLOCKED_ON_COMPOUND_SHAPE`

DNR-3C2 built the register/dataflow rewrite that DNR-3C1 said was required. It is not just structural: the candidate
launches and computes the synthetic gate/up fixture correctly.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr3c2_dataflow_emitter_probe.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c2_dataflow_emitter_result.json
```

## Result

| gate | result |
|---|---:|
| DNR-3C1 blocked on dataflow | pass |
| candidate launches | pass |
| candidate correct | pass |
| global-load budget closed | pass |
| dot4 preserved | pass |
| single store preserved | pass |
| oracle compound shape reached | fail |

Correctness:

| output | max abs | mean abs |
|---|---:|---:|
| gate | `0.00048828125` | `0.00018835067749023438` |
| up | `0.000274658203125` | `0.0001456737518310547` |

Grouped count movement:

| grouped count | DNR-2 native | DNR-3C2 candidate | hipcc/LLD oracle |
|---|---:|---:|---:|
| dot4 | `16` | `16` | `16` |
| global load | `22` | `10` | `11` |
| ds | `10` | `10` | `7` |
| branch | `0` | `0` | `5` |
| waitcnt | `17` | `10` | `20` |
| `s_clause` | `0` | `0` | `3` |
| `s_delay_alu` | `0` | `0` | `30` |

## What Changed

The scalar inner loop's sixteen `global_load_b32` instructions are replaced with four coalesced preloads:

| stream | loads | dest regs | lanes |
|---|---:|---|---|
| q4 | `global_load_b128` offset `0` | `v[80:83]` | `0..3` |
| q4 | `global_load_b128` offset `16` | `v[84:87]` | `4..7` |
| q8 | `global_load_b128` offset `0` | `v[88:91]` | `0..3` |
| q8 | `global_load_b128` offset `16` | `v[92:95]` | `4..7` |

Each dot4 lane is remapped from reused `v[8]`/`v[9]` to its own preload registers. The q4 nibble-select mutation stays
per lane, so later lanes are not clobbered.

## Next Blocker

DNR-3C2 closes the load-shape primitive. The remaining blocker is now DNR-3C3 compound shape:

1. derive branch/exec policy from lane role semantics;
2. reduce LDS/reduction shape from native `10` toward oracle `7`;
3. insert `s_clause`/`s_delay_alu` from semantic latency/resource boundaries;
4. launch and recheck correctness;
5. only then time against the q8 oracle.

No renderer defaults changed and no performance claim is made here.
