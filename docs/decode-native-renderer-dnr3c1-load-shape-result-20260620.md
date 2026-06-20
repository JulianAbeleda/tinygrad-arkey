# Decode Native Renderer DNR-3C1 Load Shape Result - 2026-06-20

## Verdict

`BLOCKED_DNR3C1_LOAD_SHAPE_NEEDS_REGISTER_DATAFLOW_EMITTER`

DNR-3C1 audited the first oracle-shaped decode rewrite after DNR-3B: reducing native grouped global loads from `22`
toward the hipcc/LLD oracle's `11`. The required opcode exists (`global_load_b128`), but the correct DNR-2 stream cannot
be fixed by local instruction substitution.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr3c1_load_shape_probe.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c1_load_shape_result.json
```

## Evidence

| gate | result |
|---|---:|
| DNR-3A plan present | pass |
| DNR-3B generic emitter blocked as expected | pass |
| `global_load_b128` opcode available | pass |
| native grouped global loads | `22` |
| oracle grouped global loads | `11` |
| scalar q4/q8 loop load pairs found | `8` |
| consumer coupling proven | pass |
| local `b32 -> b128` substitution safe | fail |

The loop has eight repeated scalar pairs:

```text
global_load_b32(v[8], v[23], ..., s[16:17])
global_load_b32(v[9], v[24], ..., s[18:19])
s_waitcnt()
... mutate/select q4 in v[8] ...
v_dot4_i32_iu8(v[4], v[8], v[9], v[4], ...)
v_dot4_i32_iu8(v[5], 0x01010101, v[9], v[5], ...)
... increment v[23]/v[24] ...
```

So the first blocker is not "missing b128." It is that the producer registers (`v[8]`, `v[9]`), q4 nibble-select
mutation, dot4 consumers, address increments, waits, and live ranges are all coupled inside the scalar loop.

## Rewrite Options

| option | status | reason |
|---|---|---|
| local `b32 -> b128` substitution | blocked | would define multiple VGPRs but the consumers still expect one q4 word in `v[8]` and one q8 word in `v[9]` per iteration |
| hoist same scalar loads | possible but not oracle-shaped | preserves semantics but still has the same grouped load count and raises VGPR pressure |
| coalesced `b128` q4/q8 preload | required path | needs a register/dataflow emitter that allocates preload registers and rewrites dot4 operands |

## Next

DNR-3C2 must be a small decode register/dataflow emitter:

1. model the eight q4/q8 dot4 lanes;
2. allocate distinct VGPR ranges for coalesced q4 and q8 preload results;
3. rewrite q4 unpack/select operations to consume those registers without clobbering later lanes;
4. rewrite q8 dot4 operands to the matching preloaded registers;
5. emit waits from the new producer-consumer edges;
6. run synthetic gate/up correctness before marker, branch, or LDS/reduction edits.

No renderer defaults changed, no kernel was launched, and no performance claim is made here.
