# Decode q8 Producer Resource Context Audit Scope - 2026-06-20

Verdict target: `PASS_OR_BLOCK_DECODE_Q8_PRODUCER_RESOURCE_CONTEXT_AUDIT`

The producer context isolation gate found:

```text
producer-only                   21.60us
producer after q4 buffers       30.94us
q4 buffer residency delta       +9.34us
controlled lifecycle gap        +8.76us
```

The next question is whether this is general resident-memory pressure, real q4 buffer content/copy-in, allocator
placement, or execution order noise.

## Tool

`extra/qk_decode_owned_q8_producer_resource_context_audit.py`

## Rows

| row | purpose |
|---|---|
| baseline | producer with only producer buffers |
| real q4 alloc-only | q4-sized buffers allocated but not copied |
| real q4 copied | real gate/up q4 bytes copied into buffers |
| dummy same alloc-only | same bytes/chunks, no copy |
| dummy same copied | same bytes/chunks, random bytes copied |
| dummy half copied | lower resident/copy size |
| dummy double copied | higher resident/copy size |

## Gates

| gate | threshold |
|---|---:|
| producer correctness | all rows pass |
| real q4 alloc reproduces slowdown | delta >= `5us` |
| q4 copy not required | copied delta within `3us` of alloc-only delta |

## Interpretation

- If real and dummy same-size buffers both slow the producer, the issue is general residency/allocator pressure.
- If only real q4 buffers slow it, inspect buffer placement/source/flags.
- If neither reproduces it, rerun context isolation with interleaving and clock provenance.
