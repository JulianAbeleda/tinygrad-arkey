# NVIDIA pp512 vocabulary many-row contract (E0)

Date: 2026-08-29  
Packet: E0  
Status: **PASS: fixture authority complete**

## Scope

The legal prefill ABI is one real final hidden row, shape `(1, 4096)`, into
all `151936` vocabulary logits, shape `(1, 151936)` (or the existing rank-3
wrapper `(1, 1, 151936)`). The full-logit tensor must remain available to the
caller. A top-1-only kernel is not an implementation of this ABI.

The two and only two E1 designs considered were:

1. **Separate Q8 producer + packed Q6 MMVQ**: convert the final hidden row to
   the established compact Q8 producer record, then consume it with a
   many-row Q6 vocabulary MMVQ. The producer owns the activation scale/zero
   metadata; the MMVQ owns canonical packed Q6 weights and writes the complete
   logits tensor. This is the selected E1 design because it matches the
   existing llama ownership boundary and can retain the current full-logit ABI.
2. **Fused quantization + many-row Q6 MMVQ**: quantize the hidden row inside
   the vocabulary kernel and consume it immediately. This reduces an exposed
   producer boundary but couples quantization metadata, vocabulary main, and
   workspace, making independent producer/main census and replay harder.

No E1 implementation or wall claim is authorized by this packet.

## Existing reference evidence

The retained full-logit qualification is `nv-vocab-nacc4-qualification-20260828`.
It contains four rows of legal-shaped logits and an independent control:

| artifact | shape | sha256 |
|---|---:|---|
| `control-logits.npz` | `(4,1,151936)` float32 | `8793aa4a492d40487a3e6ebd13fbc883e310881cc385d3f86aa255da3254283a` |
| `candidate-logits.npz` | `(4,1,151936)` float32 | `b807437f08c0fe199c283db3960158a591bd65ea5a1f7bb6bf56ab6b1f1ad23` |
| `candidate-census.npz` | census payload | `ff4687081470093293025412872be7602deb3a104392aa9418ffff4edfe32b06` |

The retained comparison passes `max_abs=7.6293945e-06`, relative L2
`2.2685432e-07`, and argmax/top-10 parity. Its final verdict is
`STOP_WALL_NEGATIVE_NO_PROMOTION`; it is reference evidence only.

Sentinel values from the retained control/candidate arrays (row 0, first,
middle, and final vocabulary element) are respectively:

```text
control: 12.12298583984375, 12.644506454467773, 5.190765857696533
candidate: 12.12298583984375, 12.64450740814209, 5.190765380859375
```

## Executable oracle fixture status

The packet manifest is
`docs/task_workflow/evidence/nv-vocab-manyrow-e0-fixture-20260829/manifest.json`.
It recovers the canonical source container at
`/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf` (SHA-256
`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`), but
does not recover the required immutable `output.weight` Q6 payload or final
hidden row.

The retained artifacts are **not** a complete E0 oracle fixture. They lack:

- the canonical real Q6 vocabulary weight payload (or an immutable packed
  weight artifact plus its source/read-only hash);
- the real final hidden row used to produce the logits;
- a fixture manifest binding hidden row, packed weight, output, tolerance,
  and sentinels into one replayable oracle command.

The existing JSON metadata is insufficient to reconstruct those inputs. The
The fixture is now complete and tied to the current composed route. Exact
artifacts are `/home/ubuntu/tinygrad-arkey/docs/task_workflow/evidence/nv-vocab-manyrow-e0-fixture-20260829/final-hidden-row.f32`,
`/home/ubuntu/tinygrad-arkey/docs/task_workflow/evidence/nv-vocab-manyrow-e0-fixture-20260829/output.weight.q6_k.bin`,
and `/home/ubuntu/tinygrad-arkey/docs/task_workflow/evidence/nv-vocab-manyrow-e0-fixture-20260829/reference-logits.f32`;
hashes, sentinels, and replay command are in the packet manifest.

## E1 handoff requirements

Before E1 starts, capture one fresh final hidden row from the exact current
pp512 composed route and the matching canonical Q6 vocabulary weight. Store
both read-only under a new packet-specific evidence directory, record hashes,
and emit a fixture manifest containing:

- input and weight paths, dtype, byte count, shape, and SHA-256;
- reference full logits path, shape, dtype, SHA-256;
- tolerance `rtol=0.02, atol=0.5`;
- sentinels at first, middle, last, argmax, and adversarial coordinates;
- an executable read-only replay command and expected PASS/FAIL behavior.

E1 may then implement only Design 1, preserving full logits, canonical packed
weight ownership, separate producer/main census, and default-off dispatch.

## Verdict

**PASS.** E1 design recommendation: **separate Q8 producer + packed Q6 MMVQ**.
Replay with the command recorded in the packet manifest before implementation changes.
