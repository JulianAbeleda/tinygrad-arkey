# D0.1 Q6-down boundary ABI

Status: **PASS**.

The executable contract freezes the exact 18 `ffn_down` Q6_K roles in layer
order `blk.0` through `blk.17`, with input `[512,12288]`, packed canonical
weight `[4096,12288]`, output `[512,4096]`, and a distinct caller-owned residual
buffer of the same shape. It defines producer, main, publication, and residual
cuts with stable marker identities and an expected 18 records at each cut.

The contract requires complete-output allclose checks, finite values, NaN
sentinel elimination, read-only input hashes, and preservation of paired K16
correction semantics. Expected lifecycle counts are 18 producers, 18 mains,
18 publications, 18 residual records, zero weight-copy kernels, zero partial
workspace bytes, and zero unknown records.

Evidence: `result.json` in the packet evidence directory. No runtime/model
files were edited and no performance claim is made by D0.1.
