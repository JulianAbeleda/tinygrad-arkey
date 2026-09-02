# NV Q6 schedule-after panel-1 Gate 10 decision

## Verdict

`REJECT_SASS_SPAN_AND_LDS`. No correctness or timing run was permitted.

## Frozen route

- Anchor cubin SHA-256: `6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137`
- All-partials fixup SHA-256: `483d997cb16bacf35db3184825833e6f921a4b23efbb6d3c6836723d18285514`
- Frozen phase token: anchor `FADD R167, R53, R36`, PC `0x9f80`, ordinal `2552`
- Source binding: `kphase=0,cg=7,n=1,p=1,r=3`
- Timing reference: `256.256 us`

The current default-off builder was recompiled in `9.027559s`. Its cubin SHA is exactly the frozen anchor SHA above, proving the admitted early route remains byte-identical.

## Integration

```python
panel1=tuple(
  q8_record[q8_epoch+Q8_WORDS+lid+i*256].load().schedule_after(weighted)
  for i in range(18)
)
```

Only the 18 immutable scalar `u32` Q8 panel-1 loads carry the opaque schedule token. No INDEX carries the phase DAG.

Focused tests under the GPU lock passed before and after the single repair:

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock env PYTHONPATH=. \
  .venv/bin/python -m pytest -q \
  test/unit/test_opaque_load_schedule.py \
  test/unit/test_nv_q6_strict_after_panel1.py
```

Results: `6 passed in 2.05s`, then `6 passed in 2.04s`.

## Single repair

The initial candidate routed scheduled loads through 18 explicit REG placeholders and joined the preload at the overwrite barrier. It compiled to cubin `cb36ea8b6a2cce09506ebc37d821e97750bee8d6007ff54131731596152b5be6`, but had span `309` and scalar LDS `184`.

The only repair removed that candidate-only REG/barrier staging. Scheduled load values feed the post-barrier STS directly, matching llama's transport role. No token, source tuple, arithmetic, publication, or route piece changed.

## Final candidate

- Compile time: `8.616186s` under a hard `240s` bound
- Cubin SHA-256: `4ffb11c5ef413fd7d828a375ad53919a203dc518ab46ff0ba337f80db743b3c7`
- Source SHA-256: `97bb6cde4148e00d9a4ddabe600bca7f14680466fd633f1e39ca1b587169e16d`
- Instructions: `5200`
- Resources: `REG255`, `STACK0`, `SHARED1024`, `LOCAL0`, `LDL0`, `STL0`

Exact chain:

```text
0x9630  FADD R126, R126, R41
0x97c0..0x98e0  18 LOP3.LUT identity schedule edges, all reading R126
0x99b0..0x9bf0  18 classified panel-1 LDG
0xac20          overwrite BAR.SYNC
0xac90..0xada0  18 classified panel-1 STS
0xadb0          panel-1 publication BAR.SYNC
```

The order gate passes, but first LDG ordinal `2459` to first STS ordinal `2761` is `302` instructions, exceeding `160`.

## Static gate table

| Gate | Required | Actual | Result |
|---|---:|---:|---|
| Default anchor cubin | frozen SHA | exact match | PASS |
| Candidate compile | `<=240s` | `8.616186s` | PASS |
| Token edges | 18 | 18 | PASS |
| Panel-1 LDG / STS | 18 / 18 | 18 / 18 | PASS |
| Producer < edges < LDG < STS | true | true | PASS |
| First LDG to first STS | `<=160` | `302` | FAIL |
| IMMA / LDSM / LDS | 256 / 32 / 176 | 256 / 32 / 184 | FAIL |
| LDG / STS / STG / BAR | 109 / 73 / 64 / 4 | 109 / 73 / 64 / 4 | PASS |
| I2FP / FMUL / FADD / FFMA | 1024 / 1544 / 1024 / 0 | exact | PASS |
| MEMBAR / ATOM | 0 / 0 | 0 / 0 | PASS |
| Stack / LDL / STL | 0 / 0 / 0 | 0 / 0 / 0 | PASS |

## Downstream disposition

- Trusted exactness: `NOT RUN`
- Partial/final uint32 identity: `NOT RUN`
- Same-process alternating locked R31: `NOT RUN`
- GPU kernel launches: `0`
- Lock: acquired for every command and released

The SASS gate failed after the single permitted repair. Gate 10 must not proceed to another integration variation. The next investigation is binary/compiler-side: explain the eight additional scalar LDS and the persistent 302-instruction transport lifetime.
