# NV Q6 strict-after Q8 panel-1 Gate 9 decision

## Verdict

`REJECT_COMPILE_TIMEOUT`. The candidate did not emit CUDA, cubin, or SASS, so no binary, correctness, or timing gate was eligible to run.

## Frozen route

- Main anchor cubin: `/home/ubuntu/tinygrad-arkey/docs/task_workflow/evidence/nv-q6-oracle-reduction-policy-20260831/artifacts/main_all_partials_ascending/main_all_partials_ascending.cubin`
- Main anchor SHA-256: `6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137`
- All-partials fixup SHA-256: `483d997cb16bacf35db3184825833e6f921a4b23efbb6d3c6836723d18285514`
- Timing reference: `256.256 us`
- Route: trusted-FP16 packed scales, one-body `+170 CTA`, combined publication, all-partials fixup.

## Exact integration

The dependency is bound only at `kphase=0,cg=7,n=1,p=1,r=3`, corresponding to the frozen anchor producer `FADD R167, R53, R36` at PC `0x9f80`, ordinal `2552`.

```python
strict_token=UOp.placeholder((1,),weighted.dtype.scalar(),91,addrspace=AddrSpace.REG)
strict_token=strict_token.after(strict_token[0].store(weighted))
panel1_base=(q8_epoch+Q8_WORDS+lid).strict_after(strict_token[0])
panel1_raw=tuple(q8_record[panel1_base+i*256].load() for i in range(18))
```

One strict base feeds all 18 address expressions. No loaded value is wrapped.

## Focused test

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock env PYTHONPATH=. \
  .venv/bin/python -m pytest -q \
  test/unit/test_strict_after.py \
  test/unit/test_nv_q6_strict_after_panel1.py
```

Result: `5 passed in 1.64s`.

## Bounded compile evidence

The direct live-token version was manually bounded after approximately 516 seconds. It was still in `to_program -> full_rewrite_to_sink -> initial symbolic`, at `simplify_valid -> valid.backward_slice -> UOp.toposort`.

The one permitted repair materialized only the exact scalar FADD value through a scalar `DEFINE_REG` store/load. Candidate-only compilation was then bounded exactly as follows:

```bash
flock -w 1200 /tmp/nv-q6-oracle-gpu.lock \
  timeout --signal=INT --kill-after=10s 240s \
  env NV_Q6_GPU_LOCK_HELD=1 PYTHONPATH=. DEV=NV \
  .venv/bin/python - <<'PY'
# Gate-7 compatibility wrapper maps true_late_tail to strict_after_q8_panel1=True.
# Only the candidate is compiled; the anchor is the frozen cubin above.
gate._compile_ast(gate._ast("true_late_tail"), "strict_after_reg_token", root)
PY
```

Result: exit `124` at 240 seconds. The interrupt stack remained in `initial symbolic`, recursively normalizing `_commutative_key(...).tuplize`. No CUDA, cubin, or disassembly was produced.

## Gate disposition

| Gate | Result |
|---|---:|
| One shared strict base / 18 dependent loads at UOp level | PASS |
| Focused core and integration tests | PASS, 5/5 |
| Candidate CUDA emission | FAIL, timeout |
| `dependency < LDG < STS` | NOT RUN |
| Exact panel-1 `18 LDG / 18 STS` | NOT RUN |
| Span `<=160` | NOT RUN |
| Static families `256/32/176/109/73/64/4` | NOT RUN |
| No added `BAR/MEMBAR` | NOT RUN |
| Registers `<=255`, stack/LDL/STL zero | NOT RUN |
| Trusted exactness and partial/final uint32 identity | NOT RUN |
| Alternating locked R31, delta `<=-3 us`, wins `>=24/31` | NOT RUN |

The lock was acquired and released for every command. GPU kernel launches: `0`.

## Proven blocker and next gate

`STRICT_AFTER` remains structurally transparent to the initial symbolic passes. Scalar register materialization did not bound the dependency graph seen by symbolic validity and commutative canonicalization.

Do not try another Q8 placement. First make strict dependency identity opaque to symbolic canonicalization after semantic reachability has been recorded. Require a large-DAG compile-scaling unit test that reaches CUDA emission within a fixed bound without introducing hardware barriers, local memory, or spills. Then rerun this unchanged one-base/18-load integration against the frozen anchor.
