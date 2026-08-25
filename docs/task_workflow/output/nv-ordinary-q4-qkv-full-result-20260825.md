# Ordinary Q4/Q4/Q4 full-grid producer result

## Verdict

`NO_GO_TOKEN_BOUNDARY_WALL`.

The workgroup-uniform full-grid substrate successfully expresses the complete
ordinary Q4/Q4/Q4 population, but this spelling does not improve production
token wall. Keep it research-only and do not replace the installed ordinary
Q plus K/V-pair route.

## Exact construction

The candidate covers blocks 13, 19, 20, 22, 23, 25, 26, 28, and 29. Every
one of 4,096 CTAs produces one Q row with the installed vector-load arithmetic.
The first 2,048 CTAs additionally produce one K-or-V row from packed
K-then-V weights. Q, K, and V write directly to separate caller-owned outputs.
Both production arms allocate the same packed weight view, eliminating
allocator and address-topology bias.

## Gates

The isolated pointer-rotated cold gate is bit-exact and improves the complete
Q plus K/V-pair span from 14.238 to 11.858 us/group at the median, recovering
2.380 us/group. Across nine groups this is a 21.420 us/token isolated ceiling.

The repaired production census sees the actual steady graphs:

| measure | installed Q + K/V pair | full-grid QKV | delta |
| --- | ---: | ---: | ---: |
| graph nodes | 490 | 480 | -10 |
| GPU node sum | 4,156.864 us | 4,125.344 us | -31.520 us |
| GPU union | 4,154.250 us | 4,122.750 us | -31.500 us |

The full depth-512, count-32, reps-7 A/B/A wall bracket is token-exact but
negative:

| arm | median wall |
| --- | ---: |
| control A | 4.384895 ms/token |
| candidate | 4.417253 ms/token |
| control C | 4.394117 ms/token |
| control midpoint | 4.389506 ms/token |

The candidate loses 27.746 us/token. Its fastest accepted sample is still
slower than the slowest control-C sample. A reps-9 confirmation is therefore
not needed to classify the direction.

## Accounting conclusion

This is not a kernel-service failure: the isolated complete span and the
production GPU timestamp ledger both improve. It is also not an incomplete
population, output-copy, allocator, or correctness wall. The missing recovery
lies outside timestamped GPU body work at the changed token/graph boundary.
The candidate removes one small graph submission, but its output/program
boundary costs enough host/replay service to reverse roughly 31.5 us of GPU
recovery into a 27.7 us wall loss.

The full-grid compiler substrate remains validated by the passing shared-Q8
producer. For ordinary Q4, this exact topology is closed. Revisit only with a
construction that also changes the program/output boundary, not another CTA
geometry variation inside the same boundary.

Evidence:

- `docs/task_workflow/evidence/nv-ordinary-q4-qkv-full-20260825/microgate-current-r7.json`
- `docs/task_workflow/evidence/nv-ordinary-q4-qkv-full-20260825/production-profile-accounted.json`
- `docs/task_workflow/evidence/nv-ordinary-q4-qkv-full-20260825/production-wall-r7.json`
