# Flash launch-bound causal result

## Verdict

The two-argument CUDA launch bound is a causal explanation of the matched
Flash score-body gap, not merely a favorable tinygrad compiler knob.

Llama ships its decode Flash kernel with
`__launch_bounds__(128, 1)`.  Removing only the second argument reduces the
compiler's register budget, destroys both K and V load walls, and slows the
kernel while preserving the grid, source arithmetic, load count, global-load
sectors, shared-memory allocation, and zero-spill status.  Adding the same
contract to tinygrad raises its register budget and makes its K schedule
llama-like.  At normalized S6/768 service, that moves tinygrad from slightly
behind llama to slightly ahead on the score primitive.

The compiler contract alone did not convert at token wall.  A graph-seam
rotation from an initial cap of 32 to 33 was then tested and promoted as the
paired capture contract; that construction books the primitive saving.  This
does **not** close the complete Flash island. Tinygrad and llama both emit a
separate partial-combine kernel, but llama services the matched partial state
faster. Tinygrad's compiler-generated V schedule also
remains staggered; a forced literal V wall is slower and is rejected.

## Same-mechanism ablation

| arm | registers/thread | spills | first 16 K loads span | next 16 V loads span | cold NCU time |
|---|---:|---:|---:|---:|---:|
| llama native `minBlocks=1` | 158 | 0 | 32 instructions | 40 instructions | 6.144 us |
| llama one-argument ablation | 96 | 0 | 279 instructions | 266 instructions | 7.808 us |
| tinygrad current S6 | 56 | 0 | 226 instructions | 310 instructions | 6.400 us |
| tinygrad S6 with `minBlocks=1` | 96 | 0 | 29 instructions | 331 instructions | 5.888 us |

The llama ablation retains 417,792 global-load sectors.  The schedule change,
not fewer requests, produces the service loss.  The exact tinygrad candidate
is bit-identical and also retains its bytes and sectors.

The llama hot A/B independently repeats the direction.  Across seven runs, a
two-Flash sequence moves from 8.639 us with the native contract to 8.813 us
without it.  Under the read-conditioned sequence it moves from 59.819 to
61.395 us.  These sequence results are supportive; the per-kernel NCU and
SASS comparison is the causal authority.

## Normalized service comparison

At S6/768, the contract changes tinygrad cold service from 6.400 to 5.888 us.
The matched llama native arm is 6.144 us.  Thus the construction crosses the
comparator rather than merely narrowing the gap:

```text
tinygrad current       6.400 us
llama native           6.144 us
tinygrad + contract    5.888 us
```

At S8/1024, llama preserves the same compact 32/40-instruction K/V cadence.
Its native cold NCU service is 6.656 us versus 7.552 us after ablation.  This
confirms that the mechanism is not peculiar to one partition count.

## Why the compiler does this

NVIDIA documents the second launch-bound argument as the minimum blocks per
multiprocessor.  It lets the compiler derive a register ceiling and, when
headroom exists, spend additional registers to reduce instructions or hide
per-thread latency.  NVIDIA's optimization guidance also notes that higher
instruction-level parallelism can cover latency at lower occupancy and must be
judged per kernel.  This is exactly the observed transformation here: more
live load destinations, no spills, compact load issue, and faster cold
service—not a byte reduction.

- CUDA Programming Guide, launch bounds:
  https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html
- CUDA C++ Best Practices Guide:
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- Nsight Compute Profiling Guide, scoreboard interpretation:
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

## Token conversion status

The primitive ceiling at S6 is approximately 0.512 us per layer, or 18.4 us
per 36-layer token before graph conversion.  The exact S8 production primitive
showed a larger 0.608-us cold recovery, a 21.9-us/token no-loss ceiling.

The first production brackets were directionally positive but drift-limited:

- control/candidate/control: candidate 6.7 us/token faster than the control
  midpoint, but it did not beat the faster first control;
- candidate/control/candidate: candidate midpoint 26.3 us/token faster than
  control, with equal token hashes, but the candidate arms drifted by 24.2 us.

The longer 24-token, nine-repetition control/candidate/control wall resolves
that ambiguity and rejects the current production application:

| arm | median us/token |
|---|---:|
| control A | 4,077.384 |
| launch-bound candidate | 4,095.214 |
| control C | 4,079.087 |

The candidate is 16.978 us/token slower than the control midpoint, or about
-1.02 tok/s in that bracket.  The controls differ by only 1.70 us and all
token hashes match.  This closed the program-only application and motivated
the graph-boundary rotation below; it is retained as the negative control, not
as the final promotion verdict.

### Where conversion is lost

A matched 57-steady-replay graph profile proves that the body saving survives
into the production kernel rows:

| production row | control | candidate | delta |
|---|---:|---:|---:|
| 36 score calls | 222.368 us | 204.640 us | **-17.728 us** |
| 36 combine calls | 50.144 us | 50.112 us | -0.032 us |
| all-kernel node sum | 3,949.856 us | 3,932.832 us | **-17.024 us** |
| device union | 3,949.500 us | 3,933.250 us | **-16.250 us** |
| first-to-last device span | 4,417.750 us | 4,482.000 us | +64.250 us |

Thus neither the score body nor combine consumes the saving.  It lands in
idle/graph pacing.  Thirty-five of the 36 score→combine pairs are adjacent in
the same graph group and retain zero measured gap.  One score call ends a
graph batch and its combine begins the next batch; that cross-group idle gap
is about 196 us in control and 219 us in the candidate profile.  A separate
producer→normalization graph-boundary gap also expands.  The latter movement
cannot be assigned solely to Flash from separate-process profiles, but the
accounting boundary is firm: kernel union improves while device span and the
clean token wall regress.

### Graph-seam discriminator and promotion

The capture splitter starts at 32 kernels and doubles each subsequent graph
cap.  In the retained token graph, the first 32-kernel seam falls between one
Flash score and its combine.  Rotating only the initial cap to 33 keeps that
pair together.  A seven-repetition control/candidate/control discriminator at
cap 33 first recovered 4.571 us/token with exact tokens.  The longer
control/candidate/control run drifted by 37 us between controls and was
inadmissible, so it was not used for promotion.

The reverse candidate/control/candidate confirmation at cap 33 passed: both
candidate arms beat the enclosed control and the candidate midpoint recovered
11.563 us/token.  The final installed test then compared the complete paired
policy—candidate cap 33 plus launch bounds—against the retained cap-32 control:

| installed arm | median us/token |
|---|---:|
| candidate A | 4,074.658 |
| cap-32 control | 4,099.756 |
| candidate C | 4,092.946 |
| candidate midpoint | **4,083.802** |

Both candidate arms beat control, all token hashes match, and the installed
midpoint recovers **15.954 us/token**, or **0.953 tok/s** in that bracket.  The
candidate-arm spread is retained in the result; the midpoint, not the fast
arm, is the booked authority.  The policy is therefore promoted as one
capture-scoped contract.  Applying launch bounds without the seam change, or
changing the global graph cap without admission, remains closed.

## Why the forced V wall loses

Forcing all V destinations live before their first consumer is not the next
lever.  At S6 scalar-Q it changes the launch-bound candidate from 5.888 to
6.272 us cold and regresses hot timing.  This is not caused by bytes, load
count, spills, or occupancy:

| field | launch-bound only | forced V wall |
|---|---:|---:|
| registers/thread | 96 | 96 |
| spills | 0 | 0 |
| DRAM read bytes | 3.183 MB | 3.183 MB |
| global-load sectors | 491,520 | 491,520 |
| dynamic instructions | 797,760 | 798,528 |
| cold long scoreboard | 61.60% | 62.96% |
| cold duration | 5.888 us | 6.272 us |

The difference is load placement.  The ordinary typed C++ loads are movable.
Ptxas progressively hoists V requests across independent score and softmax
work: most values have roughly 230--445 instruction slots from load to first
use, with the late values retaining roughly 44--154 slots.  The inline-PTX
construction fixes the sixteen V loads as a late compact group.  Its earliest
values have only roughly 52--98 slots before use, and it forfeits the earlier
load/compute overlap.

Three discriminators establish that this is a compiler-visibility wall rather
than a useful V topology:

1. moving the forced preload before the K source loop does not move the V wall
   earlier in SASS and worsens cold service to 6.752 us;
2. splitting the monolithic asm into sixteen volatile statements produces the
   exact same machine-code hash and remains slow;
3. removing `volatile` from the split statements again produces the exact same
   machine-code hash and remains slow.

Thus the compiler needs typed load/dependency information to distribute the V
requests.  Llama's compact V wall comes with different probability ownership,
packed PV arithmetic, and live ranges; the visual adjacency of its loads is
not independently portable.  The forced-PTX result closes that construction,
not every V-scheduling theory.

The remaining Flash work is consequently separated:

1. retain the paired launch-bound/graph-seam promotion and its rollback gate;
2. account for tinygrad's slower separate combine and its handoff versus
   llama's separate `flash_attn_combine_results`;
3. only revisit V scheduling through a topology-preserving construction, not
   another forced preload spelling.

## Evidence

- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/llama-launch-bounds-timing-r7.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/llama-launch-bounds-first-cold-r1.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/llama-s8-tc1024-launch-bounds-r7.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/tiny-s6-scalar-lb-r11.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/tiny-s6-scalar-vwall-r11.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/tiny-s6-scalar-vearly-r11.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/tiny-s6-scalar-vsplit-r11.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/tiny-s6-scalar-vmovable-r11.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/s8-lb-scalarq-final.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/production-wall-r9.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/production-wall-cac-r9.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/production-wall-count24-r9.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/production-wall-jit33-r7.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/production-wall-jit33-count24-r9.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/production-wall-jit33-aca-count24-r9.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/installed-flash-load-schedule-aca-r9.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/profile-control.json`
- `docs/task_workflow/evidence/nv-flash-load-wall-20260826/profile-candidate.json`
- `extra/llm_research/decode/nv_llama_flash_launch_bounds_ab.py`
- `extra/llm_research/decode/nv_flash_load_wall_probe.py`
