# Q8 payload vector-load closure (2026-09-07)

The selected generated gate/up body has eight scalar `uint32` Q8-record payload loads in two contiguous four-dword runs. Both effective bases are 16-byte aligned: every term in `alu88` is divisible by four dwords, `alu6` is divisible by four, and `alu90` is divisible by sixteen; offsets 4 and 2308 preserve 16-byte alignment.

A `uint4` replacement reduced static global loads from 152 to 104 but failed one-call numerical correctness (`max_abs=0.718`), so its apparent timing was discarded. The narrower `uint2` form maps x/y exactly to scalar dwords `[alu91+4,+5]`, `[+6,+7]`, and the corresponding `[+2308..+2311]` run. It was bit-exact in an alternating one-call R15 and read-only, with median 324.701 us versus 336.152 us control (+11.451 us/call).

The required composed replay gate rejects it. Eight calls in one graph repeated for ten cycles produced exact control every cycle, but the candidate corrupted cycles 3, 5, and 7 (`max_abs` 0.479847, 0.558603, 0.572678). This is the same composition-sensitive vector-load hazard seen for aligned metadata carriers. No production transform or enable path is retained. Further shared-publication vector aliases must first explain this multi-call nondeterminism.

## Cache/program identity audit

The injected control and first candidate had the same `ProgramInfo.name` (`q4_qo_streamk`) but distinct source, binary, and UOp hashes. To exclude a name-keyed runtime collision, the candidate was rebuilt with a distinct exported symbol and ProgramInfo name `q4_qo_streamk_q8uint2`. Its composed replay still failed cycles 4, 5, and 7 (`max_abs` 0.440012, 0.635214, 0.411940), while all control cycles stayed exact. Cache/program identity therefore does not explain the composition hazard, and the rejection stands.
