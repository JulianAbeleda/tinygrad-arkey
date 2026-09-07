# Q-only 64x32 compiler geometry transfer

The generated K/V winner's 64x32x64, four-warp geometry was transferred to the
N=4096 attention-Q compiler gate behind `NV_COMPILER_Q4_Q_K_GEOMETRY=1`.
The real `blk.0.attn_q.weight` gate passes all 2,097,152 outputs against the
static arithmetic oracle (max abs 0.00048828125, mean 9.135e-6), preserves the
canonical weight and Q8 record, and has zero unwritten outputs or local
loads/stores.

Executed ProgramInfo is grid `(128,8,1)` (1024 CTAs), block `(32,2,2)`, 7,680
bytes shared, eight static IMMA sites and two barriers. The harness previously
reported a hard-coded geometry; it now serializes executed ProgramInfo and the
candidate LDS window.

The transferred geometry measures 368.866 us median / 363.036 us minimum. The
retained current Q gate measures 238.639 / 232.357 us under the same gate
harness (`nv-compiler-q4k-qo-20260828`). The transfer therefore regresses about
130.2 us (54.6%) per call and is rejected before tile-Q8 or full-model work.
The static oracle timing in the JSON is a correctness reference, not the
selected generated-Q comparator.
