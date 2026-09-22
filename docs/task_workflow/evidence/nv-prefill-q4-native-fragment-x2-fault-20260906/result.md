# Q4 B/x2 outer-UNROLL seam result

A narrow, default-off prototype added a phase-local Q4 K32 `PackedFragmentSpec`, retained a marker through the B CONTRACT, removed only its descriptor element axes during outer-UNROLL expansion, and lowered each remaining char8 carrier through `ldmatrix.x2` before devectorization. Six focused native-fragment structural tests passed. The env-on Q gate then passed UOp verification and NVRTC compilation and emitted eight x2 sites, advancing beyond the earlier nested-STACK construction failure.

Execution faulted all SMs at the x2 instruction (`ESR=4`, warp PC `0x200e801510`) and timed out after 30 seconds, so no arithmetic or timing is claimed. The emitted dword address was `alu12 + {2560,2568,...}`, where `alu12=(lidx2*640)+((lidx0>>2)*20)+(lidx0&3)`. The `(lane&3)` term makes adjacent elected addresses differ by four bytes, violating the cooperative row alignment expected by ldmatrix.

This is an operand-layout incompatibility, not an out-of-range LDS address. The proven Q4 native oracle stages Q4 as MMA A/x4 with `(lane&15)*76+(lane>>4)*4`; the proven B/x2 K16 substrate uses `(lane&15)*4+(lane>>4)*2`. The current generated gate stages Q4 as MMA B under a scalar descriptor contract. There is no evidence that substituting either oracle lane formula preserves its B fragment. A correct next hook must restage/transpose the Q4 operand under an explicit descriptor mapping, or swap its MMA operand role, and prove all 32 lanes and both K32 substeps before execution.

The prototype was fully reverted. The default generated route and renderer are unchanged.
