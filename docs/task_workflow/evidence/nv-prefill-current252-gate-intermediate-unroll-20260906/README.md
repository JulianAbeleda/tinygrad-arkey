# Current252 gate intermediate-unroll screen

The actual generated gate/up Stream-K main was screened at U6, U10, and U12 to
probe the register/ILP interval between the closed U4/U8/U16 powers-of-two sweep.
Every arm passes the current252 structural census, token 198, and recurrent
correctness.

With three warmups and R9, medians are 56.612738, 57.632307, and 60.482363 ms.
The samples descend as clocks warm, so these medians are screening values rather
than strict endpoint comparisons. Their late minima are 54.761712, 55.847137,
and 58.196185 ms, all slower than the fully warmed U8 authority at 53.302259 ms.
No candidate advances to a full bracket. U8 remains selected; the extra accepted
unroll values remain research-only controls.
