# Current252 gate parity-bank double buffer

The candidate expands the generated gate/up main's 20 KiB shared tile to two
K-parity banks and removes only the pre-write recycle barrier. The per-epoch
post-write publish barrier remains. Alternating banks plus that barrier prevent
a faster warp from overwriting storage still read by the preceding epoch.
All loads, IMMA operations, Stream-K ownership, and fixup order are unchanged.

The candidate is structurally correct, replays token 198 for 20/20 cycles, and
is bit-exact to the single-bank route. Single/double/single medians are
53.302259, 53.506390, and 52.979670 ms. The candidate regresses 0.365426 ms or
0.688% against the mean controls. The likely occupancy cost of 40 KiB shared
outweighs removing one barrier per K64 epoch. The mechanism stays default off.
