# Q4-down x4 plus packed publication: rejected

The exact Q4 publication transfer replaced 32 scalar nibble stores with eight aligned uint stores. Against the existing x4/interleave Q4-down body, SASS changed PRMT 612 to 396 and LOP3 489 to 265; IMMA 288, LDG 234, LDS 432, STS 72 and zero spill were unchanged. One-call R15 was bit exact/read-only with one producer/main/fixup and improved 363.924 to 356.590 us.

The deep model smoke passed 252/234/126 topology, canonical weights, zero copies/overlays, token 198 and exact three-cycle replay. Stable R9 controls were 44.486996 and 44.760981 ms; their midpoint 44.6239885 versus candidate 44.313110 is a 0.310879 ms median win. Minimum midpoint 44.343481 versus candidate 44.117402 is a 0.226079 ms win. Both miss the frozen 0.5 ms promotion threshold, so all integration code was reverted.
