# Current252 gate shared-load staging negatives (2026-09-06)

Both candidates preserve the current252 census, canonical weights, 20 KiB source shared tile, and pass 20/20 exact replay. Moving all 320 shared scalar loads to their pack reduces the main from 255 registers / 88 stack bytes to 254 registers / zero stack, but regresses the full wall to 56.741611 ms. Moving only the 192 fragment bytes retains 255 registers / 88 stack bytes and measures 53.215871 ms, without clearing the frozen 0.5 ms investment threshold against adjacent current252 controls. Both modes remain default-off.
