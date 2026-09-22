# Current252 gate IMMA/update interleave (2026-09-06)

The candidate groups eight unique WMMA results and eight unique weight-scale conversions with their 16 unchanged accumulator expressions. It retains the 20 KiB source shared tile and reduces the main from 255 registers / 88 stack bytes to 225 registers / zero stack.

The control/candidate/control medians are 52.832851/52.734764/52.960819 ms. Against the 52.896835 ms mean control, the candidate wins 0.162071 ms. This does not clear the frozen 0.5 ms population investment threshold, so the mechanism remains default-off. The candidate retains the 252 generated mains/producers, zero overlays, bit-identical full logits/token against the closing control, and 20/20 exact recurrent replay.
