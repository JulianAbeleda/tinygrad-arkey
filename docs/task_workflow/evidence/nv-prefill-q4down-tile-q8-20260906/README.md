# Generated Q4 down tile-major Q8 transfer

The retained tile-major Q8 carrier and a K=12288 specialization of the FP16 DS4 producer were connected to the exact generated Q4-down Stream-K path. Static contracts passed: canonical packed Q4 input, 512x12288 activation, 144-byte tile-major records, generated main source, 170-owner launch and existing fixup map.

The first complete current252 capture did not reach its timing gate. HCQ synchronization timed out after 30 seconds with the timeline waiting for value 3148 at 3146. The failure occurs after construction and submission of the composed route, making the K=12288 tile-major main incompatible with the current Stream-K/barrier lifecycle. No timing or correctness credit is claimed. The experiment is reverted; Q4 down retains its qualified flat record.
