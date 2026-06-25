# How the tinygrad-arkey fork actually started

## Phase 0: before any code, just getting a machine that could run this (through May 29, 2026)

The fork really began as a hardware problem, not a software one.

My first plan was to use the Mac mini. The catch was I couldn't even connect to it. It needed a newer PCIe setup, a UTG4 adapter, and Thunderbolt 4 just to talk to the thing. Once I got past that, the BIOS got in the way. I had to turn on security permissions on both the Mac and in the BIOS before it would come up at all. Eventually it worked, but it was absurdly slow, and it kept shutting itself off whenever it went idle. So it was never going to be the real machine.

That's when I sacked the PC and turned it into a dedicated Linux workstation instead. More BIOS headaches followed. The machine acted like it didn't recognize the hardware, and I went down a whole security rabbit hole around GART trying to figure out why. Those exact days are still timestamped in the repo: the `ubuntu-amdgpu-wedge` snapshot is dated May 28, 2026 at 22:50, and the `linux-mmhub-gart-snapshot` is May 29, 2026 at 11:25. The real cause turned out to be small. It was a flag I had disabled myself earlier while troubleshooting something else, and it was still off. Fable helped me spot it. It wasn't some deep failure, it was my own old workaround still sitting there.

This is the part that never shows up in the git log, but it's the whole foundation. The entire project assumes I have full low-level access to the GPU, and that access had to be earned first, physically and electrically, before a single kernel mattered.

There was also an idea that justified pushing through all of it. I tied it back to Alec Radford's point, roughly that for capability to emerge I mostly just need the right substrate. That reframed the goal. Instead of fighting the abstraction, expose the entire GPU and see what's actually possible. DeepSeek had already pulled this off on Nvidia, which was proof the substrate-first approach works, and it meant the Tensile problem was solvable rather than a dead end.

## May 21, 2026: first real baseline

The first Q4_K baseline benchmark landed. The point was getting an honest "before" number, even while the GART and BIOS battle above was still going on in parallel.

## June 11, 2026: build the quantized decode paths

The first Q6_K runtime primitive went in (the ffn_down path), and the custom Q4_K and Q6_K decode primitives started taking real shape. This is the jump from toy speeds toward something that felt like real local inference.

## June 15, 2026: decode gets fast and ships

The Q6_K and Q4_K decode primitives turned on by default, recording a 2.2x decode speedup with exact output. Same day, I corrected the synthesis and proved the post-fix decode was GPU-bound, only about 3% host, not a runtime overhead problem. First big win that actually shipped.

## June 16, 2026: study llama, and crack Tensile

I profiled what llama actually does at the kernel level and found its prefill GEMM is rocBLAS Tensile WMMA, MT128x128 with 25KB LDS. After the tensile.co deep dive to really understand the kernel, Tensile stopped being a mystery and became something I could reason about. llama turned into evidence about what was missing, not a thing to blindly copy.

## June 19, 2026: prefill, the hard way

The Route A/A3 prefill experiments, all four levers of the parametric LDS GEMM (double buffering, occupancy, padding, BK), got refuted. Isolated GEMM speed just didn't survive once it hit the whole prefill path. Writing down what didn't transfer was the actual point.

## June 22, 2026: the breakthrough

The real decode wall wasn't the kernel at all. The cache was being quietly re-materialized on every read. Reading the whole cache buffer instead removed that hidden tax, and the result was 13 to 19 percent faster, byte for byte identical, with decode at parity with or ahead of llama.cpp. It was a dataflow insight, not faster assembly.

## June 25, 2026: make the search honest (where things are now)

Warp-level GEMV got promoted to default for the attn q/o path, worth another 1.6%, and the same day the codegen warp-reduce lowering started handling mixed serial and group reduce for real GEMV K. The focus has shifted to teaching the compiler to express these wins on its own. The substrate is exposed now. The open question is whether the machine can find the wins by itself.

---

The short version: through May 29 it was a Mac mini I couldn't connect to and a sacked, rebuilt Linux workstation I had to fight the BIOS and GART to wake up. The first baseline hit May 21, decode shipped fast on June 15, Tensile got cracked open June 16, and the June 22 cache-identity fix put quantized decode on par with llama.cpp.
