# TC checklist (NV sm_120, Nemotron BF16) -- final
1. [x] Ceilings: bf16 mma 248.0-251.8 TF, f16 247.4-251.8; read 1693 GB/s, copy ~1495 GB/s.
2. [x] Qwen sm120 set already promoted (98ed11fa5 / dff21c0cb: 6.1k tok/s pp512); not redone.
3. [x] bf16 precontract gate (landed inside ff1367a88), N pad, row chunking.
4. [x] v1 reuse 9a2c02bdc; M=16 lowering aa1cabc73; search routes bd3d3ec7f; 48-route v3 24e528f88.
5. [x] hook 75040fbcc/ad88f2c3c/e27e91bd4; nemotron_h.py edit DENIED -> patch at scratchpad/nemotron_h_integration.patch
   e2e (scratch tree w/ patch): B=64 184.2 -> 153.6 ms/step; prefill L=2048 1.15 -> 0.70 s; L=10000 OOM (+5.7GB resident); B=128 OOM both arms.
6. [x] report
