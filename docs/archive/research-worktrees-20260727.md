# Tinygrad research worktree archive

Created: 2026-07-27

This ref preserves reachability for research branches and detached worktree tips removed after the 14B decode campaign.
Its file tree is production master plus this manifest; parent commits retain the research histories without merging them into production.

## Branch tips

| branch | commit | subject |
|---|---|---|
| `chore/luna-profiler-harness` | `b1a28b70a4221c5e255a8d03210095baa6ecd5df` | [luna] align profiler index positive-control test |
| `codex/kfd-profiler-preflight` | `626cde9da1442c07199afa2d726b2d190294e326` | [docs] add KFD profiler preflight |
| `feature/14b-decode-ctx128-and-depth-decay` | `4d4600c725d3cb812e9d5439b45771a759b5d8e0` | [archive] preserve G5 decode and KFD observability work |
| `fix/14b-short-prefill-vector-store` | `c9c19090b0bd400795dd8c8fc4528cc7c6995df5` | [archive] preserve launch observer and PMC probes |
| `integration/ctx128-production-safety` | `c965ccbd3ea13dfc3189f07a3bf7511cf15cb840` | [docs] record ctx128 production integration scope |
| `luna-14b-profile-binding` | `54070758c41fbbc2d190f8647b43364204ddcbca` | [prefill] fail closed on model profile mismatch |
| `luna-authority-diff` | `2f5301d06c82ea8eca3e76cdc20ab7c0ae582e52` | [docs] compare tinygrad authority lifecycle artifacts |
| `luna-campaign-closeout` | `42fc0d9138e2817016daac7aaf2226bcd4ae7bab` | [docs] close blocked Luna campaign |
| `luna-ctx128-recovery` | `1b4c11ee088c77c96b1648d9e799a8ed0c032e21` | [docs] record blocked ctx128 repair design review |
| `luna-decode-tree-diff` | `3dda0567a3d12d3fa0031f6e6395e2e0ba857565` | [docs] record LUNA decode tree diff |
| `luna-fix-ctx128-codegen-path` | `f54f06e27e3db50f287480705d8c668106712fce` | [codegen] scalarize mixed output projection stores |
| `luna-fix-ctx128-compile` | `e9ba0b42f1b929862e14b1c784352a00fdb34853` | [codegen] lower scalar output projection stores |
| `luna-foundation` | `9f76c5f1a8e5459f54867647b795235270c21bc1` | [bench] add Luna foundation censuses and manifest collector |
| `luna-llama-source-map` | `c43fdcc23bb3f37bb013d9ac7ff674edb93dfe89` | [docs] map llama lifecycle quant KV and HIP backend |
| `luna-llama-trace-128` | `fa0b5f1abc38603ae023674e8cf2d66c9871a6ee` | [bench] retain LUNA-021 ctx128 tool-failure artifacts |
| `luna-prefill-compile-diff` | `cd63052ecab21cafab373093da57699351df3640` | [docs] diagnose decode first-prefill compile boundary |
| `luna-prefill-postgraph` | `d41c896e7f850f86999c53923cb2236335aa3c5a` | [bench] document single-call prefill matrix |
| `luna-prefill-stage-instrument` | `f61e44804f690eb42009d6ae64550d17d1c8fd8e` | [bench] add prefill authority lifecycle markers |
| `luna-route-observer` | `a9218aab4f31e8c0645ad705c892df6fa38c94d8` | [llm] add opt-in decode route uop observer |
| `luna-smoke-artifact-boundary` | `88afc520cea71112d0d4db078b7e72c14dee7010` | [bench] make prefill smoke artifact completion observable |
| `luna-static-synthesis` | `9afc642b111f17557ae228d5f4307b14759a8e14` | [docs] synthesize blocked Luna static comparison |
| `luna-tinygrad-worker-smoke` | `1476f5fb4be0da338377aa63ac641e911baf2ee9` | [archive] preserve Luna worker smoke artifacts |
| `luna-warmup-static` | `44fb67684ce7d2261607a258760545aff7c738ec` | [bench] trace decode warmup lifecycle stages |
| `luna/profiler-llama128` | `8d4bbe6e672ccf779c6415f275035be09bfba378` | [docs] capture llama ctx512 and ctx4096 dispatch traces |
| `luna/tinygrad-g5-trace` | `7a59e699512a49e7e6e59c54f946c9b9504614a0` | [bench] record 14b decode rocprof trace failure |
| `luna/tinygrad-profiler-prep` | `47eb2ad0d3637db7766c5c3da765f070ef1a4530` | [docs] prepare LUNA rocprofv3 tinygrad reopening |
| `luna/tinygrad-workload` | `f0eba36bd27a0ff545702c44e7d111ed4490773a` | [docs] map tinygrad decode routes and fixture |
| `luna/worker-harness-diagnosis` | `3b58c8535c1465b2ac0392defa5ab97986c25df9` | [docs] diagnose LUNA-030 worker artifact conflict |
| `prefill-authority-vs-bench` | `585bc028d8c8c2b4bb4a54907df580c0f15fcdb4` | [bench] compare prefill authority and attention replay |
| `recovery/luna-post-reset-recovery` | `c6aeb1344451efbd935425c9e9a3395adb61d8dd` | [bench] make prefill smoke artifact completion observable |
| `review/luna-g5-decay` | `cf958dbd53714ba2a05e9a419e7107b8a1bab8e9` | [docs] record G5 depth-decay candidate review |

## Detached worktree tips

| path | commit |
|---|---|
| `/home/ubuntu/tinygrad-0716` | `dbec46337650f0714c1bb0c839f0ed0db3679ebe` |
| `/home/ubuntu/worktrees/luna-tinygrad-trace-8b128` | `0bfe3b5b8315d78ae277276a645b92372f825099` |
