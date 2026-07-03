#!/usr/bin/env python3
import argparse, pathlib

BASE_TRACE = r'''#!/usr/bin/env bpftrace

/*
 * Generated Linux-good PSP deep trace.
 * Fixed offsets from Ubuntu 6.8.0-117 /sys/kernel/btf/amdgpu:
 *   struct psp_context.fw_pri_mc_addr: 80
 *   struct psp_bin_desc.size_bytes: 8
 *   struct psp_bin_desc.start_addr: 16
 */

BEGIN
{
  printf("psp deep trace start\n");
  printf("columns: time event details\n");
}

kprobe:psp_hw_start
{
  printf("%llu psp_hw_start enter psp=%p\n", nsecs, arg0);
}

kretprobe:psp_hw_start
{
  printf("%llu psp_hw_start ret=%d\n", nsecs, retval);
}

kprobe:psp_v13_0_bootloader_load_component
{
  $fw_pri_mc = *(uint64 *)(arg0 + 80);
  $size = *(uint32 *)(arg1 + 8);
  $start = *(uint64 *)(arg1 + 16);
  printf("%llu bl_load enter psp=%p desc=%p cmd=0x%x fw_pri_mc=0x%llx c2p36=0x%llx size=0x%x start=%p\n",
    nsecs, arg0, arg1, arg2, $fw_pri_mc, $fw_pri_mc >> 20, $size, $start);
}

kretprobe:psp_v13_0_bootloader_load_component
{
  printf("%llu bl_load ret=%d\n", nsecs, retval);
}

kprobe:psp_v13_0_wait_for_bootloader
{
  @wait_bl_start[tid] = nsecs;
  @wait_bl_reads[tid] = 0;
  printf("%llu wait_bl enter psp=%p\n", nsecs, arg0);
}

kretprobe:psp_v13_0_wait_for_bootloader
{
  $start = @wait_bl_start[tid];
  $reads = @wait_bl_reads[tid];
  printf("%llu wait_bl ret=%d duration_ns=%llu reads=%llu\n", nsecs, retval, $start ? nsecs - $start : 0, $reads);
  delete(@wait_bl_start[tid]);
  delete(@wait_bl_reads[tid]);
}

kprobe:psp_v13_0_memory_training_send_msg
{
  $fw_pri_mc = *(uint64 *)(arg0 + 80);
  printf("%llu mem_train_msg psp=%p msg=0x%x fw_pri_mc=0x%llx c2p36=0x%llx\n",
    nsecs, arg0, arg1, $fw_pri_mc, $fw_pri_mc >> 20);
}

kretprobe:psp_v13_0_memory_training_send_msg
{
  printf("%llu mem_train_msg ret=%d\n", nsecs, retval);
}

kprobe:amdgpu_gart_map
{
  @gart_offset[tid] = arg1;
  @gart_pages[tid] = arg2;
  @gart_dma[tid] = arg3;
  @gart_flags[tid] = arg4;
  @gart_dst[tid] = arg5;

  if (arg1 == 0x700000 && arg2 == 256) {
    $dma0 = *(uint64 *)(arg3);
    $dmalast = *(uint64 *)(arg3 + ((arg2 - 1) << 3));
    printf("%llu gart_map enter adev=%p offset=0x%llx pages=%d dma=%p dma0=0x%llx dma_last=0x%llx flags=0x%llx dst=%p\n",
      nsecs, arg0, arg1, arg2, arg3, $dma0, $dmalast, arg4, arg5);
  }
}

kretprobe:amdgpu_gart_map
{
  $offset = @gart_offset[tid];
  $pages = @gart_pages[tid];
  $dst = @gart_dst[tid];
  if ($offset == 0x700000 && $pages == 256) {
    $first_idx = $offset >> 12;
    $last_idx = $first_idx + $pages - 1;
    $pte0 = *(uint64 *)($dst + ($first_idx << 3));
    $ptelast = *(uint64 *)($dst + ($last_idx << 3));
    printf("%llu gart_map ret offset=0x%llx pages=%d flags=0x%llx first_idx=0x%llx last_idx=0x%llx pte0=0x%016llx pte_last=0x%016llx\n",
      nsecs, $offset, $pages, @gart_flags[tid], $first_idx, $last_idx, $pte0, $ptelast);
  }
  delete(@gart_offset[tid]);
  delete(@gart_pages[tid]);
  delete(@gart_dma[tid]);
  delete(@gart_flags[tid]);
  delete(@gart_dst[tid]);
}
'''

REG_TRACE = r'''
kprobe:amdgpu_device_wreg
{
  if ((arg1 >= 0x16040 && arg1 <= 0x160bf) || (arg1 >= 0x1a700 && arg1 <= 0x1a800) || (arg1 >= 0x1a8d0 && arg1 <= 0x1a900) ||
      arg1 == 0xc0 || arg1 == 0x106 || arg1 == 0x107 || arg1 == 0x12d || arg1 == 0x12e ||
      arg1 == 0xcf6e || arg1 == 0xd102 || arg1 == 0xd114 || arg1 == 0xe8ad) {
    printf("%llu wreg adev=%p reg=0x%x val=0x%x\n", nsecs, arg0, arg1, arg2);
  }
}

kprobe:amdgpu_device_rreg
{
  if ((arg1 >= 0x16040 && arg1 <= 0x160bf) || (arg1 >= 0x1a700 && arg1 <= 0x1a800) || (arg1 >= 0x1a8d0 && arg1 <= 0x1a900) ||
      arg1 == 0xc0 || arg1 == 0x106 || arg1 == 0x107 || arg1 == 0x12d || arg1 == 0x12e ||
      arg1 == 0xcf6e || arg1 == 0xd102 || arg1 == 0xd114 || arg1 == 0xe8ad) {
    @rreg_reg[tid] = arg1;
  }
}

kretprobe:amdgpu_device_rreg
/@rreg_reg[tid]/
{
  printf("%llu rreg reg=0x%x val=0x%x\n", nsecs, @rreg_reg[tid], retval);
  if (@wait_bl_start[tid] && @rreg_reg[tid] == 0x16063) {
    @wait_bl_reads[tid] = @wait_bl_reads[tid] + 1;
    printf("%llu wait_bl_rreg dt_ns=%llu read=%llu reg=0x%x val=0x%x\n",
      nsecs, nsecs - @wait_bl_start[tid], @wait_bl_reads[tid], @rreg_reg[tid], retval);
  }
  delete(@rreg_reg[tid]);
}
'''

BO_TRACE = r'''
kprobe:amdgpu_bo_create_kernel
{
  @bo_size[tid] = arg1;
  @bo_align[tid] = arg2;
  @bo_domain[tid] = arg3;
  @bo_ptrp[tid] = arg4;
  @bo_gpu_addrp[tid] = arg5;

  if (arg1 == 0x100000 || arg2 == 0x100000) {
    printf("%llu bo_create_kernel enter adev=%p size=0x%llx align=0x%llx domain=0x%x bo_ptrp=%p gpu_addrp=%p\n",
      nsecs, arg0, arg1, arg2, arg3, arg4, arg5);
  }
}

kretprobe:amdgpu_bo_create_kernel
/@bo_size[tid]/
{
  $size = @bo_size[tid];
  $align = @bo_align[tid];
  $domain = @bo_domain[tid];
  $bo_ptrp = @bo_ptrp[tid];
  $gpu_addrp = @bo_gpu_addrp[tid];

  if ($size == 0x100000 || $align == 0x100000) {
    $gpu_addr = $gpu_addrp ? *(uint64 *)$gpu_addrp : 0;
    $bo = $bo_ptrp ? *(uint64 *)$bo_ptrp : 0;
    printf("%llu bo_create_kernel ret=%d size=0x%llx align=0x%llx domain=0x%x bo=%p gpu_addr=0x%llx\n",
      nsecs, retval, $size, $align, $domain, $bo, $gpu_addr);
  }

  delete(@bo_size[tid]);
  delete(@bo_align[tid]);
  delete(@bo_domain[tid]);
  delete(@bo_ptrp[tid]);
  delete(@bo_gpu_addrp[tid]);
}
'''

OPTIONAL_PROBES = [
  "psp_v13_0_init_microcode", "psp_v13_0_sw_init", "psp_v13_0_hw_init", "psp_v13_0_hw_start",
  "psp_v13_0_bootloader_load_kdb", "psp_v13_0_bootloader_load_spl", "psp_v13_0_bootloader_load_sysdrv",
  "psp_v13_0_bootloader_load_sos", "psp_v13_0_wait_for_vmbx_ready",
  "psp_v13_0_memory_training", "psp_v13_0_memory_training_init",
  "psp_v13_0_ring_create", "psp_v13_0_ring_stop", "psp_v13_0_ring_destroy",
  "psp_v13_0_ring_get_wptr", "psp_v13_0_ring_set_wptr",
  "psp_tmr_init", "psp_tmr_load", "psp_load_toc", "psp_load_smu_fw",
  "psp_load_non_psp_fw", "psp_execute_ip_fw_load", "psp_ring_cmd_submit",
]

def symbols_from_kallsyms(path:pathlib.Path) -> set[str]:
  symbols = set()
  with path.open("r", errors="replace") as f:
    for line in f:
      parts = line.split()
      if len(parts) >= 3: symbols.add(parts[2])
  return symbols

def optional_entry_ret(sym:str) -> str:
  return f'''
kprobe:{sym}
{{
  printf("%llu {sym} enter arg0=%p arg1=0x%llx arg2=0x%llx arg3=0x%llx\\n", nsecs, arg0, arg1, arg2, arg3);
}}

kretprobe:{sym}
{{
  printf("%llu {sym} ret=%d\\n", nsecs, retval);
}}
'''

def main():
  parser = argparse.ArgumentParser(description="Generate symbol-safe Linux AMD PSP deep bpftrace script")
  parser.add_argument("--kallsyms", default="/proc/kallsyms")
  parser.add_argument("--out", required=True)
  parser.add_argument("--symbols-out", required=True)
  parser.add_argument("--optional-probes", action="store_true", help="include broad PSP optional entry/return probes")
  args = parser.parse_args()

  symbols = symbols_from_kallsyms(pathlib.Path(args.kallsyms))
  required = ["psp_hw_start", "psp_v13_0_bootloader_load_component", "psp_v13_0_wait_for_bootloader",
              "psp_v13_0_memory_training_send_msg", "amdgpu_gart_map"]
  missing = [sym for sym in required if sym not in symbols]
  if missing: raise SystemExit(f"missing required symbols: {', '.join(missing)}")

  emitted = ["required:"]
  body = [BASE_TRACE]
  if {"amdgpu_device_wreg", "amdgpu_device_rreg"} <= symbols:
    body.append(REG_TRACE)
    emitted.append("amdgpu_device_wreg")
    emitted.append("amdgpu_device_rreg")
  else:
    emitted.append("skip amdgpu_device_[rw]reg")

  if "amdgpu_bo_create_kernel" in symbols:
    body.append(BO_TRACE)
    emitted.append("amdgpu_bo_create_kernel")
  else:
    emitted.append("skip amdgpu_bo_create_kernel")

  emitted.append("optional:")
  if args.optional_probes:
    for sym in OPTIONAL_PROBES:
      if sym in symbols:
        body.append(optional_entry_ret(sym))
        emitted.append(sym)
  else:
    emitted.append("skip optional probes")

  body.append('''
END
{
  printf("psp deep trace end\\n");
}
''')

  pathlib.Path(args.out).write_text("\n".join(body))
  pathlib.Path(args.symbols_out).write_text("\n".join(emitted) + "\n")

if __name__ == "__main__":
  main()
