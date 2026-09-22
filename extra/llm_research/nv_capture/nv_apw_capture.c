// Research-only sidecar: inventory NVIDIA userspace mappings without changing the NVIDIA module.
#include <linux/module.h>
#include <linux/kprobes.h>
#include <linux/mm.h>
#include <linux/sched.h>
#include <linux/version.h>

/* nvidia_mmap(file, vma): the second SysV x86-64 argument is %rsi.  Keep the
 * probe ABI identical to the open driver's entry point instead of reconstructing
 * its mmap logic or touching the production module. */
static __always_inline struct vm_area_struct *nvidia_mmap_vma(struct pt_regs *regs)
{
#if defined(CONFIG_X86_64)
  return (struct vm_area_struct *)regs->si;
#else
#error "nv_apw_capture currently supports x86-64 only"
#endif
}

struct mmap_call { struct vm_area_struct *vma; unsigned long start,end,pgoff,flags; };
static int entry(struct kretprobe_instance *ri,struct pt_regs *regs) {
  struct mmap_call *c=(struct mmap_call *)ri->data;struct vm_area_struct *vma=nvidia_mmap_vma(regs);
  c->vma=vma;c->start=vma->vm_start;c->end=vma->vm_end;c->pgoff=vma->vm_pgoff;c->flags=vma->vm_flags;return 0;
}
static int leave(struct kretprobe_instance *ri,struct pt_regs *regs) {
  struct mmap_call *c=(struct mmap_call *)ri->data;long rc=regs_return_value(regs);
  if(rc==0)pr_info("nv_apw_capture pid=%d comm=%s start=%lx end=%lx bytes=%lx pgoff=%lx flags=%lx\n",current->tgid,current->comm,c->start,c->end,c->end-c->start,c->pgoff,c->flags);
  return 0;
}
static struct kretprobe probe={.kp.symbol_name="nvidia_mmap",.entry_handler=entry,.handler=leave,.data_size=sizeof(struct mmap_call),.maxactive=64};
static int __init begin(void){int rc=register_kretprobe(&probe);pr_info("nv_apw_capture loaded rc=%d\n",rc);return rc;}
static void __exit end(void){unregister_kretprobe(&probe);pr_info("nv_apw_capture unloaded missed=%d\n",probe.nmissed);}
module_init(begin);module_exit(end);MODULE_LICENSE("GPL");MODULE_DESCRIPTION("Research-only NVIDIA mmap inventory for APW lowering");
