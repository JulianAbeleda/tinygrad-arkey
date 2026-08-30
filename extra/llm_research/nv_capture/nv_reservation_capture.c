#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/* Research-only CUDA reservation capture.  It leaves the driver untouched and
 * protects NVIDIA's 4 KiB userspace mappings only for cudaDeviceSetLimit(). */
struct page { void *addr; size_t len; int prot, armed, hit; unsigned char before[65536]; };
static struct page pages[64]; static int npages, in_hook;
static pthread_mutex_t mu=PTHREAD_MUTEX_INITIALIZER;
static void *(*real_mmap)(void*,size_t,int,int,int,off_t);
static int (*real_mprotect)(void*,size_t,int);
static int (*real_setlimit)(int,size_t);
static int (*real_sync)(void); static int pending;

static int is_nvidia_fd(int fd) {
  char p[64], out[256]; snprintf(p,sizeof(p),"/proc/self/fd/%d",fd);
  ssize_t n=readlink(p,out,sizeof(out)-1); if(n<0)return 0; out[n]=0;
  return strstr(out,"/dev/nvidia") != NULL;
}
static void log_line(const char *s) { write(STDERR_FILENO,s,strlen(s)); }
static void segv(int sig, siginfo_t *si, void *ctx) {
  (void)sig;(void)ctx; uintptr_t a=(uintptr_t)si->si_addr;
  for(int i=0;i<npages;i++) if(pages[i].armed && a>=(uintptr_t)pages[i].addr && a<(uintptr_t)pages[i].addr+pages[i].len) {
    pages[i].hit=1; pages[i].armed=0; real_mprotect(pages[i].addr,pages[i].len,pages[i].prot);
    char b[160]; int n=snprintf(b,sizeof(b),"NV_RESERVE_WRITE page=%p fault=%p\n",pages[i].addr,si->si_addr);write(2,b,n);return;
  }
  signal(SIGSEGV,SIG_DFL); raise(SIGSEGV);
}
static void init(void) {
  if(real_mmap)return; in_hook=1;
  real_mmap=dlsym(RTLD_NEXT,"mmap"); real_mprotect=dlsym(RTLD_NEXT,"mprotect");
  struct sigaction sa={0};sa.sa_sigaction=segv;sa.sa_flags=SA_SIGINFO|SA_NODEFER;sigemptyset(&sa.sa_mask);sigaction(SIGSEGV,&sa,0);
  in_hook=0;
}
void *mmap(void *a,size_t n,int prot,int flags,int fd,off_t off) {
  if(!real_mmap)init(); void *r=real_mmap(a,n,prot,flags,fd,off);
  if(!in_hook && r!=MAP_FAILED && n<=65536 && (prot&PROT_WRITE) && is_nvidia_fd(fd)) {
    pthread_mutex_lock(&mu);if(npages<64){pages[npages++]=(struct page){.addr=r,.len=n,.prot=prot};fprintf(stderr,"NV_RESERVE_MAP page=%p off=%lx prot=%x\n",r,(unsigned long)off,prot);}pthread_mutex_unlock(&mu);
  } return r;
}
void *mmap64(void *a,size_t n,int prot,int flags,int fd,off64_t off) {
  return mmap(a,n,prot,flags,fd,(off_t)off);
}
int cudaDeviceSetLimit(int limit,size_t value) {
  if(!real_mmap)init(); if(!real_setlimit)real_setlimit=dlsym(RTLD_NEXT,"cudaDeviceSetLimit");
  fprintf(stderr,"NV_RESERVE_BEGIN limit=%d value=%zu pages=%d\n",limit,value,npages);
  for(int i=0;i<npages;i++){memcpy(pages[i].before,pages[i].addr,pages[i].len);int rc=real_mprotect(pages[i].addr,pages[i].len,PROT_READ);pages[i].armed=(rc==0);fprintf(stderr,"NV_RESERVE_ARM page=%p len=%zu rc=%d errno=%d\n",pages[i].addr,pages[i].len,rc,rc?errno:0);}
  int rc=real_setlimit(limit,value);
  pending=1; fprintf(stderr,"NV_RESERVE_CACHED rc=%d\n",rc);return rc;
}
int cudaDeviceSynchronize(void) {
  if(!real_sync)real_sync=dlsym(RTLD_NEXT,"cudaDeviceSynchronize");
  int rc=real_sync(); if(!pending)return rc; pending=0;
  for(int i=0;i<npages;i++){if(pages[i].armed)real_mprotect(pages[i].addr,pages[i].len,pages[i].prot);pages[i].armed=0;int changes=0;for(size_t j=0;j<pages[i].len;j++)if(pages[i].before[j]!=((unsigned char*)pages[i].addr)[j]){if(changes<16)fprintf(stderr,"NV_RESERVE_DIFF page=%p off=%03zx %02x>%02x\n",pages[i].addr,j,pages[i].before[j],((unsigned char*)pages[i].addr)[j]);changes++;}if(changes||pages[i].hit)fprintf(stderr,"NV_RESERVE_PAGE page=%p hit=%d changes=%d\n",pages[i].addr,pages[i].hit,changes);}
  fprintf(stderr,"NV_RESERVE_END sync_rc=%d\n",rc);return rc;
}
