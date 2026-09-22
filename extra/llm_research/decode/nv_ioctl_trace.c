#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
static unsigned long seq;
static __thread uint32_t pending_memory;
struct mapping { uint32_t memory; void *cpu; size_t length; };
struct channel { uint32_t buffer; uint64_t gpfifo; uint32_t entries; uint64_t userd; };
static struct mapping mappings[128]; static size_t nmappings;
static struct channel channels[64]; static size_t nchannels;
static void hex(FILE *f,const unsigned char *p,size_t n){for(size_t i=0;i<n;i++)fprintf(f,"%02x",p[i]);}
int ioctl(int fd,unsigned long req,...) {
  static int (*real_ioctl)(int,unsigned long,void*);if(!real_ioctl)real_ioctl=dlsym(RTLD_NEXT,"ioctl");
  va_list ap;va_start(ap,req);void *arg=va_arg(ap,void*);va_end(ap);
  unsigned char outer[64]={0},outer_post[64]={0},nested[512]={0};size_t osz=_IOC_SIZE(req);if(osz>sizeof(outer))osz=sizeof(outer);if(arg&&osz)memcpy(outer,arg,osz);
  uint32_t cmd=0,psz=0;uintptr_t pp=0;unsigned nr=req&0xff;int rm=(nr==0x2a && arg && osz>=32),alloc=(nr==0x2b&&arg&&osz>=32);
  if(rm){memcpy(&cmd,outer+8,4);memcpy(&pp,outer+16,8);memcpy(&psz,outer+24,4);if(psz>sizeof(nested))psz=sizeof(nested);if(pp&&psz)memcpy(nested,(void*)pp,psz);}
  if(alloc){memcpy(&cmd,outer+12,4);memcpy(&pp,outer+16,8);if(osz>=48)memcpy(&psz,outer+32,4);else if(osz>=28)memcpy(&psz,outer+24,4);if(cmd==0xc96f&&psz<368)psz=368;if(cmd==0xc661&&psz<8)psz=8;if(cmd==0x3e&&psz<120)psz=120;if(psz>sizeof(nested))psz=sizeof(nested);if(pp&&psz)memcpy(nested,(void*)pp,psz);}
  int rc=real_ioctl(fd,req,arg);if(arg&&osz)memcpy(outer_post,arg,osz);
  if(!rc&&nr==0x4e&&osz>=12)memcpy(&pending_memory,outer_post+8,4);
  if(!rc&&alloc&&cmd==0xc96f&&pp&&psz>=72&&nchannels<64){struct channel *c=&channels[nchannels++];memcpy(&c->buffer,(void*)pp+4,4);memcpy(&c->gpfifo,(void*)pp+8,8);memcpy(&c->entries,(void*)pp+16,4);memcpy(&c->userd,(void*)pp+64,8);}
  const char *path=getenv("NV_IOCTL_TRACE");if(!path)return rc;
  pthread_mutex_lock(&lock);FILE *f=fopen(path,"a");if(f){fprintf(f,"%lu fd=%d req=%lx rc=%d osz=%zu outer_pre=",seq++,fd,req,rc,osz);hex(f,outer,osz);fprintf(f," outer_post=");hex(f,outer_post,osz);if(rm||alloc){fprintf(f," %s=%08x psz=%u pre=",alloc?"class":"cmd",cmd,psz);hex(f,nested,psz);if(pp&&psz){fprintf(f," post=");hex(f,(unsigned char*)pp,psz);}}fputc('\n',f);fclose(f);}pthread_mutex_unlock(&lock);return rc;
}

static void record_mmap(void *ret,size_t length,int prot,int flags,int fd,off64_t offset){
  const char *path=getenv("NV_IOCTL_TRACE");
  pthread_mutex_lock(&lock);uint32_t mem=pending_memory;pending_memory=0;if(mem&&ret!=MAP_FAILED&&nmappings<128)mappings[nmappings++]=(struct mapping){mem,ret,length};
  if(path){FILE *f=fopen(path,"a");if(f){fprintf(f,"%lu mmap memory=%08x cpu=%p length=%zu prot=%x flags=%x fd=%d offset=%lld\n",seq++,mem,ret,length,prot,flags,fd,(long long)offset);fclose(f);}}pthread_mutex_unlock(&lock);
}
void *mmap(void *addr,size_t length,int prot,int flags,int fd,off_t offset){
  static void *(*real_mmap)(void*,size_t,int,int,int,off_t);if(!real_mmap)real_mmap=dlsym(RTLD_NEXT,"mmap");
  void *ret=real_mmap(addr,length,prot,flags,fd,offset);record_mmap(ret,length,prot,flags,fd,offset);return ret;
}
void *mmap64(void *addr,size_t length,int prot,int flags,int fd,off64_t offset){
  static void *(*real_mmap64)(void*,size_t,int,int,int,off64_t);if(!real_mmap64)real_mmap64=dlsym(RTLD_NEXT,"mmap64");
  void *ret=real_mmap64(addr,length,prot,flags,fd,offset);record_mmap(ret,length,prot,flags,fd,offset);return ret;
}

void nv_ioctl_trace_snapshot(const char *label){
  const char *path=getenv("NV_IOCTL_TRACE");if(!path)return;pthread_mutex_lock(&lock);FILE *f=fopen(path,"a");if(!f){pthread_mutex_unlock(&lock);return;}
  fprintf(f,"%lu snapshot_begin label=%s channels=%zu mappings=%zu\n",seq++,label?label:"",nchannels,nmappings);
  uint64_t first_gpfifo=nchannels?channels[0].gpfifo:0;
  for(size_t i=0;i<nchannels;i++)for(size_t j=0;j<nmappings;j++)if(channels[i].buffer==mappings[j].memory){
    unsigned char *base=mappings[j].cpu;if(channels[i].userd+0x90<=mappings[j].length){uint32_t put=0;memcpy(&put,base+channels[i].userd+0x8c,4);fprintf(f,"%lu snapshot label=%s channel=%zu buffer=%08x gpfifo=%llu entries=%u userd=%llu gpput=%u",seq++,label?label:"",i,channels[i].buffer,(unsigned long long)channels[i].gpfifo,channels[i].entries,(unsigned long long)channels[i].userd,put);
      uint64_t ring_off=channels[i].gpfifo-first_gpfifo;if(ring_off+channels[i].entries*8<=mappings[j].length)for(unsigned k=4;k;k--){uint64_t e=0;uint32_t idx=(put-k)%channels[i].entries;memcpy(&e,base+ring_off+idx*8,8);fprintf(f," e%u=%016llx",idx,(unsigned long long)e);}fputc('\n',f);}
  }fclose(f);pthread_mutex_unlock(&lock);
}

void nv_ioctl_trace_dump_active_pb(const char *label){
  const char *path=getenv("NV_IOCTL_TRACE");if(!path)return;pthread_mutex_lock(&lock);FILE *f=fopen(path,"a");if(!f){pthread_mutex_unlock(&lock);return;}
  if(nchannels>1)for(size_t r=0;r<nmappings;r++)if(mappings[r].memory==channels[1].buffer){unsigned char *ring=mappings[r].cpu+(channels[1].gpfifo-channels[0].gpfifo);uint32_t put=0;memcpy(&put,mappings[r].cpu+channels[1].userd+0x8c,4);uint64_t e0=0;memcpy(&e0,ring,8);uint64_t pbbase=e0&((1ULL<<42)-1);
    for(unsigned back=3;back;back--){uint32_t idx=(put-back)%channels[1].entries;uint64_t e=0;memcpy(&e,ring+idx*8,8);uint64_t addr=e&((1ULL<<42)-1),n=(e>>42)&0x1fffff;
      for(size_t m=0;m<nmappings;m++)if(mappings[m].length>=0x500000&&mappings[m].length<=0x800000&&addr>=pbbase&&addr+n*4<=pbbase+mappings[m].length){unsigned cap=n<256?n:256;fprintf(f,"%lu pbdump label=%s idx=%u addr=%llx words=%llu data=",seq++,label?label:"",idx,(unsigned long long)addr,(unsigned long long)n);hex(f,mappings[m].cpu+(addr-pbbase),cap*4);fputc('\n',f);}
    }
  }fclose(f);pthread_mutex_unlock(&lock);
}

uint64_t nv_ioctl_trace_active_pb(unsigned back,uint32_t *words){
  uint64_t out=0;uint32_t n=0;pthread_mutex_lock(&lock);if(nchannels>1&&back>0)for(size_t r=0;r<nmappings;r++)if(mappings[r].memory==channels[1].buffer){uint32_t put=0;memcpy(&put,mappings[r].cpu+channels[1].userd+0x8c,4);uint64_t e=0;memcpy(&e,mappings[r].cpu+(channels[1].gpfifo-channels[0].gpfifo)+((put-back)%channels[1].entries)*8,8);out=e&((1ULL<<42)-1);n=(e>>42)&0x1fffff;break;}pthread_mutex_unlock(&lock);if(words)*words=n;return out;
}

void nv_ioctl_trace_dump_mappings(const char *label){
  const char *path=getenv("NV_IOCTL_TRACE");if(!path||!label||!strstr(label,"n208-rep"))return;int memfd=open("/proc/self/mem",O_RDONLY);if(memfd<0)return;pthread_mutex_lock(&lock);FILE *f=fopen(path,"a");if(!f){pthread_mutex_unlock(&lock);close(memfd);return;}
  unsigned char page[4096];for(size_t m=0;m<nmappings;m++){size_t nonzero=0,shown=0;fprintf(f,"%lu mapdump label=%s memory=%08x cpu=%p length=%zu first=",seq++,label,mappings[m].memory,mappings[m].cpu,mappings[m].length);
    for(size_t off=0;off<mappings[m].length;off+=sizeof(page)){size_t want=mappings[m].length-off<sizeof(page)?mappings[m].length-off:sizeof(page);ssize_t got=pread(memfd,page,want,(off_t)((uintptr_t)mappings[m].cpu+off));if(got<=0)continue;for(ssize_t i=0;i<got;i++)if(page[i]){nonzero++;if(shown++<32)fprintf(f,"%zx:%02x,",off+(size_t)i,page[i]);}}
    fprintf(f," nonzero=%zu\n",nonzero);
  }fclose(f);pthread_mutex_unlock(&lock);close(memfd);
}
