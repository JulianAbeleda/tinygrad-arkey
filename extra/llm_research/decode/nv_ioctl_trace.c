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
  if(alloc){memcpy(&cmd,outer+12,4);memcpy(&pp,outer+16,8);psz=(cmd==0xc96f)?368:(cmd==0xc661?8:0);if(pp&&psz)memcpy(nested,(void*)pp,psz);}
  int rc=real_ioctl(fd,req,arg);if(arg&&osz)memcpy(outer_post,arg,osz);
  if(!rc&&nr==0x4e&&osz>=12)memcpy(&pending_memory,outer_post+8,4);
  if(!rc&&alloc&&cmd==0xc96f&&pp&&psz>=72&&nchannels<64){struct channel *c=&channels[nchannels++];memcpy(&c->buffer,(void*)pp+4,4);memcpy(&c->gpfifo,(void*)pp+8,8);memcpy(&c->entries,(void*)pp+16,4);memcpy(&c->userd,(void*)pp+64,8);}
  const char *path=getenv("NV_IOCTL_TRACE");if(!path)return rc;
  pthread_mutex_lock(&lock);FILE *f=fopen(path,"a");if(f){fprintf(f,"%lu fd=%d req=%lx rc=%d osz=%zu outer_pre=",seq++,fd,req,rc,osz);hex(f,outer,osz);fprintf(f," outer_post=");hex(f,outer_post,osz);if(rm||alloc){fprintf(f," %s=%08x psz=%u pre=",alloc?"class":"cmd",cmd,psz);hex(f,nested,psz);if(pp&&psz){fprintf(f," post=");hex(f,(unsigned char*)pp,psz);}}fputc('\n',f);fclose(f);}pthread_mutex_unlock(&lock);return rc;
}

void *mmap(void *addr,size_t length,int prot,int flags,int fd,off_t offset){
  static void *(*real_mmap)(void*,size_t,int,int,int,off_t);if(!real_mmap)real_mmap=dlsym(RTLD_NEXT,"mmap");
  void *ret=real_mmap(addr,length,prot,flags,fd,offset);const char *path=getenv("NV_IOCTL_TRACE");
  pthread_mutex_lock(&lock);uint32_t mem=pending_memory;pending_memory=0;if(mem&&ret!=MAP_FAILED&&nmappings<128)mappings[nmappings++]=(struct mapping){mem,ret,length};
  if(path){FILE *f=fopen(path,"a");if(f){fprintf(f,"%lu mmap memory=%08x cpu=%p length=%zu prot=%x flags=%x fd=%d offset=%lld\n",seq++,mem,ret,length,prot,flags,fd,(long long)offset);fclose(f);}}pthread_mutex_unlock(&lock);return ret;
}

void nv_ioctl_trace_snapshot(const char *label){
  const char *path=getenv("NV_IOCTL_TRACE");if(!path)return;pthread_mutex_lock(&lock);FILE *f=fopen(path,"a");if(!f){pthread_mutex_unlock(&lock);return;}
  fprintf(f,"%lu snapshot_begin label=%s channels=%zu mappings=%zu\n",seq++,label?label:"",nchannels,nmappings);
  for(size_t i=0;i<nchannels;i++)for(size_t j=0;j<nmappings;j++)if(channels[i].buffer==mappings[j].memory){
    unsigned char *base=mappings[j].cpu;if(channels[i].userd+0x90<=mappings[j].length){uint32_t put=0;memcpy(&put,base+channels[i].userd+0x8c,4);fprintf(f,"%lu snapshot label=%s channel=%zu buffer=%08x gpfifo=%llu entries=%u userd=%llu gpput=%u\n",seq++,label?label:"",i,channels[i].buffer,(unsigned long long)channels[i].gpfifo,channels[i].entries,(unsigned long long)channels[i].userd,put);}
  }fclose(f);pthread_mutex_unlock(&lock);
}
