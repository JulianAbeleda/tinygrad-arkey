// Variant-capture shim: like qk_tensile_kernarg_capture.cpp but captures EVERY distinct dispatched kernel SYMBOL
// (not first-per-role), so a rocBLAS solution-sweep over one shape yields all variants' kernargs.
//   g++ -std=c++17 -D__HIP_PLATFORM_AMD__=1 -shared -fPIC -I/opt/rocm-7.2.4/include extra/qk_tensile_kernarg_capture_all.cpp -ldl -o /tmp/qk_kacap_all.so
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <map>
#include <string>
#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE    ((void*)0x02)
#define HIP_LAUNCH_PARAM_END            ((void*)0x03)
typedef hipError_t (*ext_t)(hipFunction_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,size_t,hipStream_t,void**,void**,hipEvent_t,hipEvent_t,uint32_t);
typedef hipError_t (*getfn_t)(hipFunction_t*,hipModule_t,const char*);
static ext_t real_ext=nullptr; static getfn_t real_getfn=nullptr;
static std::map<void*,std::string> g_names; static std::map<std::string,bool> g_done;
extern "C" hipError_t hipModuleGetFunction(hipFunction_t* f,hipModule_t m,const char* name){
  if(!real_getfn) real_getfn=(getfn_t)dlsym(RTLD_NEXT,"hipModuleGetFunction");
  hipError_t e=real_getfn(f,m,name); if(e==hipSuccess&&f&&name) g_names[(void*)*f]=name; return e; }
extern "C" hipError_t hipExtModuleLaunchKernel(hipFunction_t f,uint32_t gx,uint32_t gy,uint32_t gz,
    uint32_t lx,uint32_t ly,uint32_t lz,size_t sh,hipStream_t st,void** kp,void** extra,hipEvent_t se,hipEvent_t ee,uint32_t fl){
  if(!real_ext) real_ext=(ext_t)dlsym(RTLD_NEXT,"hipExtModuleLaunchKernel");
  void* karg=nullptr; size_t ksz=0;
  if(extra) for(int i=0;extra[i]!=HIP_LAUNCH_PARAM_END&&i<8;i+=2){
    if(extra[i]==HIP_LAUNCH_PARAM_BUFFER_POINTER) karg=extra[i+1];
    else if(extra[i]==HIP_LAUNCH_PARAM_BUFFER_SIZE) ksz=*(size_t*)extra[i+1]; }
  if(karg&&ksz>=104){
    uint8_t buf[256]; size_t n=ksz>256?256:ksz; memcpy(buf,karg,n);
    int32_t M=*(int32_t*)(buf+88),N=*(int32_t*)(buf+92),K=*(int32_t*)(buf+100);
    auto it=g_names.find((void*)f); std::string sym=(it!=g_names.end())?it->second:"UNKNOWN";
    if(!g_done[sym]){ g_done[sym]=true;
      const char* outp=getenv("QK_KACAP"); std::string path=outp?outp:"/tmp/kernargs_all.jsonl";
      FILE* fp=fopen(path.c_str(),"a");
      fprintf(fp,"{\"kernel_symbol\":\"%s\",\"global\":[%u,%u,%u],\"local\":[%u,%u,%u],\"kernarg_size\":%zu,\"M\":%d,\"N\":%d,\"K\":%d,\"kernarg_bytes\":[",
              sym.c_str(),gx,gy,gz,lx,ly,lz,ksz,M,N,K);
      for(size_t i=0;i<n;i++) fprintf(fp,"%s%u",i?",":"",buf[i]);
      fprintf(fp,"]}\n"); fclose(fp);
      fprintf(stderr,"[KACAP] sym=%.50s g(%u,%u,%u) l(%u,%u,%u) ksz=%zu\n",sym.c_str(),gx,gy,gz,lx,ly,lz,ksz);
    } }
  return real_ext(f,gx,gy,gz,lx,ly,lz,sh,st,kp,extra,se,ee,fl); }
