#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <unistd.h>

static unsigned read_put(int channel) {
  volatile unsigned *p=(volatile unsigned *)(0x200400000ull+(unsigned long long)channel*0x3000+0x2000+0x8c);
  return *p;
}
static void dump_new_entries(const unsigned before[16]) {
  for(int c=0;c<16;c++) { unsigned after=read_put(c); if(after==before[c])continue;
    printf("channel=%d put=%u>%u",c,before[c],after);
    volatile unsigned long long *ring=(volatile unsigned long long *)(0x200400000ull+(unsigned long long)c*0x3000);
    for(unsigned p=before[c];p!=after;p=(p+1)&1023) { unsigned long long e=ring[p&1023];
      unsigned long long pb=(e&((1ull<<40)-1))&~3ull;unsigned words=(e>>42)&((1u<<20)-1);
      printf(" entry[%u]=%016llx pb=%llx words=%u",p,e,pb,words);
      volatile unsigned *q=(volatile unsigned *)pb;for(unsigned i=0;i<words;i++)printf(" w[%u]=%08x",i,q[i]);
    }
    printf("\n");
  }
}

__global__ void tick(unsigned *p) { if (threadIdx.x == 0) *p += 1; }

int main(int argc, char **argv) {
  const bool reserve = argc == 2 && !strcmp(argv[1], "reserve");
  if (argc != 2 || (!reserve && strcmp(argv[1], "control"))) {
    fprintf(stderr, "usage: %s control|reserve\n", argv[0]); return 2;
  }
  cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, 0);
  unsigned *p = nullptr; cudaMalloc(&p, sizeof(*p)); cudaMemset(p, 0, sizeof(*p));
  tick<<<1, 32>>>(p); cudaDeviceSynchronize(); // force channel construction first
  unsigned puts[16];for(int c=0;c<16;c++)puts[c]=read_put(c);
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                     reserve ? prop.persistingL2CacheMaxSize : 0);
  dump_new_entries(puts);
  tick<<<1, 32>>>(p); cudaDeviceSynchronize();
  printf("mode=%s pid=%d reserve_bytes=%zu\n", argv[1], getpid(),
         reserve ? prop.persistingL2CacheMaxSize : 0UL);
  cudaFree(p); return 0;
}
