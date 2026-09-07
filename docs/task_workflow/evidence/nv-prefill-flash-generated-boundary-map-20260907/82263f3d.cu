#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
extern "C" __global__ void __launch_bounds__(128) E_2048_4_8_16_2_82263f3d7c6516dd0d8b78f8d75f712cfb3098cdcd6c9a1b11ff2c17e7eeb030(half* data0_2097152, float* data1_2097152, float* data2_65536) {
  int gidx0 = blockIdx.x; /* 4 */
  int gidx1 = blockIdx.y; /* 2048 */
  int lidx0 = threadIdx.x; /* 8 */
  int lidx1 = threadIdx.y; /* 16 */
  int alu0 = (lidx1+(gidx0<<4));
  int alu1 = (lidx0<<7);
  int alu2 = (alu0+(gidx1<<10)+alu1);
  float val0 = (*(data1_2097152+alu2));
  int alu3 = (alu2+64);
  float val1 = (*(data1_2097152+alu3));
  int alu4 = (alu0+alu1+((gidx1&63)<<10));
  float val2 = (*(data2_65536+alu4));
  float val3 = (*(data2_65536+(alu4+64)));
  *(data0_2097152+alu2) = ((half)(((val0*val2)-(val1*val3))));
  *(data0_2097152+alu3) = ((half)(((val1*val2)+(val0*val3))));
}