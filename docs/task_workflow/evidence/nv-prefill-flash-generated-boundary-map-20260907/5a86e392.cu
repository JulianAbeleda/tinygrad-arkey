#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
struct __align__(8) half4 { half x, y, z, w; }; __device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x, y, z, w}; return r; }
extern "C" __global__ void __launch_bounds__(32) E_4096_32_4_5a86e3920e805a3a101a901d317f8e77ce35e37c5c5673e0ab78d740016ef947(half* data0_524288, half* data1_1048576) {
  int gidx0 = blockIdx.x; /* 4096 */
  int lidx0 = threadIdx.x; /* 32 */
  int alu0 = ((gidx0<<7)+(lidx0<<2));
  half4 val0 = (*((half4*)((data1_1048576+alu0))));
  *((half4*)((data0_524288+alu0))) = val0;
}