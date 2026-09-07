#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
struct __align__(8) half4 { half x, y, z, w; }; __device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x, y, z, w}; return r; }
extern "C" __global__ void __launch_bounds__(128) E_512_4_2_8_16_4_284de8f2c9c98fcb873331166e5366c16b6d3ea5594fe638257749ba7f83929d(half* data0_2097152, half* data1_2097152) {
  int gidx0 = blockIdx.x; /* 2 */
  int gidx1 = blockIdx.y; /* 4 */
  int gidx2 = blockIdx.z; /* 512 */
  int lidx0 = threadIdx.x; /* 8 */
  int lidx1 = threadIdx.y; /* 16 */
  int alu0 = ((gidx0<<6)+(lidx1<<2));
  half4 val0 = (*((half4*)((data1_2097152+(alu0+(gidx2<<7)+(gidx1<<19)+(lidx0<<16))))));
  *((half4*)((data0_2097152+(alu0+(gidx1<<10)+(lidx0<<7)+(gidx2<<12))))) = val0;
}