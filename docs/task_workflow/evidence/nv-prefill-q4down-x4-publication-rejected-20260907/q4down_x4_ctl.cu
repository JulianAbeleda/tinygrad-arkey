#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
__device__ __forceinline__ uint4 tg_ldmatrix_x4(const void *p) {
  uint4 r; asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
    : "=r"(r.x),"=r"(r.y),"=r"(r.z),"=r"(r.w) : "l"(p)); return r;
}
#include <cuda_fp16.h>
struct __align__(128) float64 { float v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63; }; __device__ float64 make_float64(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16, float v17, float v18, float v19, float v20, float v21, float v22, float v23, float v24, float v25, float v26, float v27, float v28, float v29, float v30, float v31, float v32, float v33, float v34, float v35, float v36, float v37, float v38, float v39, float v40, float v41, float v42, float v43, float v44, float v45, float v46, float v47, float v48, float v49, float v50, float v51, float v52, float v53, float v54, float v55, float v56, float v57, float v58, float v59, float v60, float v61, float v62, float v63) { float64 r={v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63}; return r; }
struct __align__(8) signed_char8 { signed char x, y, z, w, a, b, c, d; }; __device__ signed_char8 make_signed_char8(signed char x, signed char y, signed char z, signed char w, signed char a, signed char b, signed char c, signed char d) { signed_char8 r={x, y, z, w, a, b, c, d}; return r; }
struct __align__(16) signed_char16 { signed char x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l; }; __device__ signed_char16 make_signed_char16(signed char x, signed char y, signed char z, signed char w, signed char a, signed char b, signed char c, signed char d, signed char e, signed char f, signed char g, signed char h, signed char i, signed char j, signed char k, signed char l) { signed_char16 r={x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l}; return r; }
__device__ int4 __WMMA_8_16_32_signed_char_int(signed_char16 a, signed_char8 b, int4 c){
  int *a_pk = (int *)(&a), *b_pk = (int *)(&b), *c_pk = (int *)(&c);
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7},"
      "{%8, %9}, {%0, %1, %2, %3};"
    : "+r"(c_pk[0]), "+r"(c_pk[1]), "+r"(c_pk[2]), "+r"(c_pk[3])
    : "r"(a_pk[0]), "r"(a_pk[1]), "r"(a_pk[2]), "r"(a_pk[3]), "r"(b_pk[0]), "r"(b_pk[1]));
  return c;
}
extern "C" __global__ void __launch_bounds__(256) q4_down_streamk(float* data0_2097152, float* partials, int* partial_ids, const unsigned int* data1_7077888, const unsigned int* data2_1966080) {
  int owner = blockIdx.x; /* 170 persistent owners */
  int lidx0 = threadIdx.x; /* 32 */
  int lidx1 = threadIdx.y; /* 2 */
  int lidx2 = threadIdx.z; /* 4 */
  float buf0[64];
  __shared__ __align__(16) signed char buf1[20480];
  int4 cast0 = make_int4(0,0,0,0);
  int alu0 = ((lidx0>>1)&1);
  int alu1 = (alu0<<2);
  unsigned int cast1 = ((unsigned int)((alu1+8)));
  unsigned int cast2 = ((unsigned int)((alu1+16)));
  unsigned int cast3 = ((unsigned int)((alu1+24)));
  unsigned int cast4 = ((unsigned int)(alu1));
  int alu2 = (lidx0>>2);
  int alu3 = (alu2*80);
  int alu4 = ((lidx1*2560)+(lidx2*640)+alu3);
  int alu5 = (lidx0&3);
  int alu6 = (alu5<<2);
  int alu7 = (alu4+alu6);
  int alu8 = (alu4+(alu5<<4));
  int alu9 = (lidx1*5120);
  int alu10 = (alu9+((lidx0&15)*80)+((lidx0>>4)<<4));
  int alu11 = (lidx2*2560);
  int alu12 = (alu11+alu3+alu6);
  int alu13 = (alu9+alu3);
  int alu14 = (alu11+(alu5*160));
  int owner_start = ((owner*24576/170)/8)*8;
  if (threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0) { partial_ids[owner*2]=-1; partial_ids[owner*2+1]=-1; }
  int owner_stop = (owner == 169) ? 24576 : ((((owner+1)*24576/170)/8)*8);
  int first_tile = owner_start/192, last_tile = (owner_stop-1)/192;
  for (int tile=first_tile; tile<=last_tile; tile++) {
    int segment_start=max(owner_start,tile*192), segment_stop=min(owner_stop,(tile+1)*192);
    int k_begin=segment_start-tile*192, k_end=segment_stop-tile*192;
    int gidx0=tile%4, gidx1=tile/4;
    bool direct=(k_begin==0 && k_end==192);
    bool owner_tail=(segment_stop==owner_stop && k_end!=192);
    bool owner_has_head=((owner_start%192)!=0);
    int slot=owner*2+((owner_tail&&owner_has_head)?1:0);
  (*(buf0+0)) = 0.0f;
  (*(buf0+1)) = 0.0f;
  (*(buf0+2)) = 0.0f;
  (*(buf0+3)) = 0.0f;
  (*(buf0+4)) = 0.0f;
  (*(buf0+5)) = 0.0f;
  (*(buf0+6)) = 0.0f;
  (*(buf0+7)) = 0.0f;
  (*(buf0+8)) = 0.0f;
  (*(buf0+9)) = 0.0f;
  (*(buf0+10)) = 0.0f;
  (*(buf0+11)) = 0.0f;
  (*(buf0+12)) = 0.0f;
  (*(buf0+13)) = 0.0f;
  (*(buf0+14)) = 0.0f;
  (*(buf0+15)) = 0.0f;
  (*(buf0+16)) = 0.0f;
  (*(buf0+17)) = 0.0f;
  (*(buf0+18)) = 0.0f;
  (*(buf0+19)) = 0.0f;
  (*(buf0+20)) = 0.0f;
  (*(buf0+21)) = 0.0f;
  (*(buf0+22)) = 0.0f;
  (*(buf0+23)) = 0.0f;
  (*(buf0+24)) = 0.0f;
  (*(buf0+25)) = 0.0f;
  (*(buf0+26)) = 0.0f;
  (*(buf0+27)) = 0.0f;
  (*(buf0+28)) = 0.0f;
  (*(buf0+29)) = 0.0f;
  (*(buf0+30)) = 0.0f;
  (*(buf0+31)) = 0.0f;
  (*(buf0+32)) = 0.0f;
  (*(buf0+33)) = 0.0f;
  (*(buf0+34)) = 0.0f;
  (*(buf0+35)) = 0.0f;
  (*(buf0+36)) = 0.0f;
  (*(buf0+37)) = 0.0f;
  (*(buf0+38)) = 0.0f;
  (*(buf0+39)) = 0.0f;
  (*(buf0+40)) = 0.0f;
  (*(buf0+41)) = 0.0f;
  (*(buf0+42)) = 0.0f;
  (*(buf0+43)) = 0.0f;
  (*(buf0+44)) = 0.0f;
  (*(buf0+45)) = 0.0f;
  (*(buf0+46)) = 0.0f;
  (*(buf0+47)) = 0.0f;
  (*(buf0+48)) = 0.0f;
  (*(buf0+49)) = 0.0f;
  (*(buf0+50)) = 0.0f;
  (*(buf0+51)) = 0.0f;
  (*(buf0+52)) = 0.0f;
  (*(buf0+53)) = 0.0f;
  (*(buf0+54)) = 0.0f;
  (*(buf0+55)) = 0.0f;
  (*(buf0+56)) = 0.0f;
  (*(buf0+57)) = 0.0f;
  (*(buf0+58)) = 0.0f;
  (*(buf0+59)) = 0.0f;
  (*(buf0+60)) = 0.0f;
  (*(buf0+61)) = 0.0f;
  (*(buf0+62)) = 0.0f;
  (*(buf0+63)) = 0.0f;
  #pragma unroll 8
  for (int Ridx0 = k_begin; Ridx0 < k_end; Ridx0++) {
    int alu79 = ((lidx1*55296)+(lidx2*13824)+(alu2*1728)+(gidx1*221184)+((Ridx0>>2)*36));
    int alu80 = (alu79+((Ridx0>>1)&1));
    unsigned int val0 = (*(data1_7077888+(alu80+1)));
    unsigned int val1 = (*(data1_7077888+(alu80+2)));
    unsigned int val2 = (*(data1_7077888+(alu80+110593)));
    unsigned int val3 = (*(data1_7077888+(alu80+110594)));
    int alu81 = (Ridx0&3);
    bool alu82 = (alu81<2);
    int alu83 = (alu82?0:((alu81<<1)+alu0+-4));
    bool alu84 = (alu83<0);
    int alu85 = (alu84?3:0);
    int alu86 = (alu79+(((alu83+alu85)>>2)-((int)((((alu83%4)!=0)&(alu84!=0))))));
    unsigned int val4 = (*(data1_7077888+(alu86+1)));
    unsigned int val5 = (*(data1_7077888+(alu86+2)));
    unsigned int val6 = (*(data1_7077888+(alu86+3)));
    unsigned int val7 = (*(data1_7077888+(alu86+110593)));
    unsigned int val8 = (*(data1_7077888+(alu86+110594)));
    unsigned int val9 = (*(data1_7077888+(alu86+110595)));
    unsigned int val10 = (*(data1_7077888+(alu79+110592)));
    unsigned int val11 = (*(data1_7077888+alu79));
    int alu87 = ((lidx1*98304)+(lidx2*24576)+(alu2*3072)+(gidx0*393216)+alu6+(Ridx0<<4));
    unsigned int val12 = (*(data2_1966080+(alu87+1)));
    unsigned int val13 = (*(data2_1966080+(alu87+2)));
    unsigned int val14 = (*(data2_1966080+(alu87+3)));
    unsigned int val15 = (*(data2_1966080+(alu87+196608)));
    unsigned int val16 = (*(data2_1966080+(alu87+196609)));
    unsigned int val17 = (*(data2_1966080+(alu87+196610)));
    unsigned int val18 = (*(data2_1966080+(alu87+196611)));
    int alu88 = ((lidx1*12288)+(lidx2*3072)+(alu2*384)+(gidx0*49152)+(Ridx0<<1)+alu0);
    unsigned int val19 = (*(data2_1966080+(alu88+1572864)));
    unsigned int val20 = (*(data2_1966080+(alu88+1597440)));
    unsigned int val21 = (*(data2_1966080+(alu88+1769472)));
    unsigned int val22 = (*(data2_1966080+(alu88+1794048)));
    unsigned int val23 = (*(data2_1966080+alu87));
    int alu89 = (alu79+(alu81<<3)+((lidx0&1)<<2));
    uint4 val24 = (*((uint4*)((data1_7077888+(alu89+4)))));
    uint4 val25 = (*((uint4*)((data1_7077888+(alu89+110596)))));
    __syncthreads();
    unsigned short cast5 = tg_bitcast<unsigned short>((half)(((half)(tg_bitcast<float>((unsigned int)(val19))))));
    unsigned short cast6 = tg_bitcast<unsigned short>((half)(((half)(tg_bitcast<float>((unsigned int)(val20))))));
    unsigned short cast7 = tg_bitcast<unsigned short>((half)(((half)(tg_bitcast<float>((unsigned int)(val21))))));
    unsigned short cast8 = tg_bitcast<unsigned short>((half)(((half)(tg_bitcast<float>((unsigned int)(val22))))));
    unsigned int cast9 = ((unsigned int)(((alu0<<3)+((Ridx0&1)<<4))));
    unsigned int cast10 = ((unsigned int)((((alu83+4)&3)<<3)));
    unsigned int cast11 = ((unsigned int)((((alu83+8)&3)<<3)));
    unsigned int alu91 = (val9>>cast11);
    float alu92 = (alu82?((float)(((val3>>cast9)&63u))):((float)((((alu91&255u)>>4u)|((((val8>>cast10)&255u)>>6u)<<4u)))));
    unsigned short cast12 = tg_bitcast<unsigned short>((half)(((half)(-(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val10>>16u)&65535u)))))))*alu92)))));
    unsigned int alu93 = (val6>>cast11);
    float alu94 = (alu82?((float)(((val1>>cast9)&63u))):((float)((((alu93&255u)>>4u)|((((val5>>cast10)&255u)>>6u)<<4u)))));
    unsigned short cast13 = tg_bitcast<unsigned short>((half)(((half)(-(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)(((val11>>16u)&65535u)))))))*alu94)))));
    unsigned int cast14 = ((unsigned int)(((alu83&3)<<3)));
    float alu95 = (alu82?((float)(((val2>>cast9)&63u))):((float)(((alu91&15u)|((((val7>>cast14)&255u)>>6u)<<4u)))));
    unsigned short cast15 = tg_bitcast<unsigned short>((half)(((half)((((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val10&65535u)))))))*alu95)))));
    float alu96 = (alu82?((float)(((val0>>cast9)&63u))):((float)(((alu93&15u)|((((val4>>cast14)&255u)>>6u)<<4u)))));
    unsigned short cast16 = tg_bitcast<unsigned short>((half)(((half)((((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val11&65535u)))))))*alu96)))));
    *(buf1+(alu7+64)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast16>>((unsigned short)(0u)))&((unsigned short)(255u)))))));
    *(buf1+(alu7+65)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast16>>((unsigned short)(8u)))&((unsigned short)(255u)))))));
    *(buf1+(alu7+66)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast13>>((unsigned short)(0u)))&((unsigned short)(255u)))))));
    *(buf1+(alu7+67)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast13>>((unsigned short)(8u)))&((unsigned short)(255u)))))));
    *(buf1+(alu7+5184)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast15>>((unsigned short)(0u)))&((unsigned short)(255u)))))));
    *(buf1+(alu7+5185)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast15>>((unsigned short)(8u)))&((unsigned short)(255u)))))));
    *(buf1+(alu7+5186)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast12>>((unsigned short)(0u)))&((unsigned short)(255u)))))));
    *(buf1+(alu7+5187)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast12>>((unsigned short)(8u)))&((unsigned short)(255u)))))));
    *(buf1+(alu7+10304)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast5>>((unsigned short)(0u)))&((unsigned short)(255u)))))));
    *(buf1+(alu7+10305)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast5>>((unsigned short)(8u)))&((unsigned short)(255u)))))));
    *(buf1+(alu7+10306)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast7>>((unsigned short)(0u)))&((unsigned short)(255u)))))));
    *(buf1+(alu7+10307)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast7>>((unsigned short)(8u)))&((unsigned short)(255u)))))));
    *(buf1+(alu7+15424)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast6>>((unsigned short)(0u)))&((unsigned short)(255u)))))));
    *(buf1+(alu7+15425)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast6>>((unsigned short)(8u)))&((unsigned short)(255u)))))));
    *(buf1+(alu7+15426)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast8>>((unsigned short)(0u)))&((unsigned short)(255u)))))));
    *(buf1+(alu7+15427)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((cast8>>((unsigned short)(8u)))&((unsigned short)(255u)))))));
    *(buf1+(alu8+1)) = ((signed char)(((val24.x>>cast1)&15u)));
    *(buf1+(alu8+2)) = ((signed char)(((val24.x>>cast2)&15u)));
    *(buf1+(alu8+3)) = ((signed char)(((val24.x>>cast3)&15u)));
    *(buf1+(alu8+4)) = ((signed char)(((val24.y>>cast4)&15u)));
    *(buf1+(alu8+5)) = ((signed char)(((val24.y>>cast1)&15u)));
    *(buf1+(alu8+6)) = ((signed char)(((val24.y>>cast2)&15u)));
    *(buf1+(alu8+7)) = ((signed char)(((val24.y>>cast3)&15u)));
    *(buf1+(alu8+8)) = ((signed char)(((val24.z>>cast4)&15u)));
    *(buf1+(alu8+9)) = ((signed char)(((val24.z>>cast1)&15u)));
    *(buf1+(alu8+10)) = ((signed char)(((val24.z>>cast2)&15u)));
    *(buf1+(alu8+11)) = ((signed char)(((val24.z>>cast3)&15u)));
    *(buf1+(alu8+12)) = ((signed char)(((val24.w>>cast4)&15u)));
    *(buf1+(alu8+13)) = ((signed char)(((val24.w>>cast1)&15u)));
    *(buf1+(alu8+14)) = ((signed char)(((val24.w>>cast2)&15u)));
    *(buf1+(alu8+15)) = ((signed char)(((val24.w>>cast3)&15u)));
    *(buf1+(alu8+5120)) = ((signed char)(((val25.x>>cast4)&15u)));
    *(buf1+(alu8+5121)) = ((signed char)(((val25.x>>cast1)&15u)));
    *(buf1+(alu8+5122)) = ((signed char)(((val25.x>>cast2)&15u)));
    *(buf1+(alu8+5123)) = ((signed char)(((val25.x>>cast3)&15u)));
    *(buf1+(alu8+5124)) = ((signed char)(((val25.y>>cast4)&15u)));
    *(buf1+(alu8+5125)) = ((signed char)(((val25.y>>cast1)&15u)));
    *(buf1+(alu8+5126)) = ((signed char)(((val25.y>>cast2)&15u)));
    *(buf1+(alu8+5127)) = ((signed char)(((val25.y>>cast3)&15u)));
    *(buf1+(alu8+5128)) = ((signed char)(((val25.z>>cast4)&15u)));
    *(buf1+(alu8+5129)) = ((signed char)(((val25.z>>cast1)&15u)));
    *(buf1+(alu8+5130)) = ((signed char)(((val25.z>>cast2)&15u)));
    *(buf1+(alu8+5131)) = ((signed char)(((val25.z>>cast3)&15u)));
    *(buf1+(alu8+5132)) = ((signed char)(((val25.w>>cast4)&15u)));
    *(buf1+(alu8+5133)) = ((signed char)(((val25.w>>cast1)&15u)));
    *(buf1+(alu8+5134)) = ((signed char)(((val25.w>>cast2)&15u)));
    *(buf1+(alu8+5135)) = ((signed char)(((val25.w>>cast3)&15u)));
    *(buf1+(alu8+10240)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val23>>0u)&255u)))));
    *(buf1+(alu8+10241)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val23>>8u)&255u)))));
    *(buf1+(alu8+10242)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val23>>16u)&255u)))));
    *(buf1+(alu8+10243)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val23>>24u)&255u)))));
    *(buf1+(alu8+10244)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val12>>0u)&255u)))));
    *(buf1+(alu8+10245)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val12>>8u)&255u)))));
    *(buf1+(alu8+10246)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val12>>16u)&255u)))));
    *(buf1+(alu8+10247)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val12>>24u)&255u)))));
    *(buf1+(alu8+10248)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val13>>0u)&255u)))));
    *(buf1+(alu8+10249)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val13>>8u)&255u)))));
    *(buf1+(alu8+10250)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val13>>16u)&255u)))));
    *(buf1+(alu8+10251)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val13>>24u)&255u)))));
    *(buf1+(alu8+10252)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val14>>0u)&255u)))));
    *(buf1+(alu8+10253)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val14>>8u)&255u)))));
    *(buf1+(alu8+10254)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val14>>16u)&255u)))));
    *(buf1+(alu8+10255)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val14>>24u)&255u)))));
    *(buf1+(alu8+15360)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val15>>0u)&255u)))));
    *(buf1+(alu8+15361)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val15>>8u)&255u)))));
    *(buf1+(alu8+15362)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val15>>16u)&255u)))));
    *(buf1+(alu8+15363)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val15>>24u)&255u)))));
    *(buf1+(alu8+15364)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val16>>0u)&255u)))));
    *(buf1+(alu8+15365)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val16>>8u)&255u)))));
    *(buf1+(alu8+15366)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val16>>16u)&255u)))));
    *(buf1+(alu8+15367)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val16>>24u)&255u)))));
    *(buf1+(alu8+15368)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val17>>0u)&255u)))));
    *(buf1+(alu8+15369)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val17>>8u)&255u)))));
    *(buf1+(alu8+15370)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val17>>16u)&255u)))));
    *(buf1+(alu8+15371)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val17>>24u)&255u)))));
    *(buf1+(alu8+15372)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val18>>0u)&255u)))));
    *(buf1+(alu8+15373)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val18>>8u)&255u)))));
    *(buf1+(alu8+15374)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val18>>16u)&255u)))));
    *(buf1+(alu8+15375)) = tg_bitcast<signed char>((unsigned char)(((unsigned char)(((val18>>24u)&255u)))));
    *(buf1+alu8) = ((signed char)(((val24.x>>cast4)&15u)));
    __syncthreads();
    signed char val26 = (*(buf1+(alu12+10240)));
    signed char val27 = (*(buf1+(alu12+10241)));
    signed char val28 = (*(buf1+(alu12+10242)));
    signed char val29 = (*(buf1+(alu12+10243)));
    signed char val30 = (*(buf1+(alu12+10256)));
    signed char val31 = (*(buf1+(alu12+10257)));
    signed char val32 = (*(buf1+(alu12+10258)));
    signed char val33 = (*(buf1+(alu12+10259)));
    signed char val34 = (*(buf1+(alu12+10272)));
    signed char val35 = (*(buf1+(alu12+10273)));
    signed char val36 = (*(buf1+(alu12+10274)));
    signed char val37 = (*(buf1+(alu12+10275)));
    signed char val38 = (*(buf1+(alu12+10288)));
    signed char val39 = (*(buf1+(alu12+10289)));
    signed char val40 = (*(buf1+(alu12+10290)));
    signed char val41 = (*(buf1+(alu12+10291)));
    signed char val42 = (*(buf1+(alu12+10880)));
    signed char val43 = (*(buf1+(alu12+10881)));
    signed char val44 = (*(buf1+(alu12+10882)));
    signed char val45 = (*(buf1+(alu12+10883)));
    signed char val46 = (*(buf1+(alu12+10896)));
    signed char val47 = (*(buf1+(alu12+10897)));
    signed char val48 = (*(buf1+(alu12+10898)));
    signed char val49 = (*(buf1+(alu12+10899)));
    signed char val50 = (*(buf1+(alu12+10912)));
    signed char val51 = (*(buf1+(alu12+10913)));
    signed char val52 = (*(buf1+(alu12+10914)));
    signed char val53 = (*(buf1+(alu12+10915)));
    signed char val54 = (*(buf1+(alu12+10928)));
    signed char val55 = (*(buf1+(alu12+10929)));
    signed char val56 = (*(buf1+(alu12+10930)));
    signed char val57 = (*(buf1+(alu12+10931)));
    signed char val58 = (*(buf1+(alu12+11520)));
    signed char val59 = (*(buf1+(alu12+11521)));
    signed char val60 = (*(buf1+(alu12+11522)));
    signed char val61 = (*(buf1+(alu12+11523)));
    signed char val62 = (*(buf1+(alu12+11536)));
    signed char val63 = (*(buf1+(alu12+11537)));
    signed char val64 = (*(buf1+(alu12+11538)));
    signed char val65 = (*(buf1+(alu12+11539)));
    signed char val66 = (*(buf1+(alu12+11552)));
    signed char val67 = (*(buf1+(alu12+11553)));
    signed char val68 = (*(buf1+(alu12+11554)));
    signed char val69 = (*(buf1+(alu12+11555)));
    signed char val70 = (*(buf1+(alu12+11568)));
    signed char val71 = (*(buf1+(alu12+11569)));
    signed char val72 = (*(buf1+(alu12+11570)));
    signed char val73 = (*(buf1+(alu12+11571)));
    signed char val74 = (*(buf1+(alu12+12160)));
    signed char val75 = (*(buf1+(alu12+12161)));
    signed char val76 = (*(buf1+(alu12+12162)));
    signed char val77 = (*(buf1+(alu12+12163)));
    signed char val78 = (*(buf1+(alu12+12176)));
    signed char val79 = (*(buf1+(alu12+12177)));
    signed char val80 = (*(buf1+(alu12+12178)));
    signed char val81 = (*(buf1+(alu12+12179)));
    signed char val82 = (*(buf1+(alu12+12192)));
    signed char val83 = (*(buf1+(alu12+12193)));
    signed char val84 = (*(buf1+(alu12+12194)));
    signed char val85 = (*(buf1+(alu12+12195)));
    signed char val86 = (*(buf1+(alu12+12208)));
    signed char val87 = (*(buf1+(alu12+12209)));
    signed char val88 = (*(buf1+(alu12+12210)));
    signed char val89 = (*(buf1+(alu12+12211)));
    signed char val90 = (*(buf1+(alu13+64)));
    signed char val91 = (*(buf1+(alu13+65)));
    signed char val92 = (*(buf1+(alu13+66)));
    signed char val93 = (*(buf1+(alu13+67)));
    signed char val94 = (*(buf1+(alu13+72)));
    signed char val95 = (*(buf1+(alu13+73)));
    signed char val96 = (*(buf1+(alu13+74)));
    signed char val97 = (*(buf1+(alu13+75)));
    signed char val98 = (*(buf1+(alu13+704)));
    signed char val99 = (*(buf1+(alu13+705)));
    signed char val100 = (*(buf1+(alu13+706)));
    signed char val101 = (*(buf1+(alu13+707)));
    signed char val102 = (*(buf1+(alu13+712)));
    signed char val103 = (*(buf1+(alu13+713)));
    signed char val104 = (*(buf1+(alu13+714)));
    signed char val105 = (*(buf1+(alu13+715)));
    signed char val106 = (*(buf1+(alu13+1344)));
    signed char val107 = (*(buf1+(alu13+1345)));
    signed char val108 = (*(buf1+(alu13+1346)));
    signed char val109 = (*(buf1+(alu13+1347)));
    signed char val110 = (*(buf1+(alu13+1352)));
    signed char val111 = (*(buf1+(alu13+1353)));
    signed char val112 = (*(buf1+(alu13+1354)));
    signed char val113 = (*(buf1+(alu13+1355)));
    signed char val114 = (*(buf1+(alu13+1984)));
    signed char val115 = (*(buf1+(alu13+1985)));
    signed char val116 = (*(buf1+(alu13+1986)));
    signed char val117 = (*(buf1+(alu13+1987)));
    signed char val118 = (*(buf1+(alu13+1992)));
    signed char val119 = (*(buf1+(alu13+1993)));
    signed char val120 = (*(buf1+(alu13+1994)));
    signed char val121 = (*(buf1+(alu13+1995)));
    signed char val122 = (*(buf1+(alu13+2624)));
    signed char val123 = (*(buf1+(alu13+2625)));
    signed char val124 = (*(buf1+(alu13+2626)));
    signed char val125 = (*(buf1+(alu13+2627)));
    signed char val126 = (*(buf1+(alu13+2632)));
    signed char val127 = (*(buf1+(alu13+2633)));
    signed char val128 = (*(buf1+(alu13+2634)));
    signed char val129 = (*(buf1+(alu13+2635)));
    signed char val130 = (*(buf1+(alu13+3264)));
    signed char val131 = (*(buf1+(alu13+3265)));
    signed char val132 = (*(buf1+(alu13+3266)));
    signed char val133 = (*(buf1+(alu13+3267)));
    signed char val134 = (*(buf1+(alu13+3272)));
    signed char val135 = (*(buf1+(alu13+3273)));
    signed char val136 = (*(buf1+(alu13+3274)));
    signed char val137 = (*(buf1+(alu13+3275)));
    signed char val138 = (*(buf1+(alu13+3904)));
    signed char val139 = (*(buf1+(alu13+3905)));
    signed char val140 = (*(buf1+(alu13+3906)));
    signed char val141 = (*(buf1+(alu13+3907)));
    signed char val142 = (*(buf1+(alu13+3912)));
    signed char val143 = (*(buf1+(alu13+3913)));
    signed char val144 = (*(buf1+(alu13+3914)));
    signed char val145 = (*(buf1+(alu13+3915)));
    signed char val146 = (*(buf1+(alu13+4544)));
    signed char val147 = (*(buf1+(alu13+4545)));
    signed char val148 = (*(buf1+(alu13+4546)));
    signed char val149 = (*(buf1+(alu13+4547)));
    signed char val150 = (*(buf1+(alu13+4552)));
    signed char val151 = (*(buf1+(alu13+4553)));
    signed char val152 = (*(buf1+(alu13+4554)));
    signed char val153 = (*(buf1+(alu13+4555)));
    signed char val154 = (*(buf1+(alu14+10304)));
    signed char val155 = (*(buf1+(alu14+10305)));
    signed char val156 = (*(buf1+(alu14+10306)));
    signed char val157 = (*(buf1+(alu14+10307)));
    signed char val158 = (*(buf1+(alu14+10312)));
    signed char val159 = (*(buf1+(alu14+10313)));
    signed char val160 = (*(buf1+(alu14+10314)));
    signed char val161 = (*(buf1+(alu14+10315)));
    signed char val162 = (*(buf1+(alu14+10384)));
    signed char val163 = (*(buf1+(alu14+10385)));
    signed char val164 = (*(buf1+(alu14+10386)));
    signed char val165 = (*(buf1+(alu14+10387)));
    signed char val166 = (*(buf1+(alu14+10392)));
    signed char val167 = (*(buf1+(alu14+10393)));
    signed char val168 = (*(buf1+(alu14+10394)));
    signed char val169 = (*(buf1+(alu14+10395)));
    signed char val170 = (*(buf1+(alu14+10944)));
    signed char val171 = (*(buf1+(alu14+10945)));
    signed char val172 = (*(buf1+(alu14+10946)));
    signed char val173 = (*(buf1+(alu14+10947)));
    signed char val174 = (*(buf1+(alu14+10952)));
    signed char val175 = (*(buf1+(alu14+10953)));
    signed char val176 = (*(buf1+(alu14+10954)));
    signed char val177 = (*(buf1+(alu14+10955)));
    signed char val178 = (*(buf1+(alu14+11024)));
    signed char val179 = (*(buf1+(alu14+11025)));
    signed char val180 = (*(buf1+(alu14+11026)));
    signed char val181 = (*(buf1+(alu14+11027)));
    signed char val182 = (*(buf1+(alu14+11032)));
    signed char val183 = (*(buf1+(alu14+11033)));
    signed char val184 = (*(buf1+(alu14+11034)));
    signed char val185 = (*(buf1+(alu14+11035)));
    signed char val186 = (*(buf1+(alu14+11584)));
    signed char val187 = (*(buf1+(alu14+11585)));
    signed char val188 = (*(buf1+(alu14+11586)));
    signed char val189 = (*(buf1+(alu14+11587)));
    signed char val190 = (*(buf1+(alu14+11592)));
    signed char val191 = (*(buf1+(alu14+11593)));
    signed char val192 = (*(buf1+(alu14+11594)));
    signed char val193 = (*(buf1+(alu14+11595)));
    signed char val194 = (*(buf1+(alu14+11664)));
    signed char val195 = (*(buf1+(alu14+11665)));
    signed char val196 = (*(buf1+(alu14+11666)));
    signed char val197 = (*(buf1+(alu14+11667)));
    signed char val198 = (*(buf1+(alu14+11672)));
    signed char val199 = (*(buf1+(alu14+11673)));
    signed char val200 = (*(buf1+(alu14+11674)));
    signed char val201 = (*(buf1+(alu14+11675)));
    signed char val202 = (*(buf1+(alu14+12224)));
    signed char val203 = (*(buf1+(alu14+12225)));
    signed char val204 = (*(buf1+(alu14+12226)));
    signed char val205 = (*(buf1+(alu14+12227)));
    signed char val206 = (*(buf1+(alu14+12232)));
    signed char val207 = (*(buf1+(alu14+12233)));
    signed char val208 = (*(buf1+(alu14+12234)));
    signed char val209 = (*(buf1+(alu14+12235)));
    signed char val210 = (*(buf1+(alu14+12304)));
    signed char val211 = (*(buf1+(alu14+12305)));
    signed char val212 = (*(buf1+(alu14+12306)));
    signed char val213 = (*(buf1+(alu14+12307)));
    signed char val214 = (*(buf1+(alu14+12312)));
    signed char val215 = (*(buf1+(alu14+12313)));
    signed char val216 = (*(buf1+(alu14+12314)));
    signed char val217 = (*(buf1+(alu14+12315)));
    signed_char8 cast17 = make_signed_char8(val26,val27,val28,val29,val30,val31,val32,val33);
    signed_char8 cast18 = make_signed_char8(val34,val35,val36,val37,val38,val39,val40,val41);
    signed_char8 cast19 = make_signed_char8(val42,val43,val44,val45,val46,val47,val48,val49);
    signed_char8 cast20 = make_signed_char8(val50,val51,val52,val53,val54,val55,val56,val57);
    signed_char8 cast21 = make_signed_char8(val58,val59,val60,val61,val62,val63,val64,val65);
    signed_char8 cast22 = make_signed_char8(val66,val67,val68,val69,val70,val71,val72,val73);
    signed_char8 cast23 = make_signed_char8(val74,val75,val76,val77,val78,val79,val80,val81);
    signed_char8 cast24 = make_signed_char8(val82,val83,val84,val85,val86,val87,val88,val89);
































































































    float cast25 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val90))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val91))))<<((unsigned short)(8u))))))));
    float cast26 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val92))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val93))))<<((unsigned short)(8u))))))));
    float cast27 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val94))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val95))))<<((unsigned short)(8u))))))));
    float cast28 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val96))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val97))))<<((unsigned short)(8u))))))));
    float cast29 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val98))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val99))))<<((unsigned short)(8u))))))));
    float cast30 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val100))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val101))))<<((unsigned short)(8u))))))));
    float cast31 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val102))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val103))))<<((unsigned short)(8u))))))));
    float cast32 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val104))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val105))))<<((unsigned short)(8u))))))));
    float cast33 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val106))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val107))))<<((unsigned short)(8u))))))));
    float cast34 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val108))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val109))))<<((unsigned short)(8u))))))));
    float cast35 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val110))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val111))))<<((unsigned short)(8u))))))));
    float cast36 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val112))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val113))))<<((unsigned short)(8u))))))));
    float cast37 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val114))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val115))))<<((unsigned short)(8u))))))));
    float cast38 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val116))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val117))))<<((unsigned short)(8u))))))));
    float cast39 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val118))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val119))))<<((unsigned short)(8u))))))));
    float cast40 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val120))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val121))))<<((unsigned short)(8u))))))));
    float cast41 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val122))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val123))))<<((unsigned short)(8u))))))));
    float cast42 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val124))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val125))))<<((unsigned short)(8u))))))));
    float cast43 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val126))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val127))))<<((unsigned short)(8u))))))));
    float cast44 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val128))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val129))))<<((unsigned short)(8u))))))));
    float cast45 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val130))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val131))))<<((unsigned short)(8u))))))));
    float cast46 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val132))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val133))))<<((unsigned short)(8u))))))));
    float cast47 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val134))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val135))))<<((unsigned short)(8u))))))));
    float cast48 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val136))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val137))))<<((unsigned short)(8u))))))));
    float cast49 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val138))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val139))))<<((unsigned short)(8u))))))));
    float cast50 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val140))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val141))))<<((unsigned short)(8u))))))));
    float cast51 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val142))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val143))))<<((unsigned short)(8u))))))));
    float cast52 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val144))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val145))))<<((unsigned short)(8u))))))));
    float cast53 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val146))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val147))))<<((unsigned short)(8u))))))));
    float cast54 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val148))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val149))))<<((unsigned short)(8u))))))));
    float cast55 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val150))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val151))))<<((unsigned short)(8u))))))));
    float cast56 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val152))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val153))))<<((unsigned short)(8u))))))));
    int4 wmma28 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+alu10)+0))), cast17, cast0);
    int4 wmma0 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+32))+0))), cast18, cast0);
    int4 wmma4 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+1280))+0))), cast17, cast0);
    int4 wmma8 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+1312))+0))), cast18, cast0);
    int4 wmma12 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+2560))+0))), cast17, cast0);
    int4 wmma16 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+2592))+0))), cast18, cast0);
    int4 wmma20 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+3840))+0))), cast17, cast0);
    int4 wmma24 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+3872))+0))), cast18, cast0);
    float cast57 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val154))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val155))))<<((unsigned short)(8u))))))));
    float cast58 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val156))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val157))))<<((unsigned short)(8u))))))));
    float cast59 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val158))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val159))))<<((unsigned short)(8u))))))));
    float cast60 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val160))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val161))))<<((unsigned short)(8u))))))));
    float cast61 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val162))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val163))))<<((unsigned short)(8u))))))));
    float cast62 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val164))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val165))))<<((unsigned short)(8u))))))));
    float cast63 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val166))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val167))))<<((unsigned short)(8u))))))));
    float cast64 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val168))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val169))))<<((unsigned short)(8u))))))));
    (*(buf0+0)) = ((*(buf0+0))+(cast25*cast57*((float)(wmma28.x)))+(cast26*cast58)+(cast27*cast59*((float)(wmma0.x)))+(cast28*cast60));
    (*(buf0+16)) = ((*(buf0+16))+(cast29*cast57*((float)(wmma28.z)))+(cast30*cast58)+(cast31*cast59*((float)(wmma0.z)))+(cast32*cast60));
    (*(buf0+32)) = ((*(buf0+32))+(cast25*cast61*((float)(wmma28.y)))+(cast26*cast62)+(cast27*cast63*((float)(wmma0.y)))+(cast28*cast64));
    (*(buf0+48)) = ((*(buf0+48))+(cast29*cast61*((float)(wmma28.w)))+(cast30*cast62)+(cast31*cast63*((float)(wmma0.w)))+(cast32*cast64));
    (*(buf0+1)) = ((*(buf0+1))+(cast33*cast57*((float)(wmma4.x)))+(cast34*cast58)+(cast35*cast59*((float)(wmma8.x)))+(cast36*cast60));
    (*(buf0+17)) = ((*(buf0+17))+(cast37*cast57*((float)(wmma4.z)))+(cast38*cast58)+(cast39*cast59*((float)(wmma8.z)))+(cast40*cast60));
    (*(buf0+33)) = ((*(buf0+33))+(cast33*cast61*((float)(wmma4.y)))+(cast34*cast62)+(cast35*cast63*((float)(wmma8.y)))+(cast36*cast64));
    (*(buf0+49)) = ((*(buf0+49))+(cast37*cast61*((float)(wmma4.w)))+(cast38*cast62)+(cast39*cast63*((float)(wmma8.w)))+(cast40*cast64));
    (*(buf0+2)) = ((*(buf0+2))+(cast41*cast57*((float)(wmma12.x)))+(cast42*cast58)+(cast43*cast59*((float)(wmma16.x)))+(cast44*cast60));
    (*(buf0+18)) = ((*(buf0+18))+(cast45*cast57*((float)(wmma12.z)))+(cast46*cast58)+(cast47*cast59*((float)(wmma16.z)))+(cast48*cast60));
    (*(buf0+34)) = ((*(buf0+34))+(cast41*cast61*((float)(wmma12.y)))+(cast42*cast62)+(cast43*cast63*((float)(wmma16.y)))+(cast44*cast64));
    (*(buf0+50)) = ((*(buf0+50))+(cast45*cast61*((float)(wmma12.w)))+(cast46*cast62)+(cast47*cast63*((float)(wmma16.w)))+(cast48*cast64));
    (*(buf0+3)) = ((*(buf0+3))+(cast49*cast57*((float)(wmma20.x)))+(cast50*cast58)+(cast51*cast59*((float)(wmma24.x)))+(cast52*cast60));
    (*(buf0+19)) = ((*(buf0+19))+(cast53*cast57*((float)(wmma20.z)))+(cast54*cast58)+(cast55*cast59*((float)(wmma24.z)))+(cast56*cast60));
    (*(buf0+35)) = ((*(buf0+35))+(cast49*cast61*((float)(wmma20.y)))+(cast50*cast62)+(cast51*cast63*((float)(wmma24.y)))+(cast52*cast64));
    (*(buf0+51)) = ((*(buf0+51))+(cast53*cast61*((float)(wmma20.w)))+(cast54*cast62)+(cast55*cast63*((float)(wmma24.w)))+(cast56*cast64));
    int4 wmma29 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+alu10)+0))), cast19, cast0);
    int4 wmma1 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+32))+0))), cast20, cast0);
    int4 wmma5 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+1280))+0))), cast19, cast0);
    int4 wmma9 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+1312))+0))), cast20, cast0);
    int4 wmma13 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+2560))+0))), cast19, cast0);
    int4 wmma17 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+2592))+0))), cast20, cast0);
    int4 wmma21 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+3840))+0))), cast19, cast0);
    int4 wmma25 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+3872))+0))), cast20, cast0);
    float cast65 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val170))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val171))))<<((unsigned short)(8u))))))));
    float cast66 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val172))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val173))))<<((unsigned short)(8u))))))));
    float cast67 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val174))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val175))))<<((unsigned short)(8u))))))));
    float cast68 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val176))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val177))))<<((unsigned short)(8u))))))));
    float cast69 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val178))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val179))))<<((unsigned short)(8u))))))));
    float cast70 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val180))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val181))))<<((unsigned short)(8u))))))));
    float cast71 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val182))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val183))))<<((unsigned short)(8u))))))));
    float cast72 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val184))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val185))))<<((unsigned short)(8u))))))));
    (*(buf0+4)) = ((*(buf0+4))+(cast25*cast65*((float)(wmma29.x)))+(cast26*cast66)+(cast27*cast67*((float)(wmma1.x)))+(cast28*cast68));
    (*(buf0+20)) = ((*(buf0+20))+(cast29*cast65*((float)(wmma29.z)))+(cast30*cast66)+(cast31*cast67*((float)(wmma1.z)))+(cast32*cast68));
    (*(buf0+36)) = ((*(buf0+36))+(cast25*cast69*((float)(wmma29.y)))+(cast26*cast70)+(cast27*cast71*((float)(wmma1.y)))+(cast28*cast72));
    (*(buf0+52)) = ((*(buf0+52))+(cast29*cast69*((float)(wmma29.w)))+(cast30*cast70)+(cast31*cast71*((float)(wmma1.w)))+(cast32*cast72));
    (*(buf0+5)) = ((*(buf0+5))+(cast33*cast65*((float)(wmma5.x)))+(cast34*cast66)+(cast35*cast67*((float)(wmma9.x)))+(cast36*cast68));
    (*(buf0+21)) = ((*(buf0+21))+(cast37*cast65*((float)(wmma5.z)))+(cast38*cast66)+(cast39*cast67*((float)(wmma9.z)))+(cast40*cast68));
    (*(buf0+37)) = ((*(buf0+37))+(cast33*cast69*((float)(wmma5.y)))+(cast34*cast70)+(cast35*cast71*((float)(wmma9.y)))+(cast36*cast72));
    (*(buf0+53)) = ((*(buf0+53))+(cast37*cast69*((float)(wmma5.w)))+(cast38*cast70)+(cast39*cast71*((float)(wmma9.w)))+(cast40*cast72));
    (*(buf0+6)) = ((*(buf0+6))+(cast41*cast65*((float)(wmma13.x)))+(cast42*cast66)+(cast43*cast67*((float)(wmma17.x)))+(cast44*cast68));
    (*(buf0+22)) = ((*(buf0+22))+(cast45*cast65*((float)(wmma13.z)))+(cast46*cast66)+(cast47*cast67*((float)(wmma17.z)))+(cast48*cast68));
    (*(buf0+38)) = ((*(buf0+38))+(cast41*cast69*((float)(wmma13.y)))+(cast42*cast70)+(cast43*cast71*((float)(wmma17.y)))+(cast44*cast72));
    (*(buf0+54)) = ((*(buf0+54))+(cast45*cast69*((float)(wmma13.w)))+(cast46*cast70)+(cast47*cast71*((float)(wmma17.w)))+(cast48*cast72));
    (*(buf0+7)) = ((*(buf0+7))+(cast49*cast65*((float)(wmma21.x)))+(cast50*cast66)+(cast51*cast67*((float)(wmma25.x)))+(cast52*cast68));
    (*(buf0+23)) = ((*(buf0+23))+(cast53*cast65*((float)(wmma21.z)))+(cast54*cast66)+(cast55*cast67*((float)(wmma25.z)))+(cast56*cast68));
    (*(buf0+39)) = ((*(buf0+39))+(cast49*cast69*((float)(wmma21.y)))+(cast50*cast70)+(cast51*cast71*((float)(wmma25.y)))+(cast52*cast72));
    (*(buf0+55)) = ((*(buf0+55))+(cast53*cast69*((float)(wmma21.w)))+(cast54*cast70)+(cast55*cast71*((float)(wmma25.w)))+(cast56*cast72));
    int4 wmma30 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+alu10)+0))), cast21, cast0);
    int4 wmma2 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+32))+0))), cast22, cast0);
    int4 wmma6 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+1280))+0))), cast21, cast0);
    int4 wmma10 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+1312))+0))), cast22, cast0);
    int4 wmma14 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+2560))+0))), cast21, cast0);
    int4 wmma18 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+2592))+0))), cast22, cast0);
    int4 wmma22 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+3840))+0))), cast21, cast0);
    int4 wmma26 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+3872))+0))), cast22, cast0);
    float cast73 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val186))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val187))))<<((unsigned short)(8u))))))));
    float cast74 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val188))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val189))))<<((unsigned short)(8u))))))));
    float cast75 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val190))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val191))))<<((unsigned short)(8u))))))));
    float cast76 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val192))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val193))))<<((unsigned short)(8u))))))));
    float cast77 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val194))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val195))))<<((unsigned short)(8u))))))));
    float cast78 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val196))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val197))))<<((unsigned short)(8u))))))));
    float cast79 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val198))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val199))))<<((unsigned short)(8u))))))));
    float cast80 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val200))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val201))))<<((unsigned short)(8u))))))));
    (*(buf0+8)) = ((*(buf0+8))+(cast25*cast73*((float)(wmma30.x)))+(cast26*cast74)+(cast27*cast75*((float)(wmma2.x)))+(cast28*cast76));
    (*(buf0+24)) = ((*(buf0+24))+(cast29*cast73*((float)(wmma30.z)))+(cast30*cast74)+(cast31*cast75*((float)(wmma2.z)))+(cast32*cast76));
    (*(buf0+40)) = ((*(buf0+40))+(cast25*cast77*((float)(wmma30.y)))+(cast26*cast78)+(cast27*cast79*((float)(wmma2.y)))+(cast28*cast80));
    (*(buf0+56)) = ((*(buf0+56))+(cast29*cast77*((float)(wmma30.w)))+(cast30*cast78)+(cast31*cast79*((float)(wmma2.w)))+(cast32*cast80));
    (*(buf0+9)) = ((*(buf0+9))+(cast33*cast73*((float)(wmma6.x)))+(cast34*cast74)+(cast35*cast75*((float)(wmma10.x)))+(cast36*cast76));
    (*(buf0+25)) = ((*(buf0+25))+(cast37*cast73*((float)(wmma6.z)))+(cast38*cast74)+(cast39*cast75*((float)(wmma10.z)))+(cast40*cast76));
    (*(buf0+41)) = ((*(buf0+41))+(cast33*cast77*((float)(wmma6.y)))+(cast34*cast78)+(cast35*cast79*((float)(wmma10.y)))+(cast36*cast80));
    (*(buf0+57)) = ((*(buf0+57))+(cast37*cast77*((float)(wmma6.w)))+(cast38*cast78)+(cast39*cast79*((float)(wmma10.w)))+(cast40*cast80));
    (*(buf0+10)) = ((*(buf0+10))+(cast41*cast73*((float)(wmma14.x)))+(cast42*cast74)+(cast43*cast75*((float)(wmma18.x)))+(cast44*cast76));
    (*(buf0+26)) = ((*(buf0+26))+(cast45*cast73*((float)(wmma14.z)))+(cast46*cast74)+(cast47*cast75*((float)(wmma18.z)))+(cast48*cast76));
    (*(buf0+42)) = ((*(buf0+42))+(cast41*cast77*((float)(wmma14.y)))+(cast42*cast78)+(cast43*cast79*((float)(wmma18.y)))+(cast44*cast80));
    (*(buf0+58)) = ((*(buf0+58))+(cast45*cast77*((float)(wmma14.w)))+(cast46*cast78)+(cast47*cast79*((float)(wmma18.w)))+(cast48*cast80));
    (*(buf0+11)) = ((*(buf0+11))+(cast49*cast73*((float)(wmma22.x)))+(cast50*cast74)+(cast51*cast75*((float)(wmma26.x)))+(cast52*cast76));
    (*(buf0+27)) = ((*(buf0+27))+(cast53*cast73*((float)(wmma22.z)))+(cast54*cast74)+(cast55*cast75*((float)(wmma26.z)))+(cast56*cast76));
    (*(buf0+43)) = ((*(buf0+43))+(cast49*cast77*((float)(wmma22.y)))+(cast50*cast78)+(cast51*cast79*((float)(wmma26.y)))+(cast52*cast80));
    (*(buf0+59)) = ((*(buf0+59))+(cast53*cast77*((float)(wmma22.w)))+(cast54*cast78)+(cast55*cast79*((float)(wmma26.w)))+(cast56*cast80));
    int4 wmma31 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+alu10)+0))), cast23, cast0);
    int4 wmma3 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+32))+0))), cast24, cast0);
    int4 wmma7 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+1280))+0))), cast23, cast0);
    int4 wmma11 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+1312))+0))), cast24, cast0);
    int4 wmma15 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+2560))+0))), cast23, cast0);
    int4 wmma19 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+2592))+0))), cast24, cast0);
    int4 wmma23 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+3840))+0))), cast23, cast0);
    int4 wmma27 = __WMMA_8_16_32_signed_char_int(tg_bitcast<signed_char16>(tg_ldmatrix_x4((const void*)((buf1+(alu10+3872))+0))), cast24, cast0);
    float cast81 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val202))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val203))))<<((unsigned short)(8u))))))));
    float cast82 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val204))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val205))))<<((unsigned short)(8u))))))));
    float cast83 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val206))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val207))))<<((unsigned short)(8u))))))));
    float cast84 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val208))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val209))))<<((unsigned short)(8u))))))));
    float cast85 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val210))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val211))))<<((unsigned short)(8u))))))));
    float cast86 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val212))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val213))))<<((unsigned short)(8u))))))));
    float cast87 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val214))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val215))))<<((unsigned short)(8u))))))));
    float cast88 = ((float)(tg_bitcast<half>((unsigned short)((((unsigned short)(tg_bitcast<unsigned char>((signed char)(val216))))|(((unsigned short)(tg_bitcast<unsigned char>((signed char)(val217))))<<((unsigned short)(8u))))))));
    (*(buf0+12)) = ((*(buf0+12))+(cast25*cast81*((float)(wmma31.x)))+(cast26*cast82)+(cast27*cast83*((float)(wmma3.x)))+(cast28*cast84));
    (*(buf0+28)) = ((*(buf0+28))+(cast29*cast81*((float)(wmma31.z)))+(cast30*cast82)+(cast31*cast83*((float)(wmma3.z)))+(cast32*cast84));
    (*(buf0+44)) = ((*(buf0+44))+(cast25*cast85*((float)(wmma31.y)))+(cast26*cast86)+(cast27*cast87*((float)(wmma3.y)))+(cast28*cast88));
    (*(buf0+60)) = ((*(buf0+60))+(cast29*cast85*((float)(wmma31.w)))+(cast30*cast86)+(cast31*cast87*((float)(wmma3.w)))+(cast32*cast88));
    (*(buf0+13)) = ((*(buf0+13))+(cast33*cast81*((float)(wmma7.x)))+(cast34*cast82)+(cast35*cast83*((float)(wmma11.x)))+(cast36*cast84));
    (*(buf0+29)) = ((*(buf0+29))+(cast37*cast81*((float)(wmma7.z)))+(cast38*cast82)+(cast39*cast83*((float)(wmma11.z)))+(cast40*cast84));
    (*(buf0+45)) = ((*(buf0+45))+(cast33*cast85*((float)(wmma7.y)))+(cast34*cast86)+(cast35*cast87*((float)(wmma11.y)))+(cast36*cast88));
    (*(buf0+61)) = ((*(buf0+61))+(cast37*cast85*((float)(wmma7.w)))+(cast38*cast86)+(cast39*cast87*((float)(wmma11.w)))+(cast40*cast88));
    (*(buf0+14)) = ((*(buf0+14))+(cast41*cast81*((float)(wmma15.x)))+(cast42*cast82)+(cast43*cast83*((float)(wmma19.x)))+(cast44*cast84));
    (*(buf0+30)) = ((*(buf0+30))+(cast45*cast81*((float)(wmma15.z)))+(cast46*cast82)+(cast47*cast83*((float)(wmma19.z)))+(cast48*cast84));
    (*(buf0+46)) = ((*(buf0+46))+(cast41*cast85*((float)(wmma15.y)))+(cast42*cast86)+(cast43*cast87*((float)(wmma19.y)))+(cast44*cast88));
    (*(buf0+62)) = ((*(buf0+62))+(cast45*cast85*((float)(wmma15.w)))+(cast46*cast86)+(cast47*cast87*((float)(wmma19.w)))+(cast48*cast88));
    (*(buf0+15)) = ((*(buf0+15))+(cast49*cast81*((float)(wmma23.x)))+(cast50*cast82)+(cast51*cast83*((float)(wmma27.x)))+(cast52*cast84));
    (*(buf0+31)) = ((*(buf0+31))+(cast53*cast81*((float)(wmma23.z)))+(cast54*cast82)+(cast55*cast83*((float)(wmma27.z)))+(cast56*cast84));
    (*(buf0+47)) = ((*(buf0+47))+(cast49*cast85*((float)(wmma23.y)))+(cast50*cast86)+(cast51*cast87*((float)(wmma27.y)))+(cast52*cast88));
    (*(buf0+63)) = ((*(buf0+63))+(cast53*cast85*((float)(wmma23.w)))+(cast54*cast86)+(cast55*cast87*((float)(wmma27.w)))+(cast56*cast88));
































































  }
    if (direct) {
  float tx0_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+4)), 4);
  float tx0_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+20)), 4);
  float tx0_0_0 = (alu2&1) ? tx0_0_sh : (*(buf0+4));
  float tx0_0_1 = (alu2&1) ? (*(buf0+20)) : tx0_0_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+8))*12288)+(gidx1*128)+((lidx1<<6)+0+alu2+7*(alu2&1))))) = make_float2(tx0_0_0,tx0_0_1);
  float tx0_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+36)), 4);
  float tx0_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+52)), 4);
  float tx0_1_0 = (alu2&1) ? tx0_1_sh : (*(buf0+36));
  float tx0_1_1 = (alu2&1) ? (*(buf0+52)) : tx0_1_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+9))*12288)+(gidx1*128)+((lidx1<<6)+0+alu2+7*(alu2&1))))) = make_float2(tx0_1_0,tx0_1_1);
  float tx1_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+8)), 4);
  float tx1_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+24)), 4);
  float tx1_0_0 = (alu2&1) ? tx1_0_sh : (*(buf0+8));
  float tx1_0_1 = (alu2&1) ? (*(buf0+24)) : tx1_0_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+16))*12288)+(gidx1*128)+((lidx1<<6)+0+alu2+7*(alu2&1))))) = make_float2(tx1_0_0,tx1_0_1);
  float tx1_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+40)), 4);
  float tx1_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+56)), 4);
  float tx1_1_0 = (alu2&1) ? tx1_1_sh : (*(buf0+40));
  float tx1_1_1 = (alu2&1) ? (*(buf0+56)) : tx1_1_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+17))*12288)+(gidx1*128)+((lidx1<<6)+0+alu2+7*(alu2&1))))) = make_float2(tx1_1_0,tx1_1_1);
  float tx2_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+12)), 4);
  float tx2_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+28)), 4);
  float tx2_0_0 = (alu2&1) ? tx2_0_sh : (*(buf0+12));
  float tx2_0_1 = (alu2&1) ? (*(buf0+28)) : tx2_0_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+24))*12288)+(gidx1*128)+((lidx1<<6)+0+alu2+7*(alu2&1))))) = make_float2(tx2_0_0,tx2_0_1);
  float tx2_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+44)), 4);
  float tx2_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+60)), 4);
  float tx2_1_0 = (alu2&1) ? tx2_1_sh : (*(buf0+44));
  float tx2_1_1 = (alu2&1) ? (*(buf0+60)) : tx2_1_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+25))*12288)+(gidx1*128)+((lidx1<<6)+0+alu2+7*(alu2&1))))) = make_float2(tx2_1_0,tx2_1_1);
  float tx3_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+1)), 4);
  float tx3_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+17)), 4);
  float tx3_0_0 = (alu2&1) ? tx3_0_sh : (*(buf0+1));
  float tx3_0_1 = (alu2&1) ? (*(buf0+17)) : tx3_0_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+0))*12288)+(gidx1*128)+((lidx1<<6)+16+alu2+7*(alu2&1))))) = make_float2(tx3_0_0,tx3_0_1);
  float tx3_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+33)), 4);
  float tx3_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+49)), 4);
  float tx3_1_0 = (alu2&1) ? tx3_1_sh : (*(buf0+33));
  float tx3_1_1 = (alu2&1) ? (*(buf0+49)) : tx3_1_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+1))*12288)+(gidx1*128)+((lidx1<<6)+16+alu2+7*(alu2&1))))) = make_float2(tx3_1_0,tx3_1_1);
  float tx4_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+5)), 4);
  float tx4_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+21)), 4);
  float tx4_0_0 = (alu2&1) ? tx4_0_sh : (*(buf0+5));
  float tx4_0_1 = (alu2&1) ? (*(buf0+21)) : tx4_0_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+8))*12288)+(gidx1*128)+((lidx1<<6)+16+alu2+7*(alu2&1))))) = make_float2(tx4_0_0,tx4_0_1);
  float tx4_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+37)), 4);
  float tx4_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+53)), 4);
  float tx4_1_0 = (alu2&1) ? tx4_1_sh : (*(buf0+37));
  float tx4_1_1 = (alu2&1) ? (*(buf0+53)) : tx4_1_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+9))*12288)+(gidx1*128)+((lidx1<<6)+16+alu2+7*(alu2&1))))) = make_float2(tx4_1_0,tx4_1_1);
  float tx5_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+9)), 4);
  float tx5_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+25)), 4);
  float tx5_0_0 = (alu2&1) ? tx5_0_sh : (*(buf0+9));
  float tx5_0_1 = (alu2&1) ? (*(buf0+25)) : tx5_0_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+16))*12288)+(gidx1*128)+((lidx1<<6)+16+alu2+7*(alu2&1))))) = make_float2(tx5_0_0,tx5_0_1);
  float tx5_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+41)), 4);
  float tx5_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+57)), 4);
  float tx5_1_0 = (alu2&1) ? tx5_1_sh : (*(buf0+41));
  float tx5_1_1 = (alu2&1) ? (*(buf0+57)) : tx5_1_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+17))*12288)+(gidx1*128)+((lidx1<<6)+16+alu2+7*(alu2&1))))) = make_float2(tx5_1_0,tx5_1_1);
  float tx6_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+13)), 4);
  float tx6_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+29)), 4);
  float tx6_0_0 = (alu2&1) ? tx6_0_sh : (*(buf0+13));
  float tx6_0_1 = (alu2&1) ? (*(buf0+29)) : tx6_0_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+24))*12288)+(gidx1*128)+((lidx1<<6)+16+alu2+7*(alu2&1))))) = make_float2(tx6_0_0,tx6_0_1);
  float tx6_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+45)), 4);
  float tx6_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+61)), 4);
  float tx6_1_0 = (alu2&1) ? tx6_1_sh : (*(buf0+45));
  float tx6_1_1 = (alu2&1) ? (*(buf0+61)) : tx6_1_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+25))*12288)+(gidx1*128)+((lidx1<<6)+16+alu2+7*(alu2&1))))) = make_float2(tx6_1_0,tx6_1_1);
  float tx7_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+2)), 4);
  float tx7_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+18)), 4);
  float tx7_0_0 = (alu2&1) ? tx7_0_sh : (*(buf0+2));
  float tx7_0_1 = (alu2&1) ? (*(buf0+18)) : tx7_0_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+0))*12288)+(gidx1*128)+((lidx1<<6)+32+alu2+7*(alu2&1))))) = make_float2(tx7_0_0,tx7_0_1);
  float tx7_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+34)), 4);
  float tx7_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+50)), 4);
  float tx7_1_0 = (alu2&1) ? tx7_1_sh : (*(buf0+34));
  float tx7_1_1 = (alu2&1) ? (*(buf0+50)) : tx7_1_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+1))*12288)+(gidx1*128)+((lidx1<<6)+32+alu2+7*(alu2&1))))) = make_float2(tx7_1_0,tx7_1_1);
  float tx8_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+6)), 4);
  float tx8_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+22)), 4);
  float tx8_0_0 = (alu2&1) ? tx8_0_sh : (*(buf0+6));
  float tx8_0_1 = (alu2&1) ? (*(buf0+22)) : tx8_0_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+8))*12288)+(gidx1*128)+((lidx1<<6)+32+alu2+7*(alu2&1))))) = make_float2(tx8_0_0,tx8_0_1);
  float tx8_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+38)), 4);
  float tx8_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+54)), 4);
  float tx8_1_0 = (alu2&1) ? tx8_1_sh : (*(buf0+38));
  float tx8_1_1 = (alu2&1) ? (*(buf0+54)) : tx8_1_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+9))*12288)+(gidx1*128)+((lidx1<<6)+32+alu2+7*(alu2&1))))) = make_float2(tx8_1_0,tx8_1_1);
  float tx9_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+10)), 4);
  float tx9_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+26)), 4);
  float tx9_0_0 = (alu2&1) ? tx9_0_sh : (*(buf0+10));
  float tx9_0_1 = (alu2&1) ? (*(buf0+26)) : tx9_0_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+16))*12288)+(gidx1*128)+((lidx1<<6)+32+alu2+7*(alu2&1))))) = make_float2(tx9_0_0,tx9_0_1);
  float tx9_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+42)), 4);
  float tx9_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+58)), 4);
  float tx9_1_0 = (alu2&1) ? tx9_1_sh : (*(buf0+42));
  float tx9_1_1 = (alu2&1) ? (*(buf0+58)) : tx9_1_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+17))*12288)+(gidx1*128)+((lidx1<<6)+32+alu2+7*(alu2&1))))) = make_float2(tx9_1_0,tx9_1_1);
  float tx10_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+14)), 4);
  float tx10_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+30)), 4);
  float tx10_0_0 = (alu2&1) ? tx10_0_sh : (*(buf0+14));
  float tx10_0_1 = (alu2&1) ? (*(buf0+30)) : tx10_0_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+24))*12288)+(gidx1*128)+((lidx1<<6)+32+alu2+7*(alu2&1))))) = make_float2(tx10_0_0,tx10_0_1);
  float tx10_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+46)), 4);
  float tx10_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+62)), 4);
  float tx10_1_0 = (alu2&1) ? tx10_1_sh : (*(buf0+46));
  float tx10_1_1 = (alu2&1) ? (*(buf0+62)) : tx10_1_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+25))*12288)+(gidx1*128)+((lidx1<<6)+32+alu2+7*(alu2&1))))) = make_float2(tx10_1_0,tx10_1_1);
  float tx11_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+3)), 4);
  float tx11_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+19)), 4);
  float tx11_0_0 = (alu2&1) ? tx11_0_sh : (*(buf0+3));
  float tx11_0_1 = (alu2&1) ? (*(buf0+19)) : tx11_0_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+0))*12288)+(gidx1*128)+((lidx1<<6)+48+alu2+7*(alu2&1))))) = make_float2(tx11_0_0,tx11_0_1);
  float tx11_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+35)), 4);
  float tx11_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+51)), 4);
  float tx11_1_0 = (alu2&1) ? tx11_1_sh : (*(buf0+35));
  float tx11_1_1 = (alu2&1) ? (*(buf0+51)) : tx11_1_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+1))*12288)+(gidx1*128)+((lidx1<<6)+48+alu2+7*(alu2&1))))) = make_float2(tx11_1_0,tx11_1_1);
  float tx12_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+7)), 4);
  float tx12_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+23)), 4);
  float tx12_0_0 = (alu2&1) ? tx12_0_sh : (*(buf0+7));
  float tx12_0_1 = (alu2&1) ? (*(buf0+23)) : tx12_0_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+8))*12288)+(gidx1*128)+((lidx1<<6)+48+alu2+7*(alu2&1))))) = make_float2(tx12_0_0,tx12_0_1);
  float tx12_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+39)), 4);
  float tx12_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+55)), 4);
  float tx12_1_0 = (alu2&1) ? tx12_1_sh : (*(buf0+39));
  float tx12_1_1 = (alu2&1) ? (*(buf0+55)) : tx12_1_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+9))*12288)+(gidx1*128)+((lidx1<<6)+48+alu2+7*(alu2&1))))) = make_float2(tx12_1_0,tx12_1_1);
  float tx13_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+11)), 4);
  float tx13_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+27)), 4);
  float tx13_0_0 = (alu2&1) ? tx13_0_sh : (*(buf0+11));
  float tx13_0_1 = (alu2&1) ? (*(buf0+27)) : tx13_0_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+16))*12288)+(gidx1*128)+((lidx1<<6)+48+alu2+7*(alu2&1))))) = make_float2(tx13_0_0,tx13_0_1);
  float tx13_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+43)), 4);
  float tx13_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+59)), 4);
  float tx13_1_0 = (alu2&1) ? tx13_1_sh : (*(buf0+43));
  float tx13_1_1 = (alu2&1) ? (*(buf0+59)) : tx13_1_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+17))*12288)+(gidx1*128)+((lidx1<<6)+48+alu2+7*(alu2&1))))) = make_float2(tx13_1_0,tx13_1_1);
  float tx14_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+15)), 4);
  float tx14_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+31)), 4);
  float tx14_0_0 = (alu2&1) ? tx14_0_sh : (*(buf0+15));
  float tx14_0_1 = (alu2&1) ? (*(buf0+31)) : tx14_0_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+24))*12288)+(gidx1*128)+((lidx1<<6)+48+alu2+7*(alu2&1))))) = make_float2(tx14_0_0,tx14_0_1);
  float tx14_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+47)), 4);
  float tx14_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+63)), 4);
  float tx14_1_0 = (alu2&1) ? tx14_1_sh : (*(buf0+47));
  float tx14_1_1 = (alu2&1) ? (*(buf0+63)) : tx14_1_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+25))*12288)+(gidx1*128)+((lidx1<<6)+48+alu2+7*(alu2&1))))) = make_float2(tx14_1_0,tx14_1_1);
  float tx15_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+0)), 4);
  float tx15_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+16)), 4);
  float tx15_0_0 = (alu2&1) ? tx15_0_sh : (*(buf0+0));
  float tx15_0_1 = (alu2&1) ? (*(buf0+16)) : tx15_0_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+0))*12288)+(gidx1*128)+((lidx1<<6)+0+alu2+7*(alu2&1))))) = make_float2(tx15_0_0,tx15_0_1);
  float tx15_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+32)), 4);
  float tx15_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+48)), 4);
  float tx15_1_0 = (alu2&1) ? tx15_1_sh : (*(buf0+32));
  float tx15_1_1 = (alu2&1) ? (*(buf0+48)) : tx15_1_sl;
  *((float2*)(data0_2097152+(((gidx0*128+((lidx2<<5)+(alu5<<1)+1))*12288)+(gidx1*128)+((lidx1<<6)+0+alu2+7*(alu2&1))))) = make_float2(tx15_1_0,tx15_1_1);
    } else {
      if (threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0) partial_ids[slot]=tile;
  float tx0_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+4)), 4);
  float tx0_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+20)), 4);
  float tx0_0_0 = (alu2&1) ? tx0_0_sh : (*(buf0+4));
  float tx0_0_1 = (alu2&1) ? (*(buf0+20)) : tx0_0_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+8))*128+((lidx1<<6)+0+alu2+7*(alu2&1))))) = make_float2(tx0_0_0,tx0_0_1);
  float tx0_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+36)), 4);
  float tx0_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+52)), 4);
  float tx0_1_0 = (alu2&1) ? tx0_1_sh : (*(buf0+36));
  float tx0_1_1 = (alu2&1) ? (*(buf0+52)) : tx0_1_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+9))*128+((lidx1<<6)+0+alu2+7*(alu2&1))))) = make_float2(tx0_1_0,tx0_1_1);
  float tx1_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+8)), 4);
  float tx1_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+24)), 4);
  float tx1_0_0 = (alu2&1) ? tx1_0_sh : (*(buf0+8));
  float tx1_0_1 = (alu2&1) ? (*(buf0+24)) : tx1_0_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+16))*128+((lidx1<<6)+0+alu2+7*(alu2&1))))) = make_float2(tx1_0_0,tx1_0_1);
  float tx1_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+40)), 4);
  float tx1_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+56)), 4);
  float tx1_1_0 = (alu2&1) ? tx1_1_sh : (*(buf0+40));
  float tx1_1_1 = (alu2&1) ? (*(buf0+56)) : tx1_1_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+17))*128+((lidx1<<6)+0+alu2+7*(alu2&1))))) = make_float2(tx1_1_0,tx1_1_1);
  float tx2_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+12)), 4);
  float tx2_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+28)), 4);
  float tx2_0_0 = (alu2&1) ? tx2_0_sh : (*(buf0+12));
  float tx2_0_1 = (alu2&1) ? (*(buf0+28)) : tx2_0_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+24))*128+((lidx1<<6)+0+alu2+7*(alu2&1))))) = make_float2(tx2_0_0,tx2_0_1);
  float tx2_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+44)), 4);
  float tx2_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+60)), 4);
  float tx2_1_0 = (alu2&1) ? tx2_1_sh : (*(buf0+44));
  float tx2_1_1 = (alu2&1) ? (*(buf0+60)) : tx2_1_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+25))*128+((lidx1<<6)+0+alu2+7*(alu2&1))))) = make_float2(tx2_1_0,tx2_1_1);
  float tx3_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+1)), 4);
  float tx3_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+17)), 4);
  float tx3_0_0 = (alu2&1) ? tx3_0_sh : (*(buf0+1));
  float tx3_0_1 = (alu2&1) ? (*(buf0+17)) : tx3_0_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+0))*128+((lidx1<<6)+16+alu2+7*(alu2&1))))) = make_float2(tx3_0_0,tx3_0_1);
  float tx3_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+33)), 4);
  float tx3_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+49)), 4);
  float tx3_1_0 = (alu2&1) ? tx3_1_sh : (*(buf0+33));
  float tx3_1_1 = (alu2&1) ? (*(buf0+49)) : tx3_1_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+1))*128+((lidx1<<6)+16+alu2+7*(alu2&1))))) = make_float2(tx3_1_0,tx3_1_1);
  float tx4_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+5)), 4);
  float tx4_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+21)), 4);
  float tx4_0_0 = (alu2&1) ? tx4_0_sh : (*(buf0+5));
  float tx4_0_1 = (alu2&1) ? (*(buf0+21)) : tx4_0_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+8))*128+((lidx1<<6)+16+alu2+7*(alu2&1))))) = make_float2(tx4_0_0,tx4_0_1);
  float tx4_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+37)), 4);
  float tx4_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+53)), 4);
  float tx4_1_0 = (alu2&1) ? tx4_1_sh : (*(buf0+37));
  float tx4_1_1 = (alu2&1) ? (*(buf0+53)) : tx4_1_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+9))*128+((lidx1<<6)+16+alu2+7*(alu2&1))))) = make_float2(tx4_1_0,tx4_1_1);
  float tx5_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+9)), 4);
  float tx5_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+25)), 4);
  float tx5_0_0 = (alu2&1) ? tx5_0_sh : (*(buf0+9));
  float tx5_0_1 = (alu2&1) ? (*(buf0+25)) : tx5_0_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+16))*128+((lidx1<<6)+16+alu2+7*(alu2&1))))) = make_float2(tx5_0_0,tx5_0_1);
  float tx5_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+41)), 4);
  float tx5_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+57)), 4);
  float tx5_1_0 = (alu2&1) ? tx5_1_sh : (*(buf0+41));
  float tx5_1_1 = (alu2&1) ? (*(buf0+57)) : tx5_1_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+17))*128+((lidx1<<6)+16+alu2+7*(alu2&1))))) = make_float2(tx5_1_0,tx5_1_1);
  float tx6_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+13)), 4);
  float tx6_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+29)), 4);
  float tx6_0_0 = (alu2&1) ? tx6_0_sh : (*(buf0+13));
  float tx6_0_1 = (alu2&1) ? (*(buf0+29)) : tx6_0_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+24))*128+((lidx1<<6)+16+alu2+7*(alu2&1))))) = make_float2(tx6_0_0,tx6_0_1);
  float tx6_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+45)), 4);
  float tx6_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+61)), 4);
  float tx6_1_0 = (alu2&1) ? tx6_1_sh : (*(buf0+45));
  float tx6_1_1 = (alu2&1) ? (*(buf0+61)) : tx6_1_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+25))*128+((lidx1<<6)+16+alu2+7*(alu2&1))))) = make_float2(tx6_1_0,tx6_1_1);
  float tx7_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+2)), 4);
  float tx7_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+18)), 4);
  float tx7_0_0 = (alu2&1) ? tx7_0_sh : (*(buf0+2));
  float tx7_0_1 = (alu2&1) ? (*(buf0+18)) : tx7_0_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+0))*128+((lidx1<<6)+32+alu2+7*(alu2&1))))) = make_float2(tx7_0_0,tx7_0_1);
  float tx7_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+34)), 4);
  float tx7_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+50)), 4);
  float tx7_1_0 = (alu2&1) ? tx7_1_sh : (*(buf0+34));
  float tx7_1_1 = (alu2&1) ? (*(buf0+50)) : tx7_1_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+1))*128+((lidx1<<6)+32+alu2+7*(alu2&1))))) = make_float2(tx7_1_0,tx7_1_1);
  float tx8_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+6)), 4);
  float tx8_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+22)), 4);
  float tx8_0_0 = (alu2&1) ? tx8_0_sh : (*(buf0+6));
  float tx8_0_1 = (alu2&1) ? (*(buf0+22)) : tx8_0_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+8))*128+((lidx1<<6)+32+alu2+7*(alu2&1))))) = make_float2(tx8_0_0,tx8_0_1);
  float tx8_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+38)), 4);
  float tx8_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+54)), 4);
  float tx8_1_0 = (alu2&1) ? tx8_1_sh : (*(buf0+38));
  float tx8_1_1 = (alu2&1) ? (*(buf0+54)) : tx8_1_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+9))*128+((lidx1<<6)+32+alu2+7*(alu2&1))))) = make_float2(tx8_1_0,tx8_1_1);
  float tx9_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+10)), 4);
  float tx9_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+26)), 4);
  float tx9_0_0 = (alu2&1) ? tx9_0_sh : (*(buf0+10));
  float tx9_0_1 = (alu2&1) ? (*(buf0+26)) : tx9_0_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+16))*128+((lidx1<<6)+32+alu2+7*(alu2&1))))) = make_float2(tx9_0_0,tx9_0_1);
  float tx9_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+42)), 4);
  float tx9_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+58)), 4);
  float tx9_1_0 = (alu2&1) ? tx9_1_sh : (*(buf0+42));
  float tx9_1_1 = (alu2&1) ? (*(buf0+58)) : tx9_1_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+17))*128+((lidx1<<6)+32+alu2+7*(alu2&1))))) = make_float2(tx9_1_0,tx9_1_1);
  float tx10_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+14)), 4);
  float tx10_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+30)), 4);
  float tx10_0_0 = (alu2&1) ? tx10_0_sh : (*(buf0+14));
  float tx10_0_1 = (alu2&1) ? (*(buf0+30)) : tx10_0_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+24))*128+((lidx1<<6)+32+alu2+7*(alu2&1))))) = make_float2(tx10_0_0,tx10_0_1);
  float tx10_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+46)), 4);
  float tx10_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+62)), 4);
  float tx10_1_0 = (alu2&1) ? tx10_1_sh : (*(buf0+46));
  float tx10_1_1 = (alu2&1) ? (*(buf0+62)) : tx10_1_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+25))*128+((lidx1<<6)+32+alu2+7*(alu2&1))))) = make_float2(tx10_1_0,tx10_1_1);
  float tx11_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+3)), 4);
  float tx11_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+19)), 4);
  float tx11_0_0 = (alu2&1) ? tx11_0_sh : (*(buf0+3));
  float tx11_0_1 = (alu2&1) ? (*(buf0+19)) : tx11_0_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+0))*128+((lidx1<<6)+48+alu2+7*(alu2&1))))) = make_float2(tx11_0_0,tx11_0_1);
  float tx11_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+35)), 4);
  float tx11_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+51)), 4);
  float tx11_1_0 = (alu2&1) ? tx11_1_sh : (*(buf0+35));
  float tx11_1_1 = (alu2&1) ? (*(buf0+51)) : tx11_1_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+1))*128+((lidx1<<6)+48+alu2+7*(alu2&1))))) = make_float2(tx11_1_0,tx11_1_1);
  float tx12_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+7)), 4);
  float tx12_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+23)), 4);
  float tx12_0_0 = (alu2&1) ? tx12_0_sh : (*(buf0+7));
  float tx12_0_1 = (alu2&1) ? (*(buf0+23)) : tx12_0_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+8))*128+((lidx1<<6)+48+alu2+7*(alu2&1))))) = make_float2(tx12_0_0,tx12_0_1);
  float tx12_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+39)), 4);
  float tx12_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+55)), 4);
  float tx12_1_0 = (alu2&1) ? tx12_1_sh : (*(buf0+39));
  float tx12_1_1 = (alu2&1) ? (*(buf0+55)) : tx12_1_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+9))*128+((lidx1<<6)+48+alu2+7*(alu2&1))))) = make_float2(tx12_1_0,tx12_1_1);
  float tx13_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+11)), 4);
  float tx13_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+27)), 4);
  float tx13_0_0 = (alu2&1) ? tx13_0_sh : (*(buf0+11));
  float tx13_0_1 = (alu2&1) ? (*(buf0+27)) : tx13_0_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+16))*128+((lidx1<<6)+48+alu2+7*(alu2&1))))) = make_float2(tx13_0_0,tx13_0_1);
  float tx13_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+43)), 4);
  float tx13_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+59)), 4);
  float tx13_1_0 = (alu2&1) ? tx13_1_sh : (*(buf0+43));
  float tx13_1_1 = (alu2&1) ? (*(buf0+59)) : tx13_1_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+17))*128+((lidx1<<6)+48+alu2+7*(alu2&1))))) = make_float2(tx13_1_0,tx13_1_1);
  float tx14_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+15)), 4);
  float tx14_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+31)), 4);
  float tx14_0_0 = (alu2&1) ? tx14_0_sh : (*(buf0+15));
  float tx14_0_1 = (alu2&1) ? (*(buf0+31)) : tx14_0_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+24))*128+((lidx1<<6)+48+alu2+7*(alu2&1))))) = make_float2(tx14_0_0,tx14_0_1);
  float tx14_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+47)), 4);
  float tx14_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+63)), 4);
  float tx14_1_0 = (alu2&1) ? tx14_1_sh : (*(buf0+47));
  float tx14_1_1 = (alu2&1) ? (*(buf0+63)) : tx14_1_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+25))*128+((lidx1<<6)+48+alu2+7*(alu2&1))))) = make_float2(tx14_1_0,tx14_1_1);
  float tx15_0_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+0)), 4);
  float tx15_0_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+16)), 4);
  float tx15_0_0 = (alu2&1) ? tx15_0_sh : (*(buf0+0));
  float tx15_0_1 = (alu2&1) ? (*(buf0+16)) : tx15_0_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+0))*128+((lidx1<<6)+0+alu2+7*(alu2&1))))) = make_float2(tx15_0_0,tx15_0_1);
  float tx15_1_sl = __shfl_xor_sync(0xffffffffu, (*(buf0+32)), 4);
  float tx15_1_sh = __shfl_xor_sync(0xffffffffu, (*(buf0+48)), 4);
  float tx15_1_0 = (alu2&1) ? tx15_1_sh : (*(buf0+32));
  float tx15_1_1 = (alu2&1) ? (*(buf0+48)) : tx15_1_sl;
  *((float2*)(partials+(slot*16384)+((((lidx2<<5)+(alu5<<1)+1))*128+((lidx1<<6)+0+alu2+7*(alu2&1))))) = make_float2(tx15_1_0,tx15_1_1);
    }
  }
}
