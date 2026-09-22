#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
__device__ __forceinline__ uint2 tg_ldmatrix_x2(const void *p) {
  uint2 r; asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1},[%2];"
    : "=r"(r.x),"=r"(r.y) : "l"(p)); return r;
}
#include <cuda_fp16.h>
struct __align__(8) signed_char8 { signed char x, y, z, w, a, b, c, d; }; __device__ signed_char8 make_signed_char8(signed char x, signed char y, signed char z, signed char w, signed char a, signed char b, signed char c, signed char d) { signed_char8 r={x, y, z, w, a, b, c, d}; return r; }
__device__ int4 __WMMA_8_16_16_signed_char_int(signed_char8 a, char4 b, int4 c){
  int *a_pk = (int *)(&a), *b_pk = (int *)(&b), *c_pk = (int *)(&c);
  asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32"
      "{%0, %1, %2, %3}, {%4, %5},"
      "{%6}, {%0, %1, %2, %3};"
    : "+r"(c_pk[0]), "+r"(c_pk[1]), "+r"(c_pk[2]), "+r"(c_pk[3])
    : "r"(a_pk[0]), "r"(a_pk[1]), "r"(b_pk[0]));
  return c;
}
extern "C" __global__ void __launch_bounds__(256) nv_generated_q6k_streamk_owner_partials(float* data0_5570560, int* data1_340, unsigned short* data2_20643840, signed char* data3_6291456, float* data4_196608) {
  int gidx0 = blockIdx.x; /* 170 */
  int lidx0 = threadIdx.x; /* 256 */
  extern __shared__ __align__(16) unsigned int buf0[];
  float buf1;
  float buf2;
  float buf3;
  float buf4;
  float buf5;
  float buf6;
  float buf7;
  float buf8;
  float buf9;
  float buf10;
  float buf11;
  float buf12;
  float buf13;
  float buf14;
  float buf15;
  float buf16;
  float buf17;
  float buf18;
  float buf19;
  float buf20;
  float buf21;
  float buf22;
  float buf23;
  float buf24;
  float buf25;
  float buf26;
  float buf27;
  float buf28;
  float buf29;
  float buf30;
  float buf31;
  float buf32;
  float buf33;
  float buf34;
  float buf35;
  float buf36;
  float buf37;
  float buf38;
  float buf39;
  float buf40;
  float buf41;
  float buf42;
  float buf43;
  float buf44;
  float buf45;
  float buf46;
  float buf47;
  float buf48;
  float buf49;
  float buf50;
  float buf51;
  float buf52;
  float buf53;
  float buf54;
  float buf55;
  float buf56;
  float buf57;
  float buf58;
  float buf59;
  float buf60;
  float buf61;
  float buf62;
  float buf63;
  float buf64;
  buf1 = 0.0f;
  buf2 = 0.0f;
  buf3 = 0.0f;
  buf4 = 0.0f;
  buf5 = 0.0f;
  buf6 = 0.0f;
  buf7 = 0.0f;
  buf8 = 0.0f;
  buf9 = 0.0f;
  buf10 = 0.0f;
  buf11 = 0.0f;
  buf12 = 0.0f;
  buf13 = 0.0f;
  buf14 = 0.0f;
  buf15 = 0.0f;
  buf16 = 0.0f;
  buf17 = 0.0f;
  buf18 = 0.0f;
  buf19 = 0.0f;
  buf20 = 0.0f;
  buf21 = 0.0f;
  buf22 = 0.0f;
  buf23 = 0.0f;
  buf24 = 0.0f;
  buf25 = 0.0f;
  buf26 = 0.0f;
  buf27 = 0.0f;
  buf28 = 0.0f;
  buf29 = 0.0f;
  buf30 = 0.0f;
  buf31 = 0.0f;
  buf32 = 0.0f;
  buf33 = 0.0f;
  buf34 = 0.0f;
  buf35 = 0.0f;
  buf36 = 0.0f;
  buf37 = 0.0f;
  buf38 = 0.0f;
  buf39 = 0.0f;
  buf40 = 0.0f;
  buf41 = 0.0f;
  buf42 = 0.0f;
  buf43 = 0.0f;
  buf44 = 0.0f;
  buf45 = 0.0f;
  buf46 = 0.0f;
  buf47 = 0.0f;
  buf48 = 0.0f;
  buf49 = 0.0f;
  buf50 = 0.0f;
  buf51 = 0.0f;
  buf52 = 0.0f;
  buf53 = 0.0f;
  buf54 = 0.0f;
  buf55 = 0.0f;
  buf56 = 0.0f;
  buf57 = 0.0f;
  buf58 = 0.0f;
  buf59 = 0.0f;
  buf60 = 0.0f;
  buf61 = 0.0f;
  buf62 = 0.0f;
  buf63 = 0.0f;
  buf64 = 0.0f;
  int4 cast0 = make_int4(0,0,0,0);
  int alu64 = (lidx0>>5);
  int alu65 = (lidx0&31);
  int alu66 = (alu65&15);
  int alu67 = ((alu66*36)+(alu64*576));
  int alu68 = (alu67+9732);
  int alu69 = (alu67+9733);
  int alu70 = (alu67+9734);
  int alu71 = (alu67+9735);
  int alu72 = (alu64>>1);
  int alu73 = (alu65>>2);
  int alu74 = ((alu72<<12)+(alu73<<7));
  int alu75 = (alu64&1);
  int alu76 = (alu65&3);
  int alu77 = ((alu75<<3)+(alu76<<1));
  int alu78 = (alu74+(gidx0<<15)+alu77);
  int alu79 = (alu72*2432);
  int alu80 = (alu79+(alu66*76));
  int alu81 = (alu75*288);
  int alu82 = ((alu73*36)+alu81+alu76);
  int alu83 = (alu82+9732);
  int alu84 = (alu82+9736);
  int alu85 = (alu82+9740);
  int alu86 = (alu82+9744);
  int alu87 = (alu82+9748);
  int alu88 = (alu82+9752);
  int alu89 = (alu82+9756);
  int alu90 = (alu82+9760);
  int alu91 = (alu82+10308);
  int alu92 = (alu82+10312);
  int alu93 = (alu82+10316);
  int alu94 = (alu82+10320);
  int alu95 = (alu82+10324);
  int alu96 = (alu82+10328);
  int alu97 = (alu82+10332);
  int alu98 = (alu82+10336);
  int alu99 = (alu82+10884);
  int alu100 = (alu82+10888);
  int alu101 = (alu82+10892);
  int alu102 = (alu82+10896);
  int alu103 = (alu82+10900);
  int alu104 = (alu82+10904);
  int alu105 = (alu82+10908);
  int alu106 = (alu82+10912);
  int alu107 = (alu82+11460);
  int alu108 = (alu82+11464);
  int alu109 = (alu82+11468);
  int alu110 = (alu82+11472);
  int alu111 = (alu82+11476);
  int alu112 = (alu82+11480);
  int alu113 = (alu82+11484);
  int alu114 = (alu82+11488);
  int alu115 = (alu82+12036);
  int alu116 = (alu82+12040);
  int alu117 = (alu82+12044);
  int alu118 = (alu82+12048);
  int alu119 = (alu82+12052);
  int alu120 = (alu82+12056);
  int alu121 = (alu82+12060);
  int alu122 = (alu82+12064);
  int alu123 = (alu82+12612);
  int alu124 = (alu82+12616);
  int alu125 = (alu82+12620);
  int alu126 = (alu82+12624);
  int alu127 = (alu82+12628);
  int alu128 = (alu82+12632);
  int alu129 = (alu82+12636);
  int alu130 = (alu82+12640);
  int alu131 = (alu82+13188);
  int alu132 = (alu82+13192);
  int alu133 = (alu82+13196);
  int alu134 = (alu82+13200);
  int alu135 = (alu82+13204);
  int alu136 = (alu82+13208);
  int alu137 = (alu82+13212);
  int alu138 = (alu82+13216);
  int alu139 = (alu82+13764);
  int alu140 = (alu82+13768);
  int alu141 = (alu82+13772);
  int alu142 = (alu82+13776);
  int alu143 = (alu82+13780);
  int alu144 = (alu82+13784);
  int alu145 = (alu82+13788);
  int alu146 = (alu82+13792);
  int alu147 = (alu79+(alu73*76));
  int alu148 = (alu147+64);
  int alu149 = (alu147+65);
  int alu150 = (alu147+66);
  int alu151 = (alu147+67);
  int alu152 = (alu147+68);
  int alu153 = (alu147+672);
  int alu154 = (alu147+673);
  int alu155 = (alu147+674);
  int alu156 = (alu147+675);
  int alu157 = (alu147+676);
  int alu158 = (alu147+1280);
  int alu159 = (alu147+1281);
  int alu160 = (alu147+1282);
  int alu161 = (alu147+1283);
  int alu162 = (alu147+1284);
  int alu163 = (alu147+1888);
  int alu164 = (alu147+1889);
  int alu165 = (alu147+1890);
  int alu166 = (alu147+1891);
  int alu167 = (alu147+1892);
  int alu168 = (alu81+(alu76*72));
  int alu169 = (alu168+9728);
  int alu170 = (alu168+9729);
  int alu171 = (alu168+9730);
  int alu172 = (alu168+9731);
  int alu173 = (alu168+9764);
  int alu174 = (alu168+9765);
  int alu175 = (alu168+9766);
  int alu176 = (alu168+9767);
  int alu177 = (alu168+10304);
  int alu178 = (alu168+10305);
  int alu179 = (alu168+10306);
  int alu180 = (alu168+10307);
  int alu181 = (alu168+10340);
  int alu182 = (alu168+10341);
  int alu183 = (alu168+10342);
  int alu184 = (alu168+10343);
  int alu185 = (alu168+10880);
  int alu186 = (alu168+10881);
  int alu187 = (alu168+10882);
  int alu188 = (alu168+10883);
  int alu189 = (alu168+10916);
  int alu190 = (alu168+10917);
  int alu191 = (alu168+10918);
  int alu192 = (alu168+10919);
  int alu193 = (alu168+11456);
  int alu194 = (alu168+11457);
  int alu195 = (alu168+11458);
  int alu196 = (alu168+11459);
  int alu197 = (alu168+11492);
  int alu198 = (alu168+11493);
  int alu199 = (alu168+11494);
  int alu200 = (alu168+11495);
  int alu201 = (alu168+12032);
  int alu202 = (alu168+12033);
  int alu203 = (alu168+12034);
  int alu204 = (alu168+12035);
  int alu205 = (alu168+12068);
  int alu206 = (alu168+12069);
  int alu207 = (alu168+12070);
  int alu208 = (alu168+12071);
  int alu209 = (alu168+12608);
  int alu210 = (alu168+12609);
  int alu211 = (alu168+12610);
  int alu212 = (alu168+12611);
  int alu213 = (alu168+12644);
  int alu214 = (alu168+12645);
  int alu215 = (alu168+12646);
  int alu216 = (alu168+12647);
  int alu217 = (alu168+13184);
  int alu218 = (alu168+13185);
  int alu219 = (alu168+13186);
  int alu220 = (alu168+13187);
  int alu221 = (alu168+13220);
  int alu222 = (alu168+13221);
  int alu223 = (alu168+13222);
  int alu224 = (alu168+13223);
  int alu225 = (alu168+13760);
  int alu226 = (alu168+13761);
  int alu227 = (alu168+13762);
  int alu228 = (alu168+13763);
  int alu229 = (alu168+13796);
  int alu230 = (alu168+13797);
  int alu231 = (alu168+13798);
  int alu232 = (alu168+13799);
  int alu233 = (alu67+9728);
  int alu234 = (alu67+9729);
  int alu235 = (alu67+9730);
  int alu236 = (alu67+9731);
  int alu237 = (alu67+9736);
  int alu238 = (alu67+9737);
  int alu239 = (alu67+9738);
  int alu240 = (alu67+9739);
  int alu241 = (alu67+9740);
  int alu242 = (alu67+9741);
  int alu243 = (alu67+9742);
  int alu244 = (alu67+9743);
  int alu245 = (alu67+9744);
  int alu246 = (alu67+9745);
  int alu247 = (alu67+9746);
  int alu248 = (alu67+9747);
  int alu249 = (alu67+9748);
  int alu250 = (alu67+9749);
  int alu251 = (alu67+9750);
  int alu252 = (alu67+9751);
  int alu253 = (alu67+9752);
  int alu254 = (alu67+9753);
  int alu255 = (alu67+9754);
  int alu256 = (alu67+9755);
  int alu257 = (alu67+9756);
  int alu258 = (alu67+9757);
  int alu259 = (alu67+9758);
  int alu260 = (alu67+9759);
  int alu261 = (alu67+9760);
  int alu262 = (alu67+9761);
  int alu263 = (alu67+9762);
  int alu264 = (alu67+9763);
  int alu265 = ((alu64<<4)+alu66);
  int alu266 = (gidx0*6144);
  int alu267 = ((gidx0+1)*6144);
  int alu268 = (alu65<<1);
  int alu269 = (alu266/170);
  bool alu270 = (alu65<1);
  bool alu271 = (alu65<16);
  for (int Ridx50 = 0; Ridx50 < ((alu267/170)-alu269); Ridx50++) {
    int alu272 = (alu269+Ridx50);
    int alu273 = (alu272/48);
    int alu274 = (alu272%48);
    for (int Lidx51 = 0; Lidx51 < 16; Lidx51++) {
      int alu275 = ((((((alu273&31)<<7)+(Lidx51<<3)+alu64)*48)+alu274)*105);
      int alu276 = (alu275+((lidx0&7)<<1)+(((lidx0>>4)&1)<<4));
      unsigned short val0 = (*(data2_20643840+(alu276+64)));
      unsigned short val1 = (*(data2_20643840+(alu276+65)));
      int alu277 = (alu275+alu268);
      unsigned short val2 = (*(data2_20643840+(alu277+1)));
      unsigned short val3 = (*(data2_20643840+alu277));
      unsigned short val4 = (*(data2_20643840+(alu275+96)));
      unsigned short val5 = (*(data2_20643840+(alu275+97)));
      unsigned short val6 = (*(data2_20643840+(alu275+98)));
      unsigned short val7 = (*(data2_20643840+(alu275+99)));
      unsigned short val8 = (*(data2_20643840+(alu275+100)));
      unsigned short val9 = (*(data2_20643840+(alu275+101)));
      unsigned short val10 = (*(data2_20643840+(alu275+102)));
      unsigned short val11 = (*(data2_20643840+(alu275+103)));
      unsigned short val12 = (*(data2_20643840+(alu275+104)));
      int alu278 = ((alu64*76)+(Lidx51*608));
      int alu279 = (alu278+(alu268-(lidx0&15)));
      unsigned int alu280 = ((((unsigned int)(val0))|(((unsigned int)(val1))<<16u))>>((unsigned int)(((alu65&8)>>2))));
      unsigned int alu281 = (((unsigned int)(val3))|(((unsigned int)(val2))<<16u));
      *(buf0+(alu279+16)) = __vsubss4((((alu281>>4u)&252645135u)|(alu280&808464432u)),538976288u);
      *(buf0+alu279) = __vsubss4(((alu281&252645135u)|((alu280<<4u)&808464432u)),538976288u);
      if (alu270) {
        *(buf0+(alu278+64)) = ((unsigned int)(val12));
      }
      if (alu270) {
        *(buf0+(alu278+65)) = (((unsigned int)(val4))|(((unsigned int)(val5))<<16u));
      }
      if ((alu65==1)) {
        *(buf0+(alu278+66)) = (((unsigned int)(val6))|(((unsigned int)(val7))<<16u));
      }
      if ((alu65==2)) {
        *(buf0+(alu278+67)) = (((unsigned int)(val8))|(((unsigned int)(val9))<<16u));
      }
      if ((alu65==3)) {
        *(buf0+(alu278+68)) = (((unsigned int)(val10))|(((unsigned int)(val11))<<16u));
      }
    }
    int alu300 = (alu274<<8);
    int alu301 = ((alu272/1536)<<7);
    signed char val13 = (*(data3_6291456+(((alu300+1)<<9)+alu301+alu265)));
    signed char val14 = (*(data3_6291456+(((alu300+2)<<9)+alu301+alu265)));
    signed char val15 = (*(data3_6291456+(((alu300+3)<<9)+alu301+alu265)));
    signed char val16 = (*(data3_6291456+(((alu300+4)<<9)+alu301+alu265)));
    signed char val17 = (*(data3_6291456+(((alu300+5)<<9)+alu301+alu265)));
    signed char val18 = (*(data3_6291456+(((alu300+6)<<9)+alu301+alu265)));
    signed char val19 = (*(data3_6291456+(((alu300+7)<<9)+alu301+alu265)));
    signed char val20 = (*(data3_6291456+(((alu300+8)<<9)+alu301+alu265)));
    signed char val21 = (*(data3_6291456+(((alu300+9)<<9)+alu301+alu265)));
    signed char val22 = (*(data3_6291456+(((alu300+10)<<9)+alu301+alu265)));
    signed char val23 = (*(data3_6291456+(((alu300+11)<<9)+alu301+alu265)));
    signed char val24 = (*(data3_6291456+(((alu300+12)<<9)+alu301+alu265)));
    signed char val25 = (*(data3_6291456+(((alu300+13)<<9)+alu301+alu265)));
    signed char val26 = (*(data3_6291456+(((alu300+14)<<9)+alu301+alu265)));
    signed char val27 = (*(data3_6291456+(((alu300+15)<<9)+alu301+alu265)));
    signed char val28 = (*(data3_6291456+(((alu300+16)<<9)+alu301+alu265)));
    signed char val29 = (*(data3_6291456+(((alu300+17)<<9)+alu301+alu265)));
    signed char val30 = (*(data3_6291456+(((alu300+18)<<9)+alu301+alu265)));
    signed char val31 = (*(data3_6291456+(((alu300+19)<<9)+alu301+alu265)));
    signed char val32 = (*(data3_6291456+(((alu300+20)<<9)+alu301+alu265)));
    signed char val33 = (*(data3_6291456+(((alu300+21)<<9)+alu301+alu265)));
    signed char val34 = (*(data3_6291456+(((alu300+22)<<9)+alu301+alu265)));
    signed char val35 = (*(data3_6291456+(((alu300+23)<<9)+alu301+alu265)));
    signed char val36 = (*(data3_6291456+(((alu300+24)<<9)+alu301+alu265)));
    signed char val37 = (*(data3_6291456+(((alu300+25)<<9)+alu301+alu265)));
    signed char val38 = (*(data3_6291456+(((alu300+26)<<9)+alu301+alu265)));
    signed char val39 = (*(data3_6291456+(((alu300+27)<<9)+alu301+alu265)));
    signed char val40 = (*(data3_6291456+(((alu300+28)<<9)+alu301+alu265)));
    signed char val41 = (*(data3_6291456+(((alu300+29)<<9)+alu301+alu265)));
    signed char val42 = (*(data3_6291456+(((alu300+30)<<9)+alu301+alu265)));
    signed char val43 = (*(data3_6291456+(((alu300+31)<<9)+alu301+alu265)));
    signed char val44 = (*(data3_6291456+(((alu300+32)<<9)+alu301+alu265)));
    signed char val45 = (*(data3_6291456+(((alu300+33)<<9)+alu301+alu265)));
    signed char val46 = (*(data3_6291456+(((alu300+34)<<9)+alu301+alu265)));
    signed char val47 = (*(data3_6291456+(((alu300+35)<<9)+alu301+alu265)));
    signed char val48 = (*(data3_6291456+(((alu300+36)<<9)+alu301+alu265)));
    signed char val49 = (*(data3_6291456+(((alu300+37)<<9)+alu301+alu265)));
    signed char val50 = (*(data3_6291456+(((alu300+38)<<9)+alu301+alu265)));
    signed char val51 = (*(data3_6291456+(((alu300+39)<<9)+alu301+alu265)));
    signed char val52 = (*(data3_6291456+(((alu300+40)<<9)+alu301+alu265)));
    signed char val53 = (*(data3_6291456+(((alu300+41)<<9)+alu301+alu265)));
    signed char val54 = (*(data3_6291456+(((alu300+42)<<9)+alu301+alu265)));
    signed char val55 = (*(data3_6291456+(((alu300+43)<<9)+alu301+alu265)));
    signed char val56 = (*(data3_6291456+(((alu300+44)<<9)+alu301+alu265)));
    signed char val57 = (*(data3_6291456+(((alu300+45)<<9)+alu301+alu265)));
    signed char val58 = (*(data3_6291456+(((alu300+46)<<9)+alu301+alu265)));
    signed char val59 = (*(data3_6291456+(((alu300+47)<<9)+alu301+alu265)));
    signed char val60 = (*(data3_6291456+(((alu300+48)<<9)+alu301+alu265)));
    signed char val61 = (*(data3_6291456+(((alu300+49)<<9)+alu301+alu265)));
    signed char val62 = (*(data3_6291456+(((alu300+50)<<9)+alu301+alu265)));
    signed char val63 = (*(data3_6291456+(((alu300+51)<<9)+alu301+alu265)));
    signed char val64 = (*(data3_6291456+(((alu300+52)<<9)+alu301+alu265)));
    signed char val65 = (*(data3_6291456+(((alu300+53)<<9)+alu301+alu265)));
    signed char val66 = (*(data3_6291456+(((alu300+54)<<9)+alu301+alu265)));
    signed char val67 = (*(data3_6291456+(((alu300+55)<<9)+alu301+alu265)));
    signed char val68 = (*(data3_6291456+(((alu300+56)<<9)+alu301+alu265)));
    signed char val69 = (*(data3_6291456+(((alu300+57)<<9)+alu301+alu265)));
    signed char val70 = (*(data3_6291456+(((alu300+58)<<9)+alu301+alu265)));
    signed char val71 = (*(data3_6291456+(((alu300+59)<<9)+alu301+alu265)));
    signed char val72 = (*(data3_6291456+(((alu300+60)<<9)+alu301+alu265)));
    signed char val73 = (*(data3_6291456+(((alu300+61)<<9)+alu301+alu265)));
    signed char val74 = (*(data3_6291456+(((alu300+62)<<9)+alu301+alu265)));
    signed char val75 = (*(data3_6291456+(((alu300+63)<<9)+alu301+alu265)));
    signed char val76 = (*(data3_6291456+(((alu300+64)<<9)+alu301+alu265)));
    signed char val77 = (*(data3_6291456+(((alu300+65)<<9)+alu301+alu265)));
    signed char val78 = (*(data3_6291456+(((alu300+66)<<9)+alu301+alu265)));
    signed char val79 = (*(data3_6291456+(((alu300+67)<<9)+alu301+alu265)));
    signed char val80 = (*(data3_6291456+(((alu300+68)<<9)+alu301+alu265)));
    signed char val81 = (*(data3_6291456+(((alu300+69)<<9)+alu301+alu265)));
    signed char val82 = (*(data3_6291456+(((alu300+70)<<9)+alu301+alu265)));
    signed char val83 = (*(data3_6291456+(((alu300+71)<<9)+alu301+alu265)));
    signed char val84 = (*(data3_6291456+(((alu300+72)<<9)+alu301+alu265)));
    signed char val85 = (*(data3_6291456+(((alu300+73)<<9)+alu301+alu265)));
    signed char val86 = (*(data3_6291456+(((alu300+74)<<9)+alu301+alu265)));
    signed char val87 = (*(data3_6291456+(((alu300+75)<<9)+alu301+alu265)));
    signed char val88 = (*(data3_6291456+(((alu300+76)<<9)+alu301+alu265)));
    signed char val89 = (*(data3_6291456+(((alu300+77)<<9)+alu301+alu265)));
    signed char val90 = (*(data3_6291456+(((alu300+78)<<9)+alu301+alu265)));
    signed char val91 = (*(data3_6291456+(((alu300+79)<<9)+alu301+alu265)));
    signed char val92 = (*(data3_6291456+(((alu300+80)<<9)+alu301+alu265)));
    signed char val93 = (*(data3_6291456+(((alu300+81)<<9)+alu301+alu265)));
    signed char val94 = (*(data3_6291456+(((alu300+82)<<9)+alu301+alu265)));
    signed char val95 = (*(data3_6291456+(((alu300+83)<<9)+alu301+alu265)));
    signed char val96 = (*(data3_6291456+(((alu300+84)<<9)+alu301+alu265)));
    signed char val97 = (*(data3_6291456+(((alu300+85)<<9)+alu301+alu265)));
    signed char val98 = (*(data3_6291456+(((alu300+86)<<9)+alu301+alu265)));
    signed char val99 = (*(data3_6291456+(((alu300+87)<<9)+alu301+alu265)));
    signed char val100 = (*(data3_6291456+(((alu300+88)<<9)+alu301+alu265)));
    signed char val101 = (*(data3_6291456+(((alu300+89)<<9)+alu301+alu265)));
    signed char val102 = (*(data3_6291456+(((alu300+90)<<9)+alu301+alu265)));
    signed char val103 = (*(data3_6291456+(((alu300+91)<<9)+alu301+alu265)));
    signed char val104 = (*(data3_6291456+(((alu300+92)<<9)+alu301+alu265)));
    signed char val105 = (*(data3_6291456+(((alu300+93)<<9)+alu301+alu265)));
    signed char val106 = (*(data3_6291456+(((alu300+94)<<9)+alu301+alu265)));
    signed char val107 = (*(data3_6291456+(((alu300+95)<<9)+alu301+alu265)));
    signed char val108 = (*(data3_6291456+(((alu300+96)<<9)+alu301+alu265)));
    signed char val109 = (*(data3_6291456+(((alu300+97)<<9)+alu301+alu265)));
    signed char val110 = (*(data3_6291456+(((alu300+98)<<9)+alu301+alu265)));
    signed char val111 = (*(data3_6291456+(((alu300+99)<<9)+alu301+alu265)));
    signed char val112 = (*(data3_6291456+(((alu300+100)<<9)+alu301+alu265)));
    signed char val113 = (*(data3_6291456+(((alu300+101)<<9)+alu301+alu265)));
    signed char val114 = (*(data3_6291456+(((alu300+102)<<9)+alu301+alu265)));
    signed char val115 = (*(data3_6291456+(((alu300+103)<<9)+alu301+alu265)));
    signed char val116 = (*(data3_6291456+(((alu300+104)<<9)+alu301+alu265)));
    signed char val117 = (*(data3_6291456+(((alu300+105)<<9)+alu301+alu265)));
    signed char val118 = (*(data3_6291456+(((alu300+106)<<9)+alu301+alu265)));
    signed char val119 = (*(data3_6291456+(((alu300+107)<<9)+alu301+alu265)));
    signed char val120 = (*(data3_6291456+(((alu300+108)<<9)+alu301+alu265)));
    signed char val121 = (*(data3_6291456+(((alu300+109)<<9)+alu301+alu265)));
    signed char val122 = (*(data3_6291456+(((alu300+110)<<9)+alu301+alu265)));
    signed char val123 = (*(data3_6291456+(((alu300+111)<<9)+alu301+alu265)));
    signed char val124 = (*(data3_6291456+(((alu300+112)<<9)+alu301+alu265)));
    signed char val125 = (*(data3_6291456+(((alu300+113)<<9)+alu301+alu265)));
    signed char val126 = (*(data3_6291456+(((alu300+114)<<9)+alu301+alu265)));
    signed char val127 = (*(data3_6291456+(((alu300+115)<<9)+alu301+alu265)));
    signed char val128 = (*(data3_6291456+(((alu300+116)<<9)+alu301+alu265)));
    signed char val129 = (*(data3_6291456+(((alu300+117)<<9)+alu301+alu265)));
    signed char val130 = (*(data3_6291456+(((alu300+118)<<9)+alu301+alu265)));
    signed char val131 = (*(data3_6291456+(((alu300+119)<<9)+alu301+alu265)));
    signed char val132 = (*(data3_6291456+(((alu300+120)<<9)+alu301+alu265)));
    signed char val133 = (*(data3_6291456+(((alu300+121)<<9)+alu301+alu265)));
    signed char val134 = (*(data3_6291456+(((alu300+122)<<9)+alu301+alu265)));
    signed char val135 = (*(data3_6291456+(((alu300+123)<<9)+alu301+alu265)));
    signed char val136 = (*(data3_6291456+(((alu300+124)<<9)+alu301+alu265)));
    signed char val137 = (*(data3_6291456+(((alu300+125)<<9)+alu301+alu265)));
    signed char val138 = (*(data3_6291456+(((alu300+126)<<9)+alu301+alu265)));
    signed char val139 = (*(data3_6291456+(((alu300+127)<<9)+alu301+alu265)));
    signed char val140 = (*(data3_6291456+(((alu300+128)<<9)+alu301+alu265)));
    signed char val141 = (*(data3_6291456+(((alu300+129)<<9)+alu301+alu265)));
    signed char val142 = (*(data3_6291456+(((alu300+130)<<9)+alu301+alu265)));
    signed char val143 = (*(data3_6291456+(((alu300+131)<<9)+alu301+alu265)));
    signed char val144 = (*(data3_6291456+(((alu300+132)<<9)+alu301+alu265)));
    signed char val145 = (*(data3_6291456+(((alu300+133)<<9)+alu301+alu265)));
    signed char val146 = (*(data3_6291456+(((alu300+134)<<9)+alu301+alu265)));
    signed char val147 = (*(data3_6291456+(((alu300+135)<<9)+alu301+alu265)));
    signed char val148 = (*(data3_6291456+(((alu300+136)<<9)+alu301+alu265)));
    signed char val149 = (*(data3_6291456+(((alu300+137)<<9)+alu301+alu265)));
    signed char val150 = (*(data3_6291456+(((alu300+138)<<9)+alu301+alu265)));
    signed char val151 = (*(data3_6291456+(((alu300+139)<<9)+alu301+alu265)));
    signed char val152 = (*(data3_6291456+(((alu300+140)<<9)+alu301+alu265)));
    signed char val153 = (*(data3_6291456+(((alu300+141)<<9)+alu301+alu265)));
    signed char val154 = (*(data3_6291456+(((alu300+142)<<9)+alu301+alu265)));
    signed char val155 = (*(data3_6291456+(((alu300+143)<<9)+alu301+alu265)));
    signed char val156 = (*(data3_6291456+(((alu300+144)<<9)+alu301+alu265)));
    signed char val157 = (*(data3_6291456+(((alu300+145)<<9)+alu301+alu265)));
    signed char val158 = (*(data3_6291456+(((alu300+146)<<9)+alu301+alu265)));
    signed char val159 = (*(data3_6291456+(((alu300+147)<<9)+alu301+alu265)));
    signed char val160 = (*(data3_6291456+(((alu300+148)<<9)+alu301+alu265)));
    signed char val161 = (*(data3_6291456+(((alu300+149)<<9)+alu301+alu265)));
    signed char val162 = (*(data3_6291456+(((alu300+150)<<9)+alu301+alu265)));
    signed char val163 = (*(data3_6291456+(((alu300+151)<<9)+alu301+alu265)));
    signed char val164 = (*(data3_6291456+(((alu300+152)<<9)+alu301+alu265)));
    signed char val165 = (*(data3_6291456+(((alu300+153)<<9)+alu301+alu265)));
    signed char val166 = (*(data3_6291456+(((alu300+154)<<9)+alu301+alu265)));
    signed char val167 = (*(data3_6291456+(((alu300+155)<<9)+alu301+alu265)));
    signed char val168 = (*(data3_6291456+(((alu300+156)<<9)+alu301+alu265)));
    signed char val169 = (*(data3_6291456+(((alu300+157)<<9)+alu301+alu265)));
    signed char val170 = (*(data3_6291456+(((alu300+158)<<9)+alu301+alu265)));
    signed char val171 = (*(data3_6291456+(((alu300+159)<<9)+alu301+alu265)));
    signed char val172 = (*(data3_6291456+(((alu300+160)<<9)+alu301+alu265)));
    signed char val173 = (*(data3_6291456+(((alu300+161)<<9)+alu301+alu265)));
    signed char val174 = (*(data3_6291456+(((alu300+162)<<9)+alu301+alu265)));
    signed char val175 = (*(data3_6291456+(((alu300+163)<<9)+alu301+alu265)));
    signed char val176 = (*(data3_6291456+(((alu300+164)<<9)+alu301+alu265)));
    signed char val177 = (*(data3_6291456+(((alu300+165)<<9)+alu301+alu265)));
    signed char val178 = (*(data3_6291456+(((alu300+166)<<9)+alu301+alu265)));
    signed char val179 = (*(data3_6291456+(((alu300+167)<<9)+alu301+alu265)));
    signed char val180 = (*(data3_6291456+(((alu300+168)<<9)+alu301+alu265)));
    signed char val181 = (*(data3_6291456+(((alu300+169)<<9)+alu301+alu265)));
    signed char val182 = (*(data3_6291456+(((alu300+170)<<9)+alu301+alu265)));
    signed char val183 = (*(data3_6291456+(((alu300+171)<<9)+alu301+alu265)));
    signed char val184 = (*(data3_6291456+(((alu300+172)<<9)+alu301+alu265)));
    signed char val185 = (*(data3_6291456+(((alu300+173)<<9)+alu301+alu265)));
    signed char val186 = (*(data3_6291456+(((alu300+174)<<9)+alu301+alu265)));
    signed char val187 = (*(data3_6291456+(((alu300+175)<<9)+alu301+alu265)));
    signed char val188 = (*(data3_6291456+(((alu300+176)<<9)+alu301+alu265)));
    signed char val189 = (*(data3_6291456+(((alu300+177)<<9)+alu301+alu265)));
    signed char val190 = (*(data3_6291456+(((alu300+178)<<9)+alu301+alu265)));
    signed char val191 = (*(data3_6291456+(((alu300+179)<<9)+alu301+alu265)));
    signed char val192 = (*(data3_6291456+(((alu300+180)<<9)+alu301+alu265)));
    signed char val193 = (*(data3_6291456+(((alu300+181)<<9)+alu301+alu265)));
    signed char val194 = (*(data3_6291456+(((alu300+182)<<9)+alu301+alu265)));
    signed char val195 = (*(data3_6291456+(((alu300+183)<<9)+alu301+alu265)));
    signed char val196 = (*(data3_6291456+(((alu300+184)<<9)+alu301+alu265)));
    signed char val197 = (*(data3_6291456+(((alu300+185)<<9)+alu301+alu265)));
    signed char val198 = (*(data3_6291456+(((alu300+186)<<9)+alu301+alu265)));
    signed char val199 = (*(data3_6291456+(((alu300+187)<<9)+alu301+alu265)));
    signed char val200 = (*(data3_6291456+(((alu300+188)<<9)+alu301+alu265)));
    signed char val201 = (*(data3_6291456+(((alu300+189)<<9)+alu301+alu265)));
    signed char val202 = (*(data3_6291456+(((alu300+190)<<9)+alu301+alu265)));
    signed char val203 = (*(data3_6291456+(((alu300+191)<<9)+alu301+alu265)));
    signed char val204 = (*(data3_6291456+(((alu300+192)<<9)+alu301+alu265)));
    signed char val205 = (*(data3_6291456+(((alu300+193)<<9)+alu301+alu265)));
    signed char val206 = (*(data3_6291456+(((alu300+194)<<9)+alu301+alu265)));
    signed char val207 = (*(data3_6291456+(((alu300+195)<<9)+alu301+alu265)));
    signed char val208 = (*(data3_6291456+(((alu300+196)<<9)+alu301+alu265)));
    signed char val209 = (*(data3_6291456+(((alu300+197)<<9)+alu301+alu265)));
    signed char val210 = (*(data3_6291456+(((alu300+198)<<9)+alu301+alu265)));
    signed char val211 = (*(data3_6291456+(((alu300+199)<<9)+alu301+alu265)));
    signed char val212 = (*(data3_6291456+(((alu300+200)<<9)+alu301+alu265)));
    signed char val213 = (*(data3_6291456+(((alu300+201)<<9)+alu301+alu265)));
    signed char val214 = (*(data3_6291456+(((alu300+202)<<9)+alu301+alu265)));
    signed char val215 = (*(data3_6291456+(((alu300+203)<<9)+alu301+alu265)));
    signed char val216 = (*(data3_6291456+(((alu300+204)<<9)+alu301+alu265)));
    signed char val217 = (*(data3_6291456+(((alu300+205)<<9)+alu301+alu265)));
    signed char val218 = (*(data3_6291456+(((alu300+206)<<9)+alu301+alu265)));
    signed char val219 = (*(data3_6291456+(((alu300+207)<<9)+alu301+alu265)));
    signed char val220 = (*(data3_6291456+(((alu300+208)<<9)+alu301+alu265)));
    signed char val221 = (*(data3_6291456+(((alu300+209)<<9)+alu301+alu265)));
    signed char val222 = (*(data3_6291456+(((alu300+210)<<9)+alu301+alu265)));
    signed char val223 = (*(data3_6291456+(((alu300+211)<<9)+alu301+alu265)));
    signed char val224 = (*(data3_6291456+(((alu300+212)<<9)+alu301+alu265)));
    signed char val225 = (*(data3_6291456+(((alu300+213)<<9)+alu301+alu265)));
    signed char val226 = (*(data3_6291456+(((alu300+214)<<9)+alu301+alu265)));
    signed char val227 = (*(data3_6291456+(((alu300+215)<<9)+alu301+alu265)));
    signed char val228 = (*(data3_6291456+(((alu300+216)<<9)+alu301+alu265)));
    signed char val229 = (*(data3_6291456+(((alu300+217)<<9)+alu301+alu265)));
    signed char val230 = (*(data3_6291456+(((alu300+218)<<9)+alu301+alu265)));
    signed char val231 = (*(data3_6291456+(((alu300+219)<<9)+alu301+alu265)));
    signed char val232 = (*(data3_6291456+(((alu300+220)<<9)+alu301+alu265)));
    signed char val233 = (*(data3_6291456+(((alu300+221)<<9)+alu301+alu265)));
    signed char val234 = (*(data3_6291456+(((alu300+222)<<9)+alu301+alu265)));
    signed char val235 = (*(data3_6291456+(((alu300+223)<<9)+alu301+alu265)));
    signed char val236 = (*(data3_6291456+(((alu300+224)<<9)+alu301+alu265)));
    signed char val237 = (*(data3_6291456+(((alu300+225)<<9)+alu301+alu265)));
    signed char val238 = (*(data3_6291456+(((alu300+226)<<9)+alu301+alu265)));
    signed char val239 = (*(data3_6291456+(((alu300+227)<<9)+alu301+alu265)));
    signed char val240 = (*(data3_6291456+(((alu300+228)<<9)+alu301+alu265)));
    signed char val241 = (*(data3_6291456+(((alu300+229)<<9)+alu301+alu265)));
    signed char val242 = (*(data3_6291456+(((alu300+230)<<9)+alu301+alu265)));
    signed char val243 = (*(data3_6291456+(((alu300+231)<<9)+alu301+alu265)));
    signed char val244 = (*(data3_6291456+(((alu300+232)<<9)+alu301+alu265)));
    signed char val245 = (*(data3_6291456+(((alu300+233)<<9)+alu301+alu265)));
    signed char val246 = (*(data3_6291456+(((alu300+234)<<9)+alu301+alu265)));
    signed char val247 = (*(data3_6291456+(((alu300+235)<<9)+alu301+alu265)));
    signed char val248 = (*(data3_6291456+(((alu300+236)<<9)+alu301+alu265)));
    signed char val249 = (*(data3_6291456+(((alu300+237)<<9)+alu301+alu265)));
    signed char val250 = (*(data3_6291456+(((alu300+238)<<9)+alu301+alu265)));
    signed char val251 = (*(data3_6291456+(((alu300+239)<<9)+alu301+alu265)));
    signed char val252 = (*(data3_6291456+(((alu300+240)<<9)+alu301+alu265)));
    signed char val253 = (*(data3_6291456+(((alu300+241)<<9)+alu301+alu265)));
    signed char val254 = (*(data3_6291456+(((alu300+242)<<9)+alu301+alu265)));
    signed char val255 = (*(data3_6291456+(((alu300+243)<<9)+alu301+alu265)));
    signed char val256 = (*(data3_6291456+(((alu300+244)<<9)+alu301+alu265)));
    signed char val257 = (*(data3_6291456+(((alu300+245)<<9)+alu301+alu265)));
    signed char val258 = (*(data3_6291456+(((alu300+246)<<9)+alu301+alu265)));
    signed char val259 = (*(data3_6291456+(((alu300+247)<<9)+alu301+alu265)));
    signed char val260 = (*(data3_6291456+(((alu300+248)<<9)+alu301+alu265)));
    signed char val261 = (*(data3_6291456+(((alu300+249)<<9)+alu301+alu265)));
    signed char val262 = (*(data3_6291456+(((alu300+250)<<9)+alu301+alu265)));
    signed char val263 = (*(data3_6291456+(((alu300+251)<<9)+alu301+alu265)));
    signed char val264 = (*(data3_6291456+(((alu300+252)<<9)+alu301+alu265)));
    signed char val265 = (*(data3_6291456+(((alu300+253)<<9)+alu301+alu265)));
    signed char val266 = (*(data3_6291456+(((alu300+254)<<9)+alu301+alu265)));
    signed char val267 = (*(data3_6291456+(((alu300+255)<<9)+alu301+alu265)));
    signed char val268 = (*(data3_6291456+((alu274<<17)+alu301+alu265)));
    int alu302 = (alu274<<3);
    float val269 = (*(data4_196608+(((alu302+1)<<9)+alu301+alu265)));
    float val270 = (*(data4_196608+(((alu302+2)<<9)+alu301+alu265)));
    float val271 = (*(data4_196608+(((alu302+3)<<9)+alu301+alu265)));
    float val272 = (*(data4_196608+((alu274<<12)+alu301+alu265)));
    if (alu271) {
      *(buf0+alu68) = ((((unsigned int)(val268))&255u)|((((unsigned int)(val13))&255u)<<8u)|((((unsigned int)(val14))&255u)<<16u)|((((unsigned int)(val15))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu69) = ((((unsigned int)(val16))&255u)|((((unsigned int)(val17))&255u)<<8u)|((((unsigned int)(val18))&255u)<<16u)|((((unsigned int)(val19))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu70) = ((((unsigned int)(val20))&255u)|((((unsigned int)(val21))&255u)<<8u)|((((unsigned int)(val22))&255u)<<16u)|((((unsigned int)(val23))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu71) = ((((unsigned int)(val24))&255u)|((((unsigned int)(val25))&255u)<<8u)|((((unsigned int)(val26))&255u)<<16u)|((((unsigned int)(val27))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu237) = ((((unsigned int)(val28))&255u)|((((unsigned int)(val29))&255u)<<8u)|((((unsigned int)(val30))&255u)<<16u)|((((unsigned int)(val31))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu238) = ((((unsigned int)(val32))&255u)|((((unsigned int)(val33))&255u)<<8u)|((((unsigned int)(val34))&255u)<<16u)|((((unsigned int)(val35))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu239) = ((((unsigned int)(val36))&255u)|((((unsigned int)(val37))&255u)<<8u)|((((unsigned int)(val38))&255u)<<16u)|((((unsigned int)(val39))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu240) = ((((unsigned int)(val40))&255u)|((((unsigned int)(val41))&255u)<<8u)|((((unsigned int)(val42))&255u)<<16u)|((((unsigned int)(val43))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu241) = ((((unsigned int)(val44))&255u)|((((unsigned int)(val45))&255u)<<8u)|((((unsigned int)(val46))&255u)<<16u)|((((unsigned int)(val47))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu242) = ((((unsigned int)(val48))&255u)|((((unsigned int)(val49))&255u)<<8u)|((((unsigned int)(val50))&255u)<<16u)|((((unsigned int)(val51))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu243) = ((((unsigned int)(val52))&255u)|((((unsigned int)(val53))&255u)<<8u)|((((unsigned int)(val54))&255u)<<16u)|((((unsigned int)(val55))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu244) = ((((unsigned int)(val56))&255u)|((((unsigned int)(val57))&255u)<<8u)|((((unsigned int)(val58))&255u)<<16u)|((((unsigned int)(val59))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu245) = ((((unsigned int)(val60))&255u)|((((unsigned int)(val61))&255u)<<8u)|((((unsigned int)(val62))&255u)<<16u)|((((unsigned int)(val63))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu246) = ((((unsigned int)(val64))&255u)|((((unsigned int)(val65))&255u)<<8u)|((((unsigned int)(val66))&255u)<<16u)|((((unsigned int)(val67))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu247) = ((((unsigned int)(val68))&255u)|((((unsigned int)(val69))&255u)<<8u)|((((unsigned int)(val70))&255u)<<16u)|((((unsigned int)(val71))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu248) = ((((unsigned int)(val72))&255u)|((((unsigned int)(val73))&255u)<<8u)|((((unsigned int)(val74))&255u)<<16u)|((((unsigned int)(val75))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu249) = ((((unsigned int)(val76))&255u)|((((unsigned int)(val77))&255u)<<8u)|((((unsigned int)(val78))&255u)<<16u)|((((unsigned int)(val79))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu250) = ((((unsigned int)(val80))&255u)|((((unsigned int)(val81))&255u)<<8u)|((((unsigned int)(val82))&255u)<<16u)|((((unsigned int)(val83))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu251) = ((((unsigned int)(val84))&255u)|((((unsigned int)(val85))&255u)<<8u)|((((unsigned int)(val86))&255u)<<16u)|((((unsigned int)(val87))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu252) = ((((unsigned int)(val88))&255u)|((((unsigned int)(val89))&255u)<<8u)|((((unsigned int)(val90))&255u)<<16u)|((((unsigned int)(val91))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu253) = ((((unsigned int)(val92))&255u)|((((unsigned int)(val93))&255u)<<8u)|((((unsigned int)(val94))&255u)<<16u)|((((unsigned int)(val95))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu254) = ((((unsigned int)(val96))&255u)|((((unsigned int)(val97))&255u)<<8u)|((((unsigned int)(val98))&255u)<<16u)|((((unsigned int)(val99))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu255) = ((((unsigned int)(val100))&255u)|((((unsigned int)(val101))&255u)<<8u)|((((unsigned int)(val102))&255u)<<16u)|((((unsigned int)(val103))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu256) = ((((unsigned int)(val104))&255u)|((((unsigned int)(val105))&255u)<<8u)|((((unsigned int)(val106))&255u)<<16u)|((((unsigned int)(val107))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu257) = ((((unsigned int)(val108))&255u)|((((unsigned int)(val109))&255u)<<8u)|((((unsigned int)(val110))&255u)<<16u)|((((unsigned int)(val111))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu258) = ((((unsigned int)(val112))&255u)|((((unsigned int)(val113))&255u)<<8u)|((((unsigned int)(val114))&255u)<<16u)|((((unsigned int)(val115))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu259) = ((((unsigned int)(val116))&255u)|((((unsigned int)(val117))&255u)<<8u)|((((unsigned int)(val118))&255u)<<16u)|((((unsigned int)(val119))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu260) = ((((unsigned int)(val120))&255u)|((((unsigned int)(val121))&255u)<<8u)|((((unsigned int)(val122))&255u)<<16u)|((((unsigned int)(val123))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu261) = ((((unsigned int)(val124))&255u)|((((unsigned int)(val125))&255u)<<8u)|((((unsigned int)(val126))&255u)<<16u)|((((unsigned int)(val127))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu262) = ((((unsigned int)(val128))&255u)|((((unsigned int)(val129))&255u)<<8u)|((((unsigned int)(val130))&255u)<<16u)|((((unsigned int)(val131))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu263) = ((((unsigned int)(val132))&255u)|((((unsigned int)(val133))&255u)<<8u)|((((unsigned int)(val134))&255u)<<16u)|((((unsigned int)(val135))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu264) = ((((unsigned int)(val136))&255u)|((((unsigned int)(val137))&255u)<<8u)|((((unsigned int)(val138))&255u)<<16u)|((((unsigned int)(val139))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu233) = tg_bitcast<unsigned int>((float)(val272));
    }
    if (alu271) {
      *(buf0+alu234) = tg_bitcast<unsigned int>((float)(val269));
    }
    if (alu271) {
      *(buf0+alu235) = tg_bitcast<unsigned int>((float)(val270));
    }
    if (alu271) {
      *(buf0+alu236) = tg_bitcast<unsigned int>((float)(val271));
    }
    __syncthreads();
    unsigned int val273 = (*(buf0+alu83));
    unsigned int val274 = (*(buf0+alu84));
    unsigned int val275 = (*(buf0+alu85));
    unsigned int val276 = (*(buf0+alu86));
    unsigned int val277 = (*(buf0+alu87));
    unsigned int val278 = (*(buf0+alu88));
    unsigned int val279 = (*(buf0+alu89));
    unsigned int val280 = (*(buf0+alu90));
    unsigned int val281 = (*(buf0+alu169));
    unsigned int val282 = (*(buf0+alu170));
    unsigned int val283 = (*(buf0+alu171));
    unsigned int val284 = (*(buf0+alu172));
    unsigned int val285 = (*(buf0+alu173));
    unsigned int val286 = (*(buf0+alu174));
    unsigned int val287 = (*(buf0+alu175));
    unsigned int val288 = (*(buf0+alu176));
    __syncthreads();
    unsigned int val289 = (*(buf0+alu148));
    unsigned int val290 = (*(buf0+alu149));
    unsigned int val291 = (*(buf0+alu150));
    unsigned int val292 = (*(buf0+alu153));
    unsigned int val293 = (*(buf0+alu154));
    unsigned int val294 = (*(buf0+alu155));
    int alu413 = (alu272+-1);
    bool alu414 = ((0<Ridx50)&(alu273!=((alu413/48)-((int)((((alu413%48)!=0)&((alu413<0)!=0)))))));
    if (alu414) {
      *(data0_5570560+alu78) = buf1;
    }
    char4 cast1 = make_char4(((signed char)(((val273>>0u)&255u))),((signed char)(((val273>>8u)&255u))),((signed char)(((val273>>16u)&255u))),((signed char)(((val273>>24u)&255u))));
    char4 cast2 = make_char4(((signed char)(((val274>>0u)&255u))),((signed char)(((val274>>8u)&255u))),((signed char)(((val274>>16u)&255u))),((signed char)(((val274>>24u)&255u))));
    char4 cast3 = make_char4(((signed char)(((val275>>0u)&255u))),((signed char)(((val275>>8u)&255u))),((signed char)(((val275>>16u)&255u))),((signed char)(((val275>>24u)&255u))));
    char4 cast4 = make_char4(((signed char)(((val276>>0u)&255u))),((signed char)(((val276>>8u)&255u))),((signed char)(((val276>>16u)&255u))),((signed char)(((val276>>24u)&255u))));
    char4 cast5 = make_char4(((signed char)(((val277>>0u)&255u))),((signed char)(((val277>>8u)&255u))),((signed char)(((val277>>16u)&255u))),((signed char)(((val277>>24u)&255u))));
    char4 cast6 = make_char4(((signed char)(((val278>>0u)&255u))),((signed char)(((val278>>8u)&255u))),((signed char)(((val278>>16u)&255u))),((signed char)(((val278>>24u)&255u))));
    char4 cast7 = make_char4(((signed char)(((val279>>0u)&255u))),((signed char)(((val279>>8u)&255u))),((signed char)(((val279>>16u)&255u))),((signed char)(((val279>>24u)&255u))));
    char4 cast8 = make_char4(((signed char)(((val280>>0u)&255u))),((signed char)(((val280>>8u)&255u))),((signed char)(((val280>>16u)&255u))),((signed char)(((val280>>24u)&255u))));
    signed_char8 alu418 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+4))))*4)));
    int4 wmma0 = __WMMA_8_16_16_signed_char_int(alu418, cast2, cast0);
    signed_char8 alu419 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+8))))*4)));
    int4 wmma1 = __WMMA_8_16_16_signed_char_int(alu419, cast3, cast0);
    signed_char8 alu420 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+12))))*4)));
    int4 wmma2 = __WMMA_8_16_16_signed_char_int(alu420, cast4, cast0);
    signed_char8 alu421 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+16))))*4)));
    int4 wmma3 = __WMMA_8_16_16_signed_char_int(alu421, cast5, cast0);
    signed_char8 alu422 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+20))))*4)));
    int4 wmma4 = __WMMA_8_16_16_signed_char_int(alu422, cast6, cast0);
    signed_char8 alu423 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+24))))*4)));
    int4 wmma5 = __WMMA_8_16_16_signed_char_int(alu423, cast7, cast0);
    signed_char8 alu424 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+28))))*4)));
    int4 wmma6 = __WMMA_8_16_16_signed_char_int(alu424, cast8, cast0);
    signed_char8 alu425 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)(alu80)))*4)));
    int4 wmma7 = __WMMA_8_16_16_signed_char_int(alu425, cast1, cast0);
    float cast9 = ((float)(((signed char)(((val290>>0u)&255u)))));
    float cast10 = ((float)(((signed char)(((val290>>8u)&255u)))));
    float cast11 = ((float)(((signed char)(((val290>>16u)&255u)))));
    float cast12 = ((float)(((signed char)(((val290>>24u)&255u)))));
    float cast13 = ((float)(((signed char)(((val291>>0u)&255u)))));
    float cast14 = ((float)(((signed char)(((val291>>8u)&255u)))));
    float cast15 = ((float)(((signed char)(((val291>>16u)&255u)))));
    float cast16 = ((float)(((signed char)(((val291>>24u)&255u)))));
    float cast17 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val289&65535u)))))));
    float cast18 = tg_bitcast<float>((unsigned int)(val281));
    float cast19 = tg_bitcast<float>((unsigned int)(val282));
    float cast20 = tg_bitcast<float>((unsigned int)(val283));
    float cast21 = tg_bitcast<float>((unsigned int)(val284));
    float alu426 = ((cast17*cast18*((cast9*((float)(wmma7.x)))+(cast10*((float)(wmma0.x)))))+(cast17*cast19*((cast11*((float)(wmma1.x)))+(cast12*((float)(wmma2.x)))))+(cast17*cast20*((cast13*((float)(wmma3.x)))+(cast14*((float)(wmma4.x)))))+(cast17*cast21*((cast15*((float)(wmma5.x)))+(cast16*((float)(wmma6.x))))));
    float alu427 = (alu414?alu426:(buf1+alu426));
    buf1 = alu427;
    if (alu414) {
      *(data0_5570560+(alu78+1)) = buf2;
    }
    float cast22 = tg_bitcast<float>((unsigned int)(val285));
    float cast23 = tg_bitcast<float>((unsigned int)(val286));
    float cast24 = tg_bitcast<float>((unsigned int)(val287));
    float cast25 = tg_bitcast<float>((unsigned int)(val288));
    float alu432 = ((cast17*cast22*((cast9*((float)(wmma7.y)))+(cast10*((float)(wmma0.y)))))+(cast17*cast23*((cast11*((float)(wmma1.y)))+(cast12*((float)(wmma2.y)))))+(cast17*cast24*((cast13*((float)(wmma3.y)))+(cast14*((float)(wmma4.y)))))+(cast17*cast25*((cast15*((float)(wmma5.y)))+(cast16*((float)(wmma6.y))))));
    float alu433 = (alu414?alu432:(buf2+alu432));
    buf2 = alu433;
    if (alu414) {
      *(data0_5570560+(alu78+1024)) = buf3;
    }
    float cast26 = ((float)(((signed char)(((val293>>0u)&255u)))));
    float cast27 = ((float)(((signed char)(((val293>>8u)&255u)))));
    float cast28 = ((float)(((signed char)(((val293>>16u)&255u)))));
    float cast29 = ((float)(((signed char)(((val293>>24u)&255u)))));
    float cast30 = ((float)(((signed char)(((val294>>0u)&255u)))));
    float cast31 = ((float)(((signed char)(((val294>>8u)&255u)))));
    float cast32 = ((float)(((signed char)(((val294>>16u)&255u)))));
    float cast33 = ((float)(((signed char)(((val294>>24u)&255u)))));
    float cast34 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val292&65535u)))))));
    float alu438 = ((cast34*cast18*((cast26*((float)(wmma7.z)))+(cast27*((float)(wmma0.z)))))+(cast34*cast19*((cast28*((float)(wmma1.z)))+(cast29*((float)(wmma2.z)))))+(cast34*cast20*((cast30*((float)(wmma3.z)))+(cast31*((float)(wmma4.z)))))+(cast34*cast21*((cast32*((float)(wmma5.z)))+(cast33*((float)(wmma6.z))))));
    float alu439 = (alu414?alu438:(buf3+alu438));
    buf3 = alu439;
    if (alu414) {
      *(data0_5570560+(alu78+1025)) = buf4;
    }
    float alu444 = ((cast34*cast22*((cast26*((float)(wmma7.w)))+(cast27*((float)(wmma0.w)))))+(cast34*cast23*((cast28*((float)(wmma1.w)))+(cast29*((float)(wmma2.w)))))+(cast34*cast24*((cast30*((float)(wmma3.w)))+(cast31*((float)(wmma4.w)))))+(cast34*cast25*((cast32*((float)(wmma5.w)))+(cast33*((float)(wmma6.w))))));
    float alu445 = (alu414?alu444:(buf4+alu444));
    buf4 = alu445;
    unsigned int val295 = (*(buf0+alu169));
    unsigned int val296 = (*(buf0+alu170));
    unsigned int val297 = (*(buf0+alu171));
    unsigned int val298 = (*(buf0+alu172));
    unsigned int val299 = (*(buf0+alu173));
    unsigned int val300 = (*(buf0+alu174));
    unsigned int val301 = (*(buf0+alu175));
    unsigned int val302 = (*(buf0+alu176));
    unsigned int val303 = (*(buf0+alu158));
    unsigned int val304 = (*(buf0+alu159));
    unsigned int val305 = (*(buf0+alu160));
    unsigned int val306 = (*(buf0+alu163));
    unsigned int val307 = (*(buf0+alu164));
    unsigned int val308 = (*(buf0+alu165));
    if (alu414) {
      *(data0_5570560+(alu78+2048)) = buf5;
    }
    signed_char8 alu450 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+1216))))*4)));
    int4 wmma8 = __WMMA_8_16_16_signed_char_int(alu450, cast1, cast0);
    signed_char8 alu451 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+1220))))*4)));
    int4 wmma9 = __WMMA_8_16_16_signed_char_int(alu451, cast2, cast0);
    signed_char8 alu452 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+1224))))*4)));
    int4 wmma10 = __WMMA_8_16_16_signed_char_int(alu452, cast3, cast0);
    signed_char8 alu453 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+1228))))*4)));
    int4 wmma11 = __WMMA_8_16_16_signed_char_int(alu453, cast4, cast0);
    signed_char8 alu454 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+1232))))*4)));
    int4 wmma12 = __WMMA_8_16_16_signed_char_int(alu454, cast5, cast0);
    signed_char8 alu455 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+1236))))*4)));
    int4 wmma13 = __WMMA_8_16_16_signed_char_int(alu455, cast6, cast0);
    signed_char8 alu456 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+1240))))*4)));
    int4 wmma14 = __WMMA_8_16_16_signed_char_int(alu456, cast7, cast0);
    signed_char8 alu457 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+1244))))*4)));
    int4 wmma15 = __WMMA_8_16_16_signed_char_int(alu457, cast8, cast0);
    float cast35 = ((float)(((signed char)(((val304>>0u)&255u)))));
    float cast36 = ((float)(((signed char)(((val304>>8u)&255u)))));
    float cast37 = ((float)(((signed char)(((val304>>16u)&255u)))));
    float cast38 = ((float)(((signed char)(((val304>>24u)&255u)))));
    float cast39 = ((float)(((signed char)(((val305>>0u)&255u)))));
    float cast40 = ((float)(((signed char)(((val305>>8u)&255u)))));
    float cast41 = ((float)(((signed char)(((val305>>16u)&255u)))));
    float cast42 = ((float)(((signed char)(((val305>>24u)&255u)))));
    float cast43 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val303&65535u)))))));
    float cast44 = tg_bitcast<float>((unsigned int)(val295));
    float cast45 = tg_bitcast<float>((unsigned int)(val296));
    float cast46 = tg_bitcast<float>((unsigned int)(val297));
    float cast47 = tg_bitcast<float>((unsigned int)(val298));
    float alu458 = ((cast43*cast44*((cast35*((float)(wmma8.x)))+(cast36*((float)(wmma9.x)))))+(cast43*cast45*((cast37*((float)(wmma10.x)))+(cast38*((float)(wmma11.x)))))+(cast43*cast46*((cast39*((float)(wmma12.x)))+(cast40*((float)(wmma13.x)))))+(cast43*cast47*((cast41*((float)(wmma14.x)))+(cast42*((float)(wmma15.x))))));
    float alu459 = (alu414?alu458:(buf5+alu458));
    buf5 = alu459;
    if (alu414) {
      *(data0_5570560+(alu78+2049)) = buf6;
    }
    float cast48 = tg_bitcast<float>((unsigned int)(val299));
    float cast49 = tg_bitcast<float>((unsigned int)(val300));
    float cast50 = tg_bitcast<float>((unsigned int)(val301));
    float cast51 = tg_bitcast<float>((unsigned int)(val302));
    float alu464 = ((cast43*cast48*((cast35*((float)(wmma8.y)))+(cast36*((float)(wmma9.y)))))+(cast43*cast49*((cast37*((float)(wmma10.y)))+(cast38*((float)(wmma11.y)))))+(cast43*cast50*((cast39*((float)(wmma12.y)))+(cast40*((float)(wmma13.y)))))+(cast43*cast51*((cast41*((float)(wmma14.y)))+(cast42*((float)(wmma15.y))))));
    float alu465 = (alu414?alu464:(buf6+alu464));
    buf6 = alu465;
    if (alu414) {
      *(data0_5570560+(alu78+3072)) = buf7;
    }
    float cast52 = ((float)(((signed char)(((val307>>0u)&255u)))));
    float cast53 = ((float)(((signed char)(((val307>>8u)&255u)))));
    float cast54 = ((float)(((signed char)(((val307>>16u)&255u)))));
    float cast55 = ((float)(((signed char)(((val307>>24u)&255u)))));
    float cast56 = ((float)(((signed char)(((val308>>0u)&255u)))));
    float cast57 = ((float)(((signed char)(((val308>>8u)&255u)))));
    float cast58 = ((float)(((signed char)(((val308>>16u)&255u)))));
    float cast59 = ((float)(((signed char)(((val308>>24u)&255u)))));
    float cast60 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val306&65535u)))))));
    float alu470 = ((cast60*cast44*((cast52*((float)(wmma8.z)))+(cast53*((float)(wmma9.z)))))+(cast60*cast45*((cast54*((float)(wmma10.z)))+(cast55*((float)(wmma11.z)))))+(cast60*cast46*((cast56*((float)(wmma12.z)))+(cast57*((float)(wmma13.z)))))+(cast60*cast47*((cast58*((float)(wmma14.z)))+(cast59*((float)(wmma15.z))))));
    float alu471 = (alu414?alu470:(buf7+alu470));
    buf7 = alu471;
    if (alu414) {
      *(data0_5570560+(alu78+3073)) = buf8;
    }
    float alu476 = ((cast60*cast48*((cast52*((float)(wmma8.w)))+(cast53*((float)(wmma9.w)))))+(cast60*cast49*((cast54*((float)(wmma10.w)))+(cast55*((float)(wmma11.w)))))+(cast60*cast50*((cast56*((float)(wmma12.w)))+(cast57*((float)(wmma13.w)))))+(cast60*cast51*((cast58*((float)(wmma14.w)))+(cast59*((float)(wmma15.w))))));
    float alu477 = (alu414?alu476:(buf8+alu476));
    buf8 = alu477;
    unsigned int val309 = (*(buf0+alu91));
    unsigned int val310 = (*(buf0+alu92));
    unsigned int val311 = (*(buf0+alu93));
    unsigned int val312 = (*(buf0+alu94));
    unsigned int val313 = (*(buf0+alu95));
    unsigned int val314 = (*(buf0+alu96));
    unsigned int val315 = (*(buf0+alu97));
    unsigned int val316 = (*(buf0+alu98));
    unsigned int val317 = (*(buf0+alu177));
    unsigned int val318 = (*(buf0+alu178));
    unsigned int val319 = (*(buf0+alu179));
    unsigned int val320 = (*(buf0+alu180));
    unsigned int val321 = (*(buf0+alu181));
    unsigned int val322 = (*(buf0+alu182));
    unsigned int val323 = (*(buf0+alu183));
    unsigned int val324 = (*(buf0+alu184));
    unsigned int val325 = (*(buf0+alu148));
    unsigned int val326 = (*(buf0+alu149));
    unsigned int val327 = (*(buf0+alu150));
    unsigned int val328 = (*(buf0+alu153));
    unsigned int val329 = (*(buf0+alu154));
    unsigned int val330 = (*(buf0+alu155));
    if (alu414) {
      *(data0_5570560+(alu78+16)) = buf9;
    }
    char4 cast61 = make_char4(((signed char)(((val309>>0u)&255u))),((signed char)(((val309>>8u)&255u))),((signed char)(((val309>>16u)&255u))),((signed char)(((val309>>24u)&255u))));
    char4 cast62 = make_char4(((signed char)(((val310>>0u)&255u))),((signed char)(((val310>>8u)&255u))),((signed char)(((val310>>16u)&255u))),((signed char)(((val310>>24u)&255u))));
    char4 cast63 = make_char4(((signed char)(((val311>>0u)&255u))),((signed char)(((val311>>8u)&255u))),((signed char)(((val311>>16u)&255u))),((signed char)(((val311>>24u)&255u))));
    char4 cast64 = make_char4(((signed char)(((val312>>0u)&255u))),((signed char)(((val312>>8u)&255u))),((signed char)(((val312>>16u)&255u))),((signed char)(((val312>>24u)&255u))));
    char4 cast65 = make_char4(((signed char)(((val313>>0u)&255u))),((signed char)(((val313>>8u)&255u))),((signed char)(((val313>>16u)&255u))),((signed char)(((val313>>24u)&255u))));
    char4 cast66 = make_char4(((signed char)(((val314>>0u)&255u))),((signed char)(((val314>>8u)&255u))),((signed char)(((val314>>16u)&255u))),((signed char)(((val314>>24u)&255u))));
    char4 cast67 = make_char4(((signed char)(((val315>>0u)&255u))),((signed char)(((val315>>8u)&255u))),((signed char)(((val315>>16u)&255u))),((signed char)(((val315>>24u)&255u))));
    char4 cast68 = make_char4(((signed char)(((val316>>0u)&255u))),((signed char)(((val316>>8u)&255u))),((signed char)(((val316>>16u)&255u))),((signed char)(((val316>>24u)&255u))));
    int4 wmma16 = __WMMA_8_16_16_signed_char_int(alu418, cast62, cast0);
    int4 wmma17 = __WMMA_8_16_16_signed_char_int(alu419, cast63, cast0);
    int4 wmma18 = __WMMA_8_16_16_signed_char_int(alu420, cast64, cast0);
    int4 wmma19 = __WMMA_8_16_16_signed_char_int(alu421, cast65, cast0);
    int4 wmma20 = __WMMA_8_16_16_signed_char_int(alu422, cast66, cast0);
    int4 wmma21 = __WMMA_8_16_16_signed_char_int(alu423, cast67, cast0);
    int4 wmma22 = __WMMA_8_16_16_signed_char_int(alu424, cast68, cast0);
    int4 wmma23 = __WMMA_8_16_16_signed_char_int(alu425, cast61, cast0);
    float cast69 = ((float)(((signed char)(((val326>>0u)&255u)))));
    float cast70 = ((float)(((signed char)(((val326>>8u)&255u)))));
    float cast71 = ((float)(((signed char)(((val326>>16u)&255u)))));
    float cast72 = ((float)(((signed char)(((val326>>24u)&255u)))));
    float cast73 = ((float)(((signed char)(((val327>>0u)&255u)))));
    float cast74 = ((float)(((signed char)(((val327>>8u)&255u)))));
    float cast75 = ((float)(((signed char)(((val327>>16u)&255u)))));
    float cast76 = ((float)(((signed char)(((val327>>24u)&255u)))));
    float cast77 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val325&65535u)))))));
    float cast78 = tg_bitcast<float>((unsigned int)(val317));
    float cast79 = tg_bitcast<float>((unsigned int)(val318));
    float cast80 = tg_bitcast<float>((unsigned int)(val319));
    float cast81 = tg_bitcast<float>((unsigned int)(val320));
    float alu482 = ((cast77*cast78*((cast69*((float)(wmma23.x)))+(cast70*((float)(wmma16.x)))))+(cast77*cast79*((cast71*((float)(wmma17.x)))+(cast72*((float)(wmma18.x)))))+(cast77*cast80*((cast73*((float)(wmma19.x)))+(cast74*((float)(wmma20.x)))))+(cast77*cast81*((cast75*((float)(wmma21.x)))+(cast76*((float)(wmma22.x))))));
    float alu483 = (alu414?alu482:(buf9+alu482));
    buf9 = alu483;
    if (alu414) {
      *(data0_5570560+(alu78+17)) = buf10;
    }
    float cast82 = tg_bitcast<float>((unsigned int)(val321));
    float cast83 = tg_bitcast<float>((unsigned int)(val322));
    float cast84 = tg_bitcast<float>((unsigned int)(val323));
    float cast85 = tg_bitcast<float>((unsigned int)(val324));
    float alu488 = ((cast77*cast82*((cast69*((float)(wmma23.y)))+(cast70*((float)(wmma16.y)))))+(cast77*cast83*((cast71*((float)(wmma17.y)))+(cast72*((float)(wmma18.y)))))+(cast77*cast84*((cast73*((float)(wmma19.y)))+(cast74*((float)(wmma20.y)))))+(cast77*cast85*((cast75*((float)(wmma21.y)))+(cast76*((float)(wmma22.y))))));
    float alu489 = (alu414?alu488:(buf10+alu488));
    buf10 = alu489;
    if (alu414) {
      *(data0_5570560+(alu78+1040)) = buf11;
    }
    float cast86 = ((float)(((signed char)(((val329>>0u)&255u)))));
    float cast87 = ((float)(((signed char)(((val329>>8u)&255u)))));
    float cast88 = ((float)(((signed char)(((val329>>16u)&255u)))));
    float cast89 = ((float)(((signed char)(((val329>>24u)&255u)))));
    float cast90 = ((float)(((signed char)(((val330>>0u)&255u)))));
    float cast91 = ((float)(((signed char)(((val330>>8u)&255u)))));
    float cast92 = ((float)(((signed char)(((val330>>16u)&255u)))));
    float cast93 = ((float)(((signed char)(((val330>>24u)&255u)))));
    float cast94 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val328&65535u)))))));
    float alu494 = ((cast94*cast78*((cast86*((float)(wmma23.z)))+(cast87*((float)(wmma16.z)))))+(cast94*cast79*((cast88*((float)(wmma17.z)))+(cast89*((float)(wmma18.z)))))+(cast94*cast80*((cast90*((float)(wmma19.z)))+(cast91*((float)(wmma20.z)))))+(cast94*cast81*((cast92*((float)(wmma21.z)))+(cast93*((float)(wmma22.z))))));
    float alu495 = (alu414?alu494:(buf11+alu494));
    buf11 = alu495;
    if (alu414) {
      *(data0_5570560+(alu78+1041)) = buf12;
    }
    float alu500 = ((cast94*cast82*((cast86*((float)(wmma23.w)))+(cast87*((float)(wmma16.w)))))+(cast94*cast83*((cast88*((float)(wmma17.w)))+(cast89*((float)(wmma18.w)))))+(cast94*cast84*((cast90*((float)(wmma19.w)))+(cast91*((float)(wmma20.w)))))+(cast94*cast85*((cast92*((float)(wmma21.w)))+(cast93*((float)(wmma22.w))))));
    float alu501 = (alu414?alu500:(buf12+alu500));
    buf12 = alu501;
    unsigned int val331 = (*(buf0+alu177));
    unsigned int val332 = (*(buf0+alu178));
    unsigned int val333 = (*(buf0+alu179));
    unsigned int val334 = (*(buf0+alu180));
    unsigned int val335 = (*(buf0+alu181));
    unsigned int val336 = (*(buf0+alu182));
    unsigned int val337 = (*(buf0+alu183));
    unsigned int val338 = (*(buf0+alu184));
    unsigned int val339 = (*(buf0+alu158));
    unsigned int val340 = (*(buf0+alu159));
    unsigned int val341 = (*(buf0+alu160));
    unsigned int val342 = (*(buf0+alu163));
    unsigned int val343 = (*(buf0+alu164));
    unsigned int val344 = (*(buf0+alu165));
    if (alu414) {
      *(data0_5570560+(alu78+2064)) = buf13;
    }
    int4 wmma24 = __WMMA_8_16_16_signed_char_int(alu450, cast61, cast0);
    int4 wmma25 = __WMMA_8_16_16_signed_char_int(alu451, cast62, cast0);
    int4 wmma26 = __WMMA_8_16_16_signed_char_int(alu452, cast63, cast0);
    int4 wmma27 = __WMMA_8_16_16_signed_char_int(alu453, cast64, cast0);
    int4 wmma28 = __WMMA_8_16_16_signed_char_int(alu454, cast65, cast0);
    int4 wmma29 = __WMMA_8_16_16_signed_char_int(alu455, cast66, cast0);
    int4 wmma30 = __WMMA_8_16_16_signed_char_int(alu456, cast67, cast0);
    int4 wmma31 = __WMMA_8_16_16_signed_char_int(alu457, cast68, cast0);
    float cast95 = ((float)(((signed char)(((val340>>0u)&255u)))));
    float cast96 = ((float)(((signed char)(((val340>>8u)&255u)))));
    float cast97 = ((float)(((signed char)(((val340>>16u)&255u)))));
    float cast98 = ((float)(((signed char)(((val340>>24u)&255u)))));
    float cast99 = ((float)(((signed char)(((val341>>0u)&255u)))));
    float cast100 = ((float)(((signed char)(((val341>>8u)&255u)))));
    float cast101 = ((float)(((signed char)(((val341>>16u)&255u)))));
    float cast102 = ((float)(((signed char)(((val341>>24u)&255u)))));
    float cast103 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val339&65535u)))))));
    float cast104 = tg_bitcast<float>((unsigned int)(val331));
    float cast105 = tg_bitcast<float>((unsigned int)(val332));
    float cast106 = tg_bitcast<float>((unsigned int)(val333));
    float cast107 = tg_bitcast<float>((unsigned int)(val334));
    float alu506 = ((cast103*cast104*((cast95*((float)(wmma24.x)))+(cast96*((float)(wmma25.x)))))+(cast103*cast105*((cast97*((float)(wmma26.x)))+(cast98*((float)(wmma27.x)))))+(cast103*cast106*((cast99*((float)(wmma28.x)))+(cast100*((float)(wmma29.x)))))+(cast103*cast107*((cast101*((float)(wmma30.x)))+(cast102*((float)(wmma31.x))))));
    float alu507 = (alu414?alu506:(buf13+alu506));
    buf13 = alu507;
    if (alu414) {
      *(data0_5570560+(alu78+2065)) = buf14;
    }
    float cast108 = tg_bitcast<float>((unsigned int)(val335));
    float cast109 = tg_bitcast<float>((unsigned int)(val336));
    float cast110 = tg_bitcast<float>((unsigned int)(val337));
    float cast111 = tg_bitcast<float>((unsigned int)(val338));
    float alu512 = ((cast103*cast108*((cast95*((float)(wmma24.y)))+(cast96*((float)(wmma25.y)))))+(cast103*cast109*((cast97*((float)(wmma26.y)))+(cast98*((float)(wmma27.y)))))+(cast103*cast110*((cast99*((float)(wmma28.y)))+(cast100*((float)(wmma29.y)))))+(cast103*cast111*((cast101*((float)(wmma30.y)))+(cast102*((float)(wmma31.y))))));
    float alu513 = (alu414?alu512:(buf14+alu512));
    buf14 = alu513;
    if (alu414) {
      *(data0_5570560+(alu78+3088)) = buf15;
    }
    float cast112 = ((float)(((signed char)(((val343>>0u)&255u)))));
    float cast113 = ((float)(((signed char)(((val343>>8u)&255u)))));
    float cast114 = ((float)(((signed char)(((val343>>16u)&255u)))));
    float cast115 = ((float)(((signed char)(((val343>>24u)&255u)))));
    float cast116 = ((float)(((signed char)(((val344>>0u)&255u)))));
    float cast117 = ((float)(((signed char)(((val344>>8u)&255u)))));
    float cast118 = ((float)(((signed char)(((val344>>16u)&255u)))));
    float cast119 = ((float)(((signed char)(((val344>>24u)&255u)))));
    float cast120 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val342&65535u)))))));
    float alu518 = ((cast120*cast104*((cast112*((float)(wmma24.z)))+(cast113*((float)(wmma25.z)))))+(cast120*cast105*((cast114*((float)(wmma26.z)))+(cast115*((float)(wmma27.z)))))+(cast120*cast106*((cast116*((float)(wmma28.z)))+(cast117*((float)(wmma29.z)))))+(cast120*cast107*((cast118*((float)(wmma30.z)))+(cast119*((float)(wmma31.z))))));
    float alu519 = (alu414?alu518:(buf15+alu518));
    buf15 = alu519;
    if (alu414) {
      *(data0_5570560+(alu78+3089)) = buf16;
    }
    float alu524 = ((cast120*cast108*((cast112*((float)(wmma24.w)))+(cast113*((float)(wmma25.w)))))+(cast120*cast109*((cast114*((float)(wmma26.w)))+(cast115*((float)(wmma27.w)))))+(cast120*cast110*((cast116*((float)(wmma28.w)))+(cast117*((float)(wmma29.w)))))+(cast120*cast111*((cast118*((float)(wmma30.w)))+(cast119*((float)(wmma31.w))))));
    float alu525 = (alu414?alu524:(buf16+alu524));
    buf16 = alu525;
    unsigned int val345 = (*(buf0+alu99));
    unsigned int val346 = (*(buf0+alu100));
    unsigned int val347 = (*(buf0+alu101));
    unsigned int val348 = (*(buf0+alu102));
    unsigned int val349 = (*(buf0+alu103));
    unsigned int val350 = (*(buf0+alu104));
    unsigned int val351 = (*(buf0+alu105));
    unsigned int val352 = (*(buf0+alu106));
    unsigned int val353 = (*(buf0+alu185));
    unsigned int val354 = (*(buf0+alu186));
    unsigned int val355 = (*(buf0+alu187));
    unsigned int val356 = (*(buf0+alu188));
    unsigned int val357 = (*(buf0+alu189));
    unsigned int val358 = (*(buf0+alu190));
    unsigned int val359 = (*(buf0+alu191));
    unsigned int val360 = (*(buf0+alu192));
    unsigned int val361 = (*(buf0+alu148));
    unsigned int val362 = (*(buf0+alu149));
    unsigned int val363 = (*(buf0+alu150));
    unsigned int val364 = (*(buf0+alu153));
    unsigned int val365 = (*(buf0+alu154));
    unsigned int val366 = (*(buf0+alu155));
    if (alu414) {
      *(data0_5570560+(alu78+32)) = buf17;
    }
    char4 cast121 = make_char4(((signed char)(((val345>>0u)&255u))),((signed char)(((val345>>8u)&255u))),((signed char)(((val345>>16u)&255u))),((signed char)(((val345>>24u)&255u))));
    char4 cast122 = make_char4(((signed char)(((val346>>0u)&255u))),((signed char)(((val346>>8u)&255u))),((signed char)(((val346>>16u)&255u))),((signed char)(((val346>>24u)&255u))));
    char4 cast123 = make_char4(((signed char)(((val347>>0u)&255u))),((signed char)(((val347>>8u)&255u))),((signed char)(((val347>>16u)&255u))),((signed char)(((val347>>24u)&255u))));
    char4 cast124 = make_char4(((signed char)(((val348>>0u)&255u))),((signed char)(((val348>>8u)&255u))),((signed char)(((val348>>16u)&255u))),((signed char)(((val348>>24u)&255u))));
    char4 cast125 = make_char4(((signed char)(((val349>>0u)&255u))),((signed char)(((val349>>8u)&255u))),((signed char)(((val349>>16u)&255u))),((signed char)(((val349>>24u)&255u))));
    char4 cast126 = make_char4(((signed char)(((val350>>0u)&255u))),((signed char)(((val350>>8u)&255u))),((signed char)(((val350>>16u)&255u))),((signed char)(((val350>>24u)&255u))));
    char4 cast127 = make_char4(((signed char)(((val351>>0u)&255u))),((signed char)(((val351>>8u)&255u))),((signed char)(((val351>>16u)&255u))),((signed char)(((val351>>24u)&255u))));
    char4 cast128 = make_char4(((signed char)(((val352>>0u)&255u))),((signed char)(((val352>>8u)&255u))),((signed char)(((val352>>16u)&255u))),((signed char)(((val352>>24u)&255u))));
    int4 wmma32 = __WMMA_8_16_16_signed_char_int(alu418, cast122, cast0);
    int4 wmma33 = __WMMA_8_16_16_signed_char_int(alu419, cast123, cast0);
    int4 wmma34 = __WMMA_8_16_16_signed_char_int(alu420, cast124, cast0);
    int4 wmma35 = __WMMA_8_16_16_signed_char_int(alu421, cast125, cast0);
    int4 wmma36 = __WMMA_8_16_16_signed_char_int(alu422, cast126, cast0);
    int4 wmma37 = __WMMA_8_16_16_signed_char_int(alu423, cast127, cast0);
    int4 wmma38 = __WMMA_8_16_16_signed_char_int(alu424, cast128, cast0);
    int4 wmma39 = __WMMA_8_16_16_signed_char_int(alu425, cast121, cast0);
    float cast129 = ((float)(((signed char)(((val362>>0u)&255u)))));
    float cast130 = ((float)(((signed char)(((val362>>8u)&255u)))));
    float cast131 = ((float)(((signed char)(((val362>>16u)&255u)))));
    float cast132 = ((float)(((signed char)(((val362>>24u)&255u)))));
    float cast133 = ((float)(((signed char)(((val363>>0u)&255u)))));
    float cast134 = ((float)(((signed char)(((val363>>8u)&255u)))));
    float cast135 = ((float)(((signed char)(((val363>>16u)&255u)))));
    float cast136 = ((float)(((signed char)(((val363>>24u)&255u)))));
    float cast137 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val361&65535u)))))));
    float cast138 = tg_bitcast<float>((unsigned int)(val353));
    float cast139 = tg_bitcast<float>((unsigned int)(val354));
    float cast140 = tg_bitcast<float>((unsigned int)(val355));
    float cast141 = tg_bitcast<float>((unsigned int)(val356));
    float alu530 = ((cast137*cast138*((cast129*((float)(wmma39.x)))+(cast130*((float)(wmma32.x)))))+(cast137*cast139*((cast131*((float)(wmma33.x)))+(cast132*((float)(wmma34.x)))))+(cast137*cast140*((cast133*((float)(wmma35.x)))+(cast134*((float)(wmma36.x)))))+(cast137*cast141*((cast135*((float)(wmma37.x)))+(cast136*((float)(wmma38.x))))));
    float alu531 = (alu414?alu530:(buf17+alu530));
    buf17 = alu531;
    if (alu414) {
      *(data0_5570560+(alu78+33)) = buf18;
    }
    float cast142 = tg_bitcast<float>((unsigned int)(val357));
    float cast143 = tg_bitcast<float>((unsigned int)(val358));
    float cast144 = tg_bitcast<float>((unsigned int)(val359));
    float cast145 = tg_bitcast<float>((unsigned int)(val360));
    float alu536 = ((cast137*cast142*((cast129*((float)(wmma39.y)))+(cast130*((float)(wmma32.y)))))+(cast137*cast143*((cast131*((float)(wmma33.y)))+(cast132*((float)(wmma34.y)))))+(cast137*cast144*((cast133*((float)(wmma35.y)))+(cast134*((float)(wmma36.y)))))+(cast137*cast145*((cast135*((float)(wmma37.y)))+(cast136*((float)(wmma38.y))))));
    float alu537 = (alu414?alu536:(buf18+alu536));
    buf18 = alu537;
    if (alu414) {
      *(data0_5570560+(alu78+1056)) = buf19;
    }
    float cast146 = ((float)(((signed char)(((val365>>0u)&255u)))));
    float cast147 = ((float)(((signed char)(((val365>>8u)&255u)))));
    float cast148 = ((float)(((signed char)(((val365>>16u)&255u)))));
    float cast149 = ((float)(((signed char)(((val365>>24u)&255u)))));
    float cast150 = ((float)(((signed char)(((val366>>0u)&255u)))));
    float cast151 = ((float)(((signed char)(((val366>>8u)&255u)))));
    float cast152 = ((float)(((signed char)(((val366>>16u)&255u)))));
    float cast153 = ((float)(((signed char)(((val366>>24u)&255u)))));
    float cast154 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val364&65535u)))))));
    float alu542 = ((cast154*cast138*((cast146*((float)(wmma39.z)))+(cast147*((float)(wmma32.z)))))+(cast154*cast139*((cast148*((float)(wmma33.z)))+(cast149*((float)(wmma34.z)))))+(cast154*cast140*((cast150*((float)(wmma35.z)))+(cast151*((float)(wmma36.z)))))+(cast154*cast141*((cast152*((float)(wmma37.z)))+(cast153*((float)(wmma38.z))))));
    float alu543 = (alu414?alu542:(buf19+alu542));
    buf19 = alu543;
    if (alu414) {
      *(data0_5570560+(alu78+1057)) = buf20;
    }
    float alu548 = ((cast154*cast142*((cast146*((float)(wmma39.w)))+(cast147*((float)(wmma32.w)))))+(cast154*cast143*((cast148*((float)(wmma33.w)))+(cast149*((float)(wmma34.w)))))+(cast154*cast144*((cast150*((float)(wmma35.w)))+(cast151*((float)(wmma36.w)))))+(cast154*cast145*((cast152*((float)(wmma37.w)))+(cast153*((float)(wmma38.w))))));
    float alu549 = (alu414?alu548:(buf20+alu548));
    buf20 = alu549;
    unsigned int val367 = (*(buf0+alu185));
    unsigned int val368 = (*(buf0+alu186));
    unsigned int val369 = (*(buf0+alu187));
    unsigned int val370 = (*(buf0+alu188));
    unsigned int val371 = (*(buf0+alu189));
    unsigned int val372 = (*(buf0+alu190));
    unsigned int val373 = (*(buf0+alu191));
    unsigned int val374 = (*(buf0+alu192));
    unsigned int val375 = (*(buf0+alu158));
    unsigned int val376 = (*(buf0+alu159));
    unsigned int val377 = (*(buf0+alu160));
    unsigned int val378 = (*(buf0+alu163));
    unsigned int val379 = (*(buf0+alu164));
    unsigned int val380 = (*(buf0+alu165));
    if (alu414) {
      *(data0_5570560+(alu78+2080)) = buf21;
    }
    int4 wmma40 = __WMMA_8_16_16_signed_char_int(alu450, cast121, cast0);
    int4 wmma41 = __WMMA_8_16_16_signed_char_int(alu451, cast122, cast0);
    int4 wmma42 = __WMMA_8_16_16_signed_char_int(alu452, cast123, cast0);
    int4 wmma43 = __WMMA_8_16_16_signed_char_int(alu453, cast124, cast0);
    int4 wmma44 = __WMMA_8_16_16_signed_char_int(alu454, cast125, cast0);
    int4 wmma45 = __WMMA_8_16_16_signed_char_int(alu455, cast126, cast0);
    int4 wmma46 = __WMMA_8_16_16_signed_char_int(alu456, cast127, cast0);
    int4 wmma47 = __WMMA_8_16_16_signed_char_int(alu457, cast128, cast0);
    float cast155 = ((float)(((signed char)(((val376>>0u)&255u)))));
    float cast156 = ((float)(((signed char)(((val376>>8u)&255u)))));
    float cast157 = ((float)(((signed char)(((val376>>16u)&255u)))));
    float cast158 = ((float)(((signed char)(((val376>>24u)&255u)))));
    float cast159 = ((float)(((signed char)(((val377>>0u)&255u)))));
    float cast160 = ((float)(((signed char)(((val377>>8u)&255u)))));
    float cast161 = ((float)(((signed char)(((val377>>16u)&255u)))));
    float cast162 = ((float)(((signed char)(((val377>>24u)&255u)))));
    float cast163 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val375&65535u)))))));
    float cast164 = tg_bitcast<float>((unsigned int)(val367));
    float cast165 = tg_bitcast<float>((unsigned int)(val368));
    float cast166 = tg_bitcast<float>((unsigned int)(val369));
    float cast167 = tg_bitcast<float>((unsigned int)(val370));
    float alu554 = ((cast163*cast164*((cast155*((float)(wmma40.x)))+(cast156*((float)(wmma41.x)))))+(cast163*cast165*((cast157*((float)(wmma42.x)))+(cast158*((float)(wmma43.x)))))+(cast163*cast166*((cast159*((float)(wmma44.x)))+(cast160*((float)(wmma45.x)))))+(cast163*cast167*((cast161*((float)(wmma46.x)))+(cast162*((float)(wmma47.x))))));
    float alu555 = (alu414?alu554:(buf21+alu554));
    buf21 = alu555;
    if (alu414) {
      *(data0_5570560+(alu78+2081)) = buf22;
    }
    float cast168 = tg_bitcast<float>((unsigned int)(val371));
    float cast169 = tg_bitcast<float>((unsigned int)(val372));
    float cast170 = tg_bitcast<float>((unsigned int)(val373));
    float cast171 = tg_bitcast<float>((unsigned int)(val374));
    float alu560 = ((cast163*cast168*((cast155*((float)(wmma40.y)))+(cast156*((float)(wmma41.y)))))+(cast163*cast169*((cast157*((float)(wmma42.y)))+(cast158*((float)(wmma43.y)))))+(cast163*cast170*((cast159*((float)(wmma44.y)))+(cast160*((float)(wmma45.y)))))+(cast163*cast171*((cast161*((float)(wmma46.y)))+(cast162*((float)(wmma47.y))))));
    float alu561 = (alu414?alu560:(buf22+alu560));
    buf22 = alu561;
    if (alu414) {
      *(data0_5570560+(alu78+3104)) = buf23;
    }
    float cast172 = ((float)(((signed char)(((val379>>0u)&255u)))));
    float cast173 = ((float)(((signed char)(((val379>>8u)&255u)))));
    float cast174 = ((float)(((signed char)(((val379>>16u)&255u)))));
    float cast175 = ((float)(((signed char)(((val379>>24u)&255u)))));
    float cast176 = ((float)(((signed char)(((val380>>0u)&255u)))));
    float cast177 = ((float)(((signed char)(((val380>>8u)&255u)))));
    float cast178 = ((float)(((signed char)(((val380>>16u)&255u)))));
    float cast179 = ((float)(((signed char)(((val380>>24u)&255u)))));
    float cast180 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val378&65535u)))))));
    float alu566 = ((cast180*cast164*((cast172*((float)(wmma40.z)))+(cast173*((float)(wmma41.z)))))+(cast180*cast165*((cast174*((float)(wmma42.z)))+(cast175*((float)(wmma43.z)))))+(cast180*cast166*((cast176*((float)(wmma44.z)))+(cast177*((float)(wmma45.z)))))+(cast180*cast167*((cast178*((float)(wmma46.z)))+(cast179*((float)(wmma47.z))))));
    float alu567 = (alu414?alu566:(buf23+alu566));
    buf23 = alu567;
    if (alu414) {
      *(data0_5570560+(alu78+3105)) = buf24;
    }
    float alu572 = ((cast180*cast168*((cast172*((float)(wmma40.w)))+(cast173*((float)(wmma41.w)))))+(cast180*cast169*((cast174*((float)(wmma42.w)))+(cast175*((float)(wmma43.w)))))+(cast180*cast170*((cast176*((float)(wmma44.w)))+(cast177*((float)(wmma45.w)))))+(cast180*cast171*((cast178*((float)(wmma46.w)))+(cast179*((float)(wmma47.w))))));
    float alu573 = (alu414?alu572:(buf24+alu572));
    buf24 = alu573;
    unsigned int val381 = (*(buf0+alu107));
    unsigned int val382 = (*(buf0+alu108));
    unsigned int val383 = (*(buf0+alu109));
    unsigned int val384 = (*(buf0+alu110));
    unsigned int val385 = (*(buf0+alu111));
    unsigned int val386 = (*(buf0+alu112));
    unsigned int val387 = (*(buf0+alu113));
    unsigned int val388 = (*(buf0+alu114));
    unsigned int val389 = (*(buf0+alu193));
    unsigned int val390 = (*(buf0+alu194));
    unsigned int val391 = (*(buf0+alu195));
    unsigned int val392 = (*(buf0+alu196));
    unsigned int val393 = (*(buf0+alu197));
    unsigned int val394 = (*(buf0+alu198));
    unsigned int val395 = (*(buf0+alu199));
    unsigned int val396 = (*(buf0+alu200));
    unsigned int val397 = (*(buf0+alu148));
    unsigned int val398 = (*(buf0+alu149));
    unsigned int val399 = (*(buf0+alu150));
    unsigned int val400 = (*(buf0+alu153));
    unsigned int val401 = (*(buf0+alu154));
    unsigned int val402 = (*(buf0+alu155));
    if (alu414) {
      *(data0_5570560+(alu78+48)) = buf25;
    }
    char4 cast181 = make_char4(((signed char)(((val381>>0u)&255u))),((signed char)(((val381>>8u)&255u))),((signed char)(((val381>>16u)&255u))),((signed char)(((val381>>24u)&255u))));
    char4 cast182 = make_char4(((signed char)(((val382>>0u)&255u))),((signed char)(((val382>>8u)&255u))),((signed char)(((val382>>16u)&255u))),((signed char)(((val382>>24u)&255u))));
    char4 cast183 = make_char4(((signed char)(((val383>>0u)&255u))),((signed char)(((val383>>8u)&255u))),((signed char)(((val383>>16u)&255u))),((signed char)(((val383>>24u)&255u))));
    char4 cast184 = make_char4(((signed char)(((val384>>0u)&255u))),((signed char)(((val384>>8u)&255u))),((signed char)(((val384>>16u)&255u))),((signed char)(((val384>>24u)&255u))));
    char4 cast185 = make_char4(((signed char)(((val385>>0u)&255u))),((signed char)(((val385>>8u)&255u))),((signed char)(((val385>>16u)&255u))),((signed char)(((val385>>24u)&255u))));
    char4 cast186 = make_char4(((signed char)(((val386>>0u)&255u))),((signed char)(((val386>>8u)&255u))),((signed char)(((val386>>16u)&255u))),((signed char)(((val386>>24u)&255u))));
    char4 cast187 = make_char4(((signed char)(((val387>>0u)&255u))),((signed char)(((val387>>8u)&255u))),((signed char)(((val387>>16u)&255u))),((signed char)(((val387>>24u)&255u))));
    char4 cast188 = make_char4(((signed char)(((val388>>0u)&255u))),((signed char)(((val388>>8u)&255u))),((signed char)(((val388>>16u)&255u))),((signed char)(((val388>>24u)&255u))));
    int4 wmma48 = __WMMA_8_16_16_signed_char_int(alu418, cast182, cast0);
    int4 wmma49 = __WMMA_8_16_16_signed_char_int(alu419, cast183, cast0);
    int4 wmma50 = __WMMA_8_16_16_signed_char_int(alu420, cast184, cast0);
    int4 wmma51 = __WMMA_8_16_16_signed_char_int(alu421, cast185, cast0);
    int4 wmma52 = __WMMA_8_16_16_signed_char_int(alu422, cast186, cast0);
    int4 wmma53 = __WMMA_8_16_16_signed_char_int(alu423, cast187, cast0);
    int4 wmma54 = __WMMA_8_16_16_signed_char_int(alu424, cast188, cast0);
    int4 wmma55 = __WMMA_8_16_16_signed_char_int(alu425, cast181, cast0);
    float cast189 = ((float)(((signed char)(((val398>>0u)&255u)))));
    float cast190 = ((float)(((signed char)(((val398>>8u)&255u)))));
    float cast191 = ((float)(((signed char)(((val398>>16u)&255u)))));
    float cast192 = ((float)(((signed char)(((val398>>24u)&255u)))));
    float cast193 = ((float)(((signed char)(((val399>>0u)&255u)))));
    float cast194 = ((float)(((signed char)(((val399>>8u)&255u)))));
    float cast195 = ((float)(((signed char)(((val399>>16u)&255u)))));
    float cast196 = ((float)(((signed char)(((val399>>24u)&255u)))));
    float cast197 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val397&65535u)))))));
    float cast198 = tg_bitcast<float>((unsigned int)(val389));
    float cast199 = tg_bitcast<float>((unsigned int)(val390));
    float cast200 = tg_bitcast<float>((unsigned int)(val391));
    float cast201 = tg_bitcast<float>((unsigned int)(val392));
    float alu578 = ((cast197*cast198*((cast189*((float)(wmma55.x)))+(cast190*((float)(wmma48.x)))))+(cast197*cast199*((cast191*((float)(wmma49.x)))+(cast192*((float)(wmma50.x)))))+(cast197*cast200*((cast193*((float)(wmma51.x)))+(cast194*((float)(wmma52.x)))))+(cast197*cast201*((cast195*((float)(wmma53.x)))+(cast196*((float)(wmma54.x))))));
    float alu579 = (alu414?alu578:(buf25+alu578));
    buf25 = alu579;
    if (alu414) {
      *(data0_5570560+(alu78+49)) = buf26;
    }
    float cast202 = tg_bitcast<float>((unsigned int)(val393));
    float cast203 = tg_bitcast<float>((unsigned int)(val394));
    float cast204 = tg_bitcast<float>((unsigned int)(val395));
    float cast205 = tg_bitcast<float>((unsigned int)(val396));
    float alu584 = ((cast197*cast202*((cast189*((float)(wmma55.y)))+(cast190*((float)(wmma48.y)))))+(cast197*cast203*((cast191*((float)(wmma49.y)))+(cast192*((float)(wmma50.y)))))+(cast197*cast204*((cast193*((float)(wmma51.y)))+(cast194*((float)(wmma52.y)))))+(cast197*cast205*((cast195*((float)(wmma53.y)))+(cast196*((float)(wmma54.y))))));
    float alu585 = (alu414?alu584:(buf26+alu584));
    buf26 = alu585;
    if (alu414) {
      *(data0_5570560+(alu78+1072)) = buf27;
    }
    float cast206 = ((float)(((signed char)(((val401>>0u)&255u)))));
    float cast207 = ((float)(((signed char)(((val401>>8u)&255u)))));
    float cast208 = ((float)(((signed char)(((val401>>16u)&255u)))));
    float cast209 = ((float)(((signed char)(((val401>>24u)&255u)))));
    float cast210 = ((float)(((signed char)(((val402>>0u)&255u)))));
    float cast211 = ((float)(((signed char)(((val402>>8u)&255u)))));
    float cast212 = ((float)(((signed char)(((val402>>16u)&255u)))));
    float cast213 = ((float)(((signed char)(((val402>>24u)&255u)))));
    float cast214 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val400&65535u)))))));
    float alu590 = ((cast214*cast198*((cast206*((float)(wmma55.z)))+(cast207*((float)(wmma48.z)))))+(cast214*cast199*((cast208*((float)(wmma49.z)))+(cast209*((float)(wmma50.z)))))+(cast214*cast200*((cast210*((float)(wmma51.z)))+(cast211*((float)(wmma52.z)))))+(cast214*cast201*((cast212*((float)(wmma53.z)))+(cast213*((float)(wmma54.z))))));
    float alu591 = (alu414?alu590:(buf27+alu590));
    buf27 = alu591;
    if (alu414) {
      *(data0_5570560+(alu78+1073)) = buf28;
    }
    float alu596 = ((cast214*cast202*((cast206*((float)(wmma55.w)))+(cast207*((float)(wmma48.w)))))+(cast214*cast203*((cast208*((float)(wmma49.w)))+(cast209*((float)(wmma50.w)))))+(cast214*cast204*((cast210*((float)(wmma51.w)))+(cast211*((float)(wmma52.w)))))+(cast214*cast205*((cast212*((float)(wmma53.w)))+(cast213*((float)(wmma54.w))))));
    float alu597 = (alu414?alu596:(buf28+alu596));
    buf28 = alu597;
    unsigned int val403 = (*(buf0+alu193));
    unsigned int val404 = (*(buf0+alu194));
    unsigned int val405 = (*(buf0+alu195));
    unsigned int val406 = (*(buf0+alu196));
    unsigned int val407 = (*(buf0+alu197));
    unsigned int val408 = (*(buf0+alu198));
    unsigned int val409 = (*(buf0+alu199));
    unsigned int val410 = (*(buf0+alu200));
    unsigned int val411 = (*(buf0+alu158));
    unsigned int val412 = (*(buf0+alu159));
    unsigned int val413 = (*(buf0+alu160));
    unsigned int val414 = (*(buf0+alu163));
    unsigned int val415 = (*(buf0+alu164));
    unsigned int val416 = (*(buf0+alu165));
    if (alu414) {
      *(data0_5570560+(alu78+2096)) = buf29;
    }
    int4 wmma56 = __WMMA_8_16_16_signed_char_int(alu450, cast181, cast0);
    int4 wmma57 = __WMMA_8_16_16_signed_char_int(alu451, cast182, cast0);
    int4 wmma58 = __WMMA_8_16_16_signed_char_int(alu452, cast183, cast0);
    int4 wmma59 = __WMMA_8_16_16_signed_char_int(alu453, cast184, cast0);
    int4 wmma60 = __WMMA_8_16_16_signed_char_int(alu454, cast185, cast0);
    int4 wmma61 = __WMMA_8_16_16_signed_char_int(alu455, cast186, cast0);
    int4 wmma62 = __WMMA_8_16_16_signed_char_int(alu456, cast187, cast0);
    int4 wmma63 = __WMMA_8_16_16_signed_char_int(alu457, cast188, cast0);
    float cast215 = ((float)(((signed char)(((val412>>0u)&255u)))));
    float cast216 = ((float)(((signed char)(((val412>>8u)&255u)))));
    float cast217 = ((float)(((signed char)(((val412>>16u)&255u)))));
    float cast218 = ((float)(((signed char)(((val412>>24u)&255u)))));
    float cast219 = ((float)(((signed char)(((val413>>0u)&255u)))));
    float cast220 = ((float)(((signed char)(((val413>>8u)&255u)))));
    float cast221 = ((float)(((signed char)(((val413>>16u)&255u)))));
    float cast222 = ((float)(((signed char)(((val413>>24u)&255u)))));
    float cast223 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val411&65535u)))))));
    float cast224 = tg_bitcast<float>((unsigned int)(val403));
    float cast225 = tg_bitcast<float>((unsigned int)(val404));
    float cast226 = tg_bitcast<float>((unsigned int)(val405));
    float cast227 = tg_bitcast<float>((unsigned int)(val406));
    float alu602 = ((cast223*cast224*((cast215*((float)(wmma56.x)))+(cast216*((float)(wmma57.x)))))+(cast223*cast225*((cast217*((float)(wmma58.x)))+(cast218*((float)(wmma59.x)))))+(cast223*cast226*((cast219*((float)(wmma60.x)))+(cast220*((float)(wmma61.x)))))+(cast223*cast227*((cast221*((float)(wmma62.x)))+(cast222*((float)(wmma63.x))))));
    float alu603 = (alu414?alu602:(buf29+alu602));
    buf29 = alu603;
    if (alu414) {
      *(data0_5570560+(alu78+2097)) = buf30;
    }
    float cast228 = tg_bitcast<float>((unsigned int)(val407));
    float cast229 = tg_bitcast<float>((unsigned int)(val408));
    float cast230 = tg_bitcast<float>((unsigned int)(val409));
    float cast231 = tg_bitcast<float>((unsigned int)(val410));
    float alu608 = ((cast223*cast228*((cast215*((float)(wmma56.y)))+(cast216*((float)(wmma57.y)))))+(cast223*cast229*((cast217*((float)(wmma58.y)))+(cast218*((float)(wmma59.y)))))+(cast223*cast230*((cast219*((float)(wmma60.y)))+(cast220*((float)(wmma61.y)))))+(cast223*cast231*((cast221*((float)(wmma62.y)))+(cast222*((float)(wmma63.y))))));
    float alu609 = (alu414?alu608:(buf30+alu608));
    buf30 = alu609;
    if (alu414) {
      *(data0_5570560+(alu78+3120)) = buf31;
    }
    float cast232 = ((float)(((signed char)(((val415>>0u)&255u)))));
    float cast233 = ((float)(((signed char)(((val415>>8u)&255u)))));
    float cast234 = ((float)(((signed char)(((val415>>16u)&255u)))));
    float cast235 = ((float)(((signed char)(((val415>>24u)&255u)))));
    float cast236 = ((float)(((signed char)(((val416>>0u)&255u)))));
    float cast237 = ((float)(((signed char)(((val416>>8u)&255u)))));
    float cast238 = ((float)(((signed char)(((val416>>16u)&255u)))));
    float cast239 = ((float)(((signed char)(((val416>>24u)&255u)))));
    float cast240 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val414&65535u)))))));
    float alu614 = ((cast240*cast224*((cast232*((float)(wmma56.z)))+(cast233*((float)(wmma57.z)))))+(cast240*cast225*((cast234*((float)(wmma58.z)))+(cast235*((float)(wmma59.z)))))+(cast240*cast226*((cast236*((float)(wmma60.z)))+(cast237*((float)(wmma61.z)))))+(cast240*cast227*((cast238*((float)(wmma62.z)))+(cast239*((float)(wmma63.z))))));
    float alu615 = (alu414?alu614:(buf31+alu614));
    buf31 = alu615;
    if (alu414) {
      *(data0_5570560+(alu78+3121)) = buf32;
    }
    float alu620 = ((cast240*cast228*((cast232*((float)(wmma56.w)))+(cast233*((float)(wmma57.w)))))+(cast240*cast229*((cast234*((float)(wmma58.w)))+(cast235*((float)(wmma59.w)))))+(cast240*cast230*((cast236*((float)(wmma60.w)))+(cast237*((float)(wmma61.w)))))+(cast240*cast231*((cast238*((float)(wmma62.w)))+(cast239*((float)(wmma63.w))))));
    float alu621 = (alu414?alu620:(buf32+alu620));
    buf32 = alu621;
    unsigned int val417 = (*(buf0+alu115));
    unsigned int val418 = (*(buf0+alu116));
    unsigned int val419 = (*(buf0+alu117));
    unsigned int val420 = (*(buf0+alu118));
    unsigned int val421 = (*(buf0+alu119));
    unsigned int val422 = (*(buf0+alu120));
    unsigned int val423 = (*(buf0+alu121));
    unsigned int val424 = (*(buf0+alu122));
    unsigned int val425 = (*(buf0+alu201));
    unsigned int val426 = (*(buf0+alu202));
    unsigned int val427 = (*(buf0+alu203));
    unsigned int val428 = (*(buf0+alu204));
    unsigned int val429 = (*(buf0+alu205));
    unsigned int val430 = (*(buf0+alu206));
    unsigned int val431 = (*(buf0+alu207));
    unsigned int val432 = (*(buf0+alu208));
    unsigned int val433 = (*(buf0+alu148));
    unsigned int val434 = (*(buf0+alu149));
    unsigned int val435 = (*(buf0+alu150));
    unsigned int val436 = (*(buf0+alu153));
    unsigned int val437 = (*(buf0+alu154));
    unsigned int val438 = (*(buf0+alu155));
    if (alu414) {
      *(data0_5570560+(alu78+64)) = buf33;
    }
    char4 cast241 = make_char4(((signed char)(((val417>>0u)&255u))),((signed char)(((val417>>8u)&255u))),((signed char)(((val417>>16u)&255u))),((signed char)(((val417>>24u)&255u))));
    char4 cast242 = make_char4(((signed char)(((val418>>0u)&255u))),((signed char)(((val418>>8u)&255u))),((signed char)(((val418>>16u)&255u))),((signed char)(((val418>>24u)&255u))));
    char4 cast243 = make_char4(((signed char)(((val419>>0u)&255u))),((signed char)(((val419>>8u)&255u))),((signed char)(((val419>>16u)&255u))),((signed char)(((val419>>24u)&255u))));
    char4 cast244 = make_char4(((signed char)(((val420>>0u)&255u))),((signed char)(((val420>>8u)&255u))),((signed char)(((val420>>16u)&255u))),((signed char)(((val420>>24u)&255u))));
    char4 cast245 = make_char4(((signed char)(((val421>>0u)&255u))),((signed char)(((val421>>8u)&255u))),((signed char)(((val421>>16u)&255u))),((signed char)(((val421>>24u)&255u))));
    char4 cast246 = make_char4(((signed char)(((val422>>0u)&255u))),((signed char)(((val422>>8u)&255u))),((signed char)(((val422>>16u)&255u))),((signed char)(((val422>>24u)&255u))));
    char4 cast247 = make_char4(((signed char)(((val423>>0u)&255u))),((signed char)(((val423>>8u)&255u))),((signed char)(((val423>>16u)&255u))),((signed char)(((val423>>24u)&255u))));
    char4 cast248 = make_char4(((signed char)(((val424>>0u)&255u))),((signed char)(((val424>>8u)&255u))),((signed char)(((val424>>16u)&255u))),((signed char)(((val424>>24u)&255u))));
    int4 wmma64 = __WMMA_8_16_16_signed_char_int(alu418, cast242, cast0);
    int4 wmma65 = __WMMA_8_16_16_signed_char_int(alu419, cast243, cast0);
    int4 wmma66 = __WMMA_8_16_16_signed_char_int(alu420, cast244, cast0);
    int4 wmma67 = __WMMA_8_16_16_signed_char_int(alu421, cast245, cast0);
    int4 wmma68 = __WMMA_8_16_16_signed_char_int(alu422, cast246, cast0);
    int4 wmma69 = __WMMA_8_16_16_signed_char_int(alu423, cast247, cast0);
    int4 wmma70 = __WMMA_8_16_16_signed_char_int(alu424, cast248, cast0);
    int4 wmma71 = __WMMA_8_16_16_signed_char_int(alu425, cast241, cast0);
    float cast249 = ((float)(((signed char)(((val434>>0u)&255u)))));
    float cast250 = ((float)(((signed char)(((val434>>8u)&255u)))));
    float cast251 = ((float)(((signed char)(((val434>>16u)&255u)))));
    float cast252 = ((float)(((signed char)(((val434>>24u)&255u)))));
    float cast253 = ((float)(((signed char)(((val435>>0u)&255u)))));
    float cast254 = ((float)(((signed char)(((val435>>8u)&255u)))));
    float cast255 = ((float)(((signed char)(((val435>>16u)&255u)))));
    float cast256 = ((float)(((signed char)(((val435>>24u)&255u)))));
    float cast257 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val433&65535u)))))));
    float cast258 = tg_bitcast<float>((unsigned int)(val425));
    float cast259 = tg_bitcast<float>((unsigned int)(val426));
    float cast260 = tg_bitcast<float>((unsigned int)(val427));
    float cast261 = tg_bitcast<float>((unsigned int)(val428));
    float alu626 = ((cast257*cast258*((cast249*((float)(wmma71.x)))+(cast250*((float)(wmma64.x)))))+(cast257*cast259*((cast251*((float)(wmma65.x)))+(cast252*((float)(wmma66.x)))))+(cast257*cast260*((cast253*((float)(wmma67.x)))+(cast254*((float)(wmma68.x)))))+(cast257*cast261*((cast255*((float)(wmma69.x)))+(cast256*((float)(wmma70.x))))));
    float alu627 = (alu414?alu626:(buf33+alu626));
    buf33 = alu627;
    if (alu414) {
      *(data0_5570560+(alu78+65)) = buf34;
    }
    float cast262 = tg_bitcast<float>((unsigned int)(val429));
    float cast263 = tg_bitcast<float>((unsigned int)(val430));
    float cast264 = tg_bitcast<float>((unsigned int)(val431));
    float cast265 = tg_bitcast<float>((unsigned int)(val432));
    float alu632 = ((cast257*cast262*((cast249*((float)(wmma71.y)))+(cast250*((float)(wmma64.y)))))+(cast257*cast263*((cast251*((float)(wmma65.y)))+(cast252*((float)(wmma66.y)))))+(cast257*cast264*((cast253*((float)(wmma67.y)))+(cast254*((float)(wmma68.y)))))+(cast257*cast265*((cast255*((float)(wmma69.y)))+(cast256*((float)(wmma70.y))))));
    float alu633 = (alu414?alu632:(buf34+alu632));
    buf34 = alu633;
    if (alu414) {
      *(data0_5570560+(alu78+1088)) = buf35;
    }
    float cast266 = ((float)(((signed char)(((val437>>0u)&255u)))));
    float cast267 = ((float)(((signed char)(((val437>>8u)&255u)))));
    float cast268 = ((float)(((signed char)(((val437>>16u)&255u)))));
    float cast269 = ((float)(((signed char)(((val437>>24u)&255u)))));
    float cast270 = ((float)(((signed char)(((val438>>0u)&255u)))));
    float cast271 = ((float)(((signed char)(((val438>>8u)&255u)))));
    float cast272 = ((float)(((signed char)(((val438>>16u)&255u)))));
    float cast273 = ((float)(((signed char)(((val438>>24u)&255u)))));
    float cast274 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val436&65535u)))))));
    float alu638 = ((cast274*cast258*((cast266*((float)(wmma71.z)))+(cast267*((float)(wmma64.z)))))+(cast274*cast259*((cast268*((float)(wmma65.z)))+(cast269*((float)(wmma66.z)))))+(cast274*cast260*((cast270*((float)(wmma67.z)))+(cast271*((float)(wmma68.z)))))+(cast274*cast261*((cast272*((float)(wmma69.z)))+(cast273*((float)(wmma70.z))))));
    float alu639 = (alu414?alu638:(buf35+alu638));
    buf35 = alu639;
    if (alu414) {
      *(data0_5570560+(alu78+1089)) = buf36;
    }
    float alu644 = ((cast274*cast262*((cast266*((float)(wmma71.w)))+(cast267*((float)(wmma64.w)))))+(cast274*cast263*((cast268*((float)(wmma65.w)))+(cast269*((float)(wmma66.w)))))+(cast274*cast264*((cast270*((float)(wmma67.w)))+(cast271*((float)(wmma68.w)))))+(cast274*cast265*((cast272*((float)(wmma69.w)))+(cast273*((float)(wmma70.w))))));
    float alu645 = (alu414?alu644:(buf36+alu644));
    buf36 = alu645;
    unsigned int val439 = (*(buf0+alu201));
    unsigned int val440 = (*(buf0+alu202));
    unsigned int val441 = (*(buf0+alu203));
    unsigned int val442 = (*(buf0+alu204));
    unsigned int val443 = (*(buf0+alu205));
    unsigned int val444 = (*(buf0+alu206));
    unsigned int val445 = (*(buf0+alu207));
    unsigned int val446 = (*(buf0+alu208));
    unsigned int val447 = (*(buf0+alu158));
    unsigned int val448 = (*(buf0+alu159));
    unsigned int val449 = (*(buf0+alu160));
    unsigned int val450 = (*(buf0+alu163));
    unsigned int val451 = (*(buf0+alu164));
    unsigned int val452 = (*(buf0+alu165));
    if (alu414) {
      *(data0_5570560+(alu78+2112)) = buf37;
    }
    int4 wmma72 = __WMMA_8_16_16_signed_char_int(alu450, cast241, cast0);
    int4 wmma73 = __WMMA_8_16_16_signed_char_int(alu451, cast242, cast0);
    int4 wmma74 = __WMMA_8_16_16_signed_char_int(alu452, cast243, cast0);
    int4 wmma75 = __WMMA_8_16_16_signed_char_int(alu453, cast244, cast0);
    int4 wmma76 = __WMMA_8_16_16_signed_char_int(alu454, cast245, cast0);
    int4 wmma77 = __WMMA_8_16_16_signed_char_int(alu455, cast246, cast0);
    int4 wmma78 = __WMMA_8_16_16_signed_char_int(alu456, cast247, cast0);
    int4 wmma79 = __WMMA_8_16_16_signed_char_int(alu457, cast248, cast0);
    float cast275 = ((float)(((signed char)(((val448>>0u)&255u)))));
    float cast276 = ((float)(((signed char)(((val448>>8u)&255u)))));
    float cast277 = ((float)(((signed char)(((val448>>16u)&255u)))));
    float cast278 = ((float)(((signed char)(((val448>>24u)&255u)))));
    float cast279 = ((float)(((signed char)(((val449>>0u)&255u)))));
    float cast280 = ((float)(((signed char)(((val449>>8u)&255u)))));
    float cast281 = ((float)(((signed char)(((val449>>16u)&255u)))));
    float cast282 = ((float)(((signed char)(((val449>>24u)&255u)))));
    float cast283 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val447&65535u)))))));
    float cast284 = tg_bitcast<float>((unsigned int)(val439));
    float cast285 = tg_bitcast<float>((unsigned int)(val440));
    float cast286 = tg_bitcast<float>((unsigned int)(val441));
    float cast287 = tg_bitcast<float>((unsigned int)(val442));
    float alu650 = ((cast283*cast284*((cast275*((float)(wmma72.x)))+(cast276*((float)(wmma73.x)))))+(cast283*cast285*((cast277*((float)(wmma74.x)))+(cast278*((float)(wmma75.x)))))+(cast283*cast286*((cast279*((float)(wmma76.x)))+(cast280*((float)(wmma77.x)))))+(cast283*cast287*((cast281*((float)(wmma78.x)))+(cast282*((float)(wmma79.x))))));
    float alu651 = (alu414?alu650:(buf37+alu650));
    buf37 = alu651;
    if (alu414) {
      *(data0_5570560+(alu78+2113)) = buf38;
    }
    float cast288 = tg_bitcast<float>((unsigned int)(val443));
    float cast289 = tg_bitcast<float>((unsigned int)(val444));
    float cast290 = tg_bitcast<float>((unsigned int)(val445));
    float cast291 = tg_bitcast<float>((unsigned int)(val446));
    float alu656 = ((cast283*cast288*((cast275*((float)(wmma72.y)))+(cast276*((float)(wmma73.y)))))+(cast283*cast289*((cast277*((float)(wmma74.y)))+(cast278*((float)(wmma75.y)))))+(cast283*cast290*((cast279*((float)(wmma76.y)))+(cast280*((float)(wmma77.y)))))+(cast283*cast291*((cast281*((float)(wmma78.y)))+(cast282*((float)(wmma79.y))))));
    float alu657 = (alu414?alu656:(buf38+alu656));
    buf38 = alu657;
    if (alu414) {
      *(data0_5570560+(alu78+3136)) = buf39;
    }
    float cast292 = ((float)(((signed char)(((val451>>0u)&255u)))));
    float cast293 = ((float)(((signed char)(((val451>>8u)&255u)))));
    float cast294 = ((float)(((signed char)(((val451>>16u)&255u)))));
    float cast295 = ((float)(((signed char)(((val451>>24u)&255u)))));
    float cast296 = ((float)(((signed char)(((val452>>0u)&255u)))));
    float cast297 = ((float)(((signed char)(((val452>>8u)&255u)))));
    float cast298 = ((float)(((signed char)(((val452>>16u)&255u)))));
    float cast299 = ((float)(((signed char)(((val452>>24u)&255u)))));
    float cast300 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val450&65535u)))))));
    float alu662 = ((cast300*cast284*((cast292*((float)(wmma72.z)))+(cast293*((float)(wmma73.z)))))+(cast300*cast285*((cast294*((float)(wmma74.z)))+(cast295*((float)(wmma75.z)))))+(cast300*cast286*((cast296*((float)(wmma76.z)))+(cast297*((float)(wmma77.z)))))+(cast300*cast287*((cast298*((float)(wmma78.z)))+(cast299*((float)(wmma79.z))))));
    float alu663 = (alu414?alu662:(buf39+alu662));
    buf39 = alu663;
    if (alu414) {
      *(data0_5570560+(alu78+3137)) = buf40;
    }
    float alu668 = ((cast300*cast288*((cast292*((float)(wmma72.w)))+(cast293*((float)(wmma73.w)))))+(cast300*cast289*((cast294*((float)(wmma74.w)))+(cast295*((float)(wmma75.w)))))+(cast300*cast290*((cast296*((float)(wmma76.w)))+(cast297*((float)(wmma77.w)))))+(cast300*cast291*((cast298*((float)(wmma78.w)))+(cast299*((float)(wmma79.w))))));
    float alu669 = (alu414?alu668:(buf40+alu668));
    buf40 = alu669;
    unsigned int val453 = (*(buf0+alu123));
    unsigned int val454 = (*(buf0+alu124));
    unsigned int val455 = (*(buf0+alu125));
    unsigned int val456 = (*(buf0+alu126));
    unsigned int val457 = (*(buf0+alu127));
    unsigned int val458 = (*(buf0+alu128));
    unsigned int val459 = (*(buf0+alu129));
    unsigned int val460 = (*(buf0+alu130));
    unsigned int val461 = (*(buf0+alu209));
    unsigned int val462 = (*(buf0+alu210));
    unsigned int val463 = (*(buf0+alu211));
    unsigned int val464 = (*(buf0+alu212));
    unsigned int val465 = (*(buf0+alu213));
    unsigned int val466 = (*(buf0+alu214));
    unsigned int val467 = (*(buf0+alu215));
    unsigned int val468 = (*(buf0+alu216));
    unsigned int val469 = (*(buf0+alu148));
    unsigned int val470 = (*(buf0+alu149));
    unsigned int val471 = (*(buf0+alu150));
    unsigned int val472 = (*(buf0+alu153));
    unsigned int val473 = (*(buf0+alu154));
    unsigned int val474 = (*(buf0+alu155));
    if (alu414) {
      *(data0_5570560+(alu78+80)) = buf41;
    }
    char4 cast301 = make_char4(((signed char)(((val453>>0u)&255u))),((signed char)(((val453>>8u)&255u))),((signed char)(((val453>>16u)&255u))),((signed char)(((val453>>24u)&255u))));
    char4 cast302 = make_char4(((signed char)(((val454>>0u)&255u))),((signed char)(((val454>>8u)&255u))),((signed char)(((val454>>16u)&255u))),((signed char)(((val454>>24u)&255u))));
    char4 cast303 = make_char4(((signed char)(((val455>>0u)&255u))),((signed char)(((val455>>8u)&255u))),((signed char)(((val455>>16u)&255u))),((signed char)(((val455>>24u)&255u))));
    char4 cast304 = make_char4(((signed char)(((val456>>0u)&255u))),((signed char)(((val456>>8u)&255u))),((signed char)(((val456>>16u)&255u))),((signed char)(((val456>>24u)&255u))));
    char4 cast305 = make_char4(((signed char)(((val457>>0u)&255u))),((signed char)(((val457>>8u)&255u))),((signed char)(((val457>>16u)&255u))),((signed char)(((val457>>24u)&255u))));
    char4 cast306 = make_char4(((signed char)(((val458>>0u)&255u))),((signed char)(((val458>>8u)&255u))),((signed char)(((val458>>16u)&255u))),((signed char)(((val458>>24u)&255u))));
    char4 cast307 = make_char4(((signed char)(((val459>>0u)&255u))),((signed char)(((val459>>8u)&255u))),((signed char)(((val459>>16u)&255u))),((signed char)(((val459>>24u)&255u))));
    char4 cast308 = make_char4(((signed char)(((val460>>0u)&255u))),((signed char)(((val460>>8u)&255u))),((signed char)(((val460>>16u)&255u))),((signed char)(((val460>>24u)&255u))));
    int4 wmma80 = __WMMA_8_16_16_signed_char_int(alu418, cast302, cast0);
    int4 wmma81 = __WMMA_8_16_16_signed_char_int(alu419, cast303, cast0);
    int4 wmma82 = __WMMA_8_16_16_signed_char_int(alu420, cast304, cast0);
    int4 wmma83 = __WMMA_8_16_16_signed_char_int(alu421, cast305, cast0);
    int4 wmma84 = __WMMA_8_16_16_signed_char_int(alu422, cast306, cast0);
    int4 wmma85 = __WMMA_8_16_16_signed_char_int(alu423, cast307, cast0);
    int4 wmma86 = __WMMA_8_16_16_signed_char_int(alu424, cast308, cast0);
    int4 wmma87 = __WMMA_8_16_16_signed_char_int(alu425, cast301, cast0);
    float cast309 = ((float)(((signed char)(((val470>>0u)&255u)))));
    float cast310 = ((float)(((signed char)(((val470>>8u)&255u)))));
    float cast311 = ((float)(((signed char)(((val470>>16u)&255u)))));
    float cast312 = ((float)(((signed char)(((val470>>24u)&255u)))));
    float cast313 = ((float)(((signed char)(((val471>>0u)&255u)))));
    float cast314 = ((float)(((signed char)(((val471>>8u)&255u)))));
    float cast315 = ((float)(((signed char)(((val471>>16u)&255u)))));
    float cast316 = ((float)(((signed char)(((val471>>24u)&255u)))));
    float cast317 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val469&65535u)))))));
    float cast318 = tg_bitcast<float>((unsigned int)(val461));
    float cast319 = tg_bitcast<float>((unsigned int)(val462));
    float cast320 = tg_bitcast<float>((unsigned int)(val463));
    float cast321 = tg_bitcast<float>((unsigned int)(val464));
    float alu674 = ((cast317*cast318*((cast309*((float)(wmma87.x)))+(cast310*((float)(wmma80.x)))))+(cast317*cast319*((cast311*((float)(wmma81.x)))+(cast312*((float)(wmma82.x)))))+(cast317*cast320*((cast313*((float)(wmma83.x)))+(cast314*((float)(wmma84.x)))))+(cast317*cast321*((cast315*((float)(wmma85.x)))+(cast316*((float)(wmma86.x))))));
    float alu675 = (alu414?alu674:(buf41+alu674));
    buf41 = alu675;
    if (alu414) {
      *(data0_5570560+(alu78+81)) = buf42;
    }
    float cast322 = tg_bitcast<float>((unsigned int)(val465));
    float cast323 = tg_bitcast<float>((unsigned int)(val466));
    float cast324 = tg_bitcast<float>((unsigned int)(val467));
    float cast325 = tg_bitcast<float>((unsigned int)(val468));
    float alu680 = ((cast317*cast322*((cast309*((float)(wmma87.y)))+(cast310*((float)(wmma80.y)))))+(cast317*cast323*((cast311*((float)(wmma81.y)))+(cast312*((float)(wmma82.y)))))+(cast317*cast324*((cast313*((float)(wmma83.y)))+(cast314*((float)(wmma84.y)))))+(cast317*cast325*((cast315*((float)(wmma85.y)))+(cast316*((float)(wmma86.y))))));
    float alu681 = (alu414?alu680:(buf42+alu680));
    buf42 = alu681;
    if (alu414) {
      *(data0_5570560+(alu78+1104)) = buf43;
    }
    float cast326 = ((float)(((signed char)(((val473>>0u)&255u)))));
    float cast327 = ((float)(((signed char)(((val473>>8u)&255u)))));
    float cast328 = ((float)(((signed char)(((val473>>16u)&255u)))));
    float cast329 = ((float)(((signed char)(((val473>>24u)&255u)))));
    float cast330 = ((float)(((signed char)(((val474>>0u)&255u)))));
    float cast331 = ((float)(((signed char)(((val474>>8u)&255u)))));
    float cast332 = ((float)(((signed char)(((val474>>16u)&255u)))));
    float cast333 = ((float)(((signed char)(((val474>>24u)&255u)))));
    float cast334 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val472&65535u)))))));
    float alu686 = ((cast334*cast318*((cast326*((float)(wmma87.z)))+(cast327*((float)(wmma80.z)))))+(cast334*cast319*((cast328*((float)(wmma81.z)))+(cast329*((float)(wmma82.z)))))+(cast334*cast320*((cast330*((float)(wmma83.z)))+(cast331*((float)(wmma84.z)))))+(cast334*cast321*((cast332*((float)(wmma85.z)))+(cast333*((float)(wmma86.z))))));
    float alu687 = (alu414?alu686:(buf43+alu686));
    buf43 = alu687;
    if (alu414) {
      *(data0_5570560+(alu78+1105)) = buf44;
    }
    float alu692 = ((cast334*cast322*((cast326*((float)(wmma87.w)))+(cast327*((float)(wmma80.w)))))+(cast334*cast323*((cast328*((float)(wmma81.w)))+(cast329*((float)(wmma82.w)))))+(cast334*cast324*((cast330*((float)(wmma83.w)))+(cast331*((float)(wmma84.w)))))+(cast334*cast325*((cast332*((float)(wmma85.w)))+(cast333*((float)(wmma86.w))))));
    float alu693 = (alu414?alu692:(buf44+alu692));
    buf44 = alu693;
    unsigned int val475 = (*(buf0+alu209));
    unsigned int val476 = (*(buf0+alu210));
    unsigned int val477 = (*(buf0+alu211));
    unsigned int val478 = (*(buf0+alu212));
    unsigned int val479 = (*(buf0+alu213));
    unsigned int val480 = (*(buf0+alu214));
    unsigned int val481 = (*(buf0+alu215));
    unsigned int val482 = (*(buf0+alu216));
    unsigned int val483 = (*(buf0+alu158));
    unsigned int val484 = (*(buf0+alu159));
    unsigned int val485 = (*(buf0+alu160));
    unsigned int val486 = (*(buf0+alu163));
    unsigned int val487 = (*(buf0+alu164));
    unsigned int val488 = (*(buf0+alu165));
    if (alu414) {
      *(data0_5570560+(alu78+2128)) = buf45;
    }
    int4 wmma88 = __WMMA_8_16_16_signed_char_int(alu450, cast301, cast0);
    int4 wmma89 = __WMMA_8_16_16_signed_char_int(alu451, cast302, cast0);
    int4 wmma90 = __WMMA_8_16_16_signed_char_int(alu452, cast303, cast0);
    int4 wmma91 = __WMMA_8_16_16_signed_char_int(alu453, cast304, cast0);
    int4 wmma92 = __WMMA_8_16_16_signed_char_int(alu454, cast305, cast0);
    int4 wmma93 = __WMMA_8_16_16_signed_char_int(alu455, cast306, cast0);
    int4 wmma94 = __WMMA_8_16_16_signed_char_int(alu456, cast307, cast0);
    int4 wmma95 = __WMMA_8_16_16_signed_char_int(alu457, cast308, cast0);
    float cast335 = ((float)(((signed char)(((val484>>0u)&255u)))));
    float cast336 = ((float)(((signed char)(((val484>>8u)&255u)))));
    float cast337 = ((float)(((signed char)(((val484>>16u)&255u)))));
    float cast338 = ((float)(((signed char)(((val484>>24u)&255u)))));
    float cast339 = ((float)(((signed char)(((val485>>0u)&255u)))));
    float cast340 = ((float)(((signed char)(((val485>>8u)&255u)))));
    float cast341 = ((float)(((signed char)(((val485>>16u)&255u)))));
    float cast342 = ((float)(((signed char)(((val485>>24u)&255u)))));
    float cast343 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val483&65535u)))))));
    float cast344 = tg_bitcast<float>((unsigned int)(val475));
    float cast345 = tg_bitcast<float>((unsigned int)(val476));
    float cast346 = tg_bitcast<float>((unsigned int)(val477));
    float cast347 = tg_bitcast<float>((unsigned int)(val478));
    float alu698 = ((cast343*cast344*((cast335*((float)(wmma88.x)))+(cast336*((float)(wmma89.x)))))+(cast343*cast345*((cast337*((float)(wmma90.x)))+(cast338*((float)(wmma91.x)))))+(cast343*cast346*((cast339*((float)(wmma92.x)))+(cast340*((float)(wmma93.x)))))+(cast343*cast347*((cast341*((float)(wmma94.x)))+(cast342*((float)(wmma95.x))))));
    float alu699 = (alu414?alu698:(buf45+alu698));
    buf45 = alu699;
    if (alu414) {
      *(data0_5570560+(alu78+2129)) = buf46;
    }
    float cast348 = tg_bitcast<float>((unsigned int)(val479));
    float cast349 = tg_bitcast<float>((unsigned int)(val480));
    float cast350 = tg_bitcast<float>((unsigned int)(val481));
    float cast351 = tg_bitcast<float>((unsigned int)(val482));
    float alu704 = ((cast343*cast348*((cast335*((float)(wmma88.y)))+(cast336*((float)(wmma89.y)))))+(cast343*cast349*((cast337*((float)(wmma90.y)))+(cast338*((float)(wmma91.y)))))+(cast343*cast350*((cast339*((float)(wmma92.y)))+(cast340*((float)(wmma93.y)))))+(cast343*cast351*((cast341*((float)(wmma94.y)))+(cast342*((float)(wmma95.y))))));
    float alu705 = (alu414?alu704:(buf46+alu704));
    buf46 = alu705;
    if (alu414) {
      *(data0_5570560+(alu78+3152)) = buf47;
    }
    float cast352 = ((float)(((signed char)(((val487>>0u)&255u)))));
    float cast353 = ((float)(((signed char)(((val487>>8u)&255u)))));
    float cast354 = ((float)(((signed char)(((val487>>16u)&255u)))));
    float cast355 = ((float)(((signed char)(((val487>>24u)&255u)))));
    float cast356 = ((float)(((signed char)(((val488>>0u)&255u)))));
    float cast357 = ((float)(((signed char)(((val488>>8u)&255u)))));
    float cast358 = ((float)(((signed char)(((val488>>16u)&255u)))));
    float cast359 = ((float)(((signed char)(((val488>>24u)&255u)))));
    float cast360 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val486&65535u)))))));
    float alu710 = ((cast360*cast344*((cast352*((float)(wmma88.z)))+(cast353*((float)(wmma89.z)))))+(cast360*cast345*((cast354*((float)(wmma90.z)))+(cast355*((float)(wmma91.z)))))+(cast360*cast346*((cast356*((float)(wmma92.z)))+(cast357*((float)(wmma93.z)))))+(cast360*cast347*((cast358*((float)(wmma94.z)))+(cast359*((float)(wmma95.z))))));
    float alu711 = (alu414?alu710:(buf47+alu710));
    buf47 = alu711;
    if (alu414) {
      *(data0_5570560+(alu78+3153)) = buf48;
    }
    float alu716 = ((cast360*cast348*((cast352*((float)(wmma88.w)))+(cast353*((float)(wmma89.w)))))+(cast360*cast349*((cast354*((float)(wmma90.w)))+(cast355*((float)(wmma91.w)))))+(cast360*cast350*((cast356*((float)(wmma92.w)))+(cast357*((float)(wmma93.w)))))+(cast360*cast351*((cast358*((float)(wmma94.w)))+(cast359*((float)(wmma95.w))))));
    float alu717 = (alu414?alu716:(buf48+alu716));
    buf48 = alu717;
    unsigned int val489 = (*(buf0+alu131));
    unsigned int val490 = (*(buf0+alu132));
    unsigned int val491 = (*(buf0+alu133));
    unsigned int val492 = (*(buf0+alu134));
    unsigned int val493 = (*(buf0+alu135));
    unsigned int val494 = (*(buf0+alu136));
    unsigned int val495 = (*(buf0+alu137));
    unsigned int val496 = (*(buf0+alu138));
    unsigned int val497 = (*(buf0+alu217));
    unsigned int val498 = (*(buf0+alu218));
    unsigned int val499 = (*(buf0+alu219));
    unsigned int val500 = (*(buf0+alu220));
    unsigned int val501 = (*(buf0+alu221));
    unsigned int val502 = (*(buf0+alu222));
    unsigned int val503 = (*(buf0+alu223));
    unsigned int val504 = (*(buf0+alu224));
    unsigned int val505 = (*(buf0+alu148));
    unsigned int val506 = (*(buf0+alu149));
    unsigned int val507 = (*(buf0+alu150));
    unsigned int val508 = (*(buf0+alu153));
    unsigned int val509 = (*(buf0+alu154));
    unsigned int val510 = (*(buf0+alu155));
    if (alu414) {
      *(data0_5570560+(alu78+96)) = buf49;
    }
    char4 cast361 = make_char4(((signed char)(((val489>>0u)&255u))),((signed char)(((val489>>8u)&255u))),((signed char)(((val489>>16u)&255u))),((signed char)(((val489>>24u)&255u))));
    char4 cast362 = make_char4(((signed char)(((val490>>0u)&255u))),((signed char)(((val490>>8u)&255u))),((signed char)(((val490>>16u)&255u))),((signed char)(((val490>>24u)&255u))));
    char4 cast363 = make_char4(((signed char)(((val491>>0u)&255u))),((signed char)(((val491>>8u)&255u))),((signed char)(((val491>>16u)&255u))),((signed char)(((val491>>24u)&255u))));
    char4 cast364 = make_char4(((signed char)(((val492>>0u)&255u))),((signed char)(((val492>>8u)&255u))),((signed char)(((val492>>16u)&255u))),((signed char)(((val492>>24u)&255u))));
    char4 cast365 = make_char4(((signed char)(((val493>>0u)&255u))),((signed char)(((val493>>8u)&255u))),((signed char)(((val493>>16u)&255u))),((signed char)(((val493>>24u)&255u))));
    char4 cast366 = make_char4(((signed char)(((val494>>0u)&255u))),((signed char)(((val494>>8u)&255u))),((signed char)(((val494>>16u)&255u))),((signed char)(((val494>>24u)&255u))));
    char4 cast367 = make_char4(((signed char)(((val495>>0u)&255u))),((signed char)(((val495>>8u)&255u))),((signed char)(((val495>>16u)&255u))),((signed char)(((val495>>24u)&255u))));
    char4 cast368 = make_char4(((signed char)(((val496>>0u)&255u))),((signed char)(((val496>>8u)&255u))),((signed char)(((val496>>16u)&255u))),((signed char)(((val496>>24u)&255u))));
    int4 wmma96 = __WMMA_8_16_16_signed_char_int(alu418, cast362, cast0);
    int4 wmma97 = __WMMA_8_16_16_signed_char_int(alu419, cast363, cast0);
    int4 wmma98 = __WMMA_8_16_16_signed_char_int(alu420, cast364, cast0);
    int4 wmma99 = __WMMA_8_16_16_signed_char_int(alu421, cast365, cast0);
    int4 wmma100 = __WMMA_8_16_16_signed_char_int(alu422, cast366, cast0);
    int4 wmma101 = __WMMA_8_16_16_signed_char_int(alu423, cast367, cast0);
    int4 wmma102 = __WMMA_8_16_16_signed_char_int(alu424, cast368, cast0);
    int4 wmma103 = __WMMA_8_16_16_signed_char_int(alu425, cast361, cast0);
    float cast369 = ((float)(((signed char)(((val506>>0u)&255u)))));
    float cast370 = ((float)(((signed char)(((val506>>8u)&255u)))));
    float cast371 = ((float)(((signed char)(((val506>>16u)&255u)))));
    float cast372 = ((float)(((signed char)(((val506>>24u)&255u)))));
    float cast373 = ((float)(((signed char)(((val507>>0u)&255u)))));
    float cast374 = ((float)(((signed char)(((val507>>8u)&255u)))));
    float cast375 = ((float)(((signed char)(((val507>>16u)&255u)))));
    float cast376 = ((float)(((signed char)(((val507>>24u)&255u)))));
    float cast377 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val505&65535u)))))));
    float cast378 = tg_bitcast<float>((unsigned int)(val497));
    float cast379 = tg_bitcast<float>((unsigned int)(val498));
    float cast380 = tg_bitcast<float>((unsigned int)(val499));
    float cast381 = tg_bitcast<float>((unsigned int)(val500));
    float alu722 = ((cast377*cast378*((cast369*((float)(wmma103.x)))+(cast370*((float)(wmma96.x)))))+(cast377*cast379*((cast371*((float)(wmma97.x)))+(cast372*((float)(wmma98.x)))))+(cast377*cast380*((cast373*((float)(wmma99.x)))+(cast374*((float)(wmma100.x)))))+(cast377*cast381*((cast375*((float)(wmma101.x)))+(cast376*((float)(wmma102.x))))));
    float alu723 = (alu414?alu722:(buf49+alu722));
    buf49 = alu723;
    if (alu414) {
      *(data0_5570560+(alu78+97)) = buf50;
    }
    float cast382 = tg_bitcast<float>((unsigned int)(val501));
    float cast383 = tg_bitcast<float>((unsigned int)(val502));
    float cast384 = tg_bitcast<float>((unsigned int)(val503));
    float cast385 = tg_bitcast<float>((unsigned int)(val504));
    float alu728 = ((cast377*cast382*((cast369*((float)(wmma103.y)))+(cast370*((float)(wmma96.y)))))+(cast377*cast383*((cast371*((float)(wmma97.y)))+(cast372*((float)(wmma98.y)))))+(cast377*cast384*((cast373*((float)(wmma99.y)))+(cast374*((float)(wmma100.y)))))+(cast377*cast385*((cast375*((float)(wmma101.y)))+(cast376*((float)(wmma102.y))))));
    float alu729 = (alu414?alu728:(buf50+alu728));
    buf50 = alu729;
    if (alu414) {
      *(data0_5570560+(alu78+1120)) = buf51;
    }
    float cast386 = ((float)(((signed char)(((val509>>0u)&255u)))));
    float cast387 = ((float)(((signed char)(((val509>>8u)&255u)))));
    float cast388 = ((float)(((signed char)(((val509>>16u)&255u)))));
    float cast389 = ((float)(((signed char)(((val509>>24u)&255u)))));
    float cast390 = ((float)(((signed char)(((val510>>0u)&255u)))));
    float cast391 = ((float)(((signed char)(((val510>>8u)&255u)))));
    float cast392 = ((float)(((signed char)(((val510>>16u)&255u)))));
    float cast393 = ((float)(((signed char)(((val510>>24u)&255u)))));
    float cast394 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val508&65535u)))))));
    float alu734 = ((cast394*cast378*((cast386*((float)(wmma103.z)))+(cast387*((float)(wmma96.z)))))+(cast394*cast379*((cast388*((float)(wmma97.z)))+(cast389*((float)(wmma98.z)))))+(cast394*cast380*((cast390*((float)(wmma99.z)))+(cast391*((float)(wmma100.z)))))+(cast394*cast381*((cast392*((float)(wmma101.z)))+(cast393*((float)(wmma102.z))))));
    float alu735 = (alu414?alu734:(buf51+alu734));
    buf51 = alu735;
    if (alu414) {
      *(data0_5570560+(alu78+1121)) = buf52;
    }
    float alu740 = ((cast394*cast382*((cast386*((float)(wmma103.w)))+(cast387*((float)(wmma96.w)))))+(cast394*cast383*((cast388*((float)(wmma97.w)))+(cast389*((float)(wmma98.w)))))+(cast394*cast384*((cast390*((float)(wmma99.w)))+(cast391*((float)(wmma100.w)))))+(cast394*cast385*((cast392*((float)(wmma101.w)))+(cast393*((float)(wmma102.w))))));
    float alu741 = (alu414?alu740:(buf52+alu740));
    buf52 = alu741;
    unsigned int val511 = (*(buf0+alu217));
    unsigned int val512 = (*(buf0+alu218));
    unsigned int val513 = (*(buf0+alu219));
    unsigned int val514 = (*(buf0+alu220));
    unsigned int val515 = (*(buf0+alu221));
    unsigned int val516 = (*(buf0+alu222));
    unsigned int val517 = (*(buf0+alu223));
    unsigned int val518 = (*(buf0+alu224));
    unsigned int val519 = (*(buf0+alu158));
    unsigned int val520 = (*(buf0+alu159));
    unsigned int val521 = (*(buf0+alu160));
    unsigned int val522 = (*(buf0+alu163));
    unsigned int val523 = (*(buf0+alu164));
    unsigned int val524 = (*(buf0+alu165));
    if (alu414) {
      *(data0_5570560+(alu78+2144)) = buf53;
    }
    int4 wmma104 = __WMMA_8_16_16_signed_char_int(alu450, cast361, cast0);
    int4 wmma105 = __WMMA_8_16_16_signed_char_int(alu451, cast362, cast0);
    int4 wmma106 = __WMMA_8_16_16_signed_char_int(alu452, cast363, cast0);
    int4 wmma107 = __WMMA_8_16_16_signed_char_int(alu453, cast364, cast0);
    int4 wmma108 = __WMMA_8_16_16_signed_char_int(alu454, cast365, cast0);
    int4 wmma109 = __WMMA_8_16_16_signed_char_int(alu455, cast366, cast0);
    int4 wmma110 = __WMMA_8_16_16_signed_char_int(alu456, cast367, cast0);
    int4 wmma111 = __WMMA_8_16_16_signed_char_int(alu457, cast368, cast0);
    float cast395 = ((float)(((signed char)(((val520>>0u)&255u)))));
    float cast396 = ((float)(((signed char)(((val520>>8u)&255u)))));
    float cast397 = ((float)(((signed char)(((val520>>16u)&255u)))));
    float cast398 = ((float)(((signed char)(((val520>>24u)&255u)))));
    float cast399 = ((float)(((signed char)(((val521>>0u)&255u)))));
    float cast400 = ((float)(((signed char)(((val521>>8u)&255u)))));
    float cast401 = ((float)(((signed char)(((val521>>16u)&255u)))));
    float cast402 = ((float)(((signed char)(((val521>>24u)&255u)))));
    float cast403 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val519&65535u)))))));
    float cast404 = tg_bitcast<float>((unsigned int)(val511));
    float cast405 = tg_bitcast<float>((unsigned int)(val512));
    float cast406 = tg_bitcast<float>((unsigned int)(val513));
    float cast407 = tg_bitcast<float>((unsigned int)(val514));
    float alu746 = ((cast403*cast404*((cast395*((float)(wmma104.x)))+(cast396*((float)(wmma105.x)))))+(cast403*cast405*((cast397*((float)(wmma106.x)))+(cast398*((float)(wmma107.x)))))+(cast403*cast406*((cast399*((float)(wmma108.x)))+(cast400*((float)(wmma109.x)))))+(cast403*cast407*((cast401*((float)(wmma110.x)))+(cast402*((float)(wmma111.x))))));
    float alu747 = (alu414?alu746:(buf53+alu746));
    buf53 = alu747;
    if (alu414) {
      *(data0_5570560+(alu78+2145)) = buf54;
    }
    float cast408 = tg_bitcast<float>((unsigned int)(val515));
    float cast409 = tg_bitcast<float>((unsigned int)(val516));
    float cast410 = tg_bitcast<float>((unsigned int)(val517));
    float cast411 = tg_bitcast<float>((unsigned int)(val518));
    float alu752 = ((cast403*cast408*((cast395*((float)(wmma104.y)))+(cast396*((float)(wmma105.y)))))+(cast403*cast409*((cast397*((float)(wmma106.y)))+(cast398*((float)(wmma107.y)))))+(cast403*cast410*((cast399*((float)(wmma108.y)))+(cast400*((float)(wmma109.y)))))+(cast403*cast411*((cast401*((float)(wmma110.y)))+(cast402*((float)(wmma111.y))))));
    float alu753 = (alu414?alu752:(buf54+alu752));
    buf54 = alu753;
    if (alu414) {
      *(data0_5570560+(alu78+3168)) = buf55;
    }
    float cast412 = ((float)(((signed char)(((val523>>0u)&255u)))));
    float cast413 = ((float)(((signed char)(((val523>>8u)&255u)))));
    float cast414 = ((float)(((signed char)(((val523>>16u)&255u)))));
    float cast415 = ((float)(((signed char)(((val523>>24u)&255u)))));
    float cast416 = ((float)(((signed char)(((val524>>0u)&255u)))));
    float cast417 = ((float)(((signed char)(((val524>>8u)&255u)))));
    float cast418 = ((float)(((signed char)(((val524>>16u)&255u)))));
    float cast419 = ((float)(((signed char)(((val524>>24u)&255u)))));
    float cast420 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val522&65535u)))))));
    float alu758 = ((cast420*cast404*((cast412*((float)(wmma104.z)))+(cast413*((float)(wmma105.z)))))+(cast420*cast405*((cast414*((float)(wmma106.z)))+(cast415*((float)(wmma107.z)))))+(cast420*cast406*((cast416*((float)(wmma108.z)))+(cast417*((float)(wmma109.z)))))+(cast420*cast407*((cast418*((float)(wmma110.z)))+(cast419*((float)(wmma111.z))))));
    float alu759 = (alu414?alu758:(buf55+alu758));
    buf55 = alu759;
    if (alu414) {
      *(data0_5570560+(alu78+3169)) = buf56;
    }
    float alu764 = ((cast420*cast408*((cast412*((float)(wmma104.w)))+(cast413*((float)(wmma105.w)))))+(cast420*cast409*((cast414*((float)(wmma106.w)))+(cast415*((float)(wmma107.w)))))+(cast420*cast410*((cast416*((float)(wmma108.w)))+(cast417*((float)(wmma109.w)))))+(cast420*cast411*((cast418*((float)(wmma110.w)))+(cast419*((float)(wmma111.w))))));
    float alu765 = (alu414?alu764:(buf56+alu764));
    buf56 = alu765;
    unsigned int val525 = (*(buf0+alu139));
    unsigned int val526 = (*(buf0+alu140));
    unsigned int val527 = (*(buf0+alu141));
    unsigned int val528 = (*(buf0+alu142));
    unsigned int val529 = (*(buf0+alu143));
    unsigned int val530 = (*(buf0+alu144));
    unsigned int val531 = (*(buf0+alu145));
    unsigned int val532 = (*(buf0+alu146));
    unsigned int val533 = (*(buf0+alu225));
    unsigned int val534 = (*(buf0+alu226));
    unsigned int val535 = (*(buf0+alu227));
    unsigned int val536 = (*(buf0+alu228));
    unsigned int val537 = (*(buf0+alu229));
    unsigned int val538 = (*(buf0+alu230));
    unsigned int val539 = (*(buf0+alu231));
    unsigned int val540 = (*(buf0+alu232));
    unsigned int val541 = (*(buf0+alu148));
    unsigned int val542 = (*(buf0+alu149));
    unsigned int val543 = (*(buf0+alu150));
    unsigned int val544 = (*(buf0+alu153));
    unsigned int val545 = (*(buf0+alu154));
    unsigned int val546 = (*(buf0+alu155));
    if (alu414) {
      *(data0_5570560+(alu78+112)) = buf57;
    }
    char4 cast421 = make_char4(((signed char)(((val525>>0u)&255u))),((signed char)(((val525>>8u)&255u))),((signed char)(((val525>>16u)&255u))),((signed char)(((val525>>24u)&255u))));
    char4 cast422 = make_char4(((signed char)(((val526>>0u)&255u))),((signed char)(((val526>>8u)&255u))),((signed char)(((val526>>16u)&255u))),((signed char)(((val526>>24u)&255u))));
    char4 cast423 = make_char4(((signed char)(((val527>>0u)&255u))),((signed char)(((val527>>8u)&255u))),((signed char)(((val527>>16u)&255u))),((signed char)(((val527>>24u)&255u))));
    char4 cast424 = make_char4(((signed char)(((val528>>0u)&255u))),((signed char)(((val528>>8u)&255u))),((signed char)(((val528>>16u)&255u))),((signed char)(((val528>>24u)&255u))));
    char4 cast425 = make_char4(((signed char)(((val529>>0u)&255u))),((signed char)(((val529>>8u)&255u))),((signed char)(((val529>>16u)&255u))),((signed char)(((val529>>24u)&255u))));
    char4 cast426 = make_char4(((signed char)(((val530>>0u)&255u))),((signed char)(((val530>>8u)&255u))),((signed char)(((val530>>16u)&255u))),((signed char)(((val530>>24u)&255u))));
    char4 cast427 = make_char4(((signed char)(((val531>>0u)&255u))),((signed char)(((val531>>8u)&255u))),((signed char)(((val531>>16u)&255u))),((signed char)(((val531>>24u)&255u))));
    char4 cast428 = make_char4(((signed char)(((val532>>0u)&255u))),((signed char)(((val532>>8u)&255u))),((signed char)(((val532>>16u)&255u))),((signed char)(((val532>>24u)&255u))));
    int4 wmma112 = __WMMA_8_16_16_signed_char_int(alu418, cast422, cast0);
    int4 wmma113 = __WMMA_8_16_16_signed_char_int(alu419, cast423, cast0);
    int4 wmma114 = __WMMA_8_16_16_signed_char_int(alu420, cast424, cast0);
    int4 wmma115 = __WMMA_8_16_16_signed_char_int(alu421, cast425, cast0);
    int4 wmma116 = __WMMA_8_16_16_signed_char_int(alu422, cast426, cast0);
    int4 wmma117 = __WMMA_8_16_16_signed_char_int(alu423, cast427, cast0);
    int4 wmma118 = __WMMA_8_16_16_signed_char_int(alu424, cast428, cast0);
    int4 wmma119 = __WMMA_8_16_16_signed_char_int(alu425, cast421, cast0);
    float cast429 = ((float)(((signed char)(((val542>>0u)&255u)))));
    float cast430 = ((float)(((signed char)(((val542>>8u)&255u)))));
    float cast431 = ((float)(((signed char)(((val542>>16u)&255u)))));
    float cast432 = ((float)(((signed char)(((val542>>24u)&255u)))));
    float cast433 = ((float)(((signed char)(((val543>>0u)&255u)))));
    float cast434 = ((float)(((signed char)(((val543>>8u)&255u)))));
    float cast435 = ((float)(((signed char)(((val543>>16u)&255u)))));
    float cast436 = ((float)(((signed char)(((val543>>24u)&255u)))));
    float cast437 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val541&65535u)))))));
    float cast438 = tg_bitcast<float>((unsigned int)(val533));
    float cast439 = tg_bitcast<float>((unsigned int)(val534));
    float cast440 = tg_bitcast<float>((unsigned int)(val535));
    float cast441 = tg_bitcast<float>((unsigned int)(val536));
    float alu770 = ((cast437*cast438*((cast429*((float)(wmma119.x)))+(cast430*((float)(wmma112.x)))))+(cast437*cast439*((cast431*((float)(wmma113.x)))+(cast432*((float)(wmma114.x)))))+(cast437*cast440*((cast433*((float)(wmma115.x)))+(cast434*((float)(wmma116.x)))))+(cast437*cast441*((cast435*((float)(wmma117.x)))+(cast436*((float)(wmma118.x))))));
    float alu771 = (alu414?alu770:(buf57+alu770));
    buf57 = alu771;
    if (alu414) {
      *(data0_5570560+(alu78+113)) = buf58;
    }
    float cast442 = tg_bitcast<float>((unsigned int)(val537));
    float cast443 = tg_bitcast<float>((unsigned int)(val538));
    float cast444 = tg_bitcast<float>((unsigned int)(val539));
    float cast445 = tg_bitcast<float>((unsigned int)(val540));
    float alu776 = ((cast437*cast442*((cast429*((float)(wmma119.y)))+(cast430*((float)(wmma112.y)))))+(cast437*cast443*((cast431*((float)(wmma113.y)))+(cast432*((float)(wmma114.y)))))+(cast437*cast444*((cast433*((float)(wmma115.y)))+(cast434*((float)(wmma116.y)))))+(cast437*cast445*((cast435*((float)(wmma117.y)))+(cast436*((float)(wmma118.y))))));
    float alu777 = (alu414?alu776:(buf58+alu776));
    buf58 = alu777;
    if (alu414) {
      *(data0_5570560+(alu78+1136)) = buf59;
    }
    float cast446 = ((float)(((signed char)(((val545>>0u)&255u)))));
    float cast447 = ((float)(((signed char)(((val545>>8u)&255u)))));
    float cast448 = ((float)(((signed char)(((val545>>16u)&255u)))));
    float cast449 = ((float)(((signed char)(((val545>>24u)&255u)))));
    float cast450 = ((float)(((signed char)(((val546>>0u)&255u)))));
    float cast451 = ((float)(((signed char)(((val546>>8u)&255u)))));
    float cast452 = ((float)(((signed char)(((val546>>16u)&255u)))));
    float cast453 = ((float)(((signed char)(((val546>>24u)&255u)))));
    float cast454 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val544&65535u)))))));
    float alu782 = ((cast454*cast438*((cast446*((float)(wmma119.z)))+(cast447*((float)(wmma112.z)))))+(cast454*cast439*((cast448*((float)(wmma113.z)))+(cast449*((float)(wmma114.z)))))+(cast454*cast440*((cast450*((float)(wmma115.z)))+(cast451*((float)(wmma116.z)))))+(cast454*cast441*((cast452*((float)(wmma117.z)))+(cast453*((float)(wmma118.z))))));
    float alu783 = (alu414?alu782:(buf59+alu782));
    buf59 = alu783;
    if (alu414) {
      *(data0_5570560+(alu78+1137)) = buf60;
    }
    float alu788 = ((cast454*cast442*((cast446*((float)(wmma119.w)))+(cast447*((float)(wmma112.w)))))+(cast454*cast443*((cast448*((float)(wmma113.w)))+(cast449*((float)(wmma114.w)))))+(cast454*cast444*((cast450*((float)(wmma115.w)))+(cast451*((float)(wmma116.w)))))+(cast454*cast445*((cast452*((float)(wmma117.w)))+(cast453*((float)(wmma118.w))))));
    float alu789 = (alu414?alu788:(buf60+alu788));
    buf60 = alu789;
    unsigned int val547 = (*(buf0+alu225));
    unsigned int val548 = (*(buf0+alu226));
    unsigned int val549 = (*(buf0+alu227));
    unsigned int val550 = (*(buf0+alu228));
    unsigned int val551 = (*(buf0+alu229));
    unsigned int val552 = (*(buf0+alu230));
    unsigned int val553 = (*(buf0+alu231));
    unsigned int val554 = (*(buf0+alu232));
    unsigned int val555 = (*(buf0+alu158));
    unsigned int val556 = (*(buf0+alu159));
    unsigned int val557 = (*(buf0+alu160));
    unsigned int val558 = (*(buf0+alu163));
    unsigned int val559 = (*(buf0+alu164));
    unsigned int val560 = (*(buf0+alu165));
    float val561 = (*(data4_196608+(((alu302+4)<<9)+alu301+alu265)));
    float val562 = (*(data4_196608+(((alu302+5)<<9)+alu301+alu265)));
    float val563 = (*(data4_196608+(((alu302+6)<<9)+alu301+alu265)));
    float val564 = (*(data4_196608+(((alu302+7)<<9)+alu301+alu265)));
    if (alu414) {
      *(data0_5570560+(alu78+2160)) = buf61;
    }
    int4 wmma120 = __WMMA_8_16_16_signed_char_int(alu450, cast421, cast0);
    int4 wmma121 = __WMMA_8_16_16_signed_char_int(alu451, cast422, cast0);
    int4 wmma122 = __WMMA_8_16_16_signed_char_int(alu452, cast423, cast0);
    int4 wmma123 = __WMMA_8_16_16_signed_char_int(alu453, cast424, cast0);
    int4 wmma124 = __WMMA_8_16_16_signed_char_int(alu454, cast425, cast0);
    int4 wmma125 = __WMMA_8_16_16_signed_char_int(alu455, cast426, cast0);
    int4 wmma126 = __WMMA_8_16_16_signed_char_int(alu456, cast427, cast0);
    int4 wmma127 = __WMMA_8_16_16_signed_char_int(alu457, cast428, cast0);
    float cast455 = ((float)(((signed char)(((val556>>0u)&255u)))));
    float cast456 = ((float)(((signed char)(((val556>>8u)&255u)))));
    float cast457 = ((float)(((signed char)(((val556>>16u)&255u)))));
    float cast458 = ((float)(((signed char)(((val556>>24u)&255u)))));
    float cast459 = ((float)(((signed char)(((val557>>0u)&255u)))));
    float cast460 = ((float)(((signed char)(((val557>>8u)&255u)))));
    float cast461 = ((float)(((signed char)(((val557>>16u)&255u)))));
    float cast462 = ((float)(((signed char)(((val557>>24u)&255u)))));
    float cast463 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val555&65535u)))))));
    float cast464 = tg_bitcast<float>((unsigned int)(val547));
    float cast465 = tg_bitcast<float>((unsigned int)(val548));
    float cast466 = tg_bitcast<float>((unsigned int)(val549));
    float cast467 = tg_bitcast<float>((unsigned int)(val550));
    float alu794 = ((cast463*cast464*((cast455*((float)(wmma120.x)))+(cast456*((float)(wmma121.x)))))+(cast463*cast465*((cast457*((float)(wmma122.x)))+(cast458*((float)(wmma123.x)))))+(cast463*cast466*((cast459*((float)(wmma124.x)))+(cast460*((float)(wmma125.x)))))+(cast463*cast467*((cast461*((float)(wmma126.x)))+(cast462*((float)(wmma127.x))))));
    float alu795 = (alu414?alu794:(buf61+alu794));
    buf61 = alu795;
    if (alu414) {
      *(data0_5570560+(alu78+2161)) = buf62;
    }
    float cast468 = tg_bitcast<float>((unsigned int)(val551));
    float cast469 = tg_bitcast<float>((unsigned int)(val552));
    float cast470 = tg_bitcast<float>((unsigned int)(val553));
    float cast471 = tg_bitcast<float>((unsigned int)(val554));
    float alu800 = ((cast463*cast468*((cast455*((float)(wmma120.y)))+(cast456*((float)(wmma121.y)))))+(cast463*cast469*((cast457*((float)(wmma122.y)))+(cast458*((float)(wmma123.y)))))+(cast463*cast470*((cast459*((float)(wmma124.y)))+(cast460*((float)(wmma125.y)))))+(cast463*cast471*((cast461*((float)(wmma126.y)))+(cast462*((float)(wmma127.y))))));
    float alu801 = (alu414?alu800:(buf62+alu800));
    buf62 = alu801;
    if (alu414) {
      *(data0_5570560+(alu78+3184)) = buf63;
    }
    float cast472 = ((float)(((signed char)(((val559>>0u)&255u)))));
    float cast473 = ((float)(((signed char)(((val559>>8u)&255u)))));
    float cast474 = ((float)(((signed char)(((val559>>16u)&255u)))));
    float cast475 = ((float)(((signed char)(((val559>>24u)&255u)))));
    float cast476 = ((float)(((signed char)(((val560>>0u)&255u)))));
    float cast477 = ((float)(((signed char)(((val560>>8u)&255u)))));
    float cast478 = ((float)(((signed char)(((val560>>16u)&255u)))));
    float cast479 = ((float)(((signed char)(((val560>>24u)&255u)))));
    float cast480 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val558&65535u)))))));
    float alu806 = ((cast480*cast464*((cast472*((float)(wmma120.z)))+(cast473*((float)(wmma121.z)))))+(cast480*cast465*((cast474*((float)(wmma122.z)))+(cast475*((float)(wmma123.z)))))+(cast480*cast466*((cast476*((float)(wmma124.z)))+(cast477*((float)(wmma125.z)))))+(cast480*cast467*((cast478*((float)(wmma126.z)))+(cast479*((float)(wmma127.z))))));
    float alu807 = (alu414?alu806:(buf63+alu806));
    buf63 = alu807;
    if (alu414) {
      *(data0_5570560+(alu78+3185)) = buf64;
    }
    float alu812 = ((cast480*cast468*((cast472*((float)(wmma120.w)))+(cast473*((float)(wmma121.w)))))+(cast480*cast469*((cast474*((float)(wmma122.w)))+(cast475*((float)(wmma123.w)))))+(cast480*cast470*((cast476*((float)(wmma124.w)))+(cast477*((float)(wmma125.w)))))+(cast480*cast471*((cast478*((float)(wmma126.w)))+(cast479*((float)(wmma127.w))))));
    float alu813 = (alu414?alu812:(buf64+alu812));
    buf64 = alu813;
    __syncthreads();
    if (alu271) {
      *(buf0+alu68) = ((((unsigned int)(val140))&255u)|((((unsigned int)(val141))&255u)<<8u)|((((unsigned int)(val142))&255u)<<16u)|((((unsigned int)(val143))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu69) = ((((unsigned int)(val144))&255u)|((((unsigned int)(val145))&255u)<<8u)|((((unsigned int)(val146))&255u)<<16u)|((((unsigned int)(val147))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu70) = ((((unsigned int)(val148))&255u)|((((unsigned int)(val149))&255u)<<8u)|((((unsigned int)(val150))&255u)<<16u)|((((unsigned int)(val151))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu71) = ((((unsigned int)(val152))&255u)|((((unsigned int)(val153))&255u)<<8u)|((((unsigned int)(val154))&255u)<<16u)|((((unsigned int)(val155))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu237) = ((((unsigned int)(val156))&255u)|((((unsigned int)(val157))&255u)<<8u)|((((unsigned int)(val158))&255u)<<16u)|((((unsigned int)(val159))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu238) = ((((unsigned int)(val160))&255u)|((((unsigned int)(val161))&255u)<<8u)|((((unsigned int)(val162))&255u)<<16u)|((((unsigned int)(val163))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu239) = ((((unsigned int)(val164))&255u)|((((unsigned int)(val165))&255u)<<8u)|((((unsigned int)(val166))&255u)<<16u)|((((unsigned int)(val167))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu240) = ((((unsigned int)(val168))&255u)|((((unsigned int)(val169))&255u)<<8u)|((((unsigned int)(val170))&255u)<<16u)|((((unsigned int)(val171))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu241) = ((((unsigned int)(val172))&255u)|((((unsigned int)(val173))&255u)<<8u)|((((unsigned int)(val174))&255u)<<16u)|((((unsigned int)(val175))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu242) = ((((unsigned int)(val176))&255u)|((((unsigned int)(val177))&255u)<<8u)|((((unsigned int)(val178))&255u)<<16u)|((((unsigned int)(val179))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu243) = ((((unsigned int)(val180))&255u)|((((unsigned int)(val181))&255u)<<8u)|((((unsigned int)(val182))&255u)<<16u)|((((unsigned int)(val183))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu244) = ((((unsigned int)(val184))&255u)|((((unsigned int)(val185))&255u)<<8u)|((((unsigned int)(val186))&255u)<<16u)|((((unsigned int)(val187))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu245) = ((((unsigned int)(val188))&255u)|((((unsigned int)(val189))&255u)<<8u)|((((unsigned int)(val190))&255u)<<16u)|((((unsigned int)(val191))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu246) = ((((unsigned int)(val192))&255u)|((((unsigned int)(val193))&255u)<<8u)|((((unsigned int)(val194))&255u)<<16u)|((((unsigned int)(val195))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu247) = ((((unsigned int)(val196))&255u)|((((unsigned int)(val197))&255u)<<8u)|((((unsigned int)(val198))&255u)<<16u)|((((unsigned int)(val199))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu248) = ((((unsigned int)(val200))&255u)|((((unsigned int)(val201))&255u)<<8u)|((((unsigned int)(val202))&255u)<<16u)|((((unsigned int)(val203))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu249) = ((((unsigned int)(val204))&255u)|((((unsigned int)(val205))&255u)<<8u)|((((unsigned int)(val206))&255u)<<16u)|((((unsigned int)(val207))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu250) = ((((unsigned int)(val208))&255u)|((((unsigned int)(val209))&255u)<<8u)|((((unsigned int)(val210))&255u)<<16u)|((((unsigned int)(val211))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu251) = ((((unsigned int)(val212))&255u)|((((unsigned int)(val213))&255u)<<8u)|((((unsigned int)(val214))&255u)<<16u)|((((unsigned int)(val215))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu252) = ((((unsigned int)(val216))&255u)|((((unsigned int)(val217))&255u)<<8u)|((((unsigned int)(val218))&255u)<<16u)|((((unsigned int)(val219))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu253) = ((((unsigned int)(val220))&255u)|((((unsigned int)(val221))&255u)<<8u)|((((unsigned int)(val222))&255u)<<16u)|((((unsigned int)(val223))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu254) = ((((unsigned int)(val224))&255u)|((((unsigned int)(val225))&255u)<<8u)|((((unsigned int)(val226))&255u)<<16u)|((((unsigned int)(val227))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu255) = ((((unsigned int)(val228))&255u)|((((unsigned int)(val229))&255u)<<8u)|((((unsigned int)(val230))&255u)<<16u)|((((unsigned int)(val231))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu256) = ((((unsigned int)(val232))&255u)|((((unsigned int)(val233))&255u)<<8u)|((((unsigned int)(val234))&255u)<<16u)|((((unsigned int)(val235))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu257) = ((((unsigned int)(val236))&255u)|((((unsigned int)(val237))&255u)<<8u)|((((unsigned int)(val238))&255u)<<16u)|((((unsigned int)(val239))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu258) = ((((unsigned int)(val240))&255u)|((((unsigned int)(val241))&255u)<<8u)|((((unsigned int)(val242))&255u)<<16u)|((((unsigned int)(val243))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu259) = ((((unsigned int)(val244))&255u)|((((unsigned int)(val245))&255u)<<8u)|((((unsigned int)(val246))&255u)<<16u)|((((unsigned int)(val247))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu260) = ((((unsigned int)(val248))&255u)|((((unsigned int)(val249))&255u)<<8u)|((((unsigned int)(val250))&255u)<<16u)|((((unsigned int)(val251))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu261) = ((((unsigned int)(val252))&255u)|((((unsigned int)(val253))&255u)<<8u)|((((unsigned int)(val254))&255u)<<16u)|((((unsigned int)(val255))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu262) = ((((unsigned int)(val256))&255u)|((((unsigned int)(val257))&255u)<<8u)|((((unsigned int)(val258))&255u)<<16u)|((((unsigned int)(val259))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu263) = ((((unsigned int)(val260))&255u)|((((unsigned int)(val261))&255u)<<8u)|((((unsigned int)(val262))&255u)<<16u)|((((unsigned int)(val263))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu264) = ((((unsigned int)(val264))&255u)|((((unsigned int)(val265))&255u)<<8u)|((((unsigned int)(val266))&255u)<<16u)|((((unsigned int)(val267))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu233) = tg_bitcast<unsigned int>((float)(val561));
    }
    if (alu271) {
      *(buf0+alu234) = tg_bitcast<unsigned int>((float)(val562));
    }
    if (alu271) {
      *(buf0+alu235) = tg_bitcast<unsigned int>((float)(val563));
    }
    if (alu271) {
      *(buf0+alu236) = tg_bitcast<unsigned int>((float)(val564));
    }
    __syncthreads();
    unsigned int val565 = (*(buf0+alu83));
    unsigned int val566 = (*(buf0+alu84));
    unsigned int val567 = (*(buf0+alu85));
    unsigned int val568 = (*(buf0+alu86));
    unsigned int val569 = (*(buf0+alu87));
    unsigned int val570 = (*(buf0+alu88));
    unsigned int val571 = (*(buf0+alu89));
    unsigned int val572 = (*(buf0+alu90));
    unsigned int val573 = (*(buf0+alu169));
    unsigned int val574 = (*(buf0+alu170));
    unsigned int val575 = (*(buf0+alu171));
    unsigned int val576 = (*(buf0+alu172));
    unsigned int val577 = (*(buf0+alu173));
    unsigned int val578 = (*(buf0+alu174));
    unsigned int val579 = (*(buf0+alu175));
    unsigned int val580 = (*(buf0+alu176));
    unsigned int val581 = (*(buf0+alu148));
    unsigned int val582 = (*(buf0+alu151));
    unsigned int val583 = (*(buf0+alu152));
    unsigned int val584 = (*(buf0+alu153));
    unsigned int val585 = (*(buf0+alu156));
    unsigned int val586 = (*(buf0+alu157));
    if (0) {
      *(data0_5570560+alu78) = buf1;
    }
    char4 cast481 = make_char4(((signed char)(((val565>>0u)&255u))),((signed char)(((val565>>8u)&255u))),((signed char)(((val565>>16u)&255u))),((signed char)(((val565>>24u)&255u))));
    char4 cast482 = make_char4(((signed char)(((val566>>0u)&255u))),((signed char)(((val566>>8u)&255u))),((signed char)(((val566>>16u)&255u))),((signed char)(((val566>>24u)&255u))));
    char4 cast483 = make_char4(((signed char)(((val567>>0u)&255u))),((signed char)(((val567>>8u)&255u))),((signed char)(((val567>>16u)&255u))),((signed char)(((val567>>24u)&255u))));
    char4 cast484 = make_char4(((signed char)(((val568>>0u)&255u))),((signed char)(((val568>>8u)&255u))),((signed char)(((val568>>16u)&255u))),((signed char)(((val568>>24u)&255u))));
    char4 cast485 = make_char4(((signed char)(((val569>>0u)&255u))),((signed char)(((val569>>8u)&255u))),((signed char)(((val569>>16u)&255u))),((signed char)(((val569>>24u)&255u))));
    char4 cast486 = make_char4(((signed char)(((val570>>0u)&255u))),((signed char)(((val570>>8u)&255u))),((signed char)(((val570>>16u)&255u))),((signed char)(((val570>>24u)&255u))));
    char4 cast487 = make_char4(((signed char)(((val571>>0u)&255u))),((signed char)(((val571>>8u)&255u))),((signed char)(((val571>>16u)&255u))),((signed char)(((val571>>24u)&255u))));
    char4 cast488 = make_char4(((signed char)(((val572>>0u)&255u))),((signed char)(((val572>>8u)&255u))),((signed char)(((val572>>16u)&255u))),((signed char)(((val572>>24u)&255u))));
    signed_char8 alu928 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+32))))*4)));
    int4 wmma128 = __WMMA_8_16_16_signed_char_int(alu928, cast481, cast0);
    signed_char8 alu929 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+36))))*4)));
    int4 wmma129 = __WMMA_8_16_16_signed_char_int(alu929, cast482, cast0);
    signed_char8 alu930 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+40))))*4)));
    int4 wmma130 = __WMMA_8_16_16_signed_char_int(alu930, cast483, cast0);
    signed_char8 alu931 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+44))))*4)));
    int4 wmma131 = __WMMA_8_16_16_signed_char_int(alu931, cast484, cast0);
    signed_char8 alu932 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+48))))*4)));
    int4 wmma132 = __WMMA_8_16_16_signed_char_int(alu932, cast485, cast0);
    signed_char8 alu933 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+52))))*4)));
    int4 wmma133 = __WMMA_8_16_16_signed_char_int(alu933, cast486, cast0);
    signed_char8 alu934 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+56))))*4)));
    int4 wmma134 = __WMMA_8_16_16_signed_char_int(alu934, cast487, cast0);
    signed_char8 alu935 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+60))))*4)));
    int4 wmma135 = __WMMA_8_16_16_signed_char_int(alu935, cast488, cast0);
    float cast489 = ((float)(((signed char)(((val582>>0u)&255u)))));
    float cast490 = ((float)(((signed char)(((val582>>8u)&255u)))));
    float cast491 = ((float)(((signed char)(((val582>>16u)&255u)))));
    float cast492 = ((float)(((signed char)(((val582>>24u)&255u)))));
    float cast493 = ((float)(((signed char)(((val583>>0u)&255u)))));
    float cast494 = ((float)(((signed char)(((val583>>8u)&255u)))));
    float cast495 = ((float)(((signed char)(((val583>>16u)&255u)))));
    float cast496 = ((float)(((signed char)(((val583>>24u)&255u)))));
    float cast497 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val581&65535u)))))));
    float cast498 = tg_bitcast<float>((unsigned int)(val573));
    float cast499 = tg_bitcast<float>((unsigned int)(val574));
    float cast500 = tg_bitcast<float>((unsigned int)(val575));
    float cast501 = tg_bitcast<float>((unsigned int)(val576));
    buf1 = (buf1+(cast497*cast498*((cast489*((float)(wmma128.x)))+(cast490*((float)(wmma129.x)))))+(cast497*cast499*((cast491*((float)(wmma130.x)))+(cast492*((float)(wmma131.x)))))+(cast497*cast500*((cast493*((float)(wmma132.x)))+(cast494*((float)(wmma133.x)))))+(cast497*cast501*((cast495*((float)(wmma134.x)))+(cast496*((float)(wmma135.x))))));
    if (0) {
      *(data0_5570560+(alu78+1)) = buf2;
    }
    float cast502 = tg_bitcast<float>((unsigned int)(val577));
    float cast503 = tg_bitcast<float>((unsigned int)(val578));
    float cast504 = tg_bitcast<float>((unsigned int)(val579));
    float cast505 = tg_bitcast<float>((unsigned int)(val580));
    buf2 = (buf2+(cast497*cast502*((cast489*((float)(wmma128.y)))+(cast490*((float)(wmma129.y)))))+(cast497*cast503*((cast491*((float)(wmma130.y)))+(cast492*((float)(wmma131.y)))))+(cast497*cast504*((cast493*((float)(wmma132.y)))+(cast494*((float)(wmma133.y)))))+(cast497*cast505*((cast495*((float)(wmma134.y)))+(cast496*((float)(wmma135.y))))));
    if (0) {
      *(data0_5570560+(alu78+1024)) = buf3;
    }
    float cast506 = ((float)(((signed char)(((val585>>0u)&255u)))));
    float cast507 = ((float)(((signed char)(((val585>>8u)&255u)))));
    float cast508 = ((float)(((signed char)(((val585>>16u)&255u)))));
    float cast509 = ((float)(((signed char)(((val585>>24u)&255u)))));
    float cast510 = ((float)(((signed char)(((val586>>0u)&255u)))));
    float cast511 = ((float)(((signed char)(((val586>>8u)&255u)))));
    float cast512 = ((float)(((signed char)(((val586>>16u)&255u)))));
    float cast513 = ((float)(((signed char)(((val586>>24u)&255u)))));
    float cast514 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val584&65535u)))))));
    buf3 = (buf3+(cast514*cast498*((cast506*((float)(wmma128.z)))+(cast507*((float)(wmma129.z)))))+(cast514*cast499*((cast508*((float)(wmma130.z)))+(cast509*((float)(wmma131.z)))))+(cast514*cast500*((cast510*((float)(wmma132.z)))+(cast511*((float)(wmma133.z)))))+(cast514*cast501*((cast512*((float)(wmma134.z)))+(cast513*((float)(wmma135.z))))));
    if (0) {
      *(data0_5570560+(alu78+1025)) = buf4;
    }
    buf4 = (buf4+(cast514*cast502*((cast506*((float)(wmma128.w)))+(cast507*((float)(wmma129.w)))))+(cast514*cast503*((cast508*((float)(wmma130.w)))+(cast509*((float)(wmma131.w)))))+(cast514*cast504*((cast510*((float)(wmma132.w)))+(cast511*((float)(wmma133.w)))))+(cast514*cast505*((cast512*((float)(wmma134.w)))+(cast513*((float)(wmma135.w))))));
    unsigned int val587 = (*(buf0+alu169));
    unsigned int val588 = (*(buf0+alu170));
    unsigned int val589 = (*(buf0+alu171));
    unsigned int val590 = (*(buf0+alu172));
    unsigned int val591 = (*(buf0+alu173));
    unsigned int val592 = (*(buf0+alu174));
    unsigned int val593 = (*(buf0+alu175));
    unsigned int val594 = (*(buf0+alu176));
    unsigned int val595 = (*(buf0+alu158));
    unsigned int val596 = (*(buf0+alu161));
    unsigned int val597 = (*(buf0+alu162));
    unsigned int val598 = (*(buf0+alu163));
    unsigned int val599 = (*(buf0+alu166));
    unsigned int val600 = (*(buf0+alu167));
    if (0) {
      *(data0_5570560+(alu78+2048)) = buf5;
    }
    signed_char8 alu952 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+1248))))*4)));
    int4 wmma136 = __WMMA_8_16_16_signed_char_int(alu952, cast481, cast0);
    signed_char8 alu953 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+1252))))*4)));
    int4 wmma137 = __WMMA_8_16_16_signed_char_int(alu953, cast482, cast0);
    signed_char8 alu954 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+1256))))*4)));
    int4 wmma138 = __WMMA_8_16_16_signed_char_int(alu954, cast483, cast0);
    signed_char8 alu955 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+1260))))*4)));
    int4 wmma139 = __WMMA_8_16_16_signed_char_int(alu955, cast484, cast0);
    signed_char8 alu956 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+1264))))*4)));
    int4 wmma140 = __WMMA_8_16_16_signed_char_int(alu956, cast485, cast0);
    signed_char8 alu957 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+1268))))*4)));
    int4 wmma141 = __WMMA_8_16_16_signed_char_int(alu957, cast486, cast0);
    signed_char8 alu958 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+1272))))*4)));
    int4 wmma142 = __WMMA_8_16_16_signed_char_int(alu958, cast487, cast0);
    signed_char8 alu959 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu80+1276))))*4)));
    int4 wmma143 = __WMMA_8_16_16_signed_char_int(alu959, cast488, cast0);
    float cast515 = ((float)(((signed char)(((val596>>0u)&255u)))));
    float cast516 = ((float)(((signed char)(((val596>>8u)&255u)))));
    float cast517 = ((float)(((signed char)(((val596>>16u)&255u)))));
    float cast518 = ((float)(((signed char)(((val596>>24u)&255u)))));
    float cast519 = ((float)(((signed char)(((val597>>0u)&255u)))));
    float cast520 = ((float)(((signed char)(((val597>>8u)&255u)))));
    float cast521 = ((float)(((signed char)(((val597>>16u)&255u)))));
    float cast522 = ((float)(((signed char)(((val597>>24u)&255u)))));
    float cast523 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val595&65535u)))))));
    float cast524 = tg_bitcast<float>((unsigned int)(val587));
    float cast525 = tg_bitcast<float>((unsigned int)(val588));
    float cast526 = tg_bitcast<float>((unsigned int)(val589));
    float cast527 = tg_bitcast<float>((unsigned int)(val590));
    buf5 = (buf5+(cast523*cast524*((cast515*((float)(wmma136.x)))+(cast516*((float)(wmma137.x)))))+(cast523*cast525*((cast517*((float)(wmma138.x)))+(cast518*((float)(wmma139.x)))))+(cast523*cast526*((cast519*((float)(wmma140.x)))+(cast520*((float)(wmma141.x)))))+(cast523*cast527*((cast521*((float)(wmma142.x)))+(cast522*((float)(wmma143.x))))));
    if (0) {
      *(data0_5570560+(alu78+2049)) = buf6;
    }
    float cast528 = tg_bitcast<float>((unsigned int)(val591));
    float cast529 = tg_bitcast<float>((unsigned int)(val592));
    float cast530 = tg_bitcast<float>((unsigned int)(val593));
    float cast531 = tg_bitcast<float>((unsigned int)(val594));
    buf6 = (buf6+(cast523*cast528*((cast515*((float)(wmma136.y)))+(cast516*((float)(wmma137.y)))))+(cast523*cast529*((cast517*((float)(wmma138.y)))+(cast518*((float)(wmma139.y)))))+(cast523*cast530*((cast519*((float)(wmma140.y)))+(cast520*((float)(wmma141.y)))))+(cast523*cast531*((cast521*((float)(wmma142.y)))+(cast522*((float)(wmma143.y))))));
    if (0) {
      *(data0_5570560+(alu78+3072)) = buf7;
    }
    float cast532 = ((float)(((signed char)(((val599>>0u)&255u)))));
    float cast533 = ((float)(((signed char)(((val599>>8u)&255u)))));
    float cast534 = ((float)(((signed char)(((val599>>16u)&255u)))));
    float cast535 = ((float)(((signed char)(((val599>>24u)&255u)))));
    float cast536 = ((float)(((signed char)(((val600>>0u)&255u)))));
    float cast537 = ((float)(((signed char)(((val600>>8u)&255u)))));
    float cast538 = ((float)(((signed char)(((val600>>16u)&255u)))));
    float cast539 = ((float)(((signed char)(((val600>>24u)&255u)))));
    float cast540 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val598&65535u)))))));
    buf7 = (buf7+(cast540*cast524*((cast532*((float)(wmma136.z)))+(cast533*((float)(wmma137.z)))))+(cast540*cast525*((cast534*((float)(wmma138.z)))+(cast535*((float)(wmma139.z)))))+(cast540*cast526*((cast536*((float)(wmma140.z)))+(cast537*((float)(wmma141.z)))))+(cast540*cast527*((cast538*((float)(wmma142.z)))+(cast539*((float)(wmma143.z))))));
    if (0) {
      *(data0_5570560+(alu78+3073)) = buf8;
    }
    buf8 = (buf8+(cast540*cast528*((cast532*((float)(wmma136.w)))+(cast533*((float)(wmma137.w)))))+(cast540*cast529*((cast534*((float)(wmma138.w)))+(cast535*((float)(wmma139.w)))))+(cast540*cast530*((cast536*((float)(wmma140.w)))+(cast537*((float)(wmma141.w)))))+(cast540*cast531*((cast538*((float)(wmma142.w)))+(cast539*((float)(wmma143.w))))));
    unsigned int val601 = (*(buf0+alu91));
    unsigned int val602 = (*(buf0+alu92));
    unsigned int val603 = (*(buf0+alu93));
    unsigned int val604 = (*(buf0+alu94));
    unsigned int val605 = (*(buf0+alu95));
    unsigned int val606 = (*(buf0+alu96));
    unsigned int val607 = (*(buf0+alu97));
    unsigned int val608 = (*(buf0+alu98));
    unsigned int val609 = (*(buf0+alu177));
    unsigned int val610 = (*(buf0+alu178));
    unsigned int val611 = (*(buf0+alu179));
    unsigned int val612 = (*(buf0+alu180));
    unsigned int val613 = (*(buf0+alu181));
    unsigned int val614 = (*(buf0+alu182));
    unsigned int val615 = (*(buf0+alu183));
    unsigned int val616 = (*(buf0+alu184));
    unsigned int val617 = (*(buf0+alu148));
    unsigned int val618 = (*(buf0+alu151));
    unsigned int val619 = (*(buf0+alu152));
    unsigned int val620 = (*(buf0+alu153));
    unsigned int val621 = (*(buf0+alu156));
    unsigned int val622 = (*(buf0+alu157));
    if (0) {
      *(data0_5570560+(alu78+16)) = buf9;
    }
    char4 cast541 = make_char4(((signed char)(((val601>>0u)&255u))),((signed char)(((val601>>8u)&255u))),((signed char)(((val601>>16u)&255u))),((signed char)(((val601>>24u)&255u))));
    char4 cast542 = make_char4(((signed char)(((val602>>0u)&255u))),((signed char)(((val602>>8u)&255u))),((signed char)(((val602>>16u)&255u))),((signed char)(((val602>>24u)&255u))));
    char4 cast543 = make_char4(((signed char)(((val603>>0u)&255u))),((signed char)(((val603>>8u)&255u))),((signed char)(((val603>>16u)&255u))),((signed char)(((val603>>24u)&255u))));
    char4 cast544 = make_char4(((signed char)(((val604>>0u)&255u))),((signed char)(((val604>>8u)&255u))),((signed char)(((val604>>16u)&255u))),((signed char)(((val604>>24u)&255u))));
    char4 cast545 = make_char4(((signed char)(((val605>>0u)&255u))),((signed char)(((val605>>8u)&255u))),((signed char)(((val605>>16u)&255u))),((signed char)(((val605>>24u)&255u))));
    char4 cast546 = make_char4(((signed char)(((val606>>0u)&255u))),((signed char)(((val606>>8u)&255u))),((signed char)(((val606>>16u)&255u))),((signed char)(((val606>>24u)&255u))));
    char4 cast547 = make_char4(((signed char)(((val607>>0u)&255u))),((signed char)(((val607>>8u)&255u))),((signed char)(((val607>>16u)&255u))),((signed char)(((val607>>24u)&255u))));
    char4 cast548 = make_char4(((signed char)(((val608>>0u)&255u))),((signed char)(((val608>>8u)&255u))),((signed char)(((val608>>16u)&255u))),((signed char)(((val608>>24u)&255u))));
    int4 wmma144 = __WMMA_8_16_16_signed_char_int(alu928, cast541, cast0);
    int4 wmma145 = __WMMA_8_16_16_signed_char_int(alu929, cast542, cast0);
    int4 wmma146 = __WMMA_8_16_16_signed_char_int(alu930, cast543, cast0);
    int4 wmma147 = __WMMA_8_16_16_signed_char_int(alu931, cast544, cast0);
    int4 wmma148 = __WMMA_8_16_16_signed_char_int(alu932, cast545, cast0);
    int4 wmma149 = __WMMA_8_16_16_signed_char_int(alu933, cast546, cast0);
    int4 wmma150 = __WMMA_8_16_16_signed_char_int(alu934, cast547, cast0);
    int4 wmma151 = __WMMA_8_16_16_signed_char_int(alu935, cast548, cast0);
    float cast549 = ((float)(((signed char)(((val618>>0u)&255u)))));
    float cast550 = ((float)(((signed char)(((val618>>8u)&255u)))));
    float cast551 = ((float)(((signed char)(((val618>>16u)&255u)))));
    float cast552 = ((float)(((signed char)(((val618>>24u)&255u)))));
    float cast553 = ((float)(((signed char)(((val619>>0u)&255u)))));
    float cast554 = ((float)(((signed char)(((val619>>8u)&255u)))));
    float cast555 = ((float)(((signed char)(((val619>>16u)&255u)))));
    float cast556 = ((float)(((signed char)(((val619>>24u)&255u)))));
    float cast557 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val617&65535u)))))));
    float cast558 = tg_bitcast<float>((unsigned int)(val609));
    float cast559 = tg_bitcast<float>((unsigned int)(val610));
    float cast560 = tg_bitcast<float>((unsigned int)(val611));
    float cast561 = tg_bitcast<float>((unsigned int)(val612));
    buf9 = (buf9+(cast557*cast558*((cast549*((float)(wmma144.x)))+(cast550*((float)(wmma145.x)))))+(cast557*cast559*((cast551*((float)(wmma146.x)))+(cast552*((float)(wmma147.x)))))+(cast557*cast560*((cast553*((float)(wmma148.x)))+(cast554*((float)(wmma149.x)))))+(cast557*cast561*((cast555*((float)(wmma150.x)))+(cast556*((float)(wmma151.x))))));
    if (0) {
      *(data0_5570560+(alu78+17)) = buf10;
    }
    float cast562 = tg_bitcast<float>((unsigned int)(val613));
    float cast563 = tg_bitcast<float>((unsigned int)(val614));
    float cast564 = tg_bitcast<float>((unsigned int)(val615));
    float cast565 = tg_bitcast<float>((unsigned int)(val616));
    buf10 = (buf10+(cast557*cast562*((cast549*((float)(wmma144.y)))+(cast550*((float)(wmma145.y)))))+(cast557*cast563*((cast551*((float)(wmma146.y)))+(cast552*((float)(wmma147.y)))))+(cast557*cast564*((cast553*((float)(wmma148.y)))+(cast554*((float)(wmma149.y)))))+(cast557*cast565*((cast555*((float)(wmma150.y)))+(cast556*((float)(wmma151.y))))));
    if (0) {
      *(data0_5570560+(alu78+1040)) = buf11;
    }
    float cast566 = ((float)(((signed char)(((val621>>0u)&255u)))));
    float cast567 = ((float)(((signed char)(((val621>>8u)&255u)))));
    float cast568 = ((float)(((signed char)(((val621>>16u)&255u)))));
    float cast569 = ((float)(((signed char)(((val621>>24u)&255u)))));
    float cast570 = ((float)(((signed char)(((val622>>0u)&255u)))));
    float cast571 = ((float)(((signed char)(((val622>>8u)&255u)))));
    float cast572 = ((float)(((signed char)(((val622>>16u)&255u)))));
    float cast573 = ((float)(((signed char)(((val622>>24u)&255u)))));
    float cast574 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val620&65535u)))))));
    buf11 = (buf11+(cast574*cast558*((cast566*((float)(wmma144.z)))+(cast567*((float)(wmma145.z)))))+(cast574*cast559*((cast568*((float)(wmma146.z)))+(cast569*((float)(wmma147.z)))))+(cast574*cast560*((cast570*((float)(wmma148.z)))+(cast571*((float)(wmma149.z)))))+(cast574*cast561*((cast572*((float)(wmma150.z)))+(cast573*((float)(wmma151.z))))));
    if (0) {
      *(data0_5570560+(alu78+1041)) = buf12;
    }
    buf12 = (buf12+(cast574*cast562*((cast566*((float)(wmma144.w)))+(cast567*((float)(wmma145.w)))))+(cast574*cast563*((cast568*((float)(wmma146.w)))+(cast569*((float)(wmma147.w)))))+(cast574*cast564*((cast570*((float)(wmma148.w)))+(cast571*((float)(wmma149.w)))))+(cast574*cast565*((cast572*((float)(wmma150.w)))+(cast573*((float)(wmma151.w))))));
    unsigned int val623 = (*(buf0+alu177));
    unsigned int val624 = (*(buf0+alu178));
    unsigned int val625 = (*(buf0+alu179));
    unsigned int val626 = (*(buf0+alu180));
    unsigned int val627 = (*(buf0+alu181));
    unsigned int val628 = (*(buf0+alu182));
    unsigned int val629 = (*(buf0+alu183));
    unsigned int val630 = (*(buf0+alu184));
    unsigned int val631 = (*(buf0+alu158));
    unsigned int val632 = (*(buf0+alu161));
    unsigned int val633 = (*(buf0+alu162));
    unsigned int val634 = (*(buf0+alu163));
    unsigned int val635 = (*(buf0+alu166));
    unsigned int val636 = (*(buf0+alu167));
    if (0) {
      *(data0_5570560+(alu78+2064)) = buf13;
    }
    int4 wmma152 = __WMMA_8_16_16_signed_char_int(alu952, cast541, cast0);
    int4 wmma153 = __WMMA_8_16_16_signed_char_int(alu953, cast542, cast0);
    int4 wmma154 = __WMMA_8_16_16_signed_char_int(alu954, cast543, cast0);
    int4 wmma155 = __WMMA_8_16_16_signed_char_int(alu955, cast544, cast0);
    int4 wmma156 = __WMMA_8_16_16_signed_char_int(alu956, cast545, cast0);
    int4 wmma157 = __WMMA_8_16_16_signed_char_int(alu957, cast546, cast0);
    int4 wmma158 = __WMMA_8_16_16_signed_char_int(alu958, cast547, cast0);
    int4 wmma159 = __WMMA_8_16_16_signed_char_int(alu959, cast548, cast0);
    float cast575 = ((float)(((signed char)(((val632>>0u)&255u)))));
    float cast576 = ((float)(((signed char)(((val632>>8u)&255u)))));
    float cast577 = ((float)(((signed char)(((val632>>16u)&255u)))));
    float cast578 = ((float)(((signed char)(((val632>>24u)&255u)))));
    float cast579 = ((float)(((signed char)(((val633>>0u)&255u)))));
    float cast580 = ((float)(((signed char)(((val633>>8u)&255u)))));
    float cast581 = ((float)(((signed char)(((val633>>16u)&255u)))));
    float cast582 = ((float)(((signed char)(((val633>>24u)&255u)))));
    float cast583 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val631&65535u)))))));
    float cast584 = tg_bitcast<float>((unsigned int)(val623));
    float cast585 = tg_bitcast<float>((unsigned int)(val624));
    float cast586 = tg_bitcast<float>((unsigned int)(val625));
    float cast587 = tg_bitcast<float>((unsigned int)(val626));
    buf13 = (buf13+(cast583*cast584*((cast575*((float)(wmma152.x)))+(cast576*((float)(wmma153.x)))))+(cast583*cast585*((cast577*((float)(wmma154.x)))+(cast578*((float)(wmma155.x)))))+(cast583*cast586*((cast579*((float)(wmma156.x)))+(cast580*((float)(wmma157.x)))))+(cast583*cast587*((cast581*((float)(wmma158.x)))+(cast582*((float)(wmma159.x))))));
    if (0) {
      *(data0_5570560+(alu78+2065)) = buf14;
    }
    float cast588 = tg_bitcast<float>((unsigned int)(val627));
    float cast589 = tg_bitcast<float>((unsigned int)(val628));
    float cast590 = tg_bitcast<float>((unsigned int)(val629));
    float cast591 = tg_bitcast<float>((unsigned int)(val630));
    buf14 = (buf14+(cast583*cast588*((cast575*((float)(wmma152.y)))+(cast576*((float)(wmma153.y)))))+(cast583*cast589*((cast577*((float)(wmma154.y)))+(cast578*((float)(wmma155.y)))))+(cast583*cast590*((cast579*((float)(wmma156.y)))+(cast580*((float)(wmma157.y)))))+(cast583*cast591*((cast581*((float)(wmma158.y)))+(cast582*((float)(wmma159.y))))));
    if (0) {
      *(data0_5570560+(alu78+3088)) = buf15;
    }
    float cast592 = ((float)(((signed char)(((val635>>0u)&255u)))));
    float cast593 = ((float)(((signed char)(((val635>>8u)&255u)))));
    float cast594 = ((float)(((signed char)(((val635>>16u)&255u)))));
    float cast595 = ((float)(((signed char)(((val635>>24u)&255u)))));
    float cast596 = ((float)(((signed char)(((val636>>0u)&255u)))));
    float cast597 = ((float)(((signed char)(((val636>>8u)&255u)))));
    float cast598 = ((float)(((signed char)(((val636>>16u)&255u)))));
    float cast599 = ((float)(((signed char)(((val636>>24u)&255u)))));
    float cast600 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val634&65535u)))))));
    buf15 = (buf15+(cast600*cast584*((cast592*((float)(wmma152.z)))+(cast593*((float)(wmma153.z)))))+(cast600*cast585*((cast594*((float)(wmma154.z)))+(cast595*((float)(wmma155.z)))))+(cast600*cast586*((cast596*((float)(wmma156.z)))+(cast597*((float)(wmma157.z)))))+(cast600*cast587*((cast598*((float)(wmma158.z)))+(cast599*((float)(wmma159.z))))));
    if (0) {
      *(data0_5570560+(alu78+3089)) = buf16;
    }
    buf16 = (buf16+(cast600*cast588*((cast592*((float)(wmma152.w)))+(cast593*((float)(wmma153.w)))))+(cast600*cast589*((cast594*((float)(wmma154.w)))+(cast595*((float)(wmma155.w)))))+(cast600*cast590*((cast596*((float)(wmma156.w)))+(cast597*((float)(wmma157.w)))))+(cast600*cast591*((cast598*((float)(wmma158.w)))+(cast599*((float)(wmma159.w))))));
    unsigned int val637 = (*(buf0+alu99));
    unsigned int val638 = (*(buf0+alu100));
    unsigned int val639 = (*(buf0+alu101));
    unsigned int val640 = (*(buf0+alu102));
    unsigned int val641 = (*(buf0+alu103));
    unsigned int val642 = (*(buf0+alu104));
    unsigned int val643 = (*(buf0+alu105));
    unsigned int val644 = (*(buf0+alu106));
    unsigned int val645 = (*(buf0+alu185));
    unsigned int val646 = (*(buf0+alu186));
    unsigned int val647 = (*(buf0+alu187));
    unsigned int val648 = (*(buf0+alu188));
    unsigned int val649 = (*(buf0+alu189));
    unsigned int val650 = (*(buf0+alu190));
    unsigned int val651 = (*(buf0+alu191));
    unsigned int val652 = (*(buf0+alu192));
    unsigned int val653 = (*(buf0+alu148));
    unsigned int val654 = (*(buf0+alu151));
    unsigned int val655 = (*(buf0+alu152));
    unsigned int val656 = (*(buf0+alu153));
    unsigned int val657 = (*(buf0+alu156));
    unsigned int val658 = (*(buf0+alu157));
    if (0) {
      *(data0_5570560+(alu78+32)) = buf17;
    }
    char4 cast601 = make_char4(((signed char)(((val637>>0u)&255u))),((signed char)(((val637>>8u)&255u))),((signed char)(((val637>>16u)&255u))),((signed char)(((val637>>24u)&255u))));
    char4 cast602 = make_char4(((signed char)(((val638>>0u)&255u))),((signed char)(((val638>>8u)&255u))),((signed char)(((val638>>16u)&255u))),((signed char)(((val638>>24u)&255u))));
    char4 cast603 = make_char4(((signed char)(((val639>>0u)&255u))),((signed char)(((val639>>8u)&255u))),((signed char)(((val639>>16u)&255u))),((signed char)(((val639>>24u)&255u))));
    char4 cast604 = make_char4(((signed char)(((val640>>0u)&255u))),((signed char)(((val640>>8u)&255u))),((signed char)(((val640>>16u)&255u))),((signed char)(((val640>>24u)&255u))));
    char4 cast605 = make_char4(((signed char)(((val641>>0u)&255u))),((signed char)(((val641>>8u)&255u))),((signed char)(((val641>>16u)&255u))),((signed char)(((val641>>24u)&255u))));
    char4 cast606 = make_char4(((signed char)(((val642>>0u)&255u))),((signed char)(((val642>>8u)&255u))),((signed char)(((val642>>16u)&255u))),((signed char)(((val642>>24u)&255u))));
    char4 cast607 = make_char4(((signed char)(((val643>>0u)&255u))),((signed char)(((val643>>8u)&255u))),((signed char)(((val643>>16u)&255u))),((signed char)(((val643>>24u)&255u))));
    char4 cast608 = make_char4(((signed char)(((val644>>0u)&255u))),((signed char)(((val644>>8u)&255u))),((signed char)(((val644>>16u)&255u))),((signed char)(((val644>>24u)&255u))));
    int4 wmma160 = __WMMA_8_16_16_signed_char_int(alu928, cast601, cast0);
    int4 wmma161 = __WMMA_8_16_16_signed_char_int(alu929, cast602, cast0);
    int4 wmma162 = __WMMA_8_16_16_signed_char_int(alu930, cast603, cast0);
    int4 wmma163 = __WMMA_8_16_16_signed_char_int(alu931, cast604, cast0);
    int4 wmma164 = __WMMA_8_16_16_signed_char_int(alu932, cast605, cast0);
    int4 wmma165 = __WMMA_8_16_16_signed_char_int(alu933, cast606, cast0);
    int4 wmma166 = __WMMA_8_16_16_signed_char_int(alu934, cast607, cast0);
    int4 wmma167 = __WMMA_8_16_16_signed_char_int(alu935, cast608, cast0);
    float cast609 = ((float)(((signed char)(((val654>>0u)&255u)))));
    float cast610 = ((float)(((signed char)(((val654>>8u)&255u)))));
    float cast611 = ((float)(((signed char)(((val654>>16u)&255u)))));
    float cast612 = ((float)(((signed char)(((val654>>24u)&255u)))));
    float cast613 = ((float)(((signed char)(((val655>>0u)&255u)))));
    float cast614 = ((float)(((signed char)(((val655>>8u)&255u)))));
    float cast615 = ((float)(((signed char)(((val655>>16u)&255u)))));
    float cast616 = ((float)(((signed char)(((val655>>24u)&255u)))));
    float cast617 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val653&65535u)))))));
    float cast618 = tg_bitcast<float>((unsigned int)(val645));
    float cast619 = tg_bitcast<float>((unsigned int)(val646));
    float cast620 = tg_bitcast<float>((unsigned int)(val647));
    float cast621 = tg_bitcast<float>((unsigned int)(val648));
    buf17 = (buf17+(cast617*cast618*((cast609*((float)(wmma160.x)))+(cast610*((float)(wmma161.x)))))+(cast617*cast619*((cast611*((float)(wmma162.x)))+(cast612*((float)(wmma163.x)))))+(cast617*cast620*((cast613*((float)(wmma164.x)))+(cast614*((float)(wmma165.x)))))+(cast617*cast621*((cast615*((float)(wmma166.x)))+(cast616*((float)(wmma167.x))))));
    if (0) {
      *(data0_5570560+(alu78+33)) = buf18;
    }
    float cast622 = tg_bitcast<float>((unsigned int)(val649));
    float cast623 = tg_bitcast<float>((unsigned int)(val650));
    float cast624 = tg_bitcast<float>((unsigned int)(val651));
    float cast625 = tg_bitcast<float>((unsigned int)(val652));
    buf18 = (buf18+(cast617*cast622*((cast609*((float)(wmma160.y)))+(cast610*((float)(wmma161.y)))))+(cast617*cast623*((cast611*((float)(wmma162.y)))+(cast612*((float)(wmma163.y)))))+(cast617*cast624*((cast613*((float)(wmma164.y)))+(cast614*((float)(wmma165.y)))))+(cast617*cast625*((cast615*((float)(wmma166.y)))+(cast616*((float)(wmma167.y))))));
    if (0) {
      *(data0_5570560+(alu78+1056)) = buf19;
    }
    float cast626 = ((float)(((signed char)(((val657>>0u)&255u)))));
    float cast627 = ((float)(((signed char)(((val657>>8u)&255u)))));
    float cast628 = ((float)(((signed char)(((val657>>16u)&255u)))));
    float cast629 = ((float)(((signed char)(((val657>>24u)&255u)))));
    float cast630 = ((float)(((signed char)(((val658>>0u)&255u)))));
    float cast631 = ((float)(((signed char)(((val658>>8u)&255u)))));
    float cast632 = ((float)(((signed char)(((val658>>16u)&255u)))));
    float cast633 = ((float)(((signed char)(((val658>>24u)&255u)))));
    float cast634 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val656&65535u)))))));
    buf19 = (buf19+(cast634*cast618*((cast626*((float)(wmma160.z)))+(cast627*((float)(wmma161.z)))))+(cast634*cast619*((cast628*((float)(wmma162.z)))+(cast629*((float)(wmma163.z)))))+(cast634*cast620*((cast630*((float)(wmma164.z)))+(cast631*((float)(wmma165.z)))))+(cast634*cast621*((cast632*((float)(wmma166.z)))+(cast633*((float)(wmma167.z))))));
    if (0) {
      *(data0_5570560+(alu78+1057)) = buf20;
    }
    buf20 = (buf20+(cast634*cast622*((cast626*((float)(wmma160.w)))+(cast627*((float)(wmma161.w)))))+(cast634*cast623*((cast628*((float)(wmma162.w)))+(cast629*((float)(wmma163.w)))))+(cast634*cast624*((cast630*((float)(wmma164.w)))+(cast631*((float)(wmma165.w)))))+(cast634*cast625*((cast632*((float)(wmma166.w)))+(cast633*((float)(wmma167.w))))));
    unsigned int val659 = (*(buf0+alu185));
    unsigned int val660 = (*(buf0+alu186));
    unsigned int val661 = (*(buf0+alu187));
    unsigned int val662 = (*(buf0+alu188));
    unsigned int val663 = (*(buf0+alu189));
    unsigned int val664 = (*(buf0+alu190));
    unsigned int val665 = (*(buf0+alu191));
    unsigned int val666 = (*(buf0+alu192));
    unsigned int val667 = (*(buf0+alu158));
    unsigned int val668 = (*(buf0+alu161));
    unsigned int val669 = (*(buf0+alu162));
    unsigned int val670 = (*(buf0+alu163));
    unsigned int val671 = (*(buf0+alu166));
    unsigned int val672 = (*(buf0+alu167));
    if (0) {
      *(data0_5570560+(alu78+2080)) = buf21;
    }
    int4 wmma168 = __WMMA_8_16_16_signed_char_int(alu952, cast601, cast0);
    int4 wmma169 = __WMMA_8_16_16_signed_char_int(alu953, cast602, cast0);
    int4 wmma170 = __WMMA_8_16_16_signed_char_int(alu954, cast603, cast0);
    int4 wmma171 = __WMMA_8_16_16_signed_char_int(alu955, cast604, cast0);
    int4 wmma172 = __WMMA_8_16_16_signed_char_int(alu956, cast605, cast0);
    int4 wmma173 = __WMMA_8_16_16_signed_char_int(alu957, cast606, cast0);
    int4 wmma174 = __WMMA_8_16_16_signed_char_int(alu958, cast607, cast0);
    int4 wmma175 = __WMMA_8_16_16_signed_char_int(alu959, cast608, cast0);
    float cast635 = ((float)(((signed char)(((val668>>0u)&255u)))));
    float cast636 = ((float)(((signed char)(((val668>>8u)&255u)))));
    float cast637 = ((float)(((signed char)(((val668>>16u)&255u)))));
    float cast638 = ((float)(((signed char)(((val668>>24u)&255u)))));
    float cast639 = ((float)(((signed char)(((val669>>0u)&255u)))));
    float cast640 = ((float)(((signed char)(((val669>>8u)&255u)))));
    float cast641 = ((float)(((signed char)(((val669>>16u)&255u)))));
    float cast642 = ((float)(((signed char)(((val669>>24u)&255u)))));
    float cast643 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val667&65535u)))))));
    float cast644 = tg_bitcast<float>((unsigned int)(val659));
    float cast645 = tg_bitcast<float>((unsigned int)(val660));
    float cast646 = tg_bitcast<float>((unsigned int)(val661));
    float cast647 = tg_bitcast<float>((unsigned int)(val662));
    buf21 = (buf21+(cast643*cast644*((cast635*((float)(wmma168.x)))+(cast636*((float)(wmma169.x)))))+(cast643*cast645*((cast637*((float)(wmma170.x)))+(cast638*((float)(wmma171.x)))))+(cast643*cast646*((cast639*((float)(wmma172.x)))+(cast640*((float)(wmma173.x)))))+(cast643*cast647*((cast641*((float)(wmma174.x)))+(cast642*((float)(wmma175.x))))));
    if (0) {
      *(data0_5570560+(alu78+2081)) = buf22;
    }
    float cast648 = tg_bitcast<float>((unsigned int)(val663));
    float cast649 = tg_bitcast<float>((unsigned int)(val664));
    float cast650 = tg_bitcast<float>((unsigned int)(val665));
    float cast651 = tg_bitcast<float>((unsigned int)(val666));
    buf22 = (buf22+(cast643*cast648*((cast635*((float)(wmma168.y)))+(cast636*((float)(wmma169.y)))))+(cast643*cast649*((cast637*((float)(wmma170.y)))+(cast638*((float)(wmma171.y)))))+(cast643*cast650*((cast639*((float)(wmma172.y)))+(cast640*((float)(wmma173.y)))))+(cast643*cast651*((cast641*((float)(wmma174.y)))+(cast642*((float)(wmma175.y))))));
    if (0) {
      *(data0_5570560+(alu78+3104)) = buf23;
    }
    float cast652 = ((float)(((signed char)(((val671>>0u)&255u)))));
    float cast653 = ((float)(((signed char)(((val671>>8u)&255u)))));
    float cast654 = ((float)(((signed char)(((val671>>16u)&255u)))));
    float cast655 = ((float)(((signed char)(((val671>>24u)&255u)))));
    float cast656 = ((float)(((signed char)(((val672>>0u)&255u)))));
    float cast657 = ((float)(((signed char)(((val672>>8u)&255u)))));
    float cast658 = ((float)(((signed char)(((val672>>16u)&255u)))));
    float cast659 = ((float)(((signed char)(((val672>>24u)&255u)))));
    float cast660 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val670&65535u)))))));
    buf23 = (buf23+(cast660*cast644*((cast652*((float)(wmma168.z)))+(cast653*((float)(wmma169.z)))))+(cast660*cast645*((cast654*((float)(wmma170.z)))+(cast655*((float)(wmma171.z)))))+(cast660*cast646*((cast656*((float)(wmma172.z)))+(cast657*((float)(wmma173.z)))))+(cast660*cast647*((cast658*((float)(wmma174.z)))+(cast659*((float)(wmma175.z))))));
    if (0) {
      *(data0_5570560+(alu78+3105)) = buf24;
    }
    buf24 = (buf24+(cast660*cast648*((cast652*((float)(wmma168.w)))+(cast653*((float)(wmma169.w)))))+(cast660*cast649*((cast654*((float)(wmma170.w)))+(cast655*((float)(wmma171.w)))))+(cast660*cast650*((cast656*((float)(wmma172.w)))+(cast657*((float)(wmma173.w)))))+(cast660*cast651*((cast658*((float)(wmma174.w)))+(cast659*((float)(wmma175.w))))));
    unsigned int val673 = (*(buf0+alu107));
    unsigned int val674 = (*(buf0+alu108));
    unsigned int val675 = (*(buf0+alu109));
    unsigned int val676 = (*(buf0+alu110));
    unsigned int val677 = (*(buf0+alu111));
    unsigned int val678 = (*(buf0+alu112));
    unsigned int val679 = (*(buf0+alu113));
    unsigned int val680 = (*(buf0+alu114));
    unsigned int val681 = (*(buf0+alu193));
    unsigned int val682 = (*(buf0+alu194));
    unsigned int val683 = (*(buf0+alu195));
    unsigned int val684 = (*(buf0+alu196));
    unsigned int val685 = (*(buf0+alu197));
    unsigned int val686 = (*(buf0+alu198));
    unsigned int val687 = (*(buf0+alu199));
    unsigned int val688 = (*(buf0+alu200));
    unsigned int val689 = (*(buf0+alu148));
    unsigned int val690 = (*(buf0+alu151));
    unsigned int val691 = (*(buf0+alu152));
    unsigned int val692 = (*(buf0+alu153));
    unsigned int val693 = (*(buf0+alu156));
    unsigned int val694 = (*(buf0+alu157));
    if (0) {
      *(data0_5570560+(alu78+48)) = buf25;
    }
    char4 cast661 = make_char4(((signed char)(((val673>>0u)&255u))),((signed char)(((val673>>8u)&255u))),((signed char)(((val673>>16u)&255u))),((signed char)(((val673>>24u)&255u))));
    char4 cast662 = make_char4(((signed char)(((val674>>0u)&255u))),((signed char)(((val674>>8u)&255u))),((signed char)(((val674>>16u)&255u))),((signed char)(((val674>>24u)&255u))));
    char4 cast663 = make_char4(((signed char)(((val675>>0u)&255u))),((signed char)(((val675>>8u)&255u))),((signed char)(((val675>>16u)&255u))),((signed char)(((val675>>24u)&255u))));
    char4 cast664 = make_char4(((signed char)(((val676>>0u)&255u))),((signed char)(((val676>>8u)&255u))),((signed char)(((val676>>16u)&255u))),((signed char)(((val676>>24u)&255u))));
    char4 cast665 = make_char4(((signed char)(((val677>>0u)&255u))),((signed char)(((val677>>8u)&255u))),((signed char)(((val677>>16u)&255u))),((signed char)(((val677>>24u)&255u))));
    char4 cast666 = make_char4(((signed char)(((val678>>0u)&255u))),((signed char)(((val678>>8u)&255u))),((signed char)(((val678>>16u)&255u))),((signed char)(((val678>>24u)&255u))));
    char4 cast667 = make_char4(((signed char)(((val679>>0u)&255u))),((signed char)(((val679>>8u)&255u))),((signed char)(((val679>>16u)&255u))),((signed char)(((val679>>24u)&255u))));
    char4 cast668 = make_char4(((signed char)(((val680>>0u)&255u))),((signed char)(((val680>>8u)&255u))),((signed char)(((val680>>16u)&255u))),((signed char)(((val680>>24u)&255u))));
    int4 wmma176 = __WMMA_8_16_16_signed_char_int(alu928, cast661, cast0);
    int4 wmma177 = __WMMA_8_16_16_signed_char_int(alu929, cast662, cast0);
    int4 wmma178 = __WMMA_8_16_16_signed_char_int(alu930, cast663, cast0);
    int4 wmma179 = __WMMA_8_16_16_signed_char_int(alu931, cast664, cast0);
    int4 wmma180 = __WMMA_8_16_16_signed_char_int(alu932, cast665, cast0);
    int4 wmma181 = __WMMA_8_16_16_signed_char_int(alu933, cast666, cast0);
    int4 wmma182 = __WMMA_8_16_16_signed_char_int(alu934, cast667, cast0);
    int4 wmma183 = __WMMA_8_16_16_signed_char_int(alu935, cast668, cast0);
    float cast669 = ((float)(((signed char)(((val690>>0u)&255u)))));
    float cast670 = ((float)(((signed char)(((val690>>8u)&255u)))));
    float cast671 = ((float)(((signed char)(((val690>>16u)&255u)))));
    float cast672 = ((float)(((signed char)(((val690>>24u)&255u)))));
    float cast673 = ((float)(((signed char)(((val691>>0u)&255u)))));
    float cast674 = ((float)(((signed char)(((val691>>8u)&255u)))));
    float cast675 = ((float)(((signed char)(((val691>>16u)&255u)))));
    float cast676 = ((float)(((signed char)(((val691>>24u)&255u)))));
    float cast677 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val689&65535u)))))));
    float cast678 = tg_bitcast<float>((unsigned int)(val681));
    float cast679 = tg_bitcast<float>((unsigned int)(val682));
    float cast680 = tg_bitcast<float>((unsigned int)(val683));
    float cast681 = tg_bitcast<float>((unsigned int)(val684));
    buf25 = (buf25+(cast677*cast678*((cast669*((float)(wmma176.x)))+(cast670*((float)(wmma177.x)))))+(cast677*cast679*((cast671*((float)(wmma178.x)))+(cast672*((float)(wmma179.x)))))+(cast677*cast680*((cast673*((float)(wmma180.x)))+(cast674*((float)(wmma181.x)))))+(cast677*cast681*((cast675*((float)(wmma182.x)))+(cast676*((float)(wmma183.x))))));
    if (0) {
      *(data0_5570560+(alu78+49)) = buf26;
    }
    float cast682 = tg_bitcast<float>((unsigned int)(val685));
    float cast683 = tg_bitcast<float>((unsigned int)(val686));
    float cast684 = tg_bitcast<float>((unsigned int)(val687));
    float cast685 = tg_bitcast<float>((unsigned int)(val688));
    buf26 = (buf26+(cast677*cast682*((cast669*((float)(wmma176.y)))+(cast670*((float)(wmma177.y)))))+(cast677*cast683*((cast671*((float)(wmma178.y)))+(cast672*((float)(wmma179.y)))))+(cast677*cast684*((cast673*((float)(wmma180.y)))+(cast674*((float)(wmma181.y)))))+(cast677*cast685*((cast675*((float)(wmma182.y)))+(cast676*((float)(wmma183.y))))));
    if (0) {
      *(data0_5570560+(alu78+1072)) = buf27;
    }
    float cast686 = ((float)(((signed char)(((val693>>0u)&255u)))));
    float cast687 = ((float)(((signed char)(((val693>>8u)&255u)))));
    float cast688 = ((float)(((signed char)(((val693>>16u)&255u)))));
    float cast689 = ((float)(((signed char)(((val693>>24u)&255u)))));
    float cast690 = ((float)(((signed char)(((val694>>0u)&255u)))));
    float cast691 = ((float)(((signed char)(((val694>>8u)&255u)))));
    float cast692 = ((float)(((signed char)(((val694>>16u)&255u)))));
    float cast693 = ((float)(((signed char)(((val694>>24u)&255u)))));
    float cast694 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val692&65535u)))))));
    buf27 = (buf27+(cast694*cast678*((cast686*((float)(wmma176.z)))+(cast687*((float)(wmma177.z)))))+(cast694*cast679*((cast688*((float)(wmma178.z)))+(cast689*((float)(wmma179.z)))))+(cast694*cast680*((cast690*((float)(wmma180.z)))+(cast691*((float)(wmma181.z)))))+(cast694*cast681*((cast692*((float)(wmma182.z)))+(cast693*((float)(wmma183.z))))));
    if (0) {
      *(data0_5570560+(alu78+1073)) = buf28;
    }
    buf28 = (buf28+(cast694*cast682*((cast686*((float)(wmma176.w)))+(cast687*((float)(wmma177.w)))))+(cast694*cast683*((cast688*((float)(wmma178.w)))+(cast689*((float)(wmma179.w)))))+(cast694*cast684*((cast690*((float)(wmma180.w)))+(cast691*((float)(wmma181.w)))))+(cast694*cast685*((cast692*((float)(wmma182.w)))+(cast693*((float)(wmma183.w))))));
    unsigned int val695 = (*(buf0+alu193));
    unsigned int val696 = (*(buf0+alu194));
    unsigned int val697 = (*(buf0+alu195));
    unsigned int val698 = (*(buf0+alu196));
    unsigned int val699 = (*(buf0+alu197));
    unsigned int val700 = (*(buf0+alu198));
    unsigned int val701 = (*(buf0+alu199));
    unsigned int val702 = (*(buf0+alu200));
    unsigned int val703 = (*(buf0+alu158));
    unsigned int val704 = (*(buf0+alu161));
    unsigned int val705 = (*(buf0+alu162));
    unsigned int val706 = (*(buf0+alu163));
    unsigned int val707 = (*(buf0+alu166));
    unsigned int val708 = (*(buf0+alu167));
    if (0) {
      *(data0_5570560+(alu78+2096)) = buf29;
    }
    int4 wmma184 = __WMMA_8_16_16_signed_char_int(alu952, cast661, cast0);
    int4 wmma185 = __WMMA_8_16_16_signed_char_int(alu953, cast662, cast0);
    int4 wmma186 = __WMMA_8_16_16_signed_char_int(alu954, cast663, cast0);
    int4 wmma187 = __WMMA_8_16_16_signed_char_int(alu955, cast664, cast0);
    int4 wmma188 = __WMMA_8_16_16_signed_char_int(alu956, cast665, cast0);
    int4 wmma189 = __WMMA_8_16_16_signed_char_int(alu957, cast666, cast0);
    int4 wmma190 = __WMMA_8_16_16_signed_char_int(alu958, cast667, cast0);
    int4 wmma191 = __WMMA_8_16_16_signed_char_int(alu959, cast668, cast0);
    float cast695 = ((float)(((signed char)(((val704>>0u)&255u)))));
    float cast696 = ((float)(((signed char)(((val704>>8u)&255u)))));
    float cast697 = ((float)(((signed char)(((val704>>16u)&255u)))));
    float cast698 = ((float)(((signed char)(((val704>>24u)&255u)))));
    float cast699 = ((float)(((signed char)(((val705>>0u)&255u)))));
    float cast700 = ((float)(((signed char)(((val705>>8u)&255u)))));
    float cast701 = ((float)(((signed char)(((val705>>16u)&255u)))));
    float cast702 = ((float)(((signed char)(((val705>>24u)&255u)))));
    float cast703 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val703&65535u)))))));
    float cast704 = tg_bitcast<float>((unsigned int)(val695));
    float cast705 = tg_bitcast<float>((unsigned int)(val696));
    float cast706 = tg_bitcast<float>((unsigned int)(val697));
    float cast707 = tg_bitcast<float>((unsigned int)(val698));
    buf29 = (buf29+(cast703*cast704*((cast695*((float)(wmma184.x)))+(cast696*((float)(wmma185.x)))))+(cast703*cast705*((cast697*((float)(wmma186.x)))+(cast698*((float)(wmma187.x)))))+(cast703*cast706*((cast699*((float)(wmma188.x)))+(cast700*((float)(wmma189.x)))))+(cast703*cast707*((cast701*((float)(wmma190.x)))+(cast702*((float)(wmma191.x))))));
    if (0) {
      *(data0_5570560+(alu78+2097)) = buf30;
    }
    float cast708 = tg_bitcast<float>((unsigned int)(val699));
    float cast709 = tg_bitcast<float>((unsigned int)(val700));
    float cast710 = tg_bitcast<float>((unsigned int)(val701));
    float cast711 = tg_bitcast<float>((unsigned int)(val702));
    buf30 = (buf30+(cast703*cast708*((cast695*((float)(wmma184.y)))+(cast696*((float)(wmma185.y)))))+(cast703*cast709*((cast697*((float)(wmma186.y)))+(cast698*((float)(wmma187.y)))))+(cast703*cast710*((cast699*((float)(wmma188.y)))+(cast700*((float)(wmma189.y)))))+(cast703*cast711*((cast701*((float)(wmma190.y)))+(cast702*((float)(wmma191.y))))));
    if (0) {
      *(data0_5570560+(alu78+3120)) = buf31;
    }
    float cast712 = ((float)(((signed char)(((val707>>0u)&255u)))));
    float cast713 = ((float)(((signed char)(((val707>>8u)&255u)))));
    float cast714 = ((float)(((signed char)(((val707>>16u)&255u)))));
    float cast715 = ((float)(((signed char)(((val707>>24u)&255u)))));
    float cast716 = ((float)(((signed char)(((val708>>0u)&255u)))));
    float cast717 = ((float)(((signed char)(((val708>>8u)&255u)))));
    float cast718 = ((float)(((signed char)(((val708>>16u)&255u)))));
    float cast719 = ((float)(((signed char)(((val708>>24u)&255u)))));
    float cast720 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val706&65535u)))))));
    buf31 = (buf31+(cast720*cast704*((cast712*((float)(wmma184.z)))+(cast713*((float)(wmma185.z)))))+(cast720*cast705*((cast714*((float)(wmma186.z)))+(cast715*((float)(wmma187.z)))))+(cast720*cast706*((cast716*((float)(wmma188.z)))+(cast717*((float)(wmma189.z)))))+(cast720*cast707*((cast718*((float)(wmma190.z)))+(cast719*((float)(wmma191.z))))));
    if (0) {
      *(data0_5570560+(alu78+3121)) = buf32;
    }
    buf32 = (buf32+(cast720*cast708*((cast712*((float)(wmma184.w)))+(cast713*((float)(wmma185.w)))))+(cast720*cast709*((cast714*((float)(wmma186.w)))+(cast715*((float)(wmma187.w)))))+(cast720*cast710*((cast716*((float)(wmma188.w)))+(cast717*((float)(wmma189.w)))))+(cast720*cast711*((cast718*((float)(wmma190.w)))+(cast719*((float)(wmma191.w))))));
    unsigned int val709 = (*(buf0+alu115));
    unsigned int val710 = (*(buf0+alu116));
    unsigned int val711 = (*(buf0+alu117));
    unsigned int val712 = (*(buf0+alu118));
    unsigned int val713 = (*(buf0+alu119));
    unsigned int val714 = (*(buf0+alu120));
    unsigned int val715 = (*(buf0+alu121));
    unsigned int val716 = (*(buf0+alu122));
    unsigned int val717 = (*(buf0+alu201));
    unsigned int val718 = (*(buf0+alu202));
    unsigned int val719 = (*(buf0+alu203));
    unsigned int val720 = (*(buf0+alu204));
    unsigned int val721 = (*(buf0+alu205));
    unsigned int val722 = (*(buf0+alu206));
    unsigned int val723 = (*(buf0+alu207));
    unsigned int val724 = (*(buf0+alu208));
    unsigned int val725 = (*(buf0+alu148));
    unsigned int val726 = (*(buf0+alu151));
    unsigned int val727 = (*(buf0+alu152));
    unsigned int val728 = (*(buf0+alu153));
    unsigned int val729 = (*(buf0+alu156));
    unsigned int val730 = (*(buf0+alu157));
    if (0) {
      *(data0_5570560+(alu78+64)) = buf33;
    }
    char4 cast721 = make_char4(((signed char)(((val709>>0u)&255u))),((signed char)(((val709>>8u)&255u))),((signed char)(((val709>>16u)&255u))),((signed char)(((val709>>24u)&255u))));
    char4 cast722 = make_char4(((signed char)(((val710>>0u)&255u))),((signed char)(((val710>>8u)&255u))),((signed char)(((val710>>16u)&255u))),((signed char)(((val710>>24u)&255u))));
    char4 cast723 = make_char4(((signed char)(((val711>>0u)&255u))),((signed char)(((val711>>8u)&255u))),((signed char)(((val711>>16u)&255u))),((signed char)(((val711>>24u)&255u))));
    char4 cast724 = make_char4(((signed char)(((val712>>0u)&255u))),((signed char)(((val712>>8u)&255u))),((signed char)(((val712>>16u)&255u))),((signed char)(((val712>>24u)&255u))));
    char4 cast725 = make_char4(((signed char)(((val713>>0u)&255u))),((signed char)(((val713>>8u)&255u))),((signed char)(((val713>>16u)&255u))),((signed char)(((val713>>24u)&255u))));
    char4 cast726 = make_char4(((signed char)(((val714>>0u)&255u))),((signed char)(((val714>>8u)&255u))),((signed char)(((val714>>16u)&255u))),((signed char)(((val714>>24u)&255u))));
    char4 cast727 = make_char4(((signed char)(((val715>>0u)&255u))),((signed char)(((val715>>8u)&255u))),((signed char)(((val715>>16u)&255u))),((signed char)(((val715>>24u)&255u))));
    char4 cast728 = make_char4(((signed char)(((val716>>0u)&255u))),((signed char)(((val716>>8u)&255u))),((signed char)(((val716>>16u)&255u))),((signed char)(((val716>>24u)&255u))));
    int4 wmma192 = __WMMA_8_16_16_signed_char_int(alu928, cast721, cast0);
    int4 wmma193 = __WMMA_8_16_16_signed_char_int(alu929, cast722, cast0);
    int4 wmma194 = __WMMA_8_16_16_signed_char_int(alu930, cast723, cast0);
    int4 wmma195 = __WMMA_8_16_16_signed_char_int(alu931, cast724, cast0);
    int4 wmma196 = __WMMA_8_16_16_signed_char_int(alu932, cast725, cast0);
    int4 wmma197 = __WMMA_8_16_16_signed_char_int(alu933, cast726, cast0);
    int4 wmma198 = __WMMA_8_16_16_signed_char_int(alu934, cast727, cast0);
    int4 wmma199 = __WMMA_8_16_16_signed_char_int(alu935, cast728, cast0);
    float cast729 = ((float)(((signed char)(((val726>>0u)&255u)))));
    float cast730 = ((float)(((signed char)(((val726>>8u)&255u)))));
    float cast731 = ((float)(((signed char)(((val726>>16u)&255u)))));
    float cast732 = ((float)(((signed char)(((val726>>24u)&255u)))));
    float cast733 = ((float)(((signed char)(((val727>>0u)&255u)))));
    float cast734 = ((float)(((signed char)(((val727>>8u)&255u)))));
    float cast735 = ((float)(((signed char)(((val727>>16u)&255u)))));
    float cast736 = ((float)(((signed char)(((val727>>24u)&255u)))));
    float cast737 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val725&65535u)))))));
    float cast738 = tg_bitcast<float>((unsigned int)(val717));
    float cast739 = tg_bitcast<float>((unsigned int)(val718));
    float cast740 = tg_bitcast<float>((unsigned int)(val719));
    float cast741 = tg_bitcast<float>((unsigned int)(val720));
    buf33 = (buf33+(cast737*cast738*((cast729*((float)(wmma192.x)))+(cast730*((float)(wmma193.x)))))+(cast737*cast739*((cast731*((float)(wmma194.x)))+(cast732*((float)(wmma195.x)))))+(cast737*cast740*((cast733*((float)(wmma196.x)))+(cast734*((float)(wmma197.x)))))+(cast737*cast741*((cast735*((float)(wmma198.x)))+(cast736*((float)(wmma199.x))))));
    if (0) {
      *(data0_5570560+(alu78+65)) = buf34;
    }
    float cast742 = tg_bitcast<float>((unsigned int)(val721));
    float cast743 = tg_bitcast<float>((unsigned int)(val722));
    float cast744 = tg_bitcast<float>((unsigned int)(val723));
    float cast745 = tg_bitcast<float>((unsigned int)(val724));
    buf34 = (buf34+(cast737*cast742*((cast729*((float)(wmma192.y)))+(cast730*((float)(wmma193.y)))))+(cast737*cast743*((cast731*((float)(wmma194.y)))+(cast732*((float)(wmma195.y)))))+(cast737*cast744*((cast733*((float)(wmma196.y)))+(cast734*((float)(wmma197.y)))))+(cast737*cast745*((cast735*((float)(wmma198.y)))+(cast736*((float)(wmma199.y))))));
    if (0) {
      *(data0_5570560+(alu78+1088)) = buf35;
    }
    float cast746 = ((float)(((signed char)(((val729>>0u)&255u)))));
    float cast747 = ((float)(((signed char)(((val729>>8u)&255u)))));
    float cast748 = ((float)(((signed char)(((val729>>16u)&255u)))));
    float cast749 = ((float)(((signed char)(((val729>>24u)&255u)))));
    float cast750 = ((float)(((signed char)(((val730>>0u)&255u)))));
    float cast751 = ((float)(((signed char)(((val730>>8u)&255u)))));
    float cast752 = ((float)(((signed char)(((val730>>16u)&255u)))));
    float cast753 = ((float)(((signed char)(((val730>>24u)&255u)))));
    float cast754 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val728&65535u)))))));
    buf35 = (buf35+(cast754*cast738*((cast746*((float)(wmma192.z)))+(cast747*((float)(wmma193.z)))))+(cast754*cast739*((cast748*((float)(wmma194.z)))+(cast749*((float)(wmma195.z)))))+(cast754*cast740*((cast750*((float)(wmma196.z)))+(cast751*((float)(wmma197.z)))))+(cast754*cast741*((cast752*((float)(wmma198.z)))+(cast753*((float)(wmma199.z))))));
    if (0) {
      *(data0_5570560+(alu78+1089)) = buf36;
    }
    buf36 = (buf36+(cast754*cast742*((cast746*((float)(wmma192.w)))+(cast747*((float)(wmma193.w)))))+(cast754*cast743*((cast748*((float)(wmma194.w)))+(cast749*((float)(wmma195.w)))))+(cast754*cast744*((cast750*((float)(wmma196.w)))+(cast751*((float)(wmma197.w)))))+(cast754*cast745*((cast752*((float)(wmma198.w)))+(cast753*((float)(wmma199.w))))));
    unsigned int val731 = (*(buf0+alu201));
    unsigned int val732 = (*(buf0+alu202));
    unsigned int val733 = (*(buf0+alu203));
    unsigned int val734 = (*(buf0+alu204));
    unsigned int val735 = (*(buf0+alu205));
    unsigned int val736 = (*(buf0+alu206));
    unsigned int val737 = (*(buf0+alu207));
    unsigned int val738 = (*(buf0+alu208));
    unsigned int val739 = (*(buf0+alu158));
    unsigned int val740 = (*(buf0+alu161));
    unsigned int val741 = (*(buf0+alu162));
    unsigned int val742 = (*(buf0+alu163));
    unsigned int val743 = (*(buf0+alu166));
    unsigned int val744 = (*(buf0+alu167));
    if (0) {
      *(data0_5570560+(alu78+2112)) = buf37;
    }
    int4 wmma200 = __WMMA_8_16_16_signed_char_int(alu952, cast721, cast0);
    int4 wmma201 = __WMMA_8_16_16_signed_char_int(alu953, cast722, cast0);
    int4 wmma202 = __WMMA_8_16_16_signed_char_int(alu954, cast723, cast0);
    int4 wmma203 = __WMMA_8_16_16_signed_char_int(alu955, cast724, cast0);
    int4 wmma204 = __WMMA_8_16_16_signed_char_int(alu956, cast725, cast0);
    int4 wmma205 = __WMMA_8_16_16_signed_char_int(alu957, cast726, cast0);
    int4 wmma206 = __WMMA_8_16_16_signed_char_int(alu958, cast727, cast0);
    int4 wmma207 = __WMMA_8_16_16_signed_char_int(alu959, cast728, cast0);
    float cast755 = ((float)(((signed char)(((val740>>0u)&255u)))));
    float cast756 = ((float)(((signed char)(((val740>>8u)&255u)))));
    float cast757 = ((float)(((signed char)(((val740>>16u)&255u)))));
    float cast758 = ((float)(((signed char)(((val740>>24u)&255u)))));
    float cast759 = ((float)(((signed char)(((val741>>0u)&255u)))));
    float cast760 = ((float)(((signed char)(((val741>>8u)&255u)))));
    float cast761 = ((float)(((signed char)(((val741>>16u)&255u)))));
    float cast762 = ((float)(((signed char)(((val741>>24u)&255u)))));
    float cast763 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val739&65535u)))))));
    float cast764 = tg_bitcast<float>((unsigned int)(val731));
    float cast765 = tg_bitcast<float>((unsigned int)(val732));
    float cast766 = tg_bitcast<float>((unsigned int)(val733));
    float cast767 = tg_bitcast<float>((unsigned int)(val734));
    buf37 = (buf37+(cast763*cast764*((cast755*((float)(wmma200.x)))+(cast756*((float)(wmma201.x)))))+(cast763*cast765*((cast757*((float)(wmma202.x)))+(cast758*((float)(wmma203.x)))))+(cast763*cast766*((cast759*((float)(wmma204.x)))+(cast760*((float)(wmma205.x)))))+(cast763*cast767*((cast761*((float)(wmma206.x)))+(cast762*((float)(wmma207.x))))));
    if (0) {
      *(data0_5570560+(alu78+2113)) = buf38;
    }
    float cast768 = tg_bitcast<float>((unsigned int)(val735));
    float cast769 = tg_bitcast<float>((unsigned int)(val736));
    float cast770 = tg_bitcast<float>((unsigned int)(val737));
    float cast771 = tg_bitcast<float>((unsigned int)(val738));
    buf38 = (buf38+(cast763*cast768*((cast755*((float)(wmma200.y)))+(cast756*((float)(wmma201.y)))))+(cast763*cast769*((cast757*((float)(wmma202.y)))+(cast758*((float)(wmma203.y)))))+(cast763*cast770*((cast759*((float)(wmma204.y)))+(cast760*((float)(wmma205.y)))))+(cast763*cast771*((cast761*((float)(wmma206.y)))+(cast762*((float)(wmma207.y))))));
    if (0) {
      *(data0_5570560+(alu78+3136)) = buf39;
    }
    float cast772 = ((float)(((signed char)(((val743>>0u)&255u)))));
    float cast773 = ((float)(((signed char)(((val743>>8u)&255u)))));
    float cast774 = ((float)(((signed char)(((val743>>16u)&255u)))));
    float cast775 = ((float)(((signed char)(((val743>>24u)&255u)))));
    float cast776 = ((float)(((signed char)(((val744>>0u)&255u)))));
    float cast777 = ((float)(((signed char)(((val744>>8u)&255u)))));
    float cast778 = ((float)(((signed char)(((val744>>16u)&255u)))));
    float cast779 = ((float)(((signed char)(((val744>>24u)&255u)))));
    float cast780 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val742&65535u)))))));
    buf39 = (buf39+(cast780*cast764*((cast772*((float)(wmma200.z)))+(cast773*((float)(wmma201.z)))))+(cast780*cast765*((cast774*((float)(wmma202.z)))+(cast775*((float)(wmma203.z)))))+(cast780*cast766*((cast776*((float)(wmma204.z)))+(cast777*((float)(wmma205.z)))))+(cast780*cast767*((cast778*((float)(wmma206.z)))+(cast779*((float)(wmma207.z))))));
    if (0) {
      *(data0_5570560+(alu78+3137)) = buf40;
    }
    buf40 = (buf40+(cast780*cast768*((cast772*((float)(wmma200.w)))+(cast773*((float)(wmma201.w)))))+(cast780*cast769*((cast774*((float)(wmma202.w)))+(cast775*((float)(wmma203.w)))))+(cast780*cast770*((cast776*((float)(wmma204.w)))+(cast777*((float)(wmma205.w)))))+(cast780*cast771*((cast778*((float)(wmma206.w)))+(cast779*((float)(wmma207.w))))));
    unsigned int val745 = (*(buf0+alu123));
    unsigned int val746 = (*(buf0+alu124));
    unsigned int val747 = (*(buf0+alu125));
    unsigned int val748 = (*(buf0+alu126));
    unsigned int val749 = (*(buf0+alu127));
    unsigned int val750 = (*(buf0+alu128));
    unsigned int val751 = (*(buf0+alu129));
    unsigned int val752 = (*(buf0+alu130));
    unsigned int val753 = (*(buf0+alu209));
    unsigned int val754 = (*(buf0+alu210));
    unsigned int val755 = (*(buf0+alu211));
    unsigned int val756 = (*(buf0+alu212));
    unsigned int val757 = (*(buf0+alu213));
    unsigned int val758 = (*(buf0+alu214));
    unsigned int val759 = (*(buf0+alu215));
    unsigned int val760 = (*(buf0+alu216));
    unsigned int val761 = (*(buf0+alu148));
    unsigned int val762 = (*(buf0+alu151));
    unsigned int val763 = (*(buf0+alu152));
    unsigned int val764 = (*(buf0+alu153));
    unsigned int val765 = (*(buf0+alu156));
    unsigned int val766 = (*(buf0+alu157));
    if (0) {
      *(data0_5570560+(alu78+80)) = buf41;
    }
    char4 cast781 = make_char4(((signed char)(((val745>>0u)&255u))),((signed char)(((val745>>8u)&255u))),((signed char)(((val745>>16u)&255u))),((signed char)(((val745>>24u)&255u))));
    char4 cast782 = make_char4(((signed char)(((val746>>0u)&255u))),((signed char)(((val746>>8u)&255u))),((signed char)(((val746>>16u)&255u))),((signed char)(((val746>>24u)&255u))));
    char4 cast783 = make_char4(((signed char)(((val747>>0u)&255u))),((signed char)(((val747>>8u)&255u))),((signed char)(((val747>>16u)&255u))),((signed char)(((val747>>24u)&255u))));
    char4 cast784 = make_char4(((signed char)(((val748>>0u)&255u))),((signed char)(((val748>>8u)&255u))),((signed char)(((val748>>16u)&255u))),((signed char)(((val748>>24u)&255u))));
    char4 cast785 = make_char4(((signed char)(((val749>>0u)&255u))),((signed char)(((val749>>8u)&255u))),((signed char)(((val749>>16u)&255u))),((signed char)(((val749>>24u)&255u))));
    char4 cast786 = make_char4(((signed char)(((val750>>0u)&255u))),((signed char)(((val750>>8u)&255u))),((signed char)(((val750>>16u)&255u))),((signed char)(((val750>>24u)&255u))));
    char4 cast787 = make_char4(((signed char)(((val751>>0u)&255u))),((signed char)(((val751>>8u)&255u))),((signed char)(((val751>>16u)&255u))),((signed char)(((val751>>24u)&255u))));
    char4 cast788 = make_char4(((signed char)(((val752>>0u)&255u))),((signed char)(((val752>>8u)&255u))),((signed char)(((val752>>16u)&255u))),((signed char)(((val752>>24u)&255u))));
    int4 wmma208 = __WMMA_8_16_16_signed_char_int(alu928, cast781, cast0);
    int4 wmma209 = __WMMA_8_16_16_signed_char_int(alu929, cast782, cast0);
    int4 wmma210 = __WMMA_8_16_16_signed_char_int(alu930, cast783, cast0);
    int4 wmma211 = __WMMA_8_16_16_signed_char_int(alu931, cast784, cast0);
    int4 wmma212 = __WMMA_8_16_16_signed_char_int(alu932, cast785, cast0);
    int4 wmma213 = __WMMA_8_16_16_signed_char_int(alu933, cast786, cast0);
    int4 wmma214 = __WMMA_8_16_16_signed_char_int(alu934, cast787, cast0);
    int4 wmma215 = __WMMA_8_16_16_signed_char_int(alu935, cast788, cast0);
    float cast789 = ((float)(((signed char)(((val762>>0u)&255u)))));
    float cast790 = ((float)(((signed char)(((val762>>8u)&255u)))));
    float cast791 = ((float)(((signed char)(((val762>>16u)&255u)))));
    float cast792 = ((float)(((signed char)(((val762>>24u)&255u)))));
    float cast793 = ((float)(((signed char)(((val763>>0u)&255u)))));
    float cast794 = ((float)(((signed char)(((val763>>8u)&255u)))));
    float cast795 = ((float)(((signed char)(((val763>>16u)&255u)))));
    float cast796 = ((float)(((signed char)(((val763>>24u)&255u)))));
    float cast797 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val761&65535u)))))));
    float cast798 = tg_bitcast<float>((unsigned int)(val753));
    float cast799 = tg_bitcast<float>((unsigned int)(val754));
    float cast800 = tg_bitcast<float>((unsigned int)(val755));
    float cast801 = tg_bitcast<float>((unsigned int)(val756));
    buf41 = (buf41+(cast797*cast798*((cast789*((float)(wmma208.x)))+(cast790*((float)(wmma209.x)))))+(cast797*cast799*((cast791*((float)(wmma210.x)))+(cast792*((float)(wmma211.x)))))+(cast797*cast800*((cast793*((float)(wmma212.x)))+(cast794*((float)(wmma213.x)))))+(cast797*cast801*((cast795*((float)(wmma214.x)))+(cast796*((float)(wmma215.x))))));
    if (0) {
      *(data0_5570560+(alu78+81)) = buf42;
    }
    float cast802 = tg_bitcast<float>((unsigned int)(val757));
    float cast803 = tg_bitcast<float>((unsigned int)(val758));
    float cast804 = tg_bitcast<float>((unsigned int)(val759));
    float cast805 = tg_bitcast<float>((unsigned int)(val760));
    buf42 = (buf42+(cast797*cast802*((cast789*((float)(wmma208.y)))+(cast790*((float)(wmma209.y)))))+(cast797*cast803*((cast791*((float)(wmma210.y)))+(cast792*((float)(wmma211.y)))))+(cast797*cast804*((cast793*((float)(wmma212.y)))+(cast794*((float)(wmma213.y)))))+(cast797*cast805*((cast795*((float)(wmma214.y)))+(cast796*((float)(wmma215.y))))));
    if (0) {
      *(data0_5570560+(alu78+1104)) = buf43;
    }
    float cast806 = ((float)(((signed char)(((val765>>0u)&255u)))));
    float cast807 = ((float)(((signed char)(((val765>>8u)&255u)))));
    float cast808 = ((float)(((signed char)(((val765>>16u)&255u)))));
    float cast809 = ((float)(((signed char)(((val765>>24u)&255u)))));
    float cast810 = ((float)(((signed char)(((val766>>0u)&255u)))));
    float cast811 = ((float)(((signed char)(((val766>>8u)&255u)))));
    float cast812 = ((float)(((signed char)(((val766>>16u)&255u)))));
    float cast813 = ((float)(((signed char)(((val766>>24u)&255u)))));
    float cast814 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val764&65535u)))))));
    buf43 = (buf43+(cast814*cast798*((cast806*((float)(wmma208.z)))+(cast807*((float)(wmma209.z)))))+(cast814*cast799*((cast808*((float)(wmma210.z)))+(cast809*((float)(wmma211.z)))))+(cast814*cast800*((cast810*((float)(wmma212.z)))+(cast811*((float)(wmma213.z)))))+(cast814*cast801*((cast812*((float)(wmma214.z)))+(cast813*((float)(wmma215.z))))));
    if (0) {
      *(data0_5570560+(alu78+1105)) = buf44;
    }
    buf44 = (buf44+(cast814*cast802*((cast806*((float)(wmma208.w)))+(cast807*((float)(wmma209.w)))))+(cast814*cast803*((cast808*((float)(wmma210.w)))+(cast809*((float)(wmma211.w)))))+(cast814*cast804*((cast810*((float)(wmma212.w)))+(cast811*((float)(wmma213.w)))))+(cast814*cast805*((cast812*((float)(wmma214.w)))+(cast813*((float)(wmma215.w))))));
    unsigned int val767 = (*(buf0+alu209));
    unsigned int val768 = (*(buf0+alu210));
    unsigned int val769 = (*(buf0+alu211));
    unsigned int val770 = (*(buf0+alu212));
    unsigned int val771 = (*(buf0+alu213));
    unsigned int val772 = (*(buf0+alu214));
    unsigned int val773 = (*(buf0+alu215));
    unsigned int val774 = (*(buf0+alu216));
    unsigned int val775 = (*(buf0+alu158));
    unsigned int val776 = (*(buf0+alu161));
    unsigned int val777 = (*(buf0+alu162));
    unsigned int val778 = (*(buf0+alu163));
    unsigned int val779 = (*(buf0+alu166));
    unsigned int val780 = (*(buf0+alu167));
    if (0) {
      *(data0_5570560+(alu78+2128)) = buf45;
    }
    int4 wmma216 = __WMMA_8_16_16_signed_char_int(alu952, cast781, cast0);
    int4 wmma217 = __WMMA_8_16_16_signed_char_int(alu953, cast782, cast0);
    int4 wmma218 = __WMMA_8_16_16_signed_char_int(alu954, cast783, cast0);
    int4 wmma219 = __WMMA_8_16_16_signed_char_int(alu955, cast784, cast0);
    int4 wmma220 = __WMMA_8_16_16_signed_char_int(alu956, cast785, cast0);
    int4 wmma221 = __WMMA_8_16_16_signed_char_int(alu957, cast786, cast0);
    int4 wmma222 = __WMMA_8_16_16_signed_char_int(alu958, cast787, cast0);
    int4 wmma223 = __WMMA_8_16_16_signed_char_int(alu959, cast788, cast0);
    float cast815 = ((float)(((signed char)(((val776>>0u)&255u)))));
    float cast816 = ((float)(((signed char)(((val776>>8u)&255u)))));
    float cast817 = ((float)(((signed char)(((val776>>16u)&255u)))));
    float cast818 = ((float)(((signed char)(((val776>>24u)&255u)))));
    float cast819 = ((float)(((signed char)(((val777>>0u)&255u)))));
    float cast820 = ((float)(((signed char)(((val777>>8u)&255u)))));
    float cast821 = ((float)(((signed char)(((val777>>16u)&255u)))));
    float cast822 = ((float)(((signed char)(((val777>>24u)&255u)))));
    float cast823 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val775&65535u)))))));
    float cast824 = tg_bitcast<float>((unsigned int)(val767));
    float cast825 = tg_bitcast<float>((unsigned int)(val768));
    float cast826 = tg_bitcast<float>((unsigned int)(val769));
    float cast827 = tg_bitcast<float>((unsigned int)(val770));
    buf45 = (buf45+(cast823*cast824*((cast815*((float)(wmma216.x)))+(cast816*((float)(wmma217.x)))))+(cast823*cast825*((cast817*((float)(wmma218.x)))+(cast818*((float)(wmma219.x)))))+(cast823*cast826*((cast819*((float)(wmma220.x)))+(cast820*((float)(wmma221.x)))))+(cast823*cast827*((cast821*((float)(wmma222.x)))+(cast822*((float)(wmma223.x))))));
    if (0) {
      *(data0_5570560+(alu78+2129)) = buf46;
    }
    float cast828 = tg_bitcast<float>((unsigned int)(val771));
    float cast829 = tg_bitcast<float>((unsigned int)(val772));
    float cast830 = tg_bitcast<float>((unsigned int)(val773));
    float cast831 = tg_bitcast<float>((unsigned int)(val774));
    buf46 = (buf46+(cast823*cast828*((cast815*((float)(wmma216.y)))+(cast816*((float)(wmma217.y)))))+(cast823*cast829*((cast817*((float)(wmma218.y)))+(cast818*((float)(wmma219.y)))))+(cast823*cast830*((cast819*((float)(wmma220.y)))+(cast820*((float)(wmma221.y)))))+(cast823*cast831*((cast821*((float)(wmma222.y)))+(cast822*((float)(wmma223.y))))));
    if (0) {
      *(data0_5570560+(alu78+3152)) = buf47;
    }
    float cast832 = ((float)(((signed char)(((val779>>0u)&255u)))));
    float cast833 = ((float)(((signed char)(((val779>>8u)&255u)))));
    float cast834 = ((float)(((signed char)(((val779>>16u)&255u)))));
    float cast835 = ((float)(((signed char)(((val779>>24u)&255u)))));
    float cast836 = ((float)(((signed char)(((val780>>0u)&255u)))));
    float cast837 = ((float)(((signed char)(((val780>>8u)&255u)))));
    float cast838 = ((float)(((signed char)(((val780>>16u)&255u)))));
    float cast839 = ((float)(((signed char)(((val780>>24u)&255u)))));
    float cast840 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val778&65535u)))))));
    buf47 = (buf47+(cast840*cast824*((cast832*((float)(wmma216.z)))+(cast833*((float)(wmma217.z)))))+(cast840*cast825*((cast834*((float)(wmma218.z)))+(cast835*((float)(wmma219.z)))))+(cast840*cast826*((cast836*((float)(wmma220.z)))+(cast837*((float)(wmma221.z)))))+(cast840*cast827*((cast838*((float)(wmma222.z)))+(cast839*((float)(wmma223.z))))));
    if (0) {
      *(data0_5570560+(alu78+3153)) = buf48;
    }
    buf48 = (buf48+(cast840*cast828*((cast832*((float)(wmma216.w)))+(cast833*((float)(wmma217.w)))))+(cast840*cast829*((cast834*((float)(wmma218.w)))+(cast835*((float)(wmma219.w)))))+(cast840*cast830*((cast836*((float)(wmma220.w)))+(cast837*((float)(wmma221.w)))))+(cast840*cast831*((cast838*((float)(wmma222.w)))+(cast839*((float)(wmma223.w))))));
    unsigned int val781 = (*(buf0+alu131));
    unsigned int val782 = (*(buf0+alu132));
    unsigned int val783 = (*(buf0+alu133));
    unsigned int val784 = (*(buf0+alu134));
    unsigned int val785 = (*(buf0+alu135));
    unsigned int val786 = (*(buf0+alu136));
    unsigned int val787 = (*(buf0+alu137));
    unsigned int val788 = (*(buf0+alu138));
    unsigned int val789 = (*(buf0+alu217));
    unsigned int val790 = (*(buf0+alu218));
    unsigned int val791 = (*(buf0+alu219));
    unsigned int val792 = (*(buf0+alu220));
    unsigned int val793 = (*(buf0+alu221));
    unsigned int val794 = (*(buf0+alu222));
    unsigned int val795 = (*(buf0+alu223));
    unsigned int val796 = (*(buf0+alu224));
    unsigned int val797 = (*(buf0+alu148));
    unsigned int val798 = (*(buf0+alu151));
    unsigned int val799 = (*(buf0+alu152));
    unsigned int val800 = (*(buf0+alu153));
    unsigned int val801 = (*(buf0+alu156));
    unsigned int val802 = (*(buf0+alu157));
    if (0) {
      *(data0_5570560+(alu78+96)) = buf49;
    }
    char4 cast841 = make_char4(((signed char)(((val781>>0u)&255u))),((signed char)(((val781>>8u)&255u))),((signed char)(((val781>>16u)&255u))),((signed char)(((val781>>24u)&255u))));
    char4 cast842 = make_char4(((signed char)(((val782>>0u)&255u))),((signed char)(((val782>>8u)&255u))),((signed char)(((val782>>16u)&255u))),((signed char)(((val782>>24u)&255u))));
    char4 cast843 = make_char4(((signed char)(((val783>>0u)&255u))),((signed char)(((val783>>8u)&255u))),((signed char)(((val783>>16u)&255u))),((signed char)(((val783>>24u)&255u))));
    char4 cast844 = make_char4(((signed char)(((val784>>0u)&255u))),((signed char)(((val784>>8u)&255u))),((signed char)(((val784>>16u)&255u))),((signed char)(((val784>>24u)&255u))));
    char4 cast845 = make_char4(((signed char)(((val785>>0u)&255u))),((signed char)(((val785>>8u)&255u))),((signed char)(((val785>>16u)&255u))),((signed char)(((val785>>24u)&255u))));
    char4 cast846 = make_char4(((signed char)(((val786>>0u)&255u))),((signed char)(((val786>>8u)&255u))),((signed char)(((val786>>16u)&255u))),((signed char)(((val786>>24u)&255u))));
    char4 cast847 = make_char4(((signed char)(((val787>>0u)&255u))),((signed char)(((val787>>8u)&255u))),((signed char)(((val787>>16u)&255u))),((signed char)(((val787>>24u)&255u))));
    char4 cast848 = make_char4(((signed char)(((val788>>0u)&255u))),((signed char)(((val788>>8u)&255u))),((signed char)(((val788>>16u)&255u))),((signed char)(((val788>>24u)&255u))));
    int4 wmma224 = __WMMA_8_16_16_signed_char_int(alu928, cast841, cast0);
    int4 wmma225 = __WMMA_8_16_16_signed_char_int(alu929, cast842, cast0);
    int4 wmma226 = __WMMA_8_16_16_signed_char_int(alu930, cast843, cast0);
    int4 wmma227 = __WMMA_8_16_16_signed_char_int(alu931, cast844, cast0);
    int4 wmma228 = __WMMA_8_16_16_signed_char_int(alu932, cast845, cast0);
    int4 wmma229 = __WMMA_8_16_16_signed_char_int(alu933, cast846, cast0);
    int4 wmma230 = __WMMA_8_16_16_signed_char_int(alu934, cast847, cast0);
    int4 wmma231 = __WMMA_8_16_16_signed_char_int(alu935, cast848, cast0);
    float cast849 = ((float)(((signed char)(((val798>>0u)&255u)))));
    float cast850 = ((float)(((signed char)(((val798>>8u)&255u)))));
    float cast851 = ((float)(((signed char)(((val798>>16u)&255u)))));
    float cast852 = ((float)(((signed char)(((val798>>24u)&255u)))));
    float cast853 = ((float)(((signed char)(((val799>>0u)&255u)))));
    float cast854 = ((float)(((signed char)(((val799>>8u)&255u)))));
    float cast855 = ((float)(((signed char)(((val799>>16u)&255u)))));
    float cast856 = ((float)(((signed char)(((val799>>24u)&255u)))));
    float cast857 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val797&65535u)))))));
    float cast858 = tg_bitcast<float>((unsigned int)(val789));
    float cast859 = tg_bitcast<float>((unsigned int)(val790));
    float cast860 = tg_bitcast<float>((unsigned int)(val791));
    float cast861 = tg_bitcast<float>((unsigned int)(val792));
    buf49 = (buf49+(cast857*cast858*((cast849*((float)(wmma224.x)))+(cast850*((float)(wmma225.x)))))+(cast857*cast859*((cast851*((float)(wmma226.x)))+(cast852*((float)(wmma227.x)))))+(cast857*cast860*((cast853*((float)(wmma228.x)))+(cast854*((float)(wmma229.x)))))+(cast857*cast861*((cast855*((float)(wmma230.x)))+(cast856*((float)(wmma231.x))))));
    if (0) {
      *(data0_5570560+(alu78+97)) = buf50;
    }
    float cast862 = tg_bitcast<float>((unsigned int)(val793));
    float cast863 = tg_bitcast<float>((unsigned int)(val794));
    float cast864 = tg_bitcast<float>((unsigned int)(val795));
    float cast865 = tg_bitcast<float>((unsigned int)(val796));
    buf50 = (buf50+(cast857*cast862*((cast849*((float)(wmma224.y)))+(cast850*((float)(wmma225.y)))))+(cast857*cast863*((cast851*((float)(wmma226.y)))+(cast852*((float)(wmma227.y)))))+(cast857*cast864*((cast853*((float)(wmma228.y)))+(cast854*((float)(wmma229.y)))))+(cast857*cast865*((cast855*((float)(wmma230.y)))+(cast856*((float)(wmma231.y))))));
    if (0) {
      *(data0_5570560+(alu78+1120)) = buf51;
    }
    float cast866 = ((float)(((signed char)(((val801>>0u)&255u)))));
    float cast867 = ((float)(((signed char)(((val801>>8u)&255u)))));
    float cast868 = ((float)(((signed char)(((val801>>16u)&255u)))));
    float cast869 = ((float)(((signed char)(((val801>>24u)&255u)))));
    float cast870 = ((float)(((signed char)(((val802>>0u)&255u)))));
    float cast871 = ((float)(((signed char)(((val802>>8u)&255u)))));
    float cast872 = ((float)(((signed char)(((val802>>16u)&255u)))));
    float cast873 = ((float)(((signed char)(((val802>>24u)&255u)))));
    float cast874 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val800&65535u)))))));
    buf51 = (buf51+(cast874*cast858*((cast866*((float)(wmma224.z)))+(cast867*((float)(wmma225.z)))))+(cast874*cast859*((cast868*((float)(wmma226.z)))+(cast869*((float)(wmma227.z)))))+(cast874*cast860*((cast870*((float)(wmma228.z)))+(cast871*((float)(wmma229.z)))))+(cast874*cast861*((cast872*((float)(wmma230.z)))+(cast873*((float)(wmma231.z))))));
    if (0) {
      *(data0_5570560+(alu78+1121)) = buf52;
    }
    buf52 = (buf52+(cast874*cast862*((cast866*((float)(wmma224.w)))+(cast867*((float)(wmma225.w)))))+(cast874*cast863*((cast868*((float)(wmma226.w)))+(cast869*((float)(wmma227.w)))))+(cast874*cast864*((cast870*((float)(wmma228.w)))+(cast871*((float)(wmma229.w)))))+(cast874*cast865*((cast872*((float)(wmma230.w)))+(cast873*((float)(wmma231.w))))));
    unsigned int val803 = (*(buf0+alu217));
    unsigned int val804 = (*(buf0+alu218));
    unsigned int val805 = (*(buf0+alu219));
    unsigned int val806 = (*(buf0+alu220));
    unsigned int val807 = (*(buf0+alu221));
    unsigned int val808 = (*(buf0+alu222));
    unsigned int val809 = (*(buf0+alu223));
    unsigned int val810 = (*(buf0+alu224));
    unsigned int val811 = (*(buf0+alu158));
    unsigned int val812 = (*(buf0+alu161));
    unsigned int val813 = (*(buf0+alu162));
    unsigned int val814 = (*(buf0+alu163));
    unsigned int val815 = (*(buf0+alu166));
    unsigned int val816 = (*(buf0+alu167));
    if (0) {
      *(data0_5570560+(alu78+2144)) = buf53;
    }
    int4 wmma232 = __WMMA_8_16_16_signed_char_int(alu952, cast841, cast0);
    int4 wmma233 = __WMMA_8_16_16_signed_char_int(alu953, cast842, cast0);
    int4 wmma234 = __WMMA_8_16_16_signed_char_int(alu954, cast843, cast0);
    int4 wmma235 = __WMMA_8_16_16_signed_char_int(alu955, cast844, cast0);
    int4 wmma236 = __WMMA_8_16_16_signed_char_int(alu956, cast845, cast0);
    int4 wmma237 = __WMMA_8_16_16_signed_char_int(alu957, cast846, cast0);
    int4 wmma238 = __WMMA_8_16_16_signed_char_int(alu958, cast847, cast0);
    int4 wmma239 = __WMMA_8_16_16_signed_char_int(alu959, cast848, cast0);
    float cast875 = ((float)(((signed char)(((val812>>0u)&255u)))));
    float cast876 = ((float)(((signed char)(((val812>>8u)&255u)))));
    float cast877 = ((float)(((signed char)(((val812>>16u)&255u)))));
    float cast878 = ((float)(((signed char)(((val812>>24u)&255u)))));
    float cast879 = ((float)(((signed char)(((val813>>0u)&255u)))));
    float cast880 = ((float)(((signed char)(((val813>>8u)&255u)))));
    float cast881 = ((float)(((signed char)(((val813>>16u)&255u)))));
    float cast882 = ((float)(((signed char)(((val813>>24u)&255u)))));
    float cast883 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val811&65535u)))))));
    float cast884 = tg_bitcast<float>((unsigned int)(val803));
    float cast885 = tg_bitcast<float>((unsigned int)(val804));
    float cast886 = tg_bitcast<float>((unsigned int)(val805));
    float cast887 = tg_bitcast<float>((unsigned int)(val806));
    buf53 = (buf53+(cast883*cast884*((cast875*((float)(wmma232.x)))+(cast876*((float)(wmma233.x)))))+(cast883*cast885*((cast877*((float)(wmma234.x)))+(cast878*((float)(wmma235.x)))))+(cast883*cast886*((cast879*((float)(wmma236.x)))+(cast880*((float)(wmma237.x)))))+(cast883*cast887*((cast881*((float)(wmma238.x)))+(cast882*((float)(wmma239.x))))));
    if (0) {
      *(data0_5570560+(alu78+2145)) = buf54;
    }
    float cast888 = tg_bitcast<float>((unsigned int)(val807));
    float cast889 = tg_bitcast<float>((unsigned int)(val808));
    float cast890 = tg_bitcast<float>((unsigned int)(val809));
    float cast891 = tg_bitcast<float>((unsigned int)(val810));
    buf54 = (buf54+(cast883*cast888*((cast875*((float)(wmma232.y)))+(cast876*((float)(wmma233.y)))))+(cast883*cast889*((cast877*((float)(wmma234.y)))+(cast878*((float)(wmma235.y)))))+(cast883*cast890*((cast879*((float)(wmma236.y)))+(cast880*((float)(wmma237.y)))))+(cast883*cast891*((cast881*((float)(wmma238.y)))+(cast882*((float)(wmma239.y))))));
    if (0) {
      *(data0_5570560+(alu78+3168)) = buf55;
    }
    float cast892 = ((float)(((signed char)(((val815>>0u)&255u)))));
    float cast893 = ((float)(((signed char)(((val815>>8u)&255u)))));
    float cast894 = ((float)(((signed char)(((val815>>16u)&255u)))));
    float cast895 = ((float)(((signed char)(((val815>>24u)&255u)))));
    float cast896 = ((float)(((signed char)(((val816>>0u)&255u)))));
    float cast897 = ((float)(((signed char)(((val816>>8u)&255u)))));
    float cast898 = ((float)(((signed char)(((val816>>16u)&255u)))));
    float cast899 = ((float)(((signed char)(((val816>>24u)&255u)))));
    float cast900 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val814&65535u)))))));
    buf55 = (buf55+(cast900*cast884*((cast892*((float)(wmma232.z)))+(cast893*((float)(wmma233.z)))))+(cast900*cast885*((cast894*((float)(wmma234.z)))+(cast895*((float)(wmma235.z)))))+(cast900*cast886*((cast896*((float)(wmma236.z)))+(cast897*((float)(wmma237.z)))))+(cast900*cast887*((cast898*((float)(wmma238.z)))+(cast899*((float)(wmma239.z))))));
    if (0) {
      *(data0_5570560+(alu78+3169)) = buf56;
    }
    buf56 = (buf56+(cast900*cast888*((cast892*((float)(wmma232.w)))+(cast893*((float)(wmma233.w)))))+(cast900*cast889*((cast894*((float)(wmma234.w)))+(cast895*((float)(wmma235.w)))))+(cast900*cast890*((cast896*((float)(wmma236.w)))+(cast897*((float)(wmma237.w)))))+(cast900*cast891*((cast898*((float)(wmma238.w)))+(cast899*((float)(wmma239.w))))));
    unsigned int val817 = (*(buf0+alu139));
    unsigned int val818 = (*(buf0+alu140));
    unsigned int val819 = (*(buf0+alu141));
    unsigned int val820 = (*(buf0+alu142));
    unsigned int val821 = (*(buf0+alu143));
    unsigned int val822 = (*(buf0+alu144));
    unsigned int val823 = (*(buf0+alu145));
    unsigned int val824 = (*(buf0+alu146));
    unsigned int val825 = (*(buf0+alu225));
    unsigned int val826 = (*(buf0+alu226));
    unsigned int val827 = (*(buf0+alu227));
    unsigned int val828 = (*(buf0+alu228));
    unsigned int val829 = (*(buf0+alu229));
    unsigned int val830 = (*(buf0+alu230));
    unsigned int val831 = (*(buf0+alu231));
    unsigned int val832 = (*(buf0+alu232));
    unsigned int val833 = (*(buf0+alu148));
    unsigned int val834 = (*(buf0+alu151));
    unsigned int val835 = (*(buf0+alu152));
    unsigned int val836 = (*(buf0+alu153));
    unsigned int val837 = (*(buf0+alu156));
    unsigned int val838 = (*(buf0+alu157));
    if (0) {
      *(data0_5570560+(alu78+112)) = buf57;
    }
    char4 cast901 = make_char4(((signed char)(((val817>>0u)&255u))),((signed char)(((val817>>8u)&255u))),((signed char)(((val817>>16u)&255u))),((signed char)(((val817>>24u)&255u))));
    char4 cast902 = make_char4(((signed char)(((val818>>0u)&255u))),((signed char)(((val818>>8u)&255u))),((signed char)(((val818>>16u)&255u))),((signed char)(((val818>>24u)&255u))));
    char4 cast903 = make_char4(((signed char)(((val819>>0u)&255u))),((signed char)(((val819>>8u)&255u))),((signed char)(((val819>>16u)&255u))),((signed char)(((val819>>24u)&255u))));
    char4 cast904 = make_char4(((signed char)(((val820>>0u)&255u))),((signed char)(((val820>>8u)&255u))),((signed char)(((val820>>16u)&255u))),((signed char)(((val820>>24u)&255u))));
    char4 cast905 = make_char4(((signed char)(((val821>>0u)&255u))),((signed char)(((val821>>8u)&255u))),((signed char)(((val821>>16u)&255u))),((signed char)(((val821>>24u)&255u))));
    char4 cast906 = make_char4(((signed char)(((val822>>0u)&255u))),((signed char)(((val822>>8u)&255u))),((signed char)(((val822>>16u)&255u))),((signed char)(((val822>>24u)&255u))));
    char4 cast907 = make_char4(((signed char)(((val823>>0u)&255u))),((signed char)(((val823>>8u)&255u))),((signed char)(((val823>>16u)&255u))),((signed char)(((val823>>24u)&255u))));
    char4 cast908 = make_char4(((signed char)(((val824>>0u)&255u))),((signed char)(((val824>>8u)&255u))),((signed char)(((val824>>16u)&255u))),((signed char)(((val824>>24u)&255u))));
    int4 wmma240 = __WMMA_8_16_16_signed_char_int(alu928, cast901, cast0);
    int4 wmma241 = __WMMA_8_16_16_signed_char_int(alu929, cast902, cast0);
    int4 wmma242 = __WMMA_8_16_16_signed_char_int(alu930, cast903, cast0);
    int4 wmma243 = __WMMA_8_16_16_signed_char_int(alu931, cast904, cast0);
    int4 wmma244 = __WMMA_8_16_16_signed_char_int(alu932, cast905, cast0);
    int4 wmma245 = __WMMA_8_16_16_signed_char_int(alu933, cast906, cast0);
    int4 wmma246 = __WMMA_8_16_16_signed_char_int(alu934, cast907, cast0);
    int4 wmma247 = __WMMA_8_16_16_signed_char_int(alu935, cast908, cast0);
    float cast909 = ((float)(((signed char)(((val834>>0u)&255u)))));
    float cast910 = ((float)(((signed char)(((val834>>8u)&255u)))));
    float cast911 = ((float)(((signed char)(((val834>>16u)&255u)))));
    float cast912 = ((float)(((signed char)(((val834>>24u)&255u)))));
    float cast913 = ((float)(((signed char)(((val835>>0u)&255u)))));
    float cast914 = ((float)(((signed char)(((val835>>8u)&255u)))));
    float cast915 = ((float)(((signed char)(((val835>>16u)&255u)))));
    float cast916 = ((float)(((signed char)(((val835>>24u)&255u)))));
    float cast917 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val833&65535u)))))));
    float cast918 = tg_bitcast<float>((unsigned int)(val825));
    float cast919 = tg_bitcast<float>((unsigned int)(val826));
    float cast920 = tg_bitcast<float>((unsigned int)(val827));
    float cast921 = tg_bitcast<float>((unsigned int)(val828));
    buf57 = (buf57+(cast917*cast918*((cast909*((float)(wmma240.x)))+(cast910*((float)(wmma241.x)))))+(cast917*cast919*((cast911*((float)(wmma242.x)))+(cast912*((float)(wmma243.x)))))+(cast917*cast920*((cast913*((float)(wmma244.x)))+(cast914*((float)(wmma245.x)))))+(cast917*cast921*((cast915*((float)(wmma246.x)))+(cast916*((float)(wmma247.x))))));
    if (0) {
      *(data0_5570560+(alu78+113)) = buf58;
    }
    float cast922 = tg_bitcast<float>((unsigned int)(val829));
    float cast923 = tg_bitcast<float>((unsigned int)(val830));
    float cast924 = tg_bitcast<float>((unsigned int)(val831));
    float cast925 = tg_bitcast<float>((unsigned int)(val832));
    buf58 = (buf58+(cast917*cast922*((cast909*((float)(wmma240.y)))+(cast910*((float)(wmma241.y)))))+(cast917*cast923*((cast911*((float)(wmma242.y)))+(cast912*((float)(wmma243.y)))))+(cast917*cast924*((cast913*((float)(wmma244.y)))+(cast914*((float)(wmma245.y)))))+(cast917*cast925*((cast915*((float)(wmma246.y)))+(cast916*((float)(wmma247.y))))));
    if (0) {
      *(data0_5570560+(alu78+1136)) = buf59;
    }
    float cast926 = ((float)(((signed char)(((val837>>0u)&255u)))));
    float cast927 = ((float)(((signed char)(((val837>>8u)&255u)))));
    float cast928 = ((float)(((signed char)(((val837>>16u)&255u)))));
    float cast929 = ((float)(((signed char)(((val837>>24u)&255u)))));
    float cast930 = ((float)(((signed char)(((val838>>0u)&255u)))));
    float cast931 = ((float)(((signed char)(((val838>>8u)&255u)))));
    float cast932 = ((float)(((signed char)(((val838>>16u)&255u)))));
    float cast933 = ((float)(((signed char)(((val838>>24u)&255u)))));
    float cast934 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val836&65535u)))))));
    buf59 = (buf59+(cast934*cast918*((cast926*((float)(wmma240.z)))+(cast927*((float)(wmma241.z)))))+(cast934*cast919*((cast928*((float)(wmma242.z)))+(cast929*((float)(wmma243.z)))))+(cast934*cast920*((cast930*((float)(wmma244.z)))+(cast931*((float)(wmma245.z)))))+(cast934*cast921*((cast932*((float)(wmma246.z)))+(cast933*((float)(wmma247.z))))));
    if (0) {
      *(data0_5570560+(alu78+1137)) = buf60;
    }
    buf60 = (buf60+(cast934*cast922*((cast926*((float)(wmma240.w)))+(cast927*((float)(wmma241.w)))))+(cast934*cast923*((cast928*((float)(wmma242.w)))+(cast929*((float)(wmma243.w)))))+(cast934*cast924*((cast930*((float)(wmma244.w)))+(cast931*((float)(wmma245.w)))))+(cast934*cast925*((cast932*((float)(wmma246.w)))+(cast933*((float)(wmma247.w))))));
    unsigned int val839 = (*(buf0+alu225));
    unsigned int val840 = (*(buf0+alu226));
    unsigned int val841 = (*(buf0+alu227));
    unsigned int val842 = (*(buf0+alu228));
    unsigned int val843 = (*(buf0+alu229));
    unsigned int val844 = (*(buf0+alu230));
    unsigned int val845 = (*(buf0+alu231));
    unsigned int val846 = (*(buf0+alu232));
    unsigned int val847 = (*(buf0+alu158));
    unsigned int val848 = (*(buf0+alu161));
    unsigned int val849 = (*(buf0+alu162));
    unsigned int val850 = (*(buf0+alu163));
    unsigned int val851 = (*(buf0+alu166));
    unsigned int val852 = (*(buf0+alu167));
    if (0) {
      *(data0_5570560+(alu78+2160)) = buf61;
    }
    int4 wmma248 = __WMMA_8_16_16_signed_char_int(alu952, cast901, cast0);
    int4 wmma249 = __WMMA_8_16_16_signed_char_int(alu953, cast902, cast0);
    int4 wmma250 = __WMMA_8_16_16_signed_char_int(alu954, cast903, cast0);
    int4 wmma251 = __WMMA_8_16_16_signed_char_int(alu955, cast904, cast0);
    int4 wmma252 = __WMMA_8_16_16_signed_char_int(alu956, cast905, cast0);
    int4 wmma253 = __WMMA_8_16_16_signed_char_int(alu957, cast906, cast0);
    int4 wmma254 = __WMMA_8_16_16_signed_char_int(alu958, cast907, cast0);
    int4 wmma255 = __WMMA_8_16_16_signed_char_int(alu959, cast908, cast0);
    float cast935 = ((float)(((signed char)(((val848>>0u)&255u)))));
    float cast936 = ((float)(((signed char)(((val848>>8u)&255u)))));
    float cast937 = ((float)(((signed char)(((val848>>16u)&255u)))));
    float cast938 = ((float)(((signed char)(((val848>>24u)&255u)))));
    float cast939 = ((float)(((signed char)(((val849>>0u)&255u)))));
    float cast940 = ((float)(((signed char)(((val849>>8u)&255u)))));
    float cast941 = ((float)(((signed char)(((val849>>16u)&255u)))));
    float cast942 = ((float)(((signed char)(((val849>>24u)&255u)))));
    float cast943 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val847&65535u)))))));
    float cast944 = tg_bitcast<float>((unsigned int)(val839));
    float cast945 = tg_bitcast<float>((unsigned int)(val840));
    float cast946 = tg_bitcast<float>((unsigned int)(val841));
    float cast947 = tg_bitcast<float>((unsigned int)(val842));
    buf61 = (buf61+(cast943*cast944*((cast935*((float)(wmma248.x)))+(cast936*((float)(wmma249.x)))))+(cast943*cast945*((cast937*((float)(wmma250.x)))+(cast938*((float)(wmma251.x)))))+(cast943*cast946*((cast939*((float)(wmma252.x)))+(cast940*((float)(wmma253.x)))))+(cast943*cast947*((cast941*((float)(wmma254.x)))+(cast942*((float)(wmma255.x))))));
    if (0) {
      *(data0_5570560+(alu78+2161)) = buf62;
    }
    float cast948 = tg_bitcast<float>((unsigned int)(val843));
    float cast949 = tg_bitcast<float>((unsigned int)(val844));
    float cast950 = tg_bitcast<float>((unsigned int)(val845));
    float cast951 = tg_bitcast<float>((unsigned int)(val846));
    buf62 = (buf62+(cast943*cast948*((cast935*((float)(wmma248.y)))+(cast936*((float)(wmma249.y)))))+(cast943*cast949*((cast937*((float)(wmma250.y)))+(cast938*((float)(wmma251.y)))))+(cast943*cast950*((cast939*((float)(wmma252.y)))+(cast940*((float)(wmma253.y)))))+(cast943*cast951*((cast941*((float)(wmma254.y)))+(cast942*((float)(wmma255.y))))));
    if (0) {
      *(data0_5570560+(alu78+3184)) = buf63;
    }
    float cast952 = ((float)(((signed char)(((val851>>0u)&255u)))));
    float cast953 = ((float)(((signed char)(((val851>>8u)&255u)))));
    float cast954 = ((float)(((signed char)(((val851>>16u)&255u)))));
    float cast955 = ((float)(((signed char)(((val851>>24u)&255u)))));
    float cast956 = ((float)(((signed char)(((val852>>0u)&255u)))));
    float cast957 = ((float)(((signed char)(((val852>>8u)&255u)))));
    float cast958 = ((float)(((signed char)(((val852>>16u)&255u)))));
    float cast959 = ((float)(((signed char)(((val852>>24u)&255u)))));
    float cast960 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val850&65535u)))))));
    buf63 = (buf63+(cast960*cast944*((cast952*((float)(wmma248.z)))+(cast953*((float)(wmma249.z)))))+(cast960*cast945*((cast954*((float)(wmma250.z)))+(cast955*((float)(wmma251.z)))))+(cast960*cast946*((cast956*((float)(wmma252.z)))+(cast957*((float)(wmma253.z)))))+(cast960*cast947*((cast958*((float)(wmma254.z)))+(cast959*((float)(wmma255.z))))));
    if (0) {
      *(data0_5570560+(alu78+3185)) = buf64;
    }
    buf64 = (buf64+(cast960*cast948*((cast952*((float)(wmma248.w)))+(cast953*((float)(wmma249.w)))))+(cast960*cast949*((cast954*((float)(wmma250.w)))+(cast955*((float)(wmma251.w)))))+(cast960*cast950*((cast956*((float)(wmma252.w)))+(cast957*((float)(wmma253.w)))))+(cast960*cast951*((cast958*((float)(wmma254.w)))+(cast959*((float)(wmma255.w))))));
  }
  int alu1198 = (gidx0<<1);
  int alu1199 = ((alu267+-170)/8160);
  int alu1200 = (alu266/8160);
  bool alu1201 = (alu1200!=alu1199);
  int alu1202 = ((alu1198+((int)(alu1201)))<<14);
  int alu1203 = (alu74+alu1202);
  int alu1204 = (alu1203+alu77);
  int alu1205 = (alu77+alu1203);
  int alu1206 = (alu1202+alu74);
  int alu1207 = (alu77+alu1206);
  int alu1208 = (alu1201?alu1199:-1);
  *(data1_340+(alu1198+1)) = alu1208;
  *(data1_340+alu1198) = alu1200;
  *(data0_5570560+(alu1204+1024)) = buf3;
  *(data0_5570560+(alu1204+2048)) = buf5;
  *(data0_5570560+(alu1204+3072)) = buf7;
  *(data0_5570560+(alu1205+1025)) = buf4;
  *(data0_5570560+(alu1205+1040)) = buf11;
  *(data0_5570560+(alu1205+1041)) = buf12;
  *(data0_5570560+(alu1205+1056)) = buf19;
  *(data0_5570560+(alu1205+1057)) = buf20;
  *(data0_5570560+(alu1205+1072)) = buf27;
  *(data0_5570560+(alu1205+1073)) = buf28;
  *(data0_5570560+(alu1205+1088)) = buf35;
  *(data0_5570560+(alu1205+1089)) = buf36;
  *(data0_5570560+(alu1205+1104)) = buf43;
  *(data0_5570560+(alu1205+1105)) = buf44;
  *(data0_5570560+(alu1205+1120)) = buf51;
  *(data0_5570560+(alu1205+1121)) = buf52;
  *(data0_5570560+(alu1205+1136)) = buf59;
  *(data0_5570560+(alu1205+1137)) = buf60;
  *(data0_5570560+(alu1205+2049)) = buf6;
  *(data0_5570560+(alu1205+2064)) = buf13;
  *(data0_5570560+(alu1205+2065)) = buf14;
  *(data0_5570560+(alu1205+2080)) = buf21;
  *(data0_5570560+(alu1205+2081)) = buf22;
  *(data0_5570560+(alu1205+2096)) = buf29;
  *(data0_5570560+(alu1205+2097)) = buf30;
  *(data0_5570560+(alu1205+2112)) = buf37;
  *(data0_5570560+(alu1205+2113)) = buf38;
  *(data0_5570560+(alu1205+2128)) = buf45;
  *(data0_5570560+(alu1205+2129)) = buf46;
  *(data0_5570560+(alu1205+2144)) = buf53;
  *(data0_5570560+(alu1205+2145)) = buf54;
  *(data0_5570560+(alu1205+2160)) = buf61;
  *(data0_5570560+(alu1205+2161)) = buf62;
  *(data0_5570560+(alu1205+3073)) = buf8;
  *(data0_5570560+(alu1205+3088)) = buf15;
  *(data0_5570560+(alu1205+3089)) = buf16;
  *(data0_5570560+(alu1205+3104)) = buf23;
  *(data0_5570560+(alu1205+3105)) = buf24;
  *(data0_5570560+(alu1205+3120)) = buf31;
  *(data0_5570560+(alu1205+3121)) = buf32;
  *(data0_5570560+(alu1205+3136)) = buf39;
  *(data0_5570560+(alu1205+3137)) = buf40;
  *(data0_5570560+(alu1205+3152)) = buf47;
  *(data0_5570560+(alu1205+3153)) = buf48;
  *(data0_5570560+(alu1205+3168)) = buf55;
  *(data0_5570560+(alu1205+3169)) = buf56;
  *(data0_5570560+(alu1205+3184)) = buf63;
  *(data0_5570560+(alu1205+3185)) = buf64;
  *(data0_5570560+(alu1207+1)) = buf2;
  *(data0_5570560+(alu1207+16)) = buf9;
  *(data0_5570560+(alu1207+17)) = buf10;
  *(data0_5570560+(alu1207+32)) = buf17;
  *(data0_5570560+(alu1207+33)) = buf18;
  *(data0_5570560+(alu1207+48)) = buf25;
  *(data0_5570560+(alu1207+49)) = buf26;
  *(data0_5570560+(alu1207+64)) = buf33;
  *(data0_5570560+(alu1207+65)) = buf34;
  *(data0_5570560+(alu1207+80)) = buf41;
  *(data0_5570560+(alu1207+81)) = buf42;
  *(data0_5570560+(alu1207+96)) = buf49;
  *(data0_5570560+(alu1207+97)) = buf50;
  *(data0_5570560+(alu1207+112)) = buf57;
  *(data0_5570560+(alu1207+113)) = buf58;
  *(data0_5570560+(alu1206+alu77)) = buf1;
}