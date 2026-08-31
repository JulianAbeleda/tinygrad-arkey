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
  int alu68 = (alu67+9728);
  int alu69 = (alu67+9729);
  int alu70 = (alu67+9730);
  int alu71 = (alu67+9731);
  int alu72 = (alu67+9732);
  int alu73 = (alu67+9733);
  int alu74 = (alu67+9734);
  int alu75 = (alu67+9735);
  int alu76 = (alu67+9736);
  int alu77 = (alu67+9737);
  int alu78 = (alu67+9738);
  int alu79 = (alu67+9739);
  int alu80 = (alu67+9740);
  int alu81 = (alu67+9741);
  int alu82 = (alu67+9742);
  int alu83 = (alu67+9743);
  int alu84 = (alu67+9744);
  int alu85 = (alu67+9745);
  int alu86 = (alu67+9746);
  int alu87 = (alu67+9747);
  int alu88 = (alu67+9748);
  int alu89 = (alu67+9749);
  int alu90 = (alu67+9750);
  int alu91 = (alu67+9751);
  int alu92 = (alu67+9752);
  int alu93 = (alu67+9753);
  int alu94 = (alu67+9754);
  int alu95 = (alu67+9755);
  int alu96 = (alu67+9756);
  int alu97 = (alu67+9757);
  int alu98 = (alu67+9758);
  int alu99 = (alu67+9759);
  int alu100 = (alu67+9760);
  int alu101 = (alu67+9761);
  int alu102 = (alu67+9762);
  int alu103 = (alu67+9763);
  int alu104 = (alu64>>1);
  int alu105 = (alu65>>2);
  int alu106 = ((alu104<<12)+(alu105<<7));
  int alu107 = (alu64&1);
  int alu108 = (alu65&3);
  int alu109 = ((alu107<<3)+(alu108<<1));
  int alu110 = (alu106+(gidx0<<15)+alu109);
  int alu111 = (alu104*2432);
  int alu112 = (alu111+(alu66*76));
  int alu113 = (alu107*288);
  int alu114 = ((alu105*36)+alu113+alu108);
  int alu115 = (alu114+9732);
  int alu116 = (alu114+9736);
  int alu117 = (alu114+9740);
  int alu118 = (alu114+9744);
  int alu119 = (alu114+9748);
  int alu120 = (alu114+9752);
  int alu121 = (alu114+9756);
  int alu122 = (alu114+9760);
  int alu123 = (alu114+10308);
  int alu124 = (alu114+10312);
  int alu125 = (alu114+10316);
  int alu126 = (alu114+10320);
  int alu127 = (alu114+10324);
  int alu128 = (alu114+10328);
  int alu129 = (alu114+10332);
  int alu130 = (alu114+10336);
  int alu131 = (alu114+10884);
  int alu132 = (alu114+10888);
  int alu133 = (alu114+10892);
  int alu134 = (alu114+10896);
  int alu135 = (alu114+10900);
  int alu136 = (alu114+10904);
  int alu137 = (alu114+10908);
  int alu138 = (alu114+10912);
  int alu139 = (alu114+11460);
  int alu140 = (alu114+11464);
  int alu141 = (alu114+11468);
  int alu142 = (alu114+11472);
  int alu143 = (alu114+11476);
  int alu144 = (alu114+11480);
  int alu145 = (alu114+11484);
  int alu146 = (alu114+11488);
  int alu147 = (alu114+12036);
  int alu148 = (alu114+12040);
  int alu149 = (alu114+12044);
  int alu150 = (alu114+12048);
  int alu151 = (alu114+12052);
  int alu152 = (alu114+12056);
  int alu153 = (alu114+12060);
  int alu154 = (alu114+12064);
  int alu155 = (alu114+12612);
  int alu156 = (alu114+12616);
  int alu157 = (alu114+12620);
  int alu158 = (alu114+12624);
  int alu159 = (alu114+12628);
  int alu160 = (alu114+12632);
  int alu161 = (alu114+12636);
  int alu162 = (alu114+12640);
  int alu163 = (alu114+13188);
  int alu164 = (alu114+13192);
  int alu165 = (alu114+13196);
  int alu166 = (alu114+13200);
  int alu167 = (alu114+13204);
  int alu168 = (alu114+13208);
  int alu169 = (alu114+13212);
  int alu170 = (alu114+13216);
  int alu171 = (alu114+13764);
  int alu172 = (alu114+13768);
  int alu173 = (alu114+13772);
  int alu174 = (alu114+13776);
  int alu175 = (alu114+13780);
  int alu176 = (alu114+13784);
  int alu177 = (alu114+13788);
  int alu178 = (alu114+13792);
  int alu179 = (alu111+(alu105*76));
  int alu180 = (alu179+64);
  int alu181 = (alu179+65);
  int alu182 = (alu179+66);
  int alu183 = (alu179+67);
  int alu184 = (alu179+68);
  int alu185 = (alu179+672);
  int alu186 = (alu179+673);
  int alu187 = (alu179+674);
  int alu188 = (alu179+675);
  int alu189 = (alu179+676);
  int alu190 = (alu179+1280);
  int alu191 = (alu179+1281);
  int alu192 = (alu179+1282);
  int alu193 = (alu179+1283);
  int alu194 = (alu179+1284);
  int alu195 = (alu179+1888);
  int alu196 = (alu179+1889);
  int alu197 = (alu179+1890);
  int alu198 = (alu179+1891);
  int alu199 = (alu179+1892);
  int alu200 = (alu113+(alu108*72));
  int alu201 = (alu200+9728);
  int alu202 = (alu200+9729);
  int alu203 = (alu200+9730);
  int alu204 = (alu200+9731);
  int alu205 = (alu200+9764);
  int alu206 = (alu200+9765);
  int alu207 = (alu200+9766);
  int alu208 = (alu200+9767);
  int alu209 = (alu200+10304);
  int alu210 = (alu200+10305);
  int alu211 = (alu200+10306);
  int alu212 = (alu200+10307);
  int alu213 = (alu200+10340);
  int alu214 = (alu200+10341);
  int alu215 = (alu200+10342);
  int alu216 = (alu200+10343);
  int alu217 = (alu200+10880);
  int alu218 = (alu200+10881);
  int alu219 = (alu200+10882);
  int alu220 = (alu200+10883);
  int alu221 = (alu200+10916);
  int alu222 = (alu200+10917);
  int alu223 = (alu200+10918);
  int alu224 = (alu200+10919);
  int alu225 = (alu200+11456);
  int alu226 = (alu200+11457);
  int alu227 = (alu200+11458);
  int alu228 = (alu200+11459);
  int alu229 = (alu200+11492);
  int alu230 = (alu200+11493);
  int alu231 = (alu200+11494);
  int alu232 = (alu200+11495);
  int alu233 = (alu200+12032);
  int alu234 = (alu200+12033);
  int alu235 = (alu200+12034);
  int alu236 = (alu200+12035);
  int alu237 = (alu200+12068);
  int alu238 = (alu200+12069);
  int alu239 = (alu200+12070);
  int alu240 = (alu200+12071);
  int alu241 = (alu200+12608);
  int alu242 = (alu200+12609);
  int alu243 = (alu200+12610);
  int alu244 = (alu200+12611);
  int alu245 = (alu200+12644);
  int alu246 = (alu200+12645);
  int alu247 = (alu200+12646);
  int alu248 = (alu200+12647);
  int alu249 = (alu200+13184);
  int alu250 = (alu200+13185);
  int alu251 = (alu200+13186);
  int alu252 = (alu200+13187);
  int alu253 = (alu200+13220);
  int alu254 = (alu200+13221);
  int alu255 = (alu200+13222);
  int alu256 = (alu200+13223);
  int alu257 = (alu200+13760);
  int alu258 = (alu200+13761);
  int alu259 = (alu200+13762);
  int alu260 = (alu200+13763);
  int alu261 = (alu200+13796);
  int alu262 = (alu200+13797);
  int alu263 = (alu200+13798);
  int alu264 = (alu200+13799);
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
      *(buf0+alu68) = tg_bitcast<unsigned int>((float)(val272));
    }
    if (alu271) {
      *(buf0+alu69) = tg_bitcast<unsigned int>((float)(val269));
    }
    if (alu271) {
      *(buf0+alu70) = tg_bitcast<unsigned int>((float)(val270));
    }
    if (alu271) {
      *(buf0+alu71) = tg_bitcast<unsigned int>((float)(val271));
    }
    if (alu271) {
      *(buf0+alu72) = ((((unsigned int)(val268))&255u)|((((unsigned int)(val13))&255u)<<8u)|((((unsigned int)(val14))&255u)<<16u)|((((unsigned int)(val15))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu73) = ((((unsigned int)(val16))&255u)|((((unsigned int)(val17))&255u)<<8u)|((((unsigned int)(val18))&255u)<<16u)|((((unsigned int)(val19))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu74) = ((((unsigned int)(val20))&255u)|((((unsigned int)(val21))&255u)<<8u)|((((unsigned int)(val22))&255u)<<16u)|((((unsigned int)(val23))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu75) = ((((unsigned int)(val24))&255u)|((((unsigned int)(val25))&255u)<<8u)|((((unsigned int)(val26))&255u)<<16u)|((((unsigned int)(val27))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu76) = ((((unsigned int)(val28))&255u)|((((unsigned int)(val29))&255u)<<8u)|((((unsigned int)(val30))&255u)<<16u)|((((unsigned int)(val31))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu77) = ((((unsigned int)(val32))&255u)|((((unsigned int)(val33))&255u)<<8u)|((((unsigned int)(val34))&255u)<<16u)|((((unsigned int)(val35))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu78) = ((((unsigned int)(val36))&255u)|((((unsigned int)(val37))&255u)<<8u)|((((unsigned int)(val38))&255u)<<16u)|((((unsigned int)(val39))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu79) = ((((unsigned int)(val40))&255u)|((((unsigned int)(val41))&255u)<<8u)|((((unsigned int)(val42))&255u)<<16u)|((((unsigned int)(val43))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu80) = ((((unsigned int)(val44))&255u)|((((unsigned int)(val45))&255u)<<8u)|((((unsigned int)(val46))&255u)<<16u)|((((unsigned int)(val47))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu81) = ((((unsigned int)(val48))&255u)|((((unsigned int)(val49))&255u)<<8u)|((((unsigned int)(val50))&255u)<<16u)|((((unsigned int)(val51))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu82) = ((((unsigned int)(val52))&255u)|((((unsigned int)(val53))&255u)<<8u)|((((unsigned int)(val54))&255u)<<16u)|((((unsigned int)(val55))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu83) = ((((unsigned int)(val56))&255u)|((((unsigned int)(val57))&255u)<<8u)|((((unsigned int)(val58))&255u)<<16u)|((((unsigned int)(val59))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu84) = ((((unsigned int)(val60))&255u)|((((unsigned int)(val61))&255u)<<8u)|((((unsigned int)(val62))&255u)<<16u)|((((unsigned int)(val63))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu85) = ((((unsigned int)(val64))&255u)|((((unsigned int)(val65))&255u)<<8u)|((((unsigned int)(val66))&255u)<<16u)|((((unsigned int)(val67))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu86) = ((((unsigned int)(val68))&255u)|((((unsigned int)(val69))&255u)<<8u)|((((unsigned int)(val70))&255u)<<16u)|((((unsigned int)(val71))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu87) = ((((unsigned int)(val72))&255u)|((((unsigned int)(val73))&255u)<<8u)|((((unsigned int)(val74))&255u)<<16u)|((((unsigned int)(val75))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu88) = ((((unsigned int)(val76))&255u)|((((unsigned int)(val77))&255u)<<8u)|((((unsigned int)(val78))&255u)<<16u)|((((unsigned int)(val79))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu89) = ((((unsigned int)(val80))&255u)|((((unsigned int)(val81))&255u)<<8u)|((((unsigned int)(val82))&255u)<<16u)|((((unsigned int)(val83))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu90) = ((((unsigned int)(val84))&255u)|((((unsigned int)(val85))&255u)<<8u)|((((unsigned int)(val86))&255u)<<16u)|((((unsigned int)(val87))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu91) = ((((unsigned int)(val88))&255u)|((((unsigned int)(val89))&255u)<<8u)|((((unsigned int)(val90))&255u)<<16u)|((((unsigned int)(val91))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu92) = ((((unsigned int)(val92))&255u)|((((unsigned int)(val93))&255u)<<8u)|((((unsigned int)(val94))&255u)<<16u)|((((unsigned int)(val95))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu93) = ((((unsigned int)(val96))&255u)|((((unsigned int)(val97))&255u)<<8u)|((((unsigned int)(val98))&255u)<<16u)|((((unsigned int)(val99))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu94) = ((((unsigned int)(val100))&255u)|((((unsigned int)(val101))&255u)<<8u)|((((unsigned int)(val102))&255u)<<16u)|((((unsigned int)(val103))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu95) = ((((unsigned int)(val104))&255u)|((((unsigned int)(val105))&255u)<<8u)|((((unsigned int)(val106))&255u)<<16u)|((((unsigned int)(val107))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu96) = ((((unsigned int)(val108))&255u)|((((unsigned int)(val109))&255u)<<8u)|((((unsigned int)(val110))&255u)<<16u)|((((unsigned int)(val111))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu97) = ((((unsigned int)(val112))&255u)|((((unsigned int)(val113))&255u)<<8u)|((((unsigned int)(val114))&255u)<<16u)|((((unsigned int)(val115))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu98) = ((((unsigned int)(val116))&255u)|((((unsigned int)(val117))&255u)<<8u)|((((unsigned int)(val118))&255u)<<16u)|((((unsigned int)(val119))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu99) = ((((unsigned int)(val120))&255u)|((((unsigned int)(val121))&255u)<<8u)|((((unsigned int)(val122))&255u)<<16u)|((((unsigned int)(val123))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu100) = ((((unsigned int)(val124))&255u)|((((unsigned int)(val125))&255u)<<8u)|((((unsigned int)(val126))&255u)<<16u)|((((unsigned int)(val127))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu101) = ((((unsigned int)(val128))&255u)|((((unsigned int)(val129))&255u)<<8u)|((((unsigned int)(val130))&255u)<<16u)|((((unsigned int)(val131))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu102) = ((((unsigned int)(val132))&255u)|((((unsigned int)(val133))&255u)<<8u)|((((unsigned int)(val134))&255u)<<16u)|((((unsigned int)(val135))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu103) = ((((unsigned int)(val136))&255u)|((((unsigned int)(val137))&255u)<<8u)|((((unsigned int)(val138))&255u)<<16u)|((((unsigned int)(val139))&255u)<<24u));
    }
    __syncthreads();
    unsigned int val273 = (*(buf0+alu115));
    unsigned int val274 = (*(buf0+alu116));
    unsigned int val275 = (*(buf0+alu117));
    unsigned int val276 = (*(buf0+alu118));
    unsigned int val277 = (*(buf0+alu119));
    unsigned int val278 = (*(buf0+alu120));
    unsigned int val279 = (*(buf0+alu121));
    unsigned int val280 = (*(buf0+alu122));
    unsigned int val281 = (*(buf0+alu201));
    unsigned int val282 = (*(buf0+alu202));
    unsigned int val283 = (*(buf0+alu203));
    unsigned int val284 = (*(buf0+alu204));
    __syncthreads();
    unsigned int val285 = (*(buf0+alu180));
    unsigned int val286 = (*(buf0+alu181));
    unsigned int val287 = (*(buf0+alu182));
    int alu413 = (alu272+-1);
    bool alu414 = ((0<Ridx50)&(alu273!=((alu413/48)-((int)((((alu413%48)!=0)&((alu413<0)!=0)))))));
    if (alu414) {
      *(data0_5570560+alu110) = buf1;
    }
    char4 cast1 = make_char4(((signed char)(((val273>>0u)&255u))),((signed char)(((val273>>8u)&255u))),((signed char)(((val273>>16u)&255u))),((signed char)(((val273>>24u)&255u))));
    char4 cast2 = make_char4(((signed char)(((val274>>0u)&255u))),((signed char)(((val274>>8u)&255u))),((signed char)(((val274>>16u)&255u))),((signed char)(((val274>>24u)&255u))));
    char4 cast3 = make_char4(((signed char)(((val275>>0u)&255u))),((signed char)(((val275>>8u)&255u))),((signed char)(((val275>>16u)&255u))),((signed char)(((val275>>24u)&255u))));
    char4 cast4 = make_char4(((signed char)(((val276>>0u)&255u))),((signed char)(((val276>>8u)&255u))),((signed char)(((val276>>16u)&255u))),((signed char)(((val276>>24u)&255u))));
    char4 cast5 = make_char4(((signed char)(((val277>>0u)&255u))),((signed char)(((val277>>8u)&255u))),((signed char)(((val277>>16u)&255u))),((signed char)(((val277>>24u)&255u))));
    char4 cast6 = make_char4(((signed char)(((val278>>0u)&255u))),((signed char)(((val278>>8u)&255u))),((signed char)(((val278>>16u)&255u))),((signed char)(((val278>>24u)&255u))));
    char4 cast7 = make_char4(((signed char)(((val279>>0u)&255u))),((signed char)(((val279>>8u)&255u))),((signed char)(((val279>>16u)&255u))),((signed char)(((val279>>24u)&255u))));
    char4 cast8 = make_char4(((signed char)(((val280>>0u)&255u))),((signed char)(((val280>>8u)&255u))),((signed char)(((val280>>16u)&255u))),((signed char)(((val280>>24u)&255u))));
    signed_char8 alu418 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+4))))*4)));
    int4 wmma0 = __WMMA_8_16_16_signed_char_int(alu418, cast2, cast0);
    signed_char8 alu419 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+8))))*4)));
    int4 wmma1 = __WMMA_8_16_16_signed_char_int(alu419, cast3, cast0);
    signed_char8 alu420 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+12))))*4)));
    int4 wmma2 = __WMMA_8_16_16_signed_char_int(alu420, cast4, cast0);
    signed_char8 alu421 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+16))))*4)));
    int4 wmma3 = __WMMA_8_16_16_signed_char_int(alu421, cast5, cast0);
    signed_char8 alu422 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+20))))*4)));
    int4 wmma4 = __WMMA_8_16_16_signed_char_int(alu422, cast6, cast0);
    signed_char8 alu423 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+24))))*4)));
    int4 wmma5 = __WMMA_8_16_16_signed_char_int(alu423, cast7, cast0);
    signed_char8 alu424 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+28))))*4)));
    int4 wmma6 = __WMMA_8_16_16_signed_char_int(alu424, cast8, cast0);
    signed_char8 alu425 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)(alu112)))*4)));
    int4 wmma7 = __WMMA_8_16_16_signed_char_int(alu425, cast1, cast0);
    float cast9 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val285&65535u)))))));
    float alu426 = ((cast9*tg_bitcast<float>((unsigned int)(val281))*((((float)(((signed char)(((val286>>0u)&255u)))))*((float)(wmma7.x)))+(((float)(((signed char)(((val286>>8u)&255u)))))*((float)(wmma0.x)))))+(cast9*tg_bitcast<float>((unsigned int)(val282))*((((float)(((signed char)(((val286>>16u)&255u)))))*((float)(wmma1.x)))+(((float)(((signed char)(((val286>>24u)&255u)))))*((float)(wmma2.x)))))+(cast9*tg_bitcast<float>((unsigned int)(val283))*((((float)(((signed char)(((val287>>0u)&255u)))))*((float)(wmma3.x)))+(((float)(((signed char)(((val287>>8u)&255u)))))*((float)(wmma4.x)))))+(cast9*tg_bitcast<float>((unsigned int)(val284))*((((float)(((signed char)(((val287>>16u)&255u)))))*((float)(wmma5.x)))+(((float)(((signed char)(((val287>>24u)&255u)))))*((float)(wmma6.x))))));
    float alu427 = (alu414?alu426:(buf1+alu426));
    buf1 = alu427;
    unsigned int val288 = (*(buf0+alu205));
    unsigned int val289 = (*(buf0+alu206));
    unsigned int val290 = (*(buf0+alu207));
    unsigned int val291 = (*(buf0+alu208));
    unsigned int val292 = (*(buf0+alu180));
    unsigned int val293 = (*(buf0+alu181));
    unsigned int val294 = (*(buf0+alu182));
    if (alu414) {
      *(data0_5570560+(alu110+1)) = buf2;
    }
    float cast10 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val292&65535u)))))));
    float alu432 = ((cast10*tg_bitcast<float>((unsigned int)(val288))*((((float)(((signed char)(((val293>>0u)&255u)))))*((float)(wmma7.y)))+(((float)(((signed char)(((val293>>8u)&255u)))))*((float)(wmma0.y)))))+(cast10*tg_bitcast<float>((unsigned int)(val289))*((((float)(((signed char)(((val293>>16u)&255u)))))*((float)(wmma1.y)))+(((float)(((signed char)(((val293>>24u)&255u)))))*((float)(wmma2.y)))))+(cast10*tg_bitcast<float>((unsigned int)(val290))*((((float)(((signed char)(((val294>>0u)&255u)))))*((float)(wmma3.y)))+(((float)(((signed char)(((val294>>8u)&255u)))))*((float)(wmma4.y)))))+(cast10*tg_bitcast<float>((unsigned int)(val291))*((((float)(((signed char)(((val294>>16u)&255u)))))*((float)(wmma5.y)))+(((float)(((signed char)(((val294>>24u)&255u)))))*((float)(wmma6.y))))));
    float alu433 = (alu414?alu432:(buf2+alu432));
    buf2 = alu433;
    unsigned int val295 = (*(buf0+alu201));
    unsigned int val296 = (*(buf0+alu202));
    unsigned int val297 = (*(buf0+alu203));
    unsigned int val298 = (*(buf0+alu204));
    unsigned int val299 = (*(buf0+alu185));
    unsigned int val300 = (*(buf0+alu186));
    unsigned int val301 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+1024)) = buf3;
    }
    float cast11 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val299&65535u)))))));
    float alu438 = ((cast11*tg_bitcast<float>((unsigned int)(val295))*((((float)(((signed char)(((val300>>0u)&255u)))))*((float)(wmma7.z)))+(((float)(((signed char)(((val300>>8u)&255u)))))*((float)(wmma0.z)))))+(cast11*tg_bitcast<float>((unsigned int)(val296))*((((float)(((signed char)(((val300>>16u)&255u)))))*((float)(wmma1.z)))+(((float)(((signed char)(((val300>>24u)&255u)))))*((float)(wmma2.z)))))+(cast11*tg_bitcast<float>((unsigned int)(val297))*((((float)(((signed char)(((val301>>0u)&255u)))))*((float)(wmma3.z)))+(((float)(((signed char)(((val301>>8u)&255u)))))*((float)(wmma4.z)))))+(cast11*tg_bitcast<float>((unsigned int)(val298))*((((float)(((signed char)(((val301>>16u)&255u)))))*((float)(wmma5.z)))+(((float)(((signed char)(((val301>>24u)&255u)))))*((float)(wmma6.z))))));
    float alu439 = (alu414?alu438:(buf3+alu438));
    buf3 = alu439;
    unsigned int val302 = (*(buf0+alu205));
    unsigned int val303 = (*(buf0+alu206));
    unsigned int val304 = (*(buf0+alu207));
    unsigned int val305 = (*(buf0+alu208));
    unsigned int val306 = (*(buf0+alu185));
    unsigned int val307 = (*(buf0+alu186));
    unsigned int val308 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+1025)) = buf4;
    }
    float cast12 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val306&65535u)))))));
    float alu444 = ((cast12*tg_bitcast<float>((unsigned int)(val302))*((((float)(((signed char)(((val307>>0u)&255u)))))*((float)(wmma7.w)))+(((float)(((signed char)(((val307>>8u)&255u)))))*((float)(wmma0.w)))))+(cast12*tg_bitcast<float>((unsigned int)(val303))*((((float)(((signed char)(((val307>>16u)&255u)))))*((float)(wmma1.w)))+(((float)(((signed char)(((val307>>24u)&255u)))))*((float)(wmma2.w)))))+(cast12*tg_bitcast<float>((unsigned int)(val304))*((((float)(((signed char)(((val308>>0u)&255u)))))*((float)(wmma3.w)))+(((float)(((signed char)(((val308>>8u)&255u)))))*((float)(wmma4.w)))))+(cast12*tg_bitcast<float>((unsigned int)(val305))*((((float)(((signed char)(((val308>>16u)&255u)))))*((float)(wmma5.w)))+(((float)(((signed char)(((val308>>24u)&255u)))))*((float)(wmma6.w))))));
    float alu445 = (alu414?alu444:(buf4+alu444));
    buf4 = alu445;
    unsigned int val309 = (*(buf0+alu201));
    unsigned int val310 = (*(buf0+alu202));
    unsigned int val311 = (*(buf0+alu203));
    unsigned int val312 = (*(buf0+alu204));
    unsigned int val313 = (*(buf0+alu190));
    unsigned int val314 = (*(buf0+alu191));
    unsigned int val315 = (*(buf0+alu192));
    if (alu414) {
      *(data0_5570560+(alu110+2048)) = buf5;
    }
    signed_char8 alu450 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1216))))*4)));
    int4 wmma8 = __WMMA_8_16_16_signed_char_int(alu450, cast1, cast0);
    signed_char8 alu451 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1220))))*4)));
    int4 wmma9 = __WMMA_8_16_16_signed_char_int(alu451, cast2, cast0);
    signed_char8 alu452 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1224))))*4)));
    int4 wmma10 = __WMMA_8_16_16_signed_char_int(alu452, cast3, cast0);
    signed_char8 alu453 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1228))))*4)));
    int4 wmma11 = __WMMA_8_16_16_signed_char_int(alu453, cast4, cast0);
    signed_char8 alu454 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1232))))*4)));
    int4 wmma12 = __WMMA_8_16_16_signed_char_int(alu454, cast5, cast0);
    signed_char8 alu455 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1236))))*4)));
    int4 wmma13 = __WMMA_8_16_16_signed_char_int(alu455, cast6, cast0);
    signed_char8 alu456 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1240))))*4)));
    int4 wmma14 = __WMMA_8_16_16_signed_char_int(alu456, cast7, cast0);
    signed_char8 alu457 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1244))))*4)));
    int4 wmma15 = __WMMA_8_16_16_signed_char_int(alu457, cast8, cast0);
    float cast13 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val313&65535u)))))));
    float alu458 = ((cast13*tg_bitcast<float>((unsigned int)(val309))*((((float)(((signed char)(((val314>>0u)&255u)))))*((float)(wmma8.x)))+(((float)(((signed char)(((val314>>8u)&255u)))))*((float)(wmma9.x)))))+(cast13*tg_bitcast<float>((unsigned int)(val310))*((((float)(((signed char)(((val314>>16u)&255u)))))*((float)(wmma10.x)))+(((float)(((signed char)(((val314>>24u)&255u)))))*((float)(wmma11.x)))))+(cast13*tg_bitcast<float>((unsigned int)(val311))*((((float)(((signed char)(((val315>>0u)&255u)))))*((float)(wmma12.x)))+(((float)(((signed char)(((val315>>8u)&255u)))))*((float)(wmma13.x)))))+(cast13*tg_bitcast<float>((unsigned int)(val312))*((((float)(((signed char)(((val315>>16u)&255u)))))*((float)(wmma14.x)))+(((float)(((signed char)(((val315>>24u)&255u)))))*((float)(wmma15.x))))));
    float alu459 = (alu414?alu458:(buf5+alu458));
    buf5 = alu459;
    unsigned int val316 = (*(buf0+alu205));
    unsigned int val317 = (*(buf0+alu206));
    unsigned int val318 = (*(buf0+alu207));
    unsigned int val319 = (*(buf0+alu208));
    unsigned int val320 = (*(buf0+alu190));
    unsigned int val321 = (*(buf0+alu191));
    unsigned int val322 = (*(buf0+alu192));
    if (alu414) {
      *(data0_5570560+(alu110+2049)) = buf6;
    }
    float cast14 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val320&65535u)))))));
    float alu464 = ((cast14*tg_bitcast<float>((unsigned int)(val316))*((((float)(((signed char)(((val321>>0u)&255u)))))*((float)(wmma8.y)))+(((float)(((signed char)(((val321>>8u)&255u)))))*((float)(wmma9.y)))))+(cast14*tg_bitcast<float>((unsigned int)(val317))*((((float)(((signed char)(((val321>>16u)&255u)))))*((float)(wmma10.y)))+(((float)(((signed char)(((val321>>24u)&255u)))))*((float)(wmma11.y)))))+(cast14*tg_bitcast<float>((unsigned int)(val318))*((((float)(((signed char)(((val322>>0u)&255u)))))*((float)(wmma12.y)))+(((float)(((signed char)(((val322>>8u)&255u)))))*((float)(wmma13.y)))))+(cast14*tg_bitcast<float>((unsigned int)(val319))*((((float)(((signed char)(((val322>>16u)&255u)))))*((float)(wmma14.y)))+(((float)(((signed char)(((val322>>24u)&255u)))))*((float)(wmma15.y))))));
    float alu465 = (alu414?alu464:(buf6+alu464));
    buf6 = alu465;
    unsigned int val323 = (*(buf0+alu201));
    unsigned int val324 = (*(buf0+alu202));
    unsigned int val325 = (*(buf0+alu203));
    unsigned int val326 = (*(buf0+alu204));
    unsigned int val327 = (*(buf0+alu195));
    unsigned int val328 = (*(buf0+alu196));
    unsigned int val329 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+3072)) = buf7;
    }
    float cast15 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val327&65535u)))))));
    float alu470 = ((cast15*tg_bitcast<float>((unsigned int)(val323))*((((float)(((signed char)(((val328>>0u)&255u)))))*((float)(wmma8.z)))+(((float)(((signed char)(((val328>>8u)&255u)))))*((float)(wmma9.z)))))+(cast15*tg_bitcast<float>((unsigned int)(val324))*((((float)(((signed char)(((val328>>16u)&255u)))))*((float)(wmma10.z)))+(((float)(((signed char)(((val328>>24u)&255u)))))*((float)(wmma11.z)))))+(cast15*tg_bitcast<float>((unsigned int)(val325))*((((float)(((signed char)(((val329>>0u)&255u)))))*((float)(wmma12.z)))+(((float)(((signed char)(((val329>>8u)&255u)))))*((float)(wmma13.z)))))+(cast15*tg_bitcast<float>((unsigned int)(val326))*((((float)(((signed char)(((val329>>16u)&255u)))))*((float)(wmma14.z)))+(((float)(((signed char)(((val329>>24u)&255u)))))*((float)(wmma15.z))))));
    float alu471 = (alu414?alu470:(buf7+alu470));
    buf7 = alu471;
    unsigned int val330 = (*(buf0+alu205));
    unsigned int val331 = (*(buf0+alu206));
    unsigned int val332 = (*(buf0+alu207));
    unsigned int val333 = (*(buf0+alu208));
    unsigned int val334 = (*(buf0+alu195));
    unsigned int val335 = (*(buf0+alu196));
    unsigned int val336 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+3073)) = buf8;
    }
    float cast16 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val334&65535u)))))));
    float alu476 = ((cast16*tg_bitcast<float>((unsigned int)(val330))*((((float)(((signed char)(((val335>>0u)&255u)))))*((float)(wmma8.w)))+(((float)(((signed char)(((val335>>8u)&255u)))))*((float)(wmma9.w)))))+(cast16*tg_bitcast<float>((unsigned int)(val331))*((((float)(((signed char)(((val335>>16u)&255u)))))*((float)(wmma10.w)))+(((float)(((signed char)(((val335>>24u)&255u)))))*((float)(wmma11.w)))))+(cast16*tg_bitcast<float>((unsigned int)(val332))*((((float)(((signed char)(((val336>>0u)&255u)))))*((float)(wmma12.w)))+(((float)(((signed char)(((val336>>8u)&255u)))))*((float)(wmma13.w)))))+(cast16*tg_bitcast<float>((unsigned int)(val333))*((((float)(((signed char)(((val336>>16u)&255u)))))*((float)(wmma14.w)))+(((float)(((signed char)(((val336>>24u)&255u)))))*((float)(wmma15.w))))));
    float alu477 = (alu414?alu476:(buf8+alu476));
    buf8 = alu477;
    unsigned int val337 = (*(buf0+alu123));
    unsigned int val338 = (*(buf0+alu124));
    unsigned int val339 = (*(buf0+alu125));
    unsigned int val340 = (*(buf0+alu126));
    unsigned int val341 = (*(buf0+alu127));
    unsigned int val342 = (*(buf0+alu128));
    unsigned int val343 = (*(buf0+alu129));
    unsigned int val344 = (*(buf0+alu130));
    unsigned int val345 = (*(buf0+alu209));
    unsigned int val346 = (*(buf0+alu210));
    unsigned int val347 = (*(buf0+alu211));
    unsigned int val348 = (*(buf0+alu212));
    unsigned int val349 = (*(buf0+alu180));
    unsigned int val350 = (*(buf0+alu181));
    unsigned int val351 = (*(buf0+alu182));
    if (alu414) {
      *(data0_5570560+(alu110+16)) = buf9;
    }
    char4 cast17 = make_char4(((signed char)(((val337>>0u)&255u))),((signed char)(((val337>>8u)&255u))),((signed char)(((val337>>16u)&255u))),((signed char)(((val337>>24u)&255u))));
    char4 cast18 = make_char4(((signed char)(((val338>>0u)&255u))),((signed char)(((val338>>8u)&255u))),((signed char)(((val338>>16u)&255u))),((signed char)(((val338>>24u)&255u))));
    char4 cast19 = make_char4(((signed char)(((val339>>0u)&255u))),((signed char)(((val339>>8u)&255u))),((signed char)(((val339>>16u)&255u))),((signed char)(((val339>>24u)&255u))));
    char4 cast20 = make_char4(((signed char)(((val340>>0u)&255u))),((signed char)(((val340>>8u)&255u))),((signed char)(((val340>>16u)&255u))),((signed char)(((val340>>24u)&255u))));
    char4 cast21 = make_char4(((signed char)(((val341>>0u)&255u))),((signed char)(((val341>>8u)&255u))),((signed char)(((val341>>16u)&255u))),((signed char)(((val341>>24u)&255u))));
    char4 cast22 = make_char4(((signed char)(((val342>>0u)&255u))),((signed char)(((val342>>8u)&255u))),((signed char)(((val342>>16u)&255u))),((signed char)(((val342>>24u)&255u))));
    char4 cast23 = make_char4(((signed char)(((val343>>0u)&255u))),((signed char)(((val343>>8u)&255u))),((signed char)(((val343>>16u)&255u))),((signed char)(((val343>>24u)&255u))));
    char4 cast24 = make_char4(((signed char)(((val344>>0u)&255u))),((signed char)(((val344>>8u)&255u))),((signed char)(((val344>>16u)&255u))),((signed char)(((val344>>24u)&255u))));
    int4 wmma16 = __WMMA_8_16_16_signed_char_int(alu418, cast18, cast0);
    int4 wmma17 = __WMMA_8_16_16_signed_char_int(alu419, cast19, cast0);
    int4 wmma18 = __WMMA_8_16_16_signed_char_int(alu420, cast20, cast0);
    int4 wmma19 = __WMMA_8_16_16_signed_char_int(alu421, cast21, cast0);
    int4 wmma20 = __WMMA_8_16_16_signed_char_int(alu422, cast22, cast0);
    int4 wmma21 = __WMMA_8_16_16_signed_char_int(alu423, cast23, cast0);
    int4 wmma22 = __WMMA_8_16_16_signed_char_int(alu424, cast24, cast0);
    int4 wmma23 = __WMMA_8_16_16_signed_char_int(alu425, cast17, cast0);
    float cast25 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val349&65535u)))))));
    float alu482 = ((cast25*tg_bitcast<float>((unsigned int)(val345))*((((float)(((signed char)(((val350>>0u)&255u)))))*((float)(wmma23.x)))+(((float)(((signed char)(((val350>>8u)&255u)))))*((float)(wmma16.x)))))+(cast25*tg_bitcast<float>((unsigned int)(val346))*((((float)(((signed char)(((val350>>16u)&255u)))))*((float)(wmma17.x)))+(((float)(((signed char)(((val350>>24u)&255u)))))*((float)(wmma18.x)))))+(cast25*tg_bitcast<float>((unsigned int)(val347))*((((float)(((signed char)(((val351>>0u)&255u)))))*((float)(wmma19.x)))+(((float)(((signed char)(((val351>>8u)&255u)))))*((float)(wmma20.x)))))+(cast25*tg_bitcast<float>((unsigned int)(val348))*((((float)(((signed char)(((val351>>16u)&255u)))))*((float)(wmma21.x)))+(((float)(((signed char)(((val351>>24u)&255u)))))*((float)(wmma22.x))))));
    float alu483 = (alu414?alu482:(buf9+alu482));
    buf9 = alu483;
    unsigned int val352 = (*(buf0+alu213));
    unsigned int val353 = (*(buf0+alu214));
    unsigned int val354 = (*(buf0+alu215));
    unsigned int val355 = (*(buf0+alu216));
    unsigned int val356 = (*(buf0+alu180));
    unsigned int val357 = (*(buf0+alu181));
    unsigned int val358 = (*(buf0+alu182));
    if (alu414) {
      *(data0_5570560+(alu110+17)) = buf10;
    }
    float cast26 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val356&65535u)))))));
    float alu488 = ((cast26*tg_bitcast<float>((unsigned int)(val352))*((((float)(((signed char)(((val357>>0u)&255u)))))*((float)(wmma23.y)))+(((float)(((signed char)(((val357>>8u)&255u)))))*((float)(wmma16.y)))))+(cast26*tg_bitcast<float>((unsigned int)(val353))*((((float)(((signed char)(((val357>>16u)&255u)))))*((float)(wmma17.y)))+(((float)(((signed char)(((val357>>24u)&255u)))))*((float)(wmma18.y)))))+(cast26*tg_bitcast<float>((unsigned int)(val354))*((((float)(((signed char)(((val358>>0u)&255u)))))*((float)(wmma19.y)))+(((float)(((signed char)(((val358>>8u)&255u)))))*((float)(wmma20.y)))))+(cast26*tg_bitcast<float>((unsigned int)(val355))*((((float)(((signed char)(((val358>>16u)&255u)))))*((float)(wmma21.y)))+(((float)(((signed char)(((val358>>24u)&255u)))))*((float)(wmma22.y))))));
    float alu489 = (alu414?alu488:(buf10+alu488));
    buf10 = alu489;
    unsigned int val359 = (*(buf0+alu209));
    unsigned int val360 = (*(buf0+alu210));
    unsigned int val361 = (*(buf0+alu211));
    unsigned int val362 = (*(buf0+alu212));
    unsigned int val363 = (*(buf0+alu185));
    unsigned int val364 = (*(buf0+alu186));
    unsigned int val365 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+1040)) = buf11;
    }
    float cast27 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val363&65535u)))))));
    float alu494 = ((cast27*tg_bitcast<float>((unsigned int)(val359))*((((float)(((signed char)(((val364>>0u)&255u)))))*((float)(wmma23.z)))+(((float)(((signed char)(((val364>>8u)&255u)))))*((float)(wmma16.z)))))+(cast27*tg_bitcast<float>((unsigned int)(val360))*((((float)(((signed char)(((val364>>16u)&255u)))))*((float)(wmma17.z)))+(((float)(((signed char)(((val364>>24u)&255u)))))*((float)(wmma18.z)))))+(cast27*tg_bitcast<float>((unsigned int)(val361))*((((float)(((signed char)(((val365>>0u)&255u)))))*((float)(wmma19.z)))+(((float)(((signed char)(((val365>>8u)&255u)))))*((float)(wmma20.z)))))+(cast27*tg_bitcast<float>((unsigned int)(val362))*((((float)(((signed char)(((val365>>16u)&255u)))))*((float)(wmma21.z)))+(((float)(((signed char)(((val365>>24u)&255u)))))*((float)(wmma22.z))))));
    float alu495 = (alu414?alu494:(buf11+alu494));
    buf11 = alu495;
    unsigned int val366 = (*(buf0+alu213));
    unsigned int val367 = (*(buf0+alu214));
    unsigned int val368 = (*(buf0+alu215));
    unsigned int val369 = (*(buf0+alu216));
    unsigned int val370 = (*(buf0+alu185));
    unsigned int val371 = (*(buf0+alu186));
    unsigned int val372 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+1041)) = buf12;
    }
    float cast28 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val370&65535u)))))));
    float alu500 = ((cast28*tg_bitcast<float>((unsigned int)(val366))*((((float)(((signed char)(((val371>>0u)&255u)))))*((float)(wmma23.w)))+(((float)(((signed char)(((val371>>8u)&255u)))))*((float)(wmma16.w)))))+(cast28*tg_bitcast<float>((unsigned int)(val367))*((((float)(((signed char)(((val371>>16u)&255u)))))*((float)(wmma17.w)))+(((float)(((signed char)(((val371>>24u)&255u)))))*((float)(wmma18.w)))))+(cast28*tg_bitcast<float>((unsigned int)(val368))*((((float)(((signed char)(((val372>>0u)&255u)))))*((float)(wmma19.w)))+(((float)(((signed char)(((val372>>8u)&255u)))))*((float)(wmma20.w)))))+(cast28*tg_bitcast<float>((unsigned int)(val369))*((((float)(((signed char)(((val372>>16u)&255u)))))*((float)(wmma21.w)))+(((float)(((signed char)(((val372>>24u)&255u)))))*((float)(wmma22.w))))));
    float alu501 = (alu414?alu500:(buf12+alu500));
    buf12 = alu501;
    unsigned int val373 = (*(buf0+alu209));
    unsigned int val374 = (*(buf0+alu210));
    unsigned int val375 = (*(buf0+alu211));
    unsigned int val376 = (*(buf0+alu212));
    unsigned int val377 = (*(buf0+alu190));
    unsigned int val378 = (*(buf0+alu191));
    unsigned int val379 = (*(buf0+alu192));
    if (alu414) {
      *(data0_5570560+(alu110+2064)) = buf13;
    }
    int4 wmma24 = __WMMA_8_16_16_signed_char_int(alu450, cast17, cast0);
    int4 wmma25 = __WMMA_8_16_16_signed_char_int(alu451, cast18, cast0);
    int4 wmma26 = __WMMA_8_16_16_signed_char_int(alu452, cast19, cast0);
    int4 wmma27 = __WMMA_8_16_16_signed_char_int(alu453, cast20, cast0);
    int4 wmma28 = __WMMA_8_16_16_signed_char_int(alu454, cast21, cast0);
    int4 wmma29 = __WMMA_8_16_16_signed_char_int(alu455, cast22, cast0);
    int4 wmma30 = __WMMA_8_16_16_signed_char_int(alu456, cast23, cast0);
    int4 wmma31 = __WMMA_8_16_16_signed_char_int(alu457, cast24, cast0);
    float cast29 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val377&65535u)))))));
    float alu506 = ((cast29*tg_bitcast<float>((unsigned int)(val373))*((((float)(((signed char)(((val378>>0u)&255u)))))*((float)(wmma24.x)))+(((float)(((signed char)(((val378>>8u)&255u)))))*((float)(wmma25.x)))))+(cast29*tg_bitcast<float>((unsigned int)(val374))*((((float)(((signed char)(((val378>>16u)&255u)))))*((float)(wmma26.x)))+(((float)(((signed char)(((val378>>24u)&255u)))))*((float)(wmma27.x)))))+(cast29*tg_bitcast<float>((unsigned int)(val375))*((((float)(((signed char)(((val379>>0u)&255u)))))*((float)(wmma28.x)))+(((float)(((signed char)(((val379>>8u)&255u)))))*((float)(wmma29.x)))))+(cast29*tg_bitcast<float>((unsigned int)(val376))*((((float)(((signed char)(((val379>>16u)&255u)))))*((float)(wmma30.x)))+(((float)(((signed char)(((val379>>24u)&255u)))))*((float)(wmma31.x))))));
    float alu507 = (alu414?alu506:(buf13+alu506));
    buf13 = alu507;
    unsigned int val380 = (*(buf0+alu213));
    unsigned int val381 = (*(buf0+alu214));
    unsigned int val382 = (*(buf0+alu215));
    unsigned int val383 = (*(buf0+alu216));
    unsigned int val384 = (*(buf0+alu190));
    unsigned int val385 = (*(buf0+alu191));
    unsigned int val386 = (*(buf0+alu192));
    if (alu414) {
      *(data0_5570560+(alu110+2065)) = buf14;
    }
    float cast30 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val384&65535u)))))));
    float alu512 = ((cast30*tg_bitcast<float>((unsigned int)(val380))*((((float)(((signed char)(((val385>>0u)&255u)))))*((float)(wmma24.y)))+(((float)(((signed char)(((val385>>8u)&255u)))))*((float)(wmma25.y)))))+(cast30*tg_bitcast<float>((unsigned int)(val381))*((((float)(((signed char)(((val385>>16u)&255u)))))*((float)(wmma26.y)))+(((float)(((signed char)(((val385>>24u)&255u)))))*((float)(wmma27.y)))))+(cast30*tg_bitcast<float>((unsigned int)(val382))*((((float)(((signed char)(((val386>>0u)&255u)))))*((float)(wmma28.y)))+(((float)(((signed char)(((val386>>8u)&255u)))))*((float)(wmma29.y)))))+(cast30*tg_bitcast<float>((unsigned int)(val383))*((((float)(((signed char)(((val386>>16u)&255u)))))*((float)(wmma30.y)))+(((float)(((signed char)(((val386>>24u)&255u)))))*((float)(wmma31.y))))));
    float alu513 = (alu414?alu512:(buf14+alu512));
    buf14 = alu513;
    unsigned int val387 = (*(buf0+alu209));
    unsigned int val388 = (*(buf0+alu210));
    unsigned int val389 = (*(buf0+alu211));
    unsigned int val390 = (*(buf0+alu212));
    unsigned int val391 = (*(buf0+alu195));
    unsigned int val392 = (*(buf0+alu196));
    unsigned int val393 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+3088)) = buf15;
    }
    float cast31 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val391&65535u)))))));
    float alu518 = ((cast31*tg_bitcast<float>((unsigned int)(val387))*((((float)(((signed char)(((val392>>0u)&255u)))))*((float)(wmma24.z)))+(((float)(((signed char)(((val392>>8u)&255u)))))*((float)(wmma25.z)))))+(cast31*tg_bitcast<float>((unsigned int)(val388))*((((float)(((signed char)(((val392>>16u)&255u)))))*((float)(wmma26.z)))+(((float)(((signed char)(((val392>>24u)&255u)))))*((float)(wmma27.z)))))+(cast31*tg_bitcast<float>((unsigned int)(val389))*((((float)(((signed char)(((val393>>0u)&255u)))))*((float)(wmma28.z)))+(((float)(((signed char)(((val393>>8u)&255u)))))*((float)(wmma29.z)))))+(cast31*tg_bitcast<float>((unsigned int)(val390))*((((float)(((signed char)(((val393>>16u)&255u)))))*((float)(wmma30.z)))+(((float)(((signed char)(((val393>>24u)&255u)))))*((float)(wmma31.z))))));
    float alu519 = (alu414?alu518:(buf15+alu518));
    buf15 = alu519;
    unsigned int val394 = (*(buf0+alu213));
    unsigned int val395 = (*(buf0+alu214));
    unsigned int val396 = (*(buf0+alu215));
    unsigned int val397 = (*(buf0+alu216));
    unsigned int val398 = (*(buf0+alu195));
    unsigned int val399 = (*(buf0+alu196));
    unsigned int val400 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+3089)) = buf16;
    }
    float cast32 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val398&65535u)))))));
    float alu524 = ((cast32*tg_bitcast<float>((unsigned int)(val394))*((((float)(((signed char)(((val399>>0u)&255u)))))*((float)(wmma24.w)))+(((float)(((signed char)(((val399>>8u)&255u)))))*((float)(wmma25.w)))))+(cast32*tg_bitcast<float>((unsigned int)(val395))*((((float)(((signed char)(((val399>>16u)&255u)))))*((float)(wmma26.w)))+(((float)(((signed char)(((val399>>24u)&255u)))))*((float)(wmma27.w)))))+(cast32*tg_bitcast<float>((unsigned int)(val396))*((((float)(((signed char)(((val400>>0u)&255u)))))*((float)(wmma28.w)))+(((float)(((signed char)(((val400>>8u)&255u)))))*((float)(wmma29.w)))))+(cast32*tg_bitcast<float>((unsigned int)(val397))*((((float)(((signed char)(((val400>>16u)&255u)))))*((float)(wmma30.w)))+(((float)(((signed char)(((val400>>24u)&255u)))))*((float)(wmma31.w))))));
    float alu525 = (alu414?alu524:(buf16+alu524));
    buf16 = alu525;
    unsigned int val401 = (*(buf0+alu131));
    unsigned int val402 = (*(buf0+alu132));
    unsigned int val403 = (*(buf0+alu133));
    unsigned int val404 = (*(buf0+alu134));
    unsigned int val405 = (*(buf0+alu135));
    unsigned int val406 = (*(buf0+alu136));
    unsigned int val407 = (*(buf0+alu137));
    unsigned int val408 = (*(buf0+alu138));
    unsigned int val409 = (*(buf0+alu217));
    unsigned int val410 = (*(buf0+alu218));
    unsigned int val411 = (*(buf0+alu219));
    unsigned int val412 = (*(buf0+alu220));
    unsigned int val413 = (*(buf0+alu180));
    unsigned int val414 = (*(buf0+alu181));
    unsigned int val415 = (*(buf0+alu182));
    if (alu414) {
      *(data0_5570560+(alu110+32)) = buf17;
    }
    char4 cast33 = make_char4(((signed char)(((val401>>0u)&255u))),((signed char)(((val401>>8u)&255u))),((signed char)(((val401>>16u)&255u))),((signed char)(((val401>>24u)&255u))));
    char4 cast34 = make_char4(((signed char)(((val402>>0u)&255u))),((signed char)(((val402>>8u)&255u))),((signed char)(((val402>>16u)&255u))),((signed char)(((val402>>24u)&255u))));
    char4 cast35 = make_char4(((signed char)(((val403>>0u)&255u))),((signed char)(((val403>>8u)&255u))),((signed char)(((val403>>16u)&255u))),((signed char)(((val403>>24u)&255u))));
    char4 cast36 = make_char4(((signed char)(((val404>>0u)&255u))),((signed char)(((val404>>8u)&255u))),((signed char)(((val404>>16u)&255u))),((signed char)(((val404>>24u)&255u))));
    char4 cast37 = make_char4(((signed char)(((val405>>0u)&255u))),((signed char)(((val405>>8u)&255u))),((signed char)(((val405>>16u)&255u))),((signed char)(((val405>>24u)&255u))));
    char4 cast38 = make_char4(((signed char)(((val406>>0u)&255u))),((signed char)(((val406>>8u)&255u))),((signed char)(((val406>>16u)&255u))),((signed char)(((val406>>24u)&255u))));
    char4 cast39 = make_char4(((signed char)(((val407>>0u)&255u))),((signed char)(((val407>>8u)&255u))),((signed char)(((val407>>16u)&255u))),((signed char)(((val407>>24u)&255u))));
    char4 cast40 = make_char4(((signed char)(((val408>>0u)&255u))),((signed char)(((val408>>8u)&255u))),((signed char)(((val408>>16u)&255u))),((signed char)(((val408>>24u)&255u))));
    int4 wmma32 = __WMMA_8_16_16_signed_char_int(alu418, cast34, cast0);
    int4 wmma33 = __WMMA_8_16_16_signed_char_int(alu419, cast35, cast0);
    int4 wmma34 = __WMMA_8_16_16_signed_char_int(alu420, cast36, cast0);
    int4 wmma35 = __WMMA_8_16_16_signed_char_int(alu421, cast37, cast0);
    int4 wmma36 = __WMMA_8_16_16_signed_char_int(alu422, cast38, cast0);
    int4 wmma37 = __WMMA_8_16_16_signed_char_int(alu423, cast39, cast0);
    int4 wmma38 = __WMMA_8_16_16_signed_char_int(alu424, cast40, cast0);
    int4 wmma39 = __WMMA_8_16_16_signed_char_int(alu425, cast33, cast0);
    float cast41 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val413&65535u)))))));
    float alu530 = ((cast41*tg_bitcast<float>((unsigned int)(val409))*((((float)(((signed char)(((val414>>0u)&255u)))))*((float)(wmma39.x)))+(((float)(((signed char)(((val414>>8u)&255u)))))*((float)(wmma32.x)))))+(cast41*tg_bitcast<float>((unsigned int)(val410))*((((float)(((signed char)(((val414>>16u)&255u)))))*((float)(wmma33.x)))+(((float)(((signed char)(((val414>>24u)&255u)))))*((float)(wmma34.x)))))+(cast41*tg_bitcast<float>((unsigned int)(val411))*((((float)(((signed char)(((val415>>0u)&255u)))))*((float)(wmma35.x)))+(((float)(((signed char)(((val415>>8u)&255u)))))*((float)(wmma36.x)))))+(cast41*tg_bitcast<float>((unsigned int)(val412))*((((float)(((signed char)(((val415>>16u)&255u)))))*((float)(wmma37.x)))+(((float)(((signed char)(((val415>>24u)&255u)))))*((float)(wmma38.x))))));
    float alu531 = (alu414?alu530:(buf17+alu530));
    buf17 = alu531;
    unsigned int val416 = (*(buf0+alu221));
    unsigned int val417 = (*(buf0+alu222));
    unsigned int val418 = (*(buf0+alu223));
    unsigned int val419 = (*(buf0+alu224));
    unsigned int val420 = (*(buf0+alu180));
    unsigned int val421 = (*(buf0+alu181));
    unsigned int val422 = (*(buf0+alu182));
    if (alu414) {
      *(data0_5570560+(alu110+33)) = buf18;
    }
    float cast42 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val420&65535u)))))));
    float alu536 = ((cast42*tg_bitcast<float>((unsigned int)(val416))*((((float)(((signed char)(((val421>>0u)&255u)))))*((float)(wmma39.y)))+(((float)(((signed char)(((val421>>8u)&255u)))))*((float)(wmma32.y)))))+(cast42*tg_bitcast<float>((unsigned int)(val417))*((((float)(((signed char)(((val421>>16u)&255u)))))*((float)(wmma33.y)))+(((float)(((signed char)(((val421>>24u)&255u)))))*((float)(wmma34.y)))))+(cast42*tg_bitcast<float>((unsigned int)(val418))*((((float)(((signed char)(((val422>>0u)&255u)))))*((float)(wmma35.y)))+(((float)(((signed char)(((val422>>8u)&255u)))))*((float)(wmma36.y)))))+(cast42*tg_bitcast<float>((unsigned int)(val419))*((((float)(((signed char)(((val422>>16u)&255u)))))*((float)(wmma37.y)))+(((float)(((signed char)(((val422>>24u)&255u)))))*((float)(wmma38.y))))));
    float alu537 = (alu414?alu536:(buf18+alu536));
    buf18 = alu537;
    unsigned int val423 = (*(buf0+alu217));
    unsigned int val424 = (*(buf0+alu218));
    unsigned int val425 = (*(buf0+alu219));
    unsigned int val426 = (*(buf0+alu220));
    unsigned int val427 = (*(buf0+alu185));
    unsigned int val428 = (*(buf0+alu186));
    unsigned int val429 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+1056)) = buf19;
    }
    float cast43 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val427&65535u)))))));
    float alu542 = ((cast43*tg_bitcast<float>((unsigned int)(val423))*((((float)(((signed char)(((val428>>0u)&255u)))))*((float)(wmma39.z)))+(((float)(((signed char)(((val428>>8u)&255u)))))*((float)(wmma32.z)))))+(cast43*tg_bitcast<float>((unsigned int)(val424))*((((float)(((signed char)(((val428>>16u)&255u)))))*((float)(wmma33.z)))+(((float)(((signed char)(((val428>>24u)&255u)))))*((float)(wmma34.z)))))+(cast43*tg_bitcast<float>((unsigned int)(val425))*((((float)(((signed char)(((val429>>0u)&255u)))))*((float)(wmma35.z)))+(((float)(((signed char)(((val429>>8u)&255u)))))*((float)(wmma36.z)))))+(cast43*tg_bitcast<float>((unsigned int)(val426))*((((float)(((signed char)(((val429>>16u)&255u)))))*((float)(wmma37.z)))+(((float)(((signed char)(((val429>>24u)&255u)))))*((float)(wmma38.z))))));
    float alu543 = (alu414?alu542:(buf19+alu542));
    buf19 = alu543;
    unsigned int val430 = (*(buf0+alu221));
    unsigned int val431 = (*(buf0+alu222));
    unsigned int val432 = (*(buf0+alu223));
    unsigned int val433 = (*(buf0+alu224));
    unsigned int val434 = (*(buf0+alu185));
    unsigned int val435 = (*(buf0+alu186));
    unsigned int val436 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+1057)) = buf20;
    }
    float cast44 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val434&65535u)))))));
    float alu548 = ((cast44*tg_bitcast<float>((unsigned int)(val430))*((((float)(((signed char)(((val435>>0u)&255u)))))*((float)(wmma39.w)))+(((float)(((signed char)(((val435>>8u)&255u)))))*((float)(wmma32.w)))))+(cast44*tg_bitcast<float>((unsigned int)(val431))*((((float)(((signed char)(((val435>>16u)&255u)))))*((float)(wmma33.w)))+(((float)(((signed char)(((val435>>24u)&255u)))))*((float)(wmma34.w)))))+(cast44*tg_bitcast<float>((unsigned int)(val432))*((((float)(((signed char)(((val436>>0u)&255u)))))*((float)(wmma35.w)))+(((float)(((signed char)(((val436>>8u)&255u)))))*((float)(wmma36.w)))))+(cast44*tg_bitcast<float>((unsigned int)(val433))*((((float)(((signed char)(((val436>>16u)&255u)))))*((float)(wmma37.w)))+(((float)(((signed char)(((val436>>24u)&255u)))))*((float)(wmma38.w))))));
    float alu549 = (alu414?alu548:(buf20+alu548));
    buf20 = alu549;
    unsigned int val437 = (*(buf0+alu217));
    unsigned int val438 = (*(buf0+alu218));
    unsigned int val439 = (*(buf0+alu219));
    unsigned int val440 = (*(buf0+alu220));
    unsigned int val441 = (*(buf0+alu190));
    unsigned int val442 = (*(buf0+alu191));
    unsigned int val443 = (*(buf0+alu192));
    if (alu414) {
      *(data0_5570560+(alu110+2080)) = buf21;
    }
    int4 wmma40 = __WMMA_8_16_16_signed_char_int(alu450, cast33, cast0);
    int4 wmma41 = __WMMA_8_16_16_signed_char_int(alu451, cast34, cast0);
    int4 wmma42 = __WMMA_8_16_16_signed_char_int(alu452, cast35, cast0);
    int4 wmma43 = __WMMA_8_16_16_signed_char_int(alu453, cast36, cast0);
    int4 wmma44 = __WMMA_8_16_16_signed_char_int(alu454, cast37, cast0);
    int4 wmma45 = __WMMA_8_16_16_signed_char_int(alu455, cast38, cast0);
    int4 wmma46 = __WMMA_8_16_16_signed_char_int(alu456, cast39, cast0);
    int4 wmma47 = __WMMA_8_16_16_signed_char_int(alu457, cast40, cast0);
    float cast45 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val441&65535u)))))));
    float alu554 = ((cast45*tg_bitcast<float>((unsigned int)(val437))*((((float)(((signed char)(((val442>>0u)&255u)))))*((float)(wmma40.x)))+(((float)(((signed char)(((val442>>8u)&255u)))))*((float)(wmma41.x)))))+(cast45*tg_bitcast<float>((unsigned int)(val438))*((((float)(((signed char)(((val442>>16u)&255u)))))*((float)(wmma42.x)))+(((float)(((signed char)(((val442>>24u)&255u)))))*((float)(wmma43.x)))))+(cast45*tg_bitcast<float>((unsigned int)(val439))*((((float)(((signed char)(((val443>>0u)&255u)))))*((float)(wmma44.x)))+(((float)(((signed char)(((val443>>8u)&255u)))))*((float)(wmma45.x)))))+(cast45*tg_bitcast<float>((unsigned int)(val440))*((((float)(((signed char)(((val443>>16u)&255u)))))*((float)(wmma46.x)))+(((float)(((signed char)(((val443>>24u)&255u)))))*((float)(wmma47.x))))));
    float alu555 = (alu414?alu554:(buf21+alu554));
    buf21 = alu555;
    unsigned int val444 = (*(buf0+alu221));
    unsigned int val445 = (*(buf0+alu222));
    unsigned int val446 = (*(buf0+alu223));
    unsigned int val447 = (*(buf0+alu224));
    unsigned int val448 = (*(buf0+alu190));
    unsigned int val449 = (*(buf0+alu191));
    unsigned int val450 = (*(buf0+alu192));
    if (alu414) {
      *(data0_5570560+(alu110+2081)) = buf22;
    }
    float cast46 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val448&65535u)))))));
    float alu560 = ((cast46*tg_bitcast<float>((unsigned int)(val444))*((((float)(((signed char)(((val449>>0u)&255u)))))*((float)(wmma40.y)))+(((float)(((signed char)(((val449>>8u)&255u)))))*((float)(wmma41.y)))))+(cast46*tg_bitcast<float>((unsigned int)(val445))*((((float)(((signed char)(((val449>>16u)&255u)))))*((float)(wmma42.y)))+(((float)(((signed char)(((val449>>24u)&255u)))))*((float)(wmma43.y)))))+(cast46*tg_bitcast<float>((unsigned int)(val446))*((((float)(((signed char)(((val450>>0u)&255u)))))*((float)(wmma44.y)))+(((float)(((signed char)(((val450>>8u)&255u)))))*((float)(wmma45.y)))))+(cast46*tg_bitcast<float>((unsigned int)(val447))*((((float)(((signed char)(((val450>>16u)&255u)))))*((float)(wmma46.y)))+(((float)(((signed char)(((val450>>24u)&255u)))))*((float)(wmma47.y))))));
    float alu561 = (alu414?alu560:(buf22+alu560));
    buf22 = alu561;
    unsigned int val451 = (*(buf0+alu217));
    unsigned int val452 = (*(buf0+alu218));
    unsigned int val453 = (*(buf0+alu219));
    unsigned int val454 = (*(buf0+alu220));
    unsigned int val455 = (*(buf0+alu195));
    unsigned int val456 = (*(buf0+alu196));
    unsigned int val457 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+3104)) = buf23;
    }
    float cast47 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val455&65535u)))))));
    float alu566 = ((cast47*tg_bitcast<float>((unsigned int)(val451))*((((float)(((signed char)(((val456>>0u)&255u)))))*((float)(wmma40.z)))+(((float)(((signed char)(((val456>>8u)&255u)))))*((float)(wmma41.z)))))+(cast47*tg_bitcast<float>((unsigned int)(val452))*((((float)(((signed char)(((val456>>16u)&255u)))))*((float)(wmma42.z)))+(((float)(((signed char)(((val456>>24u)&255u)))))*((float)(wmma43.z)))))+(cast47*tg_bitcast<float>((unsigned int)(val453))*((((float)(((signed char)(((val457>>0u)&255u)))))*((float)(wmma44.z)))+(((float)(((signed char)(((val457>>8u)&255u)))))*((float)(wmma45.z)))))+(cast47*tg_bitcast<float>((unsigned int)(val454))*((((float)(((signed char)(((val457>>16u)&255u)))))*((float)(wmma46.z)))+(((float)(((signed char)(((val457>>24u)&255u)))))*((float)(wmma47.z))))));
    float alu567 = (alu414?alu566:(buf23+alu566));
    buf23 = alu567;
    unsigned int val458 = (*(buf0+alu221));
    unsigned int val459 = (*(buf0+alu222));
    unsigned int val460 = (*(buf0+alu223));
    unsigned int val461 = (*(buf0+alu224));
    unsigned int val462 = (*(buf0+alu195));
    unsigned int val463 = (*(buf0+alu196));
    unsigned int val464 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+3105)) = buf24;
    }
    float cast48 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val462&65535u)))))));
    float alu572 = ((cast48*tg_bitcast<float>((unsigned int)(val458))*((((float)(((signed char)(((val463>>0u)&255u)))))*((float)(wmma40.w)))+(((float)(((signed char)(((val463>>8u)&255u)))))*((float)(wmma41.w)))))+(cast48*tg_bitcast<float>((unsigned int)(val459))*((((float)(((signed char)(((val463>>16u)&255u)))))*((float)(wmma42.w)))+(((float)(((signed char)(((val463>>24u)&255u)))))*((float)(wmma43.w)))))+(cast48*tg_bitcast<float>((unsigned int)(val460))*((((float)(((signed char)(((val464>>0u)&255u)))))*((float)(wmma44.w)))+(((float)(((signed char)(((val464>>8u)&255u)))))*((float)(wmma45.w)))))+(cast48*tg_bitcast<float>((unsigned int)(val461))*((((float)(((signed char)(((val464>>16u)&255u)))))*((float)(wmma46.w)))+(((float)(((signed char)(((val464>>24u)&255u)))))*((float)(wmma47.w))))));
    float alu573 = (alu414?alu572:(buf24+alu572));
    buf24 = alu573;
    unsigned int val465 = (*(buf0+alu139));
    unsigned int val466 = (*(buf0+alu140));
    unsigned int val467 = (*(buf0+alu141));
    unsigned int val468 = (*(buf0+alu142));
    unsigned int val469 = (*(buf0+alu143));
    unsigned int val470 = (*(buf0+alu144));
    unsigned int val471 = (*(buf0+alu145));
    unsigned int val472 = (*(buf0+alu146));
    unsigned int val473 = (*(buf0+alu225));
    unsigned int val474 = (*(buf0+alu226));
    unsigned int val475 = (*(buf0+alu227));
    unsigned int val476 = (*(buf0+alu228));
    unsigned int val477 = (*(buf0+alu180));
    unsigned int val478 = (*(buf0+alu181));
    unsigned int val479 = (*(buf0+alu182));
    if (alu414) {
      *(data0_5570560+(alu110+48)) = buf25;
    }
    char4 cast49 = make_char4(((signed char)(((val465>>0u)&255u))),((signed char)(((val465>>8u)&255u))),((signed char)(((val465>>16u)&255u))),((signed char)(((val465>>24u)&255u))));
    char4 cast50 = make_char4(((signed char)(((val466>>0u)&255u))),((signed char)(((val466>>8u)&255u))),((signed char)(((val466>>16u)&255u))),((signed char)(((val466>>24u)&255u))));
    char4 cast51 = make_char4(((signed char)(((val467>>0u)&255u))),((signed char)(((val467>>8u)&255u))),((signed char)(((val467>>16u)&255u))),((signed char)(((val467>>24u)&255u))));
    char4 cast52 = make_char4(((signed char)(((val468>>0u)&255u))),((signed char)(((val468>>8u)&255u))),((signed char)(((val468>>16u)&255u))),((signed char)(((val468>>24u)&255u))));
    char4 cast53 = make_char4(((signed char)(((val469>>0u)&255u))),((signed char)(((val469>>8u)&255u))),((signed char)(((val469>>16u)&255u))),((signed char)(((val469>>24u)&255u))));
    char4 cast54 = make_char4(((signed char)(((val470>>0u)&255u))),((signed char)(((val470>>8u)&255u))),((signed char)(((val470>>16u)&255u))),((signed char)(((val470>>24u)&255u))));
    char4 cast55 = make_char4(((signed char)(((val471>>0u)&255u))),((signed char)(((val471>>8u)&255u))),((signed char)(((val471>>16u)&255u))),((signed char)(((val471>>24u)&255u))));
    char4 cast56 = make_char4(((signed char)(((val472>>0u)&255u))),((signed char)(((val472>>8u)&255u))),((signed char)(((val472>>16u)&255u))),((signed char)(((val472>>24u)&255u))));
    int4 wmma48 = __WMMA_8_16_16_signed_char_int(alu418, cast50, cast0);
    int4 wmma49 = __WMMA_8_16_16_signed_char_int(alu419, cast51, cast0);
    int4 wmma50 = __WMMA_8_16_16_signed_char_int(alu420, cast52, cast0);
    int4 wmma51 = __WMMA_8_16_16_signed_char_int(alu421, cast53, cast0);
    int4 wmma52 = __WMMA_8_16_16_signed_char_int(alu422, cast54, cast0);
    int4 wmma53 = __WMMA_8_16_16_signed_char_int(alu423, cast55, cast0);
    int4 wmma54 = __WMMA_8_16_16_signed_char_int(alu424, cast56, cast0);
    int4 wmma55 = __WMMA_8_16_16_signed_char_int(alu425, cast49, cast0);
    float cast57 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val477&65535u)))))));
    float alu578 = ((cast57*tg_bitcast<float>((unsigned int)(val473))*((((float)(((signed char)(((val478>>0u)&255u)))))*((float)(wmma55.x)))+(((float)(((signed char)(((val478>>8u)&255u)))))*((float)(wmma48.x)))))+(cast57*tg_bitcast<float>((unsigned int)(val474))*((((float)(((signed char)(((val478>>16u)&255u)))))*((float)(wmma49.x)))+(((float)(((signed char)(((val478>>24u)&255u)))))*((float)(wmma50.x)))))+(cast57*tg_bitcast<float>((unsigned int)(val475))*((((float)(((signed char)(((val479>>0u)&255u)))))*((float)(wmma51.x)))+(((float)(((signed char)(((val479>>8u)&255u)))))*((float)(wmma52.x)))))+(cast57*tg_bitcast<float>((unsigned int)(val476))*((((float)(((signed char)(((val479>>16u)&255u)))))*((float)(wmma53.x)))+(((float)(((signed char)(((val479>>24u)&255u)))))*((float)(wmma54.x))))));
    float alu579 = (alu414?alu578:(buf25+alu578));
    buf25 = alu579;
    unsigned int val480 = (*(buf0+alu229));
    unsigned int val481 = (*(buf0+alu230));
    unsigned int val482 = (*(buf0+alu231));
    unsigned int val483 = (*(buf0+alu232));
    unsigned int val484 = (*(buf0+alu180));
    unsigned int val485 = (*(buf0+alu181));
    unsigned int val486 = (*(buf0+alu182));
    if (alu414) {
      *(data0_5570560+(alu110+49)) = buf26;
    }
    float cast58 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val484&65535u)))))));
    float alu584 = ((cast58*tg_bitcast<float>((unsigned int)(val480))*((((float)(((signed char)(((val485>>0u)&255u)))))*((float)(wmma55.y)))+(((float)(((signed char)(((val485>>8u)&255u)))))*((float)(wmma48.y)))))+(cast58*tg_bitcast<float>((unsigned int)(val481))*((((float)(((signed char)(((val485>>16u)&255u)))))*((float)(wmma49.y)))+(((float)(((signed char)(((val485>>24u)&255u)))))*((float)(wmma50.y)))))+(cast58*tg_bitcast<float>((unsigned int)(val482))*((((float)(((signed char)(((val486>>0u)&255u)))))*((float)(wmma51.y)))+(((float)(((signed char)(((val486>>8u)&255u)))))*((float)(wmma52.y)))))+(cast58*tg_bitcast<float>((unsigned int)(val483))*((((float)(((signed char)(((val486>>16u)&255u)))))*((float)(wmma53.y)))+(((float)(((signed char)(((val486>>24u)&255u)))))*((float)(wmma54.y))))));
    float alu585 = (alu414?alu584:(buf26+alu584));
    buf26 = alu585;
    unsigned int val487 = (*(buf0+alu225));
    unsigned int val488 = (*(buf0+alu226));
    unsigned int val489 = (*(buf0+alu227));
    unsigned int val490 = (*(buf0+alu228));
    unsigned int val491 = (*(buf0+alu185));
    unsigned int val492 = (*(buf0+alu186));
    unsigned int val493 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+1072)) = buf27;
    }
    float cast59 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val491&65535u)))))));
    float alu590 = ((cast59*tg_bitcast<float>((unsigned int)(val487))*((((float)(((signed char)(((val492>>0u)&255u)))))*((float)(wmma55.z)))+(((float)(((signed char)(((val492>>8u)&255u)))))*((float)(wmma48.z)))))+(cast59*tg_bitcast<float>((unsigned int)(val488))*((((float)(((signed char)(((val492>>16u)&255u)))))*((float)(wmma49.z)))+(((float)(((signed char)(((val492>>24u)&255u)))))*((float)(wmma50.z)))))+(cast59*tg_bitcast<float>((unsigned int)(val489))*((((float)(((signed char)(((val493>>0u)&255u)))))*((float)(wmma51.z)))+(((float)(((signed char)(((val493>>8u)&255u)))))*((float)(wmma52.z)))))+(cast59*tg_bitcast<float>((unsigned int)(val490))*((((float)(((signed char)(((val493>>16u)&255u)))))*((float)(wmma53.z)))+(((float)(((signed char)(((val493>>24u)&255u)))))*((float)(wmma54.z))))));
    float alu591 = (alu414?alu590:(buf27+alu590));
    buf27 = alu591;
    unsigned int val494 = (*(buf0+alu229));
    unsigned int val495 = (*(buf0+alu230));
    unsigned int val496 = (*(buf0+alu231));
    unsigned int val497 = (*(buf0+alu232));
    unsigned int val498 = (*(buf0+alu185));
    unsigned int val499 = (*(buf0+alu186));
    unsigned int val500 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+1073)) = buf28;
    }
    float cast60 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val498&65535u)))))));
    float alu596 = ((cast60*tg_bitcast<float>((unsigned int)(val494))*((((float)(((signed char)(((val499>>0u)&255u)))))*((float)(wmma55.w)))+(((float)(((signed char)(((val499>>8u)&255u)))))*((float)(wmma48.w)))))+(cast60*tg_bitcast<float>((unsigned int)(val495))*((((float)(((signed char)(((val499>>16u)&255u)))))*((float)(wmma49.w)))+(((float)(((signed char)(((val499>>24u)&255u)))))*((float)(wmma50.w)))))+(cast60*tg_bitcast<float>((unsigned int)(val496))*((((float)(((signed char)(((val500>>0u)&255u)))))*((float)(wmma51.w)))+(((float)(((signed char)(((val500>>8u)&255u)))))*((float)(wmma52.w)))))+(cast60*tg_bitcast<float>((unsigned int)(val497))*((((float)(((signed char)(((val500>>16u)&255u)))))*((float)(wmma53.w)))+(((float)(((signed char)(((val500>>24u)&255u)))))*((float)(wmma54.w))))));
    float alu597 = (alu414?alu596:(buf28+alu596));
    buf28 = alu597;
    unsigned int val501 = (*(buf0+alu225));
    unsigned int val502 = (*(buf0+alu226));
    unsigned int val503 = (*(buf0+alu227));
    unsigned int val504 = (*(buf0+alu228));
    unsigned int val505 = (*(buf0+alu190));
    unsigned int val506 = (*(buf0+alu191));
    unsigned int val507 = (*(buf0+alu192));
    if (alu414) {
      *(data0_5570560+(alu110+2096)) = buf29;
    }
    int4 wmma56 = __WMMA_8_16_16_signed_char_int(alu450, cast49, cast0);
    int4 wmma57 = __WMMA_8_16_16_signed_char_int(alu451, cast50, cast0);
    int4 wmma58 = __WMMA_8_16_16_signed_char_int(alu452, cast51, cast0);
    int4 wmma59 = __WMMA_8_16_16_signed_char_int(alu453, cast52, cast0);
    int4 wmma60 = __WMMA_8_16_16_signed_char_int(alu454, cast53, cast0);
    int4 wmma61 = __WMMA_8_16_16_signed_char_int(alu455, cast54, cast0);
    int4 wmma62 = __WMMA_8_16_16_signed_char_int(alu456, cast55, cast0);
    int4 wmma63 = __WMMA_8_16_16_signed_char_int(alu457, cast56, cast0);
    float cast61 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val505&65535u)))))));
    float alu602 = ((cast61*tg_bitcast<float>((unsigned int)(val501))*((((float)(((signed char)(((val506>>0u)&255u)))))*((float)(wmma56.x)))+(((float)(((signed char)(((val506>>8u)&255u)))))*((float)(wmma57.x)))))+(cast61*tg_bitcast<float>((unsigned int)(val502))*((((float)(((signed char)(((val506>>16u)&255u)))))*((float)(wmma58.x)))+(((float)(((signed char)(((val506>>24u)&255u)))))*((float)(wmma59.x)))))+(cast61*tg_bitcast<float>((unsigned int)(val503))*((((float)(((signed char)(((val507>>0u)&255u)))))*((float)(wmma60.x)))+(((float)(((signed char)(((val507>>8u)&255u)))))*((float)(wmma61.x)))))+(cast61*tg_bitcast<float>((unsigned int)(val504))*((((float)(((signed char)(((val507>>16u)&255u)))))*((float)(wmma62.x)))+(((float)(((signed char)(((val507>>24u)&255u)))))*((float)(wmma63.x))))));
    float alu603 = (alu414?alu602:(buf29+alu602));
    buf29 = alu603;
    unsigned int val508 = (*(buf0+alu229));
    unsigned int val509 = (*(buf0+alu230));
    unsigned int val510 = (*(buf0+alu231));
    unsigned int val511 = (*(buf0+alu232));
    unsigned int val512 = (*(buf0+alu190));
    unsigned int val513 = (*(buf0+alu191));
    unsigned int val514 = (*(buf0+alu192));
    if (alu414) {
      *(data0_5570560+(alu110+2097)) = buf30;
    }
    float cast62 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val512&65535u)))))));
    float alu608 = ((cast62*tg_bitcast<float>((unsigned int)(val508))*((((float)(((signed char)(((val513>>0u)&255u)))))*((float)(wmma56.y)))+(((float)(((signed char)(((val513>>8u)&255u)))))*((float)(wmma57.y)))))+(cast62*tg_bitcast<float>((unsigned int)(val509))*((((float)(((signed char)(((val513>>16u)&255u)))))*((float)(wmma58.y)))+(((float)(((signed char)(((val513>>24u)&255u)))))*((float)(wmma59.y)))))+(cast62*tg_bitcast<float>((unsigned int)(val510))*((((float)(((signed char)(((val514>>0u)&255u)))))*((float)(wmma60.y)))+(((float)(((signed char)(((val514>>8u)&255u)))))*((float)(wmma61.y)))))+(cast62*tg_bitcast<float>((unsigned int)(val511))*((((float)(((signed char)(((val514>>16u)&255u)))))*((float)(wmma62.y)))+(((float)(((signed char)(((val514>>24u)&255u)))))*((float)(wmma63.y))))));
    float alu609 = (alu414?alu608:(buf30+alu608));
    buf30 = alu609;
    unsigned int val515 = (*(buf0+alu225));
    unsigned int val516 = (*(buf0+alu226));
    unsigned int val517 = (*(buf0+alu227));
    unsigned int val518 = (*(buf0+alu228));
    unsigned int val519 = (*(buf0+alu195));
    unsigned int val520 = (*(buf0+alu196));
    unsigned int val521 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+3120)) = buf31;
    }
    float cast63 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val519&65535u)))))));
    float alu614 = ((cast63*tg_bitcast<float>((unsigned int)(val515))*((((float)(((signed char)(((val520>>0u)&255u)))))*((float)(wmma56.z)))+(((float)(((signed char)(((val520>>8u)&255u)))))*((float)(wmma57.z)))))+(cast63*tg_bitcast<float>((unsigned int)(val516))*((((float)(((signed char)(((val520>>16u)&255u)))))*((float)(wmma58.z)))+(((float)(((signed char)(((val520>>24u)&255u)))))*((float)(wmma59.z)))))+(cast63*tg_bitcast<float>((unsigned int)(val517))*((((float)(((signed char)(((val521>>0u)&255u)))))*((float)(wmma60.z)))+(((float)(((signed char)(((val521>>8u)&255u)))))*((float)(wmma61.z)))))+(cast63*tg_bitcast<float>((unsigned int)(val518))*((((float)(((signed char)(((val521>>16u)&255u)))))*((float)(wmma62.z)))+(((float)(((signed char)(((val521>>24u)&255u)))))*((float)(wmma63.z))))));
    float alu615 = (alu414?alu614:(buf31+alu614));
    buf31 = alu615;
    unsigned int val522 = (*(buf0+alu229));
    unsigned int val523 = (*(buf0+alu230));
    unsigned int val524 = (*(buf0+alu231));
    unsigned int val525 = (*(buf0+alu232));
    unsigned int val526 = (*(buf0+alu195));
    unsigned int val527 = (*(buf0+alu196));
    unsigned int val528 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+3121)) = buf32;
    }
    float cast64 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val526&65535u)))))));
    float alu620 = ((cast64*tg_bitcast<float>((unsigned int)(val522))*((((float)(((signed char)(((val527>>0u)&255u)))))*((float)(wmma56.w)))+(((float)(((signed char)(((val527>>8u)&255u)))))*((float)(wmma57.w)))))+(cast64*tg_bitcast<float>((unsigned int)(val523))*((((float)(((signed char)(((val527>>16u)&255u)))))*((float)(wmma58.w)))+(((float)(((signed char)(((val527>>24u)&255u)))))*((float)(wmma59.w)))))+(cast64*tg_bitcast<float>((unsigned int)(val524))*((((float)(((signed char)(((val528>>0u)&255u)))))*((float)(wmma60.w)))+(((float)(((signed char)(((val528>>8u)&255u)))))*((float)(wmma61.w)))))+(cast64*tg_bitcast<float>((unsigned int)(val525))*((((float)(((signed char)(((val528>>16u)&255u)))))*((float)(wmma62.w)))+(((float)(((signed char)(((val528>>24u)&255u)))))*((float)(wmma63.w))))));
    float alu621 = (alu414?alu620:(buf32+alu620));
    buf32 = alu621;
    unsigned int val529 = (*(buf0+alu147));
    unsigned int val530 = (*(buf0+alu148));
    unsigned int val531 = (*(buf0+alu149));
    unsigned int val532 = (*(buf0+alu150));
    unsigned int val533 = (*(buf0+alu151));
    unsigned int val534 = (*(buf0+alu152));
    unsigned int val535 = (*(buf0+alu153));
    unsigned int val536 = (*(buf0+alu154));
    unsigned int val537 = (*(buf0+alu233));
    unsigned int val538 = (*(buf0+alu234));
    unsigned int val539 = (*(buf0+alu235));
    unsigned int val540 = (*(buf0+alu236));
    unsigned int val541 = (*(buf0+alu180));
    unsigned int val542 = (*(buf0+alu181));
    unsigned int val543 = (*(buf0+alu182));
    if (alu414) {
      *(data0_5570560+(alu110+64)) = buf33;
    }
    char4 cast65 = make_char4(((signed char)(((val529>>0u)&255u))),((signed char)(((val529>>8u)&255u))),((signed char)(((val529>>16u)&255u))),((signed char)(((val529>>24u)&255u))));
    char4 cast66 = make_char4(((signed char)(((val530>>0u)&255u))),((signed char)(((val530>>8u)&255u))),((signed char)(((val530>>16u)&255u))),((signed char)(((val530>>24u)&255u))));
    char4 cast67 = make_char4(((signed char)(((val531>>0u)&255u))),((signed char)(((val531>>8u)&255u))),((signed char)(((val531>>16u)&255u))),((signed char)(((val531>>24u)&255u))));
    char4 cast68 = make_char4(((signed char)(((val532>>0u)&255u))),((signed char)(((val532>>8u)&255u))),((signed char)(((val532>>16u)&255u))),((signed char)(((val532>>24u)&255u))));
    char4 cast69 = make_char4(((signed char)(((val533>>0u)&255u))),((signed char)(((val533>>8u)&255u))),((signed char)(((val533>>16u)&255u))),((signed char)(((val533>>24u)&255u))));
    char4 cast70 = make_char4(((signed char)(((val534>>0u)&255u))),((signed char)(((val534>>8u)&255u))),((signed char)(((val534>>16u)&255u))),((signed char)(((val534>>24u)&255u))));
    char4 cast71 = make_char4(((signed char)(((val535>>0u)&255u))),((signed char)(((val535>>8u)&255u))),((signed char)(((val535>>16u)&255u))),((signed char)(((val535>>24u)&255u))));
    char4 cast72 = make_char4(((signed char)(((val536>>0u)&255u))),((signed char)(((val536>>8u)&255u))),((signed char)(((val536>>16u)&255u))),((signed char)(((val536>>24u)&255u))));
    int4 wmma64 = __WMMA_8_16_16_signed_char_int(alu418, cast66, cast0);
    int4 wmma65 = __WMMA_8_16_16_signed_char_int(alu419, cast67, cast0);
    int4 wmma66 = __WMMA_8_16_16_signed_char_int(alu420, cast68, cast0);
    int4 wmma67 = __WMMA_8_16_16_signed_char_int(alu421, cast69, cast0);
    int4 wmma68 = __WMMA_8_16_16_signed_char_int(alu422, cast70, cast0);
    int4 wmma69 = __WMMA_8_16_16_signed_char_int(alu423, cast71, cast0);
    int4 wmma70 = __WMMA_8_16_16_signed_char_int(alu424, cast72, cast0);
    int4 wmma71 = __WMMA_8_16_16_signed_char_int(alu425, cast65, cast0);
    float cast73 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val541&65535u)))))));
    float alu626 = ((cast73*tg_bitcast<float>((unsigned int)(val537))*((((float)(((signed char)(((val542>>0u)&255u)))))*((float)(wmma71.x)))+(((float)(((signed char)(((val542>>8u)&255u)))))*((float)(wmma64.x)))))+(cast73*tg_bitcast<float>((unsigned int)(val538))*((((float)(((signed char)(((val542>>16u)&255u)))))*((float)(wmma65.x)))+(((float)(((signed char)(((val542>>24u)&255u)))))*((float)(wmma66.x)))))+(cast73*tg_bitcast<float>((unsigned int)(val539))*((((float)(((signed char)(((val543>>0u)&255u)))))*((float)(wmma67.x)))+(((float)(((signed char)(((val543>>8u)&255u)))))*((float)(wmma68.x)))))+(cast73*tg_bitcast<float>((unsigned int)(val540))*((((float)(((signed char)(((val543>>16u)&255u)))))*((float)(wmma69.x)))+(((float)(((signed char)(((val543>>24u)&255u)))))*((float)(wmma70.x))))));
    float alu627 = (alu414?alu626:(buf33+alu626));
    buf33 = alu627;
    unsigned int val544 = (*(buf0+alu237));
    unsigned int val545 = (*(buf0+alu238));
    unsigned int val546 = (*(buf0+alu239));
    unsigned int val547 = (*(buf0+alu240));
    unsigned int val548 = (*(buf0+alu180));
    unsigned int val549 = (*(buf0+alu181));
    unsigned int val550 = (*(buf0+alu182));
    if (alu414) {
      *(data0_5570560+(alu110+65)) = buf34;
    }
    float cast74 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val548&65535u)))))));
    float alu632 = ((cast74*tg_bitcast<float>((unsigned int)(val544))*((((float)(((signed char)(((val549>>0u)&255u)))))*((float)(wmma71.y)))+(((float)(((signed char)(((val549>>8u)&255u)))))*((float)(wmma64.y)))))+(cast74*tg_bitcast<float>((unsigned int)(val545))*((((float)(((signed char)(((val549>>16u)&255u)))))*((float)(wmma65.y)))+(((float)(((signed char)(((val549>>24u)&255u)))))*((float)(wmma66.y)))))+(cast74*tg_bitcast<float>((unsigned int)(val546))*((((float)(((signed char)(((val550>>0u)&255u)))))*((float)(wmma67.y)))+(((float)(((signed char)(((val550>>8u)&255u)))))*((float)(wmma68.y)))))+(cast74*tg_bitcast<float>((unsigned int)(val547))*((((float)(((signed char)(((val550>>16u)&255u)))))*((float)(wmma69.y)))+(((float)(((signed char)(((val550>>24u)&255u)))))*((float)(wmma70.y))))));
    float alu633 = (alu414?alu632:(buf34+alu632));
    buf34 = alu633;
    unsigned int val551 = (*(buf0+alu233));
    unsigned int val552 = (*(buf0+alu234));
    unsigned int val553 = (*(buf0+alu235));
    unsigned int val554 = (*(buf0+alu236));
    unsigned int val555 = (*(buf0+alu185));
    unsigned int val556 = (*(buf0+alu186));
    unsigned int val557 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+1088)) = buf35;
    }
    float cast75 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val555&65535u)))))));
    float alu638 = ((cast75*tg_bitcast<float>((unsigned int)(val551))*((((float)(((signed char)(((val556>>0u)&255u)))))*((float)(wmma71.z)))+(((float)(((signed char)(((val556>>8u)&255u)))))*((float)(wmma64.z)))))+(cast75*tg_bitcast<float>((unsigned int)(val552))*((((float)(((signed char)(((val556>>16u)&255u)))))*((float)(wmma65.z)))+(((float)(((signed char)(((val556>>24u)&255u)))))*((float)(wmma66.z)))))+(cast75*tg_bitcast<float>((unsigned int)(val553))*((((float)(((signed char)(((val557>>0u)&255u)))))*((float)(wmma67.z)))+(((float)(((signed char)(((val557>>8u)&255u)))))*((float)(wmma68.z)))))+(cast75*tg_bitcast<float>((unsigned int)(val554))*((((float)(((signed char)(((val557>>16u)&255u)))))*((float)(wmma69.z)))+(((float)(((signed char)(((val557>>24u)&255u)))))*((float)(wmma70.z))))));
    float alu639 = (alu414?alu638:(buf35+alu638));
    buf35 = alu639;
    unsigned int val558 = (*(buf0+alu237));
    unsigned int val559 = (*(buf0+alu238));
    unsigned int val560 = (*(buf0+alu239));
    unsigned int val561 = (*(buf0+alu240));
    unsigned int val562 = (*(buf0+alu185));
    unsigned int val563 = (*(buf0+alu186));
    unsigned int val564 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+1089)) = buf36;
    }
    float cast76 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val562&65535u)))))));
    float alu644 = ((cast76*tg_bitcast<float>((unsigned int)(val558))*((((float)(((signed char)(((val563>>0u)&255u)))))*((float)(wmma71.w)))+(((float)(((signed char)(((val563>>8u)&255u)))))*((float)(wmma64.w)))))+(cast76*tg_bitcast<float>((unsigned int)(val559))*((((float)(((signed char)(((val563>>16u)&255u)))))*((float)(wmma65.w)))+(((float)(((signed char)(((val563>>24u)&255u)))))*((float)(wmma66.w)))))+(cast76*tg_bitcast<float>((unsigned int)(val560))*((((float)(((signed char)(((val564>>0u)&255u)))))*((float)(wmma67.w)))+(((float)(((signed char)(((val564>>8u)&255u)))))*((float)(wmma68.w)))))+(cast76*tg_bitcast<float>((unsigned int)(val561))*((((float)(((signed char)(((val564>>16u)&255u)))))*((float)(wmma69.w)))+(((float)(((signed char)(((val564>>24u)&255u)))))*((float)(wmma70.w))))));
    float alu645 = (alu414?alu644:(buf36+alu644));
    buf36 = alu645;
    unsigned int val565 = (*(buf0+alu233));
    unsigned int val566 = (*(buf0+alu234));
    unsigned int val567 = (*(buf0+alu235));
    unsigned int val568 = (*(buf0+alu236));
    unsigned int val569 = (*(buf0+alu190));
    unsigned int val570 = (*(buf0+alu191));
    unsigned int val571 = (*(buf0+alu192));
    if (alu414) {
      *(data0_5570560+(alu110+2112)) = buf37;
    }
    int4 wmma72 = __WMMA_8_16_16_signed_char_int(alu450, cast65, cast0);
    int4 wmma73 = __WMMA_8_16_16_signed_char_int(alu451, cast66, cast0);
    int4 wmma74 = __WMMA_8_16_16_signed_char_int(alu452, cast67, cast0);
    int4 wmma75 = __WMMA_8_16_16_signed_char_int(alu453, cast68, cast0);
    int4 wmma76 = __WMMA_8_16_16_signed_char_int(alu454, cast69, cast0);
    int4 wmma77 = __WMMA_8_16_16_signed_char_int(alu455, cast70, cast0);
    int4 wmma78 = __WMMA_8_16_16_signed_char_int(alu456, cast71, cast0);
    int4 wmma79 = __WMMA_8_16_16_signed_char_int(alu457, cast72, cast0);
    float cast77 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val569&65535u)))))));
    float alu650 = ((cast77*tg_bitcast<float>((unsigned int)(val565))*((((float)(((signed char)(((val570>>0u)&255u)))))*((float)(wmma72.x)))+(((float)(((signed char)(((val570>>8u)&255u)))))*((float)(wmma73.x)))))+(cast77*tg_bitcast<float>((unsigned int)(val566))*((((float)(((signed char)(((val570>>16u)&255u)))))*((float)(wmma74.x)))+(((float)(((signed char)(((val570>>24u)&255u)))))*((float)(wmma75.x)))))+(cast77*tg_bitcast<float>((unsigned int)(val567))*((((float)(((signed char)(((val571>>0u)&255u)))))*((float)(wmma76.x)))+(((float)(((signed char)(((val571>>8u)&255u)))))*((float)(wmma77.x)))))+(cast77*tg_bitcast<float>((unsigned int)(val568))*((((float)(((signed char)(((val571>>16u)&255u)))))*((float)(wmma78.x)))+(((float)(((signed char)(((val571>>24u)&255u)))))*((float)(wmma79.x))))));
    float alu651 = (alu414?alu650:(buf37+alu650));
    buf37 = alu651;
    unsigned int val572 = (*(buf0+alu237));
    unsigned int val573 = (*(buf0+alu238));
    unsigned int val574 = (*(buf0+alu239));
    unsigned int val575 = (*(buf0+alu240));
    unsigned int val576 = (*(buf0+alu190));
    unsigned int val577 = (*(buf0+alu191));
    unsigned int val578 = (*(buf0+alu192));
    if (alu414) {
      *(data0_5570560+(alu110+2113)) = buf38;
    }
    float cast78 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val576&65535u)))))));
    float alu656 = ((cast78*tg_bitcast<float>((unsigned int)(val572))*((((float)(((signed char)(((val577>>0u)&255u)))))*((float)(wmma72.y)))+(((float)(((signed char)(((val577>>8u)&255u)))))*((float)(wmma73.y)))))+(cast78*tg_bitcast<float>((unsigned int)(val573))*((((float)(((signed char)(((val577>>16u)&255u)))))*((float)(wmma74.y)))+(((float)(((signed char)(((val577>>24u)&255u)))))*((float)(wmma75.y)))))+(cast78*tg_bitcast<float>((unsigned int)(val574))*((((float)(((signed char)(((val578>>0u)&255u)))))*((float)(wmma76.y)))+(((float)(((signed char)(((val578>>8u)&255u)))))*((float)(wmma77.y)))))+(cast78*tg_bitcast<float>((unsigned int)(val575))*((((float)(((signed char)(((val578>>16u)&255u)))))*((float)(wmma78.y)))+(((float)(((signed char)(((val578>>24u)&255u)))))*((float)(wmma79.y))))));
    float alu657 = (alu414?alu656:(buf38+alu656));
    buf38 = alu657;
    unsigned int val579 = (*(buf0+alu233));
    unsigned int val580 = (*(buf0+alu234));
    unsigned int val581 = (*(buf0+alu235));
    unsigned int val582 = (*(buf0+alu236));
    unsigned int val583 = (*(buf0+alu195));
    unsigned int val584 = (*(buf0+alu196));
    unsigned int val585 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+3136)) = buf39;
    }
    float cast79 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val583&65535u)))))));
    float alu662 = ((cast79*tg_bitcast<float>((unsigned int)(val579))*((((float)(((signed char)(((val584>>0u)&255u)))))*((float)(wmma72.z)))+(((float)(((signed char)(((val584>>8u)&255u)))))*((float)(wmma73.z)))))+(cast79*tg_bitcast<float>((unsigned int)(val580))*((((float)(((signed char)(((val584>>16u)&255u)))))*((float)(wmma74.z)))+(((float)(((signed char)(((val584>>24u)&255u)))))*((float)(wmma75.z)))))+(cast79*tg_bitcast<float>((unsigned int)(val581))*((((float)(((signed char)(((val585>>0u)&255u)))))*((float)(wmma76.z)))+(((float)(((signed char)(((val585>>8u)&255u)))))*((float)(wmma77.z)))))+(cast79*tg_bitcast<float>((unsigned int)(val582))*((((float)(((signed char)(((val585>>16u)&255u)))))*((float)(wmma78.z)))+(((float)(((signed char)(((val585>>24u)&255u)))))*((float)(wmma79.z))))));
    float alu663 = (alu414?alu662:(buf39+alu662));
    buf39 = alu663;
    unsigned int val586 = (*(buf0+alu237));
    unsigned int val587 = (*(buf0+alu238));
    unsigned int val588 = (*(buf0+alu239));
    unsigned int val589 = (*(buf0+alu240));
    unsigned int val590 = (*(buf0+alu195));
    unsigned int val591 = (*(buf0+alu196));
    unsigned int val592 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+3137)) = buf40;
    }
    float cast80 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val590&65535u)))))));
    float alu668 = ((cast80*tg_bitcast<float>((unsigned int)(val586))*((((float)(((signed char)(((val591>>0u)&255u)))))*((float)(wmma72.w)))+(((float)(((signed char)(((val591>>8u)&255u)))))*((float)(wmma73.w)))))+(cast80*tg_bitcast<float>((unsigned int)(val587))*((((float)(((signed char)(((val591>>16u)&255u)))))*((float)(wmma74.w)))+(((float)(((signed char)(((val591>>24u)&255u)))))*((float)(wmma75.w)))))+(cast80*tg_bitcast<float>((unsigned int)(val588))*((((float)(((signed char)(((val592>>0u)&255u)))))*((float)(wmma76.w)))+(((float)(((signed char)(((val592>>8u)&255u)))))*((float)(wmma77.w)))))+(cast80*tg_bitcast<float>((unsigned int)(val589))*((((float)(((signed char)(((val592>>16u)&255u)))))*((float)(wmma78.w)))+(((float)(((signed char)(((val592>>24u)&255u)))))*((float)(wmma79.w))))));
    float alu669 = (alu414?alu668:(buf40+alu668));
    buf40 = alu669;
    unsigned int val593 = (*(buf0+alu155));
    unsigned int val594 = (*(buf0+alu156));
    unsigned int val595 = (*(buf0+alu157));
    unsigned int val596 = (*(buf0+alu158));
    unsigned int val597 = (*(buf0+alu159));
    unsigned int val598 = (*(buf0+alu160));
    unsigned int val599 = (*(buf0+alu161));
    unsigned int val600 = (*(buf0+alu162));
    unsigned int val601 = (*(buf0+alu241));
    unsigned int val602 = (*(buf0+alu242));
    unsigned int val603 = (*(buf0+alu243));
    unsigned int val604 = (*(buf0+alu244));
    unsigned int val605 = (*(buf0+alu180));
    unsigned int val606 = (*(buf0+alu181));
    unsigned int val607 = (*(buf0+alu182));
    if (alu414) {
      *(data0_5570560+(alu110+80)) = buf41;
    }
    char4 cast81 = make_char4(((signed char)(((val593>>0u)&255u))),((signed char)(((val593>>8u)&255u))),((signed char)(((val593>>16u)&255u))),((signed char)(((val593>>24u)&255u))));
    char4 cast82 = make_char4(((signed char)(((val594>>0u)&255u))),((signed char)(((val594>>8u)&255u))),((signed char)(((val594>>16u)&255u))),((signed char)(((val594>>24u)&255u))));
    char4 cast83 = make_char4(((signed char)(((val595>>0u)&255u))),((signed char)(((val595>>8u)&255u))),((signed char)(((val595>>16u)&255u))),((signed char)(((val595>>24u)&255u))));
    char4 cast84 = make_char4(((signed char)(((val596>>0u)&255u))),((signed char)(((val596>>8u)&255u))),((signed char)(((val596>>16u)&255u))),((signed char)(((val596>>24u)&255u))));
    char4 cast85 = make_char4(((signed char)(((val597>>0u)&255u))),((signed char)(((val597>>8u)&255u))),((signed char)(((val597>>16u)&255u))),((signed char)(((val597>>24u)&255u))));
    char4 cast86 = make_char4(((signed char)(((val598>>0u)&255u))),((signed char)(((val598>>8u)&255u))),((signed char)(((val598>>16u)&255u))),((signed char)(((val598>>24u)&255u))));
    char4 cast87 = make_char4(((signed char)(((val599>>0u)&255u))),((signed char)(((val599>>8u)&255u))),((signed char)(((val599>>16u)&255u))),((signed char)(((val599>>24u)&255u))));
    char4 cast88 = make_char4(((signed char)(((val600>>0u)&255u))),((signed char)(((val600>>8u)&255u))),((signed char)(((val600>>16u)&255u))),((signed char)(((val600>>24u)&255u))));
    int4 wmma80 = __WMMA_8_16_16_signed_char_int(alu418, cast82, cast0);
    int4 wmma81 = __WMMA_8_16_16_signed_char_int(alu419, cast83, cast0);
    int4 wmma82 = __WMMA_8_16_16_signed_char_int(alu420, cast84, cast0);
    int4 wmma83 = __WMMA_8_16_16_signed_char_int(alu421, cast85, cast0);
    int4 wmma84 = __WMMA_8_16_16_signed_char_int(alu422, cast86, cast0);
    int4 wmma85 = __WMMA_8_16_16_signed_char_int(alu423, cast87, cast0);
    int4 wmma86 = __WMMA_8_16_16_signed_char_int(alu424, cast88, cast0);
    int4 wmma87 = __WMMA_8_16_16_signed_char_int(alu425, cast81, cast0);
    float cast89 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val605&65535u)))))));
    float alu674 = ((cast89*tg_bitcast<float>((unsigned int)(val601))*((((float)(((signed char)(((val606>>0u)&255u)))))*((float)(wmma87.x)))+(((float)(((signed char)(((val606>>8u)&255u)))))*((float)(wmma80.x)))))+(cast89*tg_bitcast<float>((unsigned int)(val602))*((((float)(((signed char)(((val606>>16u)&255u)))))*((float)(wmma81.x)))+(((float)(((signed char)(((val606>>24u)&255u)))))*((float)(wmma82.x)))))+(cast89*tg_bitcast<float>((unsigned int)(val603))*((((float)(((signed char)(((val607>>0u)&255u)))))*((float)(wmma83.x)))+(((float)(((signed char)(((val607>>8u)&255u)))))*((float)(wmma84.x)))))+(cast89*tg_bitcast<float>((unsigned int)(val604))*((((float)(((signed char)(((val607>>16u)&255u)))))*((float)(wmma85.x)))+(((float)(((signed char)(((val607>>24u)&255u)))))*((float)(wmma86.x))))));
    float alu675 = (alu414?alu674:(buf41+alu674));
    buf41 = alu675;
    unsigned int val608 = (*(buf0+alu245));
    unsigned int val609 = (*(buf0+alu246));
    unsigned int val610 = (*(buf0+alu247));
    unsigned int val611 = (*(buf0+alu248));
    unsigned int val612 = (*(buf0+alu180));
    unsigned int val613 = (*(buf0+alu181));
    unsigned int val614 = (*(buf0+alu182));
    if (alu414) {
      *(data0_5570560+(alu110+81)) = buf42;
    }
    float cast90 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val612&65535u)))))));
    float alu680 = ((cast90*tg_bitcast<float>((unsigned int)(val608))*((((float)(((signed char)(((val613>>0u)&255u)))))*((float)(wmma87.y)))+(((float)(((signed char)(((val613>>8u)&255u)))))*((float)(wmma80.y)))))+(cast90*tg_bitcast<float>((unsigned int)(val609))*((((float)(((signed char)(((val613>>16u)&255u)))))*((float)(wmma81.y)))+(((float)(((signed char)(((val613>>24u)&255u)))))*((float)(wmma82.y)))))+(cast90*tg_bitcast<float>((unsigned int)(val610))*((((float)(((signed char)(((val614>>0u)&255u)))))*((float)(wmma83.y)))+(((float)(((signed char)(((val614>>8u)&255u)))))*((float)(wmma84.y)))))+(cast90*tg_bitcast<float>((unsigned int)(val611))*((((float)(((signed char)(((val614>>16u)&255u)))))*((float)(wmma85.y)))+(((float)(((signed char)(((val614>>24u)&255u)))))*((float)(wmma86.y))))));
    float alu681 = (alu414?alu680:(buf42+alu680));
    buf42 = alu681;
    unsigned int val615 = (*(buf0+alu241));
    unsigned int val616 = (*(buf0+alu242));
    unsigned int val617 = (*(buf0+alu243));
    unsigned int val618 = (*(buf0+alu244));
    unsigned int val619 = (*(buf0+alu185));
    unsigned int val620 = (*(buf0+alu186));
    unsigned int val621 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+1104)) = buf43;
    }
    float cast91 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val619&65535u)))))));
    float alu686 = ((cast91*tg_bitcast<float>((unsigned int)(val615))*((((float)(((signed char)(((val620>>0u)&255u)))))*((float)(wmma87.z)))+(((float)(((signed char)(((val620>>8u)&255u)))))*((float)(wmma80.z)))))+(cast91*tg_bitcast<float>((unsigned int)(val616))*((((float)(((signed char)(((val620>>16u)&255u)))))*((float)(wmma81.z)))+(((float)(((signed char)(((val620>>24u)&255u)))))*((float)(wmma82.z)))))+(cast91*tg_bitcast<float>((unsigned int)(val617))*((((float)(((signed char)(((val621>>0u)&255u)))))*((float)(wmma83.z)))+(((float)(((signed char)(((val621>>8u)&255u)))))*((float)(wmma84.z)))))+(cast91*tg_bitcast<float>((unsigned int)(val618))*((((float)(((signed char)(((val621>>16u)&255u)))))*((float)(wmma85.z)))+(((float)(((signed char)(((val621>>24u)&255u)))))*((float)(wmma86.z))))));
    float alu687 = (alu414?alu686:(buf43+alu686));
    buf43 = alu687;
    unsigned int val622 = (*(buf0+alu245));
    unsigned int val623 = (*(buf0+alu246));
    unsigned int val624 = (*(buf0+alu247));
    unsigned int val625 = (*(buf0+alu248));
    unsigned int val626 = (*(buf0+alu185));
    unsigned int val627 = (*(buf0+alu186));
    unsigned int val628 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+1105)) = buf44;
    }
    float cast92 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val626&65535u)))))));
    float alu692 = ((cast92*tg_bitcast<float>((unsigned int)(val622))*((((float)(((signed char)(((val627>>0u)&255u)))))*((float)(wmma87.w)))+(((float)(((signed char)(((val627>>8u)&255u)))))*((float)(wmma80.w)))))+(cast92*tg_bitcast<float>((unsigned int)(val623))*((((float)(((signed char)(((val627>>16u)&255u)))))*((float)(wmma81.w)))+(((float)(((signed char)(((val627>>24u)&255u)))))*((float)(wmma82.w)))))+(cast92*tg_bitcast<float>((unsigned int)(val624))*((((float)(((signed char)(((val628>>0u)&255u)))))*((float)(wmma83.w)))+(((float)(((signed char)(((val628>>8u)&255u)))))*((float)(wmma84.w)))))+(cast92*tg_bitcast<float>((unsigned int)(val625))*((((float)(((signed char)(((val628>>16u)&255u)))))*((float)(wmma85.w)))+(((float)(((signed char)(((val628>>24u)&255u)))))*((float)(wmma86.w))))));
    float alu693 = (alu414?alu692:(buf44+alu692));
    buf44 = alu693;
    unsigned int val629 = (*(buf0+alu241));
    unsigned int val630 = (*(buf0+alu242));
    unsigned int val631 = (*(buf0+alu243));
    unsigned int val632 = (*(buf0+alu244));
    unsigned int val633 = (*(buf0+alu190));
    unsigned int val634 = (*(buf0+alu191));
    unsigned int val635 = (*(buf0+alu192));
    if (alu414) {
      *(data0_5570560+(alu110+2128)) = buf45;
    }
    int4 wmma88 = __WMMA_8_16_16_signed_char_int(alu450, cast81, cast0);
    int4 wmma89 = __WMMA_8_16_16_signed_char_int(alu451, cast82, cast0);
    int4 wmma90 = __WMMA_8_16_16_signed_char_int(alu452, cast83, cast0);
    int4 wmma91 = __WMMA_8_16_16_signed_char_int(alu453, cast84, cast0);
    int4 wmma92 = __WMMA_8_16_16_signed_char_int(alu454, cast85, cast0);
    int4 wmma93 = __WMMA_8_16_16_signed_char_int(alu455, cast86, cast0);
    int4 wmma94 = __WMMA_8_16_16_signed_char_int(alu456, cast87, cast0);
    int4 wmma95 = __WMMA_8_16_16_signed_char_int(alu457, cast88, cast0);
    float cast93 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val633&65535u)))))));
    float alu698 = ((cast93*tg_bitcast<float>((unsigned int)(val629))*((((float)(((signed char)(((val634>>0u)&255u)))))*((float)(wmma88.x)))+(((float)(((signed char)(((val634>>8u)&255u)))))*((float)(wmma89.x)))))+(cast93*tg_bitcast<float>((unsigned int)(val630))*((((float)(((signed char)(((val634>>16u)&255u)))))*((float)(wmma90.x)))+(((float)(((signed char)(((val634>>24u)&255u)))))*((float)(wmma91.x)))))+(cast93*tg_bitcast<float>((unsigned int)(val631))*((((float)(((signed char)(((val635>>0u)&255u)))))*((float)(wmma92.x)))+(((float)(((signed char)(((val635>>8u)&255u)))))*((float)(wmma93.x)))))+(cast93*tg_bitcast<float>((unsigned int)(val632))*((((float)(((signed char)(((val635>>16u)&255u)))))*((float)(wmma94.x)))+(((float)(((signed char)(((val635>>24u)&255u)))))*((float)(wmma95.x))))));
    float alu699 = (alu414?alu698:(buf45+alu698));
    buf45 = alu699;
    unsigned int val636 = (*(buf0+alu245));
    unsigned int val637 = (*(buf0+alu246));
    unsigned int val638 = (*(buf0+alu247));
    unsigned int val639 = (*(buf0+alu248));
    unsigned int val640 = (*(buf0+alu190));
    unsigned int val641 = (*(buf0+alu191));
    unsigned int val642 = (*(buf0+alu192));
    if (alu414) {
      *(data0_5570560+(alu110+2129)) = buf46;
    }
    float cast94 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val640&65535u)))))));
    float alu704 = ((cast94*tg_bitcast<float>((unsigned int)(val636))*((((float)(((signed char)(((val641>>0u)&255u)))))*((float)(wmma88.y)))+(((float)(((signed char)(((val641>>8u)&255u)))))*((float)(wmma89.y)))))+(cast94*tg_bitcast<float>((unsigned int)(val637))*((((float)(((signed char)(((val641>>16u)&255u)))))*((float)(wmma90.y)))+(((float)(((signed char)(((val641>>24u)&255u)))))*((float)(wmma91.y)))))+(cast94*tg_bitcast<float>((unsigned int)(val638))*((((float)(((signed char)(((val642>>0u)&255u)))))*((float)(wmma92.y)))+(((float)(((signed char)(((val642>>8u)&255u)))))*((float)(wmma93.y)))))+(cast94*tg_bitcast<float>((unsigned int)(val639))*((((float)(((signed char)(((val642>>16u)&255u)))))*((float)(wmma94.y)))+(((float)(((signed char)(((val642>>24u)&255u)))))*((float)(wmma95.y))))));
    float alu705 = (alu414?alu704:(buf46+alu704));
    buf46 = alu705;
    unsigned int val643 = (*(buf0+alu241));
    unsigned int val644 = (*(buf0+alu242));
    unsigned int val645 = (*(buf0+alu243));
    unsigned int val646 = (*(buf0+alu244));
    unsigned int val647 = (*(buf0+alu195));
    unsigned int val648 = (*(buf0+alu196));
    unsigned int val649 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+3152)) = buf47;
    }
    float cast95 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val647&65535u)))))));
    float alu710 = ((cast95*tg_bitcast<float>((unsigned int)(val643))*((((float)(((signed char)(((val648>>0u)&255u)))))*((float)(wmma88.z)))+(((float)(((signed char)(((val648>>8u)&255u)))))*((float)(wmma89.z)))))+(cast95*tg_bitcast<float>((unsigned int)(val644))*((((float)(((signed char)(((val648>>16u)&255u)))))*((float)(wmma90.z)))+(((float)(((signed char)(((val648>>24u)&255u)))))*((float)(wmma91.z)))))+(cast95*tg_bitcast<float>((unsigned int)(val645))*((((float)(((signed char)(((val649>>0u)&255u)))))*((float)(wmma92.z)))+(((float)(((signed char)(((val649>>8u)&255u)))))*((float)(wmma93.z)))))+(cast95*tg_bitcast<float>((unsigned int)(val646))*((((float)(((signed char)(((val649>>16u)&255u)))))*((float)(wmma94.z)))+(((float)(((signed char)(((val649>>24u)&255u)))))*((float)(wmma95.z))))));
    float alu711 = (alu414?alu710:(buf47+alu710));
    buf47 = alu711;
    unsigned int val650 = (*(buf0+alu245));
    unsigned int val651 = (*(buf0+alu246));
    unsigned int val652 = (*(buf0+alu247));
    unsigned int val653 = (*(buf0+alu248));
    unsigned int val654 = (*(buf0+alu195));
    unsigned int val655 = (*(buf0+alu196));
    unsigned int val656 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+3153)) = buf48;
    }
    float cast96 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val654&65535u)))))));
    float alu716 = ((cast96*tg_bitcast<float>((unsigned int)(val650))*((((float)(((signed char)(((val655>>0u)&255u)))))*((float)(wmma88.w)))+(((float)(((signed char)(((val655>>8u)&255u)))))*((float)(wmma89.w)))))+(cast96*tg_bitcast<float>((unsigned int)(val651))*((((float)(((signed char)(((val655>>16u)&255u)))))*((float)(wmma90.w)))+(((float)(((signed char)(((val655>>24u)&255u)))))*((float)(wmma91.w)))))+(cast96*tg_bitcast<float>((unsigned int)(val652))*((((float)(((signed char)(((val656>>0u)&255u)))))*((float)(wmma92.w)))+(((float)(((signed char)(((val656>>8u)&255u)))))*((float)(wmma93.w)))))+(cast96*tg_bitcast<float>((unsigned int)(val653))*((((float)(((signed char)(((val656>>16u)&255u)))))*((float)(wmma94.w)))+(((float)(((signed char)(((val656>>24u)&255u)))))*((float)(wmma95.w))))));
    float alu717 = (alu414?alu716:(buf48+alu716));
    buf48 = alu717;
    unsigned int val657 = (*(buf0+alu163));
    unsigned int val658 = (*(buf0+alu164));
    unsigned int val659 = (*(buf0+alu165));
    unsigned int val660 = (*(buf0+alu166));
    unsigned int val661 = (*(buf0+alu167));
    unsigned int val662 = (*(buf0+alu168));
    unsigned int val663 = (*(buf0+alu169));
    unsigned int val664 = (*(buf0+alu170));
    unsigned int val665 = (*(buf0+alu249));
    unsigned int val666 = (*(buf0+alu250));
    unsigned int val667 = (*(buf0+alu251));
    unsigned int val668 = (*(buf0+alu252));
    unsigned int val669 = (*(buf0+alu180));
    unsigned int val670 = (*(buf0+alu181));
    unsigned int val671 = (*(buf0+alu182));
    if (alu414) {
      *(data0_5570560+(alu110+96)) = buf49;
    }
    char4 cast97 = make_char4(((signed char)(((val657>>0u)&255u))),((signed char)(((val657>>8u)&255u))),((signed char)(((val657>>16u)&255u))),((signed char)(((val657>>24u)&255u))));
    char4 cast98 = make_char4(((signed char)(((val658>>0u)&255u))),((signed char)(((val658>>8u)&255u))),((signed char)(((val658>>16u)&255u))),((signed char)(((val658>>24u)&255u))));
    char4 cast99 = make_char4(((signed char)(((val659>>0u)&255u))),((signed char)(((val659>>8u)&255u))),((signed char)(((val659>>16u)&255u))),((signed char)(((val659>>24u)&255u))));
    char4 cast100 = make_char4(((signed char)(((val660>>0u)&255u))),((signed char)(((val660>>8u)&255u))),((signed char)(((val660>>16u)&255u))),((signed char)(((val660>>24u)&255u))));
    char4 cast101 = make_char4(((signed char)(((val661>>0u)&255u))),((signed char)(((val661>>8u)&255u))),((signed char)(((val661>>16u)&255u))),((signed char)(((val661>>24u)&255u))));
    char4 cast102 = make_char4(((signed char)(((val662>>0u)&255u))),((signed char)(((val662>>8u)&255u))),((signed char)(((val662>>16u)&255u))),((signed char)(((val662>>24u)&255u))));
    char4 cast103 = make_char4(((signed char)(((val663>>0u)&255u))),((signed char)(((val663>>8u)&255u))),((signed char)(((val663>>16u)&255u))),((signed char)(((val663>>24u)&255u))));
    char4 cast104 = make_char4(((signed char)(((val664>>0u)&255u))),((signed char)(((val664>>8u)&255u))),((signed char)(((val664>>16u)&255u))),((signed char)(((val664>>24u)&255u))));
    int4 wmma96 = __WMMA_8_16_16_signed_char_int(alu418, cast98, cast0);
    int4 wmma97 = __WMMA_8_16_16_signed_char_int(alu419, cast99, cast0);
    int4 wmma98 = __WMMA_8_16_16_signed_char_int(alu420, cast100, cast0);
    int4 wmma99 = __WMMA_8_16_16_signed_char_int(alu421, cast101, cast0);
    int4 wmma100 = __WMMA_8_16_16_signed_char_int(alu422, cast102, cast0);
    int4 wmma101 = __WMMA_8_16_16_signed_char_int(alu423, cast103, cast0);
    int4 wmma102 = __WMMA_8_16_16_signed_char_int(alu424, cast104, cast0);
    int4 wmma103 = __WMMA_8_16_16_signed_char_int(alu425, cast97, cast0);
    float cast105 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val669&65535u)))))));
    float alu722 = ((cast105*tg_bitcast<float>((unsigned int)(val665))*((((float)(((signed char)(((val670>>0u)&255u)))))*((float)(wmma103.x)))+(((float)(((signed char)(((val670>>8u)&255u)))))*((float)(wmma96.x)))))+(cast105*tg_bitcast<float>((unsigned int)(val666))*((((float)(((signed char)(((val670>>16u)&255u)))))*((float)(wmma97.x)))+(((float)(((signed char)(((val670>>24u)&255u)))))*((float)(wmma98.x)))))+(cast105*tg_bitcast<float>((unsigned int)(val667))*((((float)(((signed char)(((val671>>0u)&255u)))))*((float)(wmma99.x)))+(((float)(((signed char)(((val671>>8u)&255u)))))*((float)(wmma100.x)))))+(cast105*tg_bitcast<float>((unsigned int)(val668))*((((float)(((signed char)(((val671>>16u)&255u)))))*((float)(wmma101.x)))+(((float)(((signed char)(((val671>>24u)&255u)))))*((float)(wmma102.x))))));
    float alu723 = (alu414?alu722:(buf49+alu722));
    buf49 = alu723;
    unsigned int val672 = (*(buf0+alu253));
    unsigned int val673 = (*(buf0+alu254));
    unsigned int val674 = (*(buf0+alu255));
    unsigned int val675 = (*(buf0+alu256));
    unsigned int val676 = (*(buf0+alu180));
    unsigned int val677 = (*(buf0+alu181));
    unsigned int val678 = (*(buf0+alu182));
    if (alu414) {
      *(data0_5570560+(alu110+97)) = buf50;
    }
    float cast106 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val676&65535u)))))));
    float alu728 = ((cast106*tg_bitcast<float>((unsigned int)(val672))*((((float)(((signed char)(((val677>>0u)&255u)))))*((float)(wmma103.y)))+(((float)(((signed char)(((val677>>8u)&255u)))))*((float)(wmma96.y)))))+(cast106*tg_bitcast<float>((unsigned int)(val673))*((((float)(((signed char)(((val677>>16u)&255u)))))*((float)(wmma97.y)))+(((float)(((signed char)(((val677>>24u)&255u)))))*((float)(wmma98.y)))))+(cast106*tg_bitcast<float>((unsigned int)(val674))*((((float)(((signed char)(((val678>>0u)&255u)))))*((float)(wmma99.y)))+(((float)(((signed char)(((val678>>8u)&255u)))))*((float)(wmma100.y)))))+(cast106*tg_bitcast<float>((unsigned int)(val675))*((((float)(((signed char)(((val678>>16u)&255u)))))*((float)(wmma101.y)))+(((float)(((signed char)(((val678>>24u)&255u)))))*((float)(wmma102.y))))));
    float alu729 = (alu414?alu728:(buf50+alu728));
    buf50 = alu729;
    unsigned int val679 = (*(buf0+alu249));
    unsigned int val680 = (*(buf0+alu250));
    unsigned int val681 = (*(buf0+alu251));
    unsigned int val682 = (*(buf0+alu252));
    unsigned int val683 = (*(buf0+alu185));
    unsigned int val684 = (*(buf0+alu186));
    unsigned int val685 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+1120)) = buf51;
    }
    float cast107 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val683&65535u)))))));
    float alu734 = ((cast107*tg_bitcast<float>((unsigned int)(val679))*((((float)(((signed char)(((val684>>0u)&255u)))))*((float)(wmma103.z)))+(((float)(((signed char)(((val684>>8u)&255u)))))*((float)(wmma96.z)))))+(cast107*tg_bitcast<float>((unsigned int)(val680))*((((float)(((signed char)(((val684>>16u)&255u)))))*((float)(wmma97.z)))+(((float)(((signed char)(((val684>>24u)&255u)))))*((float)(wmma98.z)))))+(cast107*tg_bitcast<float>((unsigned int)(val681))*((((float)(((signed char)(((val685>>0u)&255u)))))*((float)(wmma99.z)))+(((float)(((signed char)(((val685>>8u)&255u)))))*((float)(wmma100.z)))))+(cast107*tg_bitcast<float>((unsigned int)(val682))*((((float)(((signed char)(((val685>>16u)&255u)))))*((float)(wmma101.z)))+(((float)(((signed char)(((val685>>24u)&255u)))))*((float)(wmma102.z))))));
    float alu735 = (alu414?alu734:(buf51+alu734));
    buf51 = alu735;
    unsigned int val686 = (*(buf0+alu253));
    unsigned int val687 = (*(buf0+alu254));
    unsigned int val688 = (*(buf0+alu255));
    unsigned int val689 = (*(buf0+alu256));
    unsigned int val690 = (*(buf0+alu185));
    unsigned int val691 = (*(buf0+alu186));
    unsigned int val692 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+1121)) = buf52;
    }
    float cast108 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val690&65535u)))))));
    float alu740 = ((cast108*tg_bitcast<float>((unsigned int)(val686))*((((float)(((signed char)(((val691>>0u)&255u)))))*((float)(wmma103.w)))+(((float)(((signed char)(((val691>>8u)&255u)))))*((float)(wmma96.w)))))+(cast108*tg_bitcast<float>((unsigned int)(val687))*((((float)(((signed char)(((val691>>16u)&255u)))))*((float)(wmma97.w)))+(((float)(((signed char)(((val691>>24u)&255u)))))*((float)(wmma98.w)))))+(cast108*tg_bitcast<float>((unsigned int)(val688))*((((float)(((signed char)(((val692>>0u)&255u)))))*((float)(wmma99.w)))+(((float)(((signed char)(((val692>>8u)&255u)))))*((float)(wmma100.w)))))+(cast108*tg_bitcast<float>((unsigned int)(val689))*((((float)(((signed char)(((val692>>16u)&255u)))))*((float)(wmma101.w)))+(((float)(((signed char)(((val692>>24u)&255u)))))*((float)(wmma102.w))))));
    float alu741 = (alu414?alu740:(buf52+alu740));
    buf52 = alu741;
    unsigned int val693 = (*(buf0+alu249));
    unsigned int val694 = (*(buf0+alu250));
    unsigned int val695 = (*(buf0+alu251));
    unsigned int val696 = (*(buf0+alu252));
    unsigned int val697 = (*(buf0+alu190));
    unsigned int val698 = (*(buf0+alu191));
    unsigned int val699 = (*(buf0+alu192));
    if (alu414) {
      *(data0_5570560+(alu110+2144)) = buf53;
    }
    int4 wmma104 = __WMMA_8_16_16_signed_char_int(alu450, cast97, cast0);
    int4 wmma105 = __WMMA_8_16_16_signed_char_int(alu451, cast98, cast0);
    int4 wmma106 = __WMMA_8_16_16_signed_char_int(alu452, cast99, cast0);
    int4 wmma107 = __WMMA_8_16_16_signed_char_int(alu453, cast100, cast0);
    int4 wmma108 = __WMMA_8_16_16_signed_char_int(alu454, cast101, cast0);
    int4 wmma109 = __WMMA_8_16_16_signed_char_int(alu455, cast102, cast0);
    int4 wmma110 = __WMMA_8_16_16_signed_char_int(alu456, cast103, cast0);
    int4 wmma111 = __WMMA_8_16_16_signed_char_int(alu457, cast104, cast0);
    float cast109 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val697&65535u)))))));
    float alu746 = ((cast109*tg_bitcast<float>((unsigned int)(val693))*((((float)(((signed char)(((val698>>0u)&255u)))))*((float)(wmma104.x)))+(((float)(((signed char)(((val698>>8u)&255u)))))*((float)(wmma105.x)))))+(cast109*tg_bitcast<float>((unsigned int)(val694))*((((float)(((signed char)(((val698>>16u)&255u)))))*((float)(wmma106.x)))+(((float)(((signed char)(((val698>>24u)&255u)))))*((float)(wmma107.x)))))+(cast109*tg_bitcast<float>((unsigned int)(val695))*((((float)(((signed char)(((val699>>0u)&255u)))))*((float)(wmma108.x)))+(((float)(((signed char)(((val699>>8u)&255u)))))*((float)(wmma109.x)))))+(cast109*tg_bitcast<float>((unsigned int)(val696))*((((float)(((signed char)(((val699>>16u)&255u)))))*((float)(wmma110.x)))+(((float)(((signed char)(((val699>>24u)&255u)))))*((float)(wmma111.x))))));
    float alu747 = (alu414?alu746:(buf53+alu746));
    buf53 = alu747;
    unsigned int val700 = (*(buf0+alu253));
    unsigned int val701 = (*(buf0+alu254));
    unsigned int val702 = (*(buf0+alu255));
    unsigned int val703 = (*(buf0+alu256));
    unsigned int val704 = (*(buf0+alu190));
    unsigned int val705 = (*(buf0+alu191));
    unsigned int val706 = (*(buf0+alu192));
    if (alu414) {
      *(data0_5570560+(alu110+2145)) = buf54;
    }
    float cast110 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val704&65535u)))))));
    float alu752 = ((cast110*tg_bitcast<float>((unsigned int)(val700))*((((float)(((signed char)(((val705>>0u)&255u)))))*((float)(wmma104.y)))+(((float)(((signed char)(((val705>>8u)&255u)))))*((float)(wmma105.y)))))+(cast110*tg_bitcast<float>((unsigned int)(val701))*((((float)(((signed char)(((val705>>16u)&255u)))))*((float)(wmma106.y)))+(((float)(((signed char)(((val705>>24u)&255u)))))*((float)(wmma107.y)))))+(cast110*tg_bitcast<float>((unsigned int)(val702))*((((float)(((signed char)(((val706>>0u)&255u)))))*((float)(wmma108.y)))+(((float)(((signed char)(((val706>>8u)&255u)))))*((float)(wmma109.y)))))+(cast110*tg_bitcast<float>((unsigned int)(val703))*((((float)(((signed char)(((val706>>16u)&255u)))))*((float)(wmma110.y)))+(((float)(((signed char)(((val706>>24u)&255u)))))*((float)(wmma111.y))))));
    float alu753 = (alu414?alu752:(buf54+alu752));
    buf54 = alu753;
    unsigned int val707 = (*(buf0+alu249));
    unsigned int val708 = (*(buf0+alu250));
    unsigned int val709 = (*(buf0+alu251));
    unsigned int val710 = (*(buf0+alu252));
    unsigned int val711 = (*(buf0+alu195));
    unsigned int val712 = (*(buf0+alu196));
    unsigned int val713 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+3168)) = buf55;
    }
    float cast111 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val711&65535u)))))));
    float alu758 = ((cast111*tg_bitcast<float>((unsigned int)(val707))*((((float)(((signed char)(((val712>>0u)&255u)))))*((float)(wmma104.z)))+(((float)(((signed char)(((val712>>8u)&255u)))))*((float)(wmma105.z)))))+(cast111*tg_bitcast<float>((unsigned int)(val708))*((((float)(((signed char)(((val712>>16u)&255u)))))*((float)(wmma106.z)))+(((float)(((signed char)(((val712>>24u)&255u)))))*((float)(wmma107.z)))))+(cast111*tg_bitcast<float>((unsigned int)(val709))*((((float)(((signed char)(((val713>>0u)&255u)))))*((float)(wmma108.z)))+(((float)(((signed char)(((val713>>8u)&255u)))))*((float)(wmma109.z)))))+(cast111*tg_bitcast<float>((unsigned int)(val710))*((((float)(((signed char)(((val713>>16u)&255u)))))*((float)(wmma110.z)))+(((float)(((signed char)(((val713>>24u)&255u)))))*((float)(wmma111.z))))));
    float alu759 = (alu414?alu758:(buf55+alu758));
    buf55 = alu759;
    unsigned int val714 = (*(buf0+alu253));
    unsigned int val715 = (*(buf0+alu254));
    unsigned int val716 = (*(buf0+alu255));
    unsigned int val717 = (*(buf0+alu256));
    unsigned int val718 = (*(buf0+alu195));
    unsigned int val719 = (*(buf0+alu196));
    unsigned int val720 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+3169)) = buf56;
    }
    float cast112 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val718&65535u)))))));
    float alu764 = ((cast112*tg_bitcast<float>((unsigned int)(val714))*((((float)(((signed char)(((val719>>0u)&255u)))))*((float)(wmma104.w)))+(((float)(((signed char)(((val719>>8u)&255u)))))*((float)(wmma105.w)))))+(cast112*tg_bitcast<float>((unsigned int)(val715))*((((float)(((signed char)(((val719>>16u)&255u)))))*((float)(wmma106.w)))+(((float)(((signed char)(((val719>>24u)&255u)))))*((float)(wmma107.w)))))+(cast112*tg_bitcast<float>((unsigned int)(val716))*((((float)(((signed char)(((val720>>0u)&255u)))))*((float)(wmma108.w)))+(((float)(((signed char)(((val720>>8u)&255u)))))*((float)(wmma109.w)))))+(cast112*tg_bitcast<float>((unsigned int)(val717))*((((float)(((signed char)(((val720>>16u)&255u)))))*((float)(wmma110.w)))+(((float)(((signed char)(((val720>>24u)&255u)))))*((float)(wmma111.w))))));
    float alu765 = (alu414?alu764:(buf56+alu764));
    buf56 = alu765;
    unsigned int val721 = (*(buf0+alu171));
    unsigned int val722 = (*(buf0+alu172));
    unsigned int val723 = (*(buf0+alu173));
    unsigned int val724 = (*(buf0+alu174));
    unsigned int val725 = (*(buf0+alu175));
    unsigned int val726 = (*(buf0+alu176));
    unsigned int val727 = (*(buf0+alu177));
    unsigned int val728 = (*(buf0+alu178));
    unsigned int val729 = (*(buf0+alu257));
    unsigned int val730 = (*(buf0+alu258));
    unsigned int val731 = (*(buf0+alu259));
    unsigned int val732 = (*(buf0+alu260));
    unsigned int val733 = (*(buf0+alu180));
    unsigned int val734 = (*(buf0+alu181));
    unsigned int val735 = (*(buf0+alu182));
    if (alu414) {
      *(data0_5570560+(alu110+112)) = buf57;
    }
    char4 cast113 = make_char4(((signed char)(((val721>>0u)&255u))),((signed char)(((val721>>8u)&255u))),((signed char)(((val721>>16u)&255u))),((signed char)(((val721>>24u)&255u))));
    char4 cast114 = make_char4(((signed char)(((val722>>0u)&255u))),((signed char)(((val722>>8u)&255u))),((signed char)(((val722>>16u)&255u))),((signed char)(((val722>>24u)&255u))));
    char4 cast115 = make_char4(((signed char)(((val723>>0u)&255u))),((signed char)(((val723>>8u)&255u))),((signed char)(((val723>>16u)&255u))),((signed char)(((val723>>24u)&255u))));
    char4 cast116 = make_char4(((signed char)(((val724>>0u)&255u))),((signed char)(((val724>>8u)&255u))),((signed char)(((val724>>16u)&255u))),((signed char)(((val724>>24u)&255u))));
    char4 cast117 = make_char4(((signed char)(((val725>>0u)&255u))),((signed char)(((val725>>8u)&255u))),((signed char)(((val725>>16u)&255u))),((signed char)(((val725>>24u)&255u))));
    char4 cast118 = make_char4(((signed char)(((val726>>0u)&255u))),((signed char)(((val726>>8u)&255u))),((signed char)(((val726>>16u)&255u))),((signed char)(((val726>>24u)&255u))));
    char4 cast119 = make_char4(((signed char)(((val727>>0u)&255u))),((signed char)(((val727>>8u)&255u))),((signed char)(((val727>>16u)&255u))),((signed char)(((val727>>24u)&255u))));
    char4 cast120 = make_char4(((signed char)(((val728>>0u)&255u))),((signed char)(((val728>>8u)&255u))),((signed char)(((val728>>16u)&255u))),((signed char)(((val728>>24u)&255u))));
    int4 wmma112 = __WMMA_8_16_16_signed_char_int(alu418, cast114, cast0);
    int4 wmma113 = __WMMA_8_16_16_signed_char_int(alu419, cast115, cast0);
    int4 wmma114 = __WMMA_8_16_16_signed_char_int(alu420, cast116, cast0);
    int4 wmma115 = __WMMA_8_16_16_signed_char_int(alu421, cast117, cast0);
    int4 wmma116 = __WMMA_8_16_16_signed_char_int(alu422, cast118, cast0);
    int4 wmma117 = __WMMA_8_16_16_signed_char_int(alu423, cast119, cast0);
    int4 wmma118 = __WMMA_8_16_16_signed_char_int(alu424, cast120, cast0);
    int4 wmma119 = __WMMA_8_16_16_signed_char_int(alu425, cast113, cast0);
    float cast121 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val733&65535u)))))));
    float alu770 = ((cast121*tg_bitcast<float>((unsigned int)(val729))*((((float)(((signed char)(((val734>>0u)&255u)))))*((float)(wmma119.x)))+(((float)(((signed char)(((val734>>8u)&255u)))))*((float)(wmma112.x)))))+(cast121*tg_bitcast<float>((unsigned int)(val730))*((((float)(((signed char)(((val734>>16u)&255u)))))*((float)(wmma113.x)))+(((float)(((signed char)(((val734>>24u)&255u)))))*((float)(wmma114.x)))))+(cast121*tg_bitcast<float>((unsigned int)(val731))*((((float)(((signed char)(((val735>>0u)&255u)))))*((float)(wmma115.x)))+(((float)(((signed char)(((val735>>8u)&255u)))))*((float)(wmma116.x)))))+(cast121*tg_bitcast<float>((unsigned int)(val732))*((((float)(((signed char)(((val735>>16u)&255u)))))*((float)(wmma117.x)))+(((float)(((signed char)(((val735>>24u)&255u)))))*((float)(wmma118.x))))));
    float alu771 = (alu414?alu770:(buf57+alu770));
    buf57 = alu771;
    unsigned int val736 = (*(buf0+alu261));
    unsigned int val737 = (*(buf0+alu262));
    unsigned int val738 = (*(buf0+alu263));
    unsigned int val739 = (*(buf0+alu264));
    unsigned int val740 = (*(buf0+alu180));
    unsigned int val741 = (*(buf0+alu181));
    unsigned int val742 = (*(buf0+alu182));
    if (alu414) {
      *(data0_5570560+(alu110+113)) = buf58;
    }
    float cast122 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val740&65535u)))))));
    float alu776 = ((cast122*tg_bitcast<float>((unsigned int)(val736))*((((float)(((signed char)(((val741>>0u)&255u)))))*((float)(wmma119.y)))+(((float)(((signed char)(((val741>>8u)&255u)))))*((float)(wmma112.y)))))+(cast122*tg_bitcast<float>((unsigned int)(val737))*((((float)(((signed char)(((val741>>16u)&255u)))))*((float)(wmma113.y)))+(((float)(((signed char)(((val741>>24u)&255u)))))*((float)(wmma114.y)))))+(cast122*tg_bitcast<float>((unsigned int)(val738))*((((float)(((signed char)(((val742>>0u)&255u)))))*((float)(wmma115.y)))+(((float)(((signed char)(((val742>>8u)&255u)))))*((float)(wmma116.y)))))+(cast122*tg_bitcast<float>((unsigned int)(val739))*((((float)(((signed char)(((val742>>16u)&255u)))))*((float)(wmma117.y)))+(((float)(((signed char)(((val742>>24u)&255u)))))*((float)(wmma118.y))))));
    float alu777 = (alu414?alu776:(buf58+alu776));
    buf58 = alu777;
    unsigned int val743 = (*(buf0+alu257));
    unsigned int val744 = (*(buf0+alu258));
    unsigned int val745 = (*(buf0+alu259));
    unsigned int val746 = (*(buf0+alu260));
    unsigned int val747 = (*(buf0+alu185));
    unsigned int val748 = (*(buf0+alu186));
    unsigned int val749 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+1136)) = buf59;
    }
    float cast123 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val747&65535u)))))));
    float alu782 = ((cast123*tg_bitcast<float>((unsigned int)(val743))*((((float)(((signed char)(((val748>>0u)&255u)))))*((float)(wmma119.z)))+(((float)(((signed char)(((val748>>8u)&255u)))))*((float)(wmma112.z)))))+(cast123*tg_bitcast<float>((unsigned int)(val744))*((((float)(((signed char)(((val748>>16u)&255u)))))*((float)(wmma113.z)))+(((float)(((signed char)(((val748>>24u)&255u)))))*((float)(wmma114.z)))))+(cast123*tg_bitcast<float>((unsigned int)(val745))*((((float)(((signed char)(((val749>>0u)&255u)))))*((float)(wmma115.z)))+(((float)(((signed char)(((val749>>8u)&255u)))))*((float)(wmma116.z)))))+(cast123*tg_bitcast<float>((unsigned int)(val746))*((((float)(((signed char)(((val749>>16u)&255u)))))*((float)(wmma117.z)))+(((float)(((signed char)(((val749>>24u)&255u)))))*((float)(wmma118.z))))));
    float alu783 = (alu414?alu782:(buf59+alu782));
    buf59 = alu783;
    unsigned int val750 = (*(buf0+alu261));
    unsigned int val751 = (*(buf0+alu262));
    unsigned int val752 = (*(buf0+alu263));
    unsigned int val753 = (*(buf0+alu264));
    unsigned int val754 = (*(buf0+alu185));
    unsigned int val755 = (*(buf0+alu186));
    unsigned int val756 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+1137)) = buf60;
    }
    float cast124 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val754&65535u)))))));
    float alu788 = ((cast124*tg_bitcast<float>((unsigned int)(val750))*((((float)(((signed char)(((val755>>0u)&255u)))))*((float)(wmma119.w)))+(((float)(((signed char)(((val755>>8u)&255u)))))*((float)(wmma112.w)))))+(cast124*tg_bitcast<float>((unsigned int)(val751))*((((float)(((signed char)(((val755>>16u)&255u)))))*((float)(wmma113.w)))+(((float)(((signed char)(((val755>>24u)&255u)))))*((float)(wmma114.w)))))+(cast124*tg_bitcast<float>((unsigned int)(val752))*((((float)(((signed char)(((val756>>0u)&255u)))))*((float)(wmma115.w)))+(((float)(((signed char)(((val756>>8u)&255u)))))*((float)(wmma116.w)))))+(cast124*tg_bitcast<float>((unsigned int)(val753))*((((float)(((signed char)(((val756>>16u)&255u)))))*((float)(wmma117.w)))+(((float)(((signed char)(((val756>>24u)&255u)))))*((float)(wmma118.w))))));
    float alu789 = (alu414?alu788:(buf60+alu788));
    buf60 = alu789;
    unsigned int val757 = (*(buf0+alu257));
    unsigned int val758 = (*(buf0+alu258));
    unsigned int val759 = (*(buf0+alu259));
    unsigned int val760 = (*(buf0+alu260));
    unsigned int val761 = (*(buf0+alu190));
    unsigned int val762 = (*(buf0+alu191));
    unsigned int val763 = (*(buf0+alu192));
    if (alu414) {
      *(data0_5570560+(alu110+2160)) = buf61;
    }
    int4 wmma120 = __WMMA_8_16_16_signed_char_int(alu450, cast113, cast0);
    int4 wmma121 = __WMMA_8_16_16_signed_char_int(alu451, cast114, cast0);
    int4 wmma122 = __WMMA_8_16_16_signed_char_int(alu452, cast115, cast0);
    int4 wmma123 = __WMMA_8_16_16_signed_char_int(alu453, cast116, cast0);
    int4 wmma124 = __WMMA_8_16_16_signed_char_int(alu454, cast117, cast0);
    int4 wmma125 = __WMMA_8_16_16_signed_char_int(alu455, cast118, cast0);
    int4 wmma126 = __WMMA_8_16_16_signed_char_int(alu456, cast119, cast0);
    int4 wmma127 = __WMMA_8_16_16_signed_char_int(alu457, cast120, cast0);
    float cast125 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val761&65535u)))))));
    float alu794 = ((cast125*tg_bitcast<float>((unsigned int)(val757))*((((float)(((signed char)(((val762>>0u)&255u)))))*((float)(wmma120.x)))+(((float)(((signed char)(((val762>>8u)&255u)))))*((float)(wmma121.x)))))+(cast125*tg_bitcast<float>((unsigned int)(val758))*((((float)(((signed char)(((val762>>16u)&255u)))))*((float)(wmma122.x)))+(((float)(((signed char)(((val762>>24u)&255u)))))*((float)(wmma123.x)))))+(cast125*tg_bitcast<float>((unsigned int)(val759))*((((float)(((signed char)(((val763>>0u)&255u)))))*((float)(wmma124.x)))+(((float)(((signed char)(((val763>>8u)&255u)))))*((float)(wmma125.x)))))+(cast125*tg_bitcast<float>((unsigned int)(val760))*((((float)(((signed char)(((val763>>16u)&255u)))))*((float)(wmma126.x)))+(((float)(((signed char)(((val763>>24u)&255u)))))*((float)(wmma127.x))))));
    float alu795 = (alu414?alu794:(buf61+alu794));
    buf61 = alu795;
    unsigned int val764 = (*(buf0+alu261));
    unsigned int val765 = (*(buf0+alu262));
    unsigned int val766 = (*(buf0+alu263));
    unsigned int val767 = (*(buf0+alu264));
    unsigned int val768 = (*(buf0+alu190));
    unsigned int val769 = (*(buf0+alu191));
    unsigned int val770 = (*(buf0+alu192));
    if (alu414) {
      *(data0_5570560+(alu110+2161)) = buf62;
    }
    float cast126 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val768&65535u)))))));
    float alu800 = ((cast126*tg_bitcast<float>((unsigned int)(val764))*((((float)(((signed char)(((val769>>0u)&255u)))))*((float)(wmma120.y)))+(((float)(((signed char)(((val769>>8u)&255u)))))*((float)(wmma121.y)))))+(cast126*tg_bitcast<float>((unsigned int)(val765))*((((float)(((signed char)(((val769>>16u)&255u)))))*((float)(wmma122.y)))+(((float)(((signed char)(((val769>>24u)&255u)))))*((float)(wmma123.y)))))+(cast126*tg_bitcast<float>((unsigned int)(val766))*((((float)(((signed char)(((val770>>0u)&255u)))))*((float)(wmma124.y)))+(((float)(((signed char)(((val770>>8u)&255u)))))*((float)(wmma125.y)))))+(cast126*tg_bitcast<float>((unsigned int)(val767))*((((float)(((signed char)(((val770>>16u)&255u)))))*((float)(wmma126.y)))+(((float)(((signed char)(((val770>>24u)&255u)))))*((float)(wmma127.y))))));
    float alu801 = (alu414?alu800:(buf62+alu800));
    buf62 = alu801;
    unsigned int val771 = (*(buf0+alu257));
    unsigned int val772 = (*(buf0+alu258));
    unsigned int val773 = (*(buf0+alu259));
    unsigned int val774 = (*(buf0+alu260));
    unsigned int val775 = (*(buf0+alu195));
    unsigned int val776 = (*(buf0+alu196));
    unsigned int val777 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+3184)) = buf63;
    }
    float cast127 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val775&65535u)))))));
    float alu806 = ((cast127*tg_bitcast<float>((unsigned int)(val771))*((((float)(((signed char)(((val776>>0u)&255u)))))*((float)(wmma120.z)))+(((float)(((signed char)(((val776>>8u)&255u)))))*((float)(wmma121.z)))))+(cast127*tg_bitcast<float>((unsigned int)(val772))*((((float)(((signed char)(((val776>>16u)&255u)))))*((float)(wmma122.z)))+(((float)(((signed char)(((val776>>24u)&255u)))))*((float)(wmma123.z)))))+(cast127*tg_bitcast<float>((unsigned int)(val773))*((((float)(((signed char)(((val777>>0u)&255u)))))*((float)(wmma124.z)))+(((float)(((signed char)(((val777>>8u)&255u)))))*((float)(wmma125.z)))))+(cast127*tg_bitcast<float>((unsigned int)(val774))*((((float)(((signed char)(((val777>>16u)&255u)))))*((float)(wmma126.z)))+(((float)(((signed char)(((val777>>24u)&255u)))))*((float)(wmma127.z))))));
    float alu807 = (alu414?alu806:(buf63+alu806));
    buf63 = alu807;
    unsigned int val778 = (*(buf0+alu261));
    unsigned int val779 = (*(buf0+alu262));
    unsigned int val780 = (*(buf0+alu263));
    unsigned int val781 = (*(buf0+alu264));
    unsigned int val782 = (*(buf0+alu195));
    unsigned int val783 = (*(buf0+alu196));
    unsigned int val784 = (*(buf0+alu197));
    float val785 = (*(data4_196608+(((alu302+4)<<9)+alu301+alu265)));
    float val786 = (*(data4_196608+(((alu302+5)<<9)+alu301+alu265)));
    float val787 = (*(data4_196608+(((alu302+6)<<9)+alu301+alu265)));
    float val788 = (*(data4_196608+(((alu302+7)<<9)+alu301+alu265)));
    if (alu414) {
      *(data0_5570560+(alu110+3185)) = buf64;
    }
    float cast128 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val782&65535u)))))));
    float alu812 = ((cast128*tg_bitcast<float>((unsigned int)(val778))*((((float)(((signed char)(((val783>>0u)&255u)))))*((float)(wmma120.w)))+(((float)(((signed char)(((val783>>8u)&255u)))))*((float)(wmma121.w)))))+(cast128*tg_bitcast<float>((unsigned int)(val779))*((((float)(((signed char)(((val783>>16u)&255u)))))*((float)(wmma122.w)))+(((float)(((signed char)(((val783>>24u)&255u)))))*((float)(wmma123.w)))))+(cast128*tg_bitcast<float>((unsigned int)(val780))*((((float)(((signed char)(((val784>>0u)&255u)))))*((float)(wmma124.w)))+(((float)(((signed char)(((val784>>8u)&255u)))))*((float)(wmma125.w)))))+(cast128*tg_bitcast<float>((unsigned int)(val781))*((((float)(((signed char)(((val784>>16u)&255u)))))*((float)(wmma126.w)))+(((float)(((signed char)(((val784>>24u)&255u)))))*((float)(wmma127.w))))));
    float alu813 = (alu414?alu812:(buf64+alu812));
    buf64 = alu813;
    __syncthreads();
    if (alu271) {
      *(buf0+alu68) = tg_bitcast<unsigned int>((float)(val785));
    }
    if (alu271) {
      *(buf0+alu69) = tg_bitcast<unsigned int>((float)(val786));
    }
    if (alu271) {
      *(buf0+alu70) = tg_bitcast<unsigned int>((float)(val787));
    }
    if (alu271) {
      *(buf0+alu71) = tg_bitcast<unsigned int>((float)(val788));
    }
    if (alu271) {
      *(buf0+alu72) = ((((unsigned int)(val140))&255u)|((((unsigned int)(val141))&255u)<<8u)|((((unsigned int)(val142))&255u)<<16u)|((((unsigned int)(val143))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu73) = ((((unsigned int)(val144))&255u)|((((unsigned int)(val145))&255u)<<8u)|((((unsigned int)(val146))&255u)<<16u)|((((unsigned int)(val147))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu74) = ((((unsigned int)(val148))&255u)|((((unsigned int)(val149))&255u)<<8u)|((((unsigned int)(val150))&255u)<<16u)|((((unsigned int)(val151))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu75) = ((((unsigned int)(val152))&255u)|((((unsigned int)(val153))&255u)<<8u)|((((unsigned int)(val154))&255u)<<16u)|((((unsigned int)(val155))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu76) = ((((unsigned int)(val156))&255u)|((((unsigned int)(val157))&255u)<<8u)|((((unsigned int)(val158))&255u)<<16u)|((((unsigned int)(val159))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu77) = ((((unsigned int)(val160))&255u)|((((unsigned int)(val161))&255u)<<8u)|((((unsigned int)(val162))&255u)<<16u)|((((unsigned int)(val163))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu78) = ((((unsigned int)(val164))&255u)|((((unsigned int)(val165))&255u)<<8u)|((((unsigned int)(val166))&255u)<<16u)|((((unsigned int)(val167))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu79) = ((((unsigned int)(val168))&255u)|((((unsigned int)(val169))&255u)<<8u)|((((unsigned int)(val170))&255u)<<16u)|((((unsigned int)(val171))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu80) = ((((unsigned int)(val172))&255u)|((((unsigned int)(val173))&255u)<<8u)|((((unsigned int)(val174))&255u)<<16u)|((((unsigned int)(val175))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu81) = ((((unsigned int)(val176))&255u)|((((unsigned int)(val177))&255u)<<8u)|((((unsigned int)(val178))&255u)<<16u)|((((unsigned int)(val179))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu82) = ((((unsigned int)(val180))&255u)|((((unsigned int)(val181))&255u)<<8u)|((((unsigned int)(val182))&255u)<<16u)|((((unsigned int)(val183))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu83) = ((((unsigned int)(val184))&255u)|((((unsigned int)(val185))&255u)<<8u)|((((unsigned int)(val186))&255u)<<16u)|((((unsigned int)(val187))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu84) = ((((unsigned int)(val188))&255u)|((((unsigned int)(val189))&255u)<<8u)|((((unsigned int)(val190))&255u)<<16u)|((((unsigned int)(val191))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu85) = ((((unsigned int)(val192))&255u)|((((unsigned int)(val193))&255u)<<8u)|((((unsigned int)(val194))&255u)<<16u)|((((unsigned int)(val195))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu86) = ((((unsigned int)(val196))&255u)|((((unsigned int)(val197))&255u)<<8u)|((((unsigned int)(val198))&255u)<<16u)|((((unsigned int)(val199))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu87) = ((((unsigned int)(val200))&255u)|((((unsigned int)(val201))&255u)<<8u)|((((unsigned int)(val202))&255u)<<16u)|((((unsigned int)(val203))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu88) = ((((unsigned int)(val204))&255u)|((((unsigned int)(val205))&255u)<<8u)|((((unsigned int)(val206))&255u)<<16u)|((((unsigned int)(val207))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu89) = ((((unsigned int)(val208))&255u)|((((unsigned int)(val209))&255u)<<8u)|((((unsigned int)(val210))&255u)<<16u)|((((unsigned int)(val211))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu90) = ((((unsigned int)(val212))&255u)|((((unsigned int)(val213))&255u)<<8u)|((((unsigned int)(val214))&255u)<<16u)|((((unsigned int)(val215))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu91) = ((((unsigned int)(val216))&255u)|((((unsigned int)(val217))&255u)<<8u)|((((unsigned int)(val218))&255u)<<16u)|((((unsigned int)(val219))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu92) = ((((unsigned int)(val220))&255u)|((((unsigned int)(val221))&255u)<<8u)|((((unsigned int)(val222))&255u)<<16u)|((((unsigned int)(val223))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu93) = ((((unsigned int)(val224))&255u)|((((unsigned int)(val225))&255u)<<8u)|((((unsigned int)(val226))&255u)<<16u)|((((unsigned int)(val227))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu94) = ((((unsigned int)(val228))&255u)|((((unsigned int)(val229))&255u)<<8u)|((((unsigned int)(val230))&255u)<<16u)|((((unsigned int)(val231))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu95) = ((((unsigned int)(val232))&255u)|((((unsigned int)(val233))&255u)<<8u)|((((unsigned int)(val234))&255u)<<16u)|((((unsigned int)(val235))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu96) = ((((unsigned int)(val236))&255u)|((((unsigned int)(val237))&255u)<<8u)|((((unsigned int)(val238))&255u)<<16u)|((((unsigned int)(val239))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu97) = ((((unsigned int)(val240))&255u)|((((unsigned int)(val241))&255u)<<8u)|((((unsigned int)(val242))&255u)<<16u)|((((unsigned int)(val243))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu98) = ((((unsigned int)(val244))&255u)|((((unsigned int)(val245))&255u)<<8u)|((((unsigned int)(val246))&255u)<<16u)|((((unsigned int)(val247))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu99) = ((((unsigned int)(val248))&255u)|((((unsigned int)(val249))&255u)<<8u)|((((unsigned int)(val250))&255u)<<16u)|((((unsigned int)(val251))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu100) = ((((unsigned int)(val252))&255u)|((((unsigned int)(val253))&255u)<<8u)|((((unsigned int)(val254))&255u)<<16u)|((((unsigned int)(val255))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu101) = ((((unsigned int)(val256))&255u)|((((unsigned int)(val257))&255u)<<8u)|((((unsigned int)(val258))&255u)<<16u)|((((unsigned int)(val259))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu102) = ((((unsigned int)(val260))&255u)|((((unsigned int)(val261))&255u)<<8u)|((((unsigned int)(val262))&255u)<<16u)|((((unsigned int)(val263))&255u)<<24u));
    }
    if (alu271) {
      *(buf0+alu103) = ((((unsigned int)(val264))&255u)|((((unsigned int)(val265))&255u)<<8u)|((((unsigned int)(val266))&255u)<<16u)|((((unsigned int)(val267))&255u)<<24u));
    }
    __syncthreads();
    unsigned int val789 = (*(buf0+alu115));
    unsigned int val790 = (*(buf0+alu116));
    unsigned int val791 = (*(buf0+alu117));
    unsigned int val792 = (*(buf0+alu118));
    unsigned int val793 = (*(buf0+alu119));
    unsigned int val794 = (*(buf0+alu120));
    unsigned int val795 = (*(buf0+alu121));
    unsigned int val796 = (*(buf0+alu122));
    unsigned int val797 = (*(buf0+alu201));
    unsigned int val798 = (*(buf0+alu202));
    unsigned int val799 = (*(buf0+alu203));
    unsigned int val800 = (*(buf0+alu204));
    unsigned int val801 = (*(buf0+alu180));
    unsigned int val802 = (*(buf0+alu183));
    unsigned int val803 = (*(buf0+alu184));
    if (0) {
      *(data0_5570560+alu110) = buf1;
    }
    char4 cast129 = make_char4(((signed char)(((val789>>0u)&255u))),((signed char)(((val789>>8u)&255u))),((signed char)(((val789>>16u)&255u))),((signed char)(((val789>>24u)&255u))));
    char4 cast130 = make_char4(((signed char)(((val790>>0u)&255u))),((signed char)(((val790>>8u)&255u))),((signed char)(((val790>>16u)&255u))),((signed char)(((val790>>24u)&255u))));
    char4 cast131 = make_char4(((signed char)(((val791>>0u)&255u))),((signed char)(((val791>>8u)&255u))),((signed char)(((val791>>16u)&255u))),((signed char)(((val791>>24u)&255u))));
    char4 cast132 = make_char4(((signed char)(((val792>>0u)&255u))),((signed char)(((val792>>8u)&255u))),((signed char)(((val792>>16u)&255u))),((signed char)(((val792>>24u)&255u))));
    char4 cast133 = make_char4(((signed char)(((val793>>0u)&255u))),((signed char)(((val793>>8u)&255u))),((signed char)(((val793>>16u)&255u))),((signed char)(((val793>>24u)&255u))));
    char4 cast134 = make_char4(((signed char)(((val794>>0u)&255u))),((signed char)(((val794>>8u)&255u))),((signed char)(((val794>>16u)&255u))),((signed char)(((val794>>24u)&255u))));
    char4 cast135 = make_char4(((signed char)(((val795>>0u)&255u))),((signed char)(((val795>>8u)&255u))),((signed char)(((val795>>16u)&255u))),((signed char)(((val795>>24u)&255u))));
    char4 cast136 = make_char4(((signed char)(((val796>>0u)&255u))),((signed char)(((val796>>8u)&255u))),((signed char)(((val796>>16u)&255u))),((signed char)(((val796>>24u)&255u))));
    signed_char8 alu928 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+32))))*4)));
    int4 wmma128 = __WMMA_8_16_16_signed_char_int(alu928, cast129, cast0);
    signed_char8 alu929 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+36))))*4)));
    int4 wmma129 = __WMMA_8_16_16_signed_char_int(alu929, cast130, cast0);
    signed_char8 alu930 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+40))))*4)));
    int4 wmma130 = __WMMA_8_16_16_signed_char_int(alu930, cast131, cast0);
    signed_char8 alu931 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+44))))*4)));
    int4 wmma131 = __WMMA_8_16_16_signed_char_int(alu931, cast132, cast0);
    signed_char8 alu932 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+48))))*4)));
    int4 wmma132 = __WMMA_8_16_16_signed_char_int(alu932, cast133, cast0);
    signed_char8 alu933 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+52))))*4)));
    int4 wmma133 = __WMMA_8_16_16_signed_char_int(alu933, cast134, cast0);
    signed_char8 alu934 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+56))))*4)));
    int4 wmma134 = __WMMA_8_16_16_signed_char_int(alu934, cast135, cast0);
    signed_char8 alu935 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+60))))*4)));
    int4 wmma135 = __WMMA_8_16_16_signed_char_int(alu935, cast136, cast0);
    float cast137 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val801&65535u)))))));
    buf1 = (buf1+(cast137*tg_bitcast<float>((unsigned int)(val797))*((((float)(((signed char)(((val802>>0u)&255u)))))*((float)(wmma128.x)))+(((float)(((signed char)(((val802>>8u)&255u)))))*((float)(wmma129.x)))))+(cast137*tg_bitcast<float>((unsigned int)(val798))*((((float)(((signed char)(((val802>>16u)&255u)))))*((float)(wmma130.x)))+(((float)(((signed char)(((val802>>24u)&255u)))))*((float)(wmma131.x)))))+(cast137*tg_bitcast<float>((unsigned int)(val799))*((((float)(((signed char)(((val803>>0u)&255u)))))*((float)(wmma132.x)))+(((float)(((signed char)(((val803>>8u)&255u)))))*((float)(wmma133.x)))))+(cast137*tg_bitcast<float>((unsigned int)(val800))*((((float)(((signed char)(((val803>>16u)&255u)))))*((float)(wmma134.x)))+(((float)(((signed char)(((val803>>24u)&255u)))))*((float)(wmma135.x))))));
    unsigned int val804 = (*(buf0+alu205));
    unsigned int val805 = (*(buf0+alu206));
    unsigned int val806 = (*(buf0+alu207));
    unsigned int val807 = (*(buf0+alu208));
    unsigned int val808 = (*(buf0+alu180));
    unsigned int val809 = (*(buf0+alu183));
    unsigned int val810 = (*(buf0+alu184));
    if (0) {
      *(data0_5570560+(alu110+1)) = buf2;
    }
    float cast138 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val808&65535u)))))));
    buf2 = (buf2+(cast138*tg_bitcast<float>((unsigned int)(val804))*((((float)(((signed char)(((val809>>0u)&255u)))))*((float)(wmma128.y)))+(((float)(((signed char)(((val809>>8u)&255u)))))*((float)(wmma129.y)))))+(cast138*tg_bitcast<float>((unsigned int)(val805))*((((float)(((signed char)(((val809>>16u)&255u)))))*((float)(wmma130.y)))+(((float)(((signed char)(((val809>>24u)&255u)))))*((float)(wmma131.y)))))+(cast138*tg_bitcast<float>((unsigned int)(val806))*((((float)(((signed char)(((val810>>0u)&255u)))))*((float)(wmma132.y)))+(((float)(((signed char)(((val810>>8u)&255u)))))*((float)(wmma133.y)))))+(cast138*tg_bitcast<float>((unsigned int)(val807))*((((float)(((signed char)(((val810>>16u)&255u)))))*((float)(wmma134.y)))+(((float)(((signed char)(((val810>>24u)&255u)))))*((float)(wmma135.y))))));
    unsigned int val811 = (*(buf0+alu201));
    unsigned int val812 = (*(buf0+alu202));
    unsigned int val813 = (*(buf0+alu203));
    unsigned int val814 = (*(buf0+alu204));
    unsigned int val815 = (*(buf0+alu185));
    unsigned int val816 = (*(buf0+alu188));
    unsigned int val817 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+1024)) = buf3;
    }
    float cast139 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val815&65535u)))))));
    buf3 = (buf3+(cast139*tg_bitcast<float>((unsigned int)(val811))*((((float)(((signed char)(((val816>>0u)&255u)))))*((float)(wmma128.z)))+(((float)(((signed char)(((val816>>8u)&255u)))))*((float)(wmma129.z)))))+(cast139*tg_bitcast<float>((unsigned int)(val812))*((((float)(((signed char)(((val816>>16u)&255u)))))*((float)(wmma130.z)))+(((float)(((signed char)(((val816>>24u)&255u)))))*((float)(wmma131.z)))))+(cast139*tg_bitcast<float>((unsigned int)(val813))*((((float)(((signed char)(((val817>>0u)&255u)))))*((float)(wmma132.z)))+(((float)(((signed char)(((val817>>8u)&255u)))))*((float)(wmma133.z)))))+(cast139*tg_bitcast<float>((unsigned int)(val814))*((((float)(((signed char)(((val817>>16u)&255u)))))*((float)(wmma134.z)))+(((float)(((signed char)(((val817>>24u)&255u)))))*((float)(wmma135.z))))));
    unsigned int val818 = (*(buf0+alu205));
    unsigned int val819 = (*(buf0+alu206));
    unsigned int val820 = (*(buf0+alu207));
    unsigned int val821 = (*(buf0+alu208));
    unsigned int val822 = (*(buf0+alu185));
    unsigned int val823 = (*(buf0+alu188));
    unsigned int val824 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+1025)) = buf4;
    }
    float cast140 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val822&65535u)))))));
    buf4 = (buf4+(cast140*tg_bitcast<float>((unsigned int)(val818))*((((float)(((signed char)(((val823>>0u)&255u)))))*((float)(wmma128.w)))+(((float)(((signed char)(((val823>>8u)&255u)))))*((float)(wmma129.w)))))+(cast140*tg_bitcast<float>((unsigned int)(val819))*((((float)(((signed char)(((val823>>16u)&255u)))))*((float)(wmma130.w)))+(((float)(((signed char)(((val823>>24u)&255u)))))*((float)(wmma131.w)))))+(cast140*tg_bitcast<float>((unsigned int)(val820))*((((float)(((signed char)(((val824>>0u)&255u)))))*((float)(wmma132.w)))+(((float)(((signed char)(((val824>>8u)&255u)))))*((float)(wmma133.w)))))+(cast140*tg_bitcast<float>((unsigned int)(val821))*((((float)(((signed char)(((val824>>16u)&255u)))))*((float)(wmma134.w)))+(((float)(((signed char)(((val824>>24u)&255u)))))*((float)(wmma135.w))))));
    unsigned int val825 = (*(buf0+alu201));
    unsigned int val826 = (*(buf0+alu202));
    unsigned int val827 = (*(buf0+alu203));
    unsigned int val828 = (*(buf0+alu204));
    unsigned int val829 = (*(buf0+alu190));
    unsigned int val830 = (*(buf0+alu193));
    unsigned int val831 = (*(buf0+alu194));
    if (0) {
      *(data0_5570560+(alu110+2048)) = buf5;
    }
    signed_char8 alu952 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1248))))*4)));
    int4 wmma136 = __WMMA_8_16_16_signed_char_int(alu952, cast129, cast0);
    signed_char8 alu953 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1252))))*4)));
    int4 wmma137 = __WMMA_8_16_16_signed_char_int(alu953, cast130, cast0);
    signed_char8 alu954 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1256))))*4)));
    int4 wmma138 = __WMMA_8_16_16_signed_char_int(alu954, cast131, cast0);
    signed_char8 alu955 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1260))))*4)));
    int4 wmma139 = __WMMA_8_16_16_signed_char_int(alu955, cast132, cast0);
    signed_char8 alu956 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1264))))*4)));
    int4 wmma140 = __WMMA_8_16_16_signed_char_int(alu956, cast133, cast0);
    signed_char8 alu957 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1268))))*4)));
    int4 wmma141 = __WMMA_8_16_16_signed_char_int(alu957, cast134, cast0);
    signed_char8 alu958 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1272))))*4)));
    int4 wmma142 = __WMMA_8_16_16_signed_char_int(alu958, cast135, cast0);
    signed_char8 alu959 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1276))))*4)));
    int4 wmma143 = __WMMA_8_16_16_signed_char_int(alu959, cast136, cast0);
    float cast141 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val829&65535u)))))));
    buf5 = (buf5+(cast141*tg_bitcast<float>((unsigned int)(val825))*((((float)(((signed char)(((val830>>0u)&255u)))))*((float)(wmma136.x)))+(((float)(((signed char)(((val830>>8u)&255u)))))*((float)(wmma137.x)))))+(cast141*tg_bitcast<float>((unsigned int)(val826))*((((float)(((signed char)(((val830>>16u)&255u)))))*((float)(wmma138.x)))+(((float)(((signed char)(((val830>>24u)&255u)))))*((float)(wmma139.x)))))+(cast141*tg_bitcast<float>((unsigned int)(val827))*((((float)(((signed char)(((val831>>0u)&255u)))))*((float)(wmma140.x)))+(((float)(((signed char)(((val831>>8u)&255u)))))*((float)(wmma141.x)))))+(cast141*tg_bitcast<float>((unsigned int)(val828))*((((float)(((signed char)(((val831>>16u)&255u)))))*((float)(wmma142.x)))+(((float)(((signed char)(((val831>>24u)&255u)))))*((float)(wmma143.x))))));
    unsigned int val832 = (*(buf0+alu205));
    unsigned int val833 = (*(buf0+alu206));
    unsigned int val834 = (*(buf0+alu207));
    unsigned int val835 = (*(buf0+alu208));
    unsigned int val836 = (*(buf0+alu190));
    unsigned int val837 = (*(buf0+alu193));
    unsigned int val838 = (*(buf0+alu194));
    if (0) {
      *(data0_5570560+(alu110+2049)) = buf6;
    }
    float cast142 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val836&65535u)))))));
    buf6 = (buf6+(cast142*tg_bitcast<float>((unsigned int)(val832))*((((float)(((signed char)(((val837>>0u)&255u)))))*((float)(wmma136.y)))+(((float)(((signed char)(((val837>>8u)&255u)))))*((float)(wmma137.y)))))+(cast142*tg_bitcast<float>((unsigned int)(val833))*((((float)(((signed char)(((val837>>16u)&255u)))))*((float)(wmma138.y)))+(((float)(((signed char)(((val837>>24u)&255u)))))*((float)(wmma139.y)))))+(cast142*tg_bitcast<float>((unsigned int)(val834))*((((float)(((signed char)(((val838>>0u)&255u)))))*((float)(wmma140.y)))+(((float)(((signed char)(((val838>>8u)&255u)))))*((float)(wmma141.y)))))+(cast142*tg_bitcast<float>((unsigned int)(val835))*((((float)(((signed char)(((val838>>16u)&255u)))))*((float)(wmma142.y)))+(((float)(((signed char)(((val838>>24u)&255u)))))*((float)(wmma143.y))))));
    unsigned int val839 = (*(buf0+alu201));
    unsigned int val840 = (*(buf0+alu202));
    unsigned int val841 = (*(buf0+alu203));
    unsigned int val842 = (*(buf0+alu204));
    unsigned int val843 = (*(buf0+alu195));
    unsigned int val844 = (*(buf0+alu198));
    unsigned int val845 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+3072)) = buf7;
    }
    float cast143 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val843&65535u)))))));
    buf7 = (buf7+(cast143*tg_bitcast<float>((unsigned int)(val839))*((((float)(((signed char)(((val844>>0u)&255u)))))*((float)(wmma136.z)))+(((float)(((signed char)(((val844>>8u)&255u)))))*((float)(wmma137.z)))))+(cast143*tg_bitcast<float>((unsigned int)(val840))*((((float)(((signed char)(((val844>>16u)&255u)))))*((float)(wmma138.z)))+(((float)(((signed char)(((val844>>24u)&255u)))))*((float)(wmma139.z)))))+(cast143*tg_bitcast<float>((unsigned int)(val841))*((((float)(((signed char)(((val845>>0u)&255u)))))*((float)(wmma140.z)))+(((float)(((signed char)(((val845>>8u)&255u)))))*((float)(wmma141.z)))))+(cast143*tg_bitcast<float>((unsigned int)(val842))*((((float)(((signed char)(((val845>>16u)&255u)))))*((float)(wmma142.z)))+(((float)(((signed char)(((val845>>24u)&255u)))))*((float)(wmma143.z))))));
    unsigned int val846 = (*(buf0+alu205));
    unsigned int val847 = (*(buf0+alu206));
    unsigned int val848 = (*(buf0+alu207));
    unsigned int val849 = (*(buf0+alu208));
    unsigned int val850 = (*(buf0+alu195));
    unsigned int val851 = (*(buf0+alu198));
    unsigned int val852 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+3073)) = buf8;
    }
    float cast144 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val850&65535u)))))));
    buf8 = (buf8+(cast144*tg_bitcast<float>((unsigned int)(val846))*((((float)(((signed char)(((val851>>0u)&255u)))))*((float)(wmma136.w)))+(((float)(((signed char)(((val851>>8u)&255u)))))*((float)(wmma137.w)))))+(cast144*tg_bitcast<float>((unsigned int)(val847))*((((float)(((signed char)(((val851>>16u)&255u)))))*((float)(wmma138.w)))+(((float)(((signed char)(((val851>>24u)&255u)))))*((float)(wmma139.w)))))+(cast144*tg_bitcast<float>((unsigned int)(val848))*((((float)(((signed char)(((val852>>0u)&255u)))))*((float)(wmma140.w)))+(((float)(((signed char)(((val852>>8u)&255u)))))*((float)(wmma141.w)))))+(cast144*tg_bitcast<float>((unsigned int)(val849))*((((float)(((signed char)(((val852>>16u)&255u)))))*((float)(wmma142.w)))+(((float)(((signed char)(((val852>>24u)&255u)))))*((float)(wmma143.w))))));
    unsigned int val853 = (*(buf0+alu123));
    unsigned int val854 = (*(buf0+alu124));
    unsigned int val855 = (*(buf0+alu125));
    unsigned int val856 = (*(buf0+alu126));
    unsigned int val857 = (*(buf0+alu127));
    unsigned int val858 = (*(buf0+alu128));
    unsigned int val859 = (*(buf0+alu129));
    unsigned int val860 = (*(buf0+alu130));
    unsigned int val861 = (*(buf0+alu209));
    unsigned int val862 = (*(buf0+alu210));
    unsigned int val863 = (*(buf0+alu211));
    unsigned int val864 = (*(buf0+alu212));
    unsigned int val865 = (*(buf0+alu180));
    unsigned int val866 = (*(buf0+alu183));
    unsigned int val867 = (*(buf0+alu184));
    if (0) {
      *(data0_5570560+(alu110+16)) = buf9;
    }
    char4 cast145 = make_char4(((signed char)(((val853>>0u)&255u))),((signed char)(((val853>>8u)&255u))),((signed char)(((val853>>16u)&255u))),((signed char)(((val853>>24u)&255u))));
    char4 cast146 = make_char4(((signed char)(((val854>>0u)&255u))),((signed char)(((val854>>8u)&255u))),((signed char)(((val854>>16u)&255u))),((signed char)(((val854>>24u)&255u))));
    char4 cast147 = make_char4(((signed char)(((val855>>0u)&255u))),((signed char)(((val855>>8u)&255u))),((signed char)(((val855>>16u)&255u))),((signed char)(((val855>>24u)&255u))));
    char4 cast148 = make_char4(((signed char)(((val856>>0u)&255u))),((signed char)(((val856>>8u)&255u))),((signed char)(((val856>>16u)&255u))),((signed char)(((val856>>24u)&255u))));
    char4 cast149 = make_char4(((signed char)(((val857>>0u)&255u))),((signed char)(((val857>>8u)&255u))),((signed char)(((val857>>16u)&255u))),((signed char)(((val857>>24u)&255u))));
    char4 cast150 = make_char4(((signed char)(((val858>>0u)&255u))),((signed char)(((val858>>8u)&255u))),((signed char)(((val858>>16u)&255u))),((signed char)(((val858>>24u)&255u))));
    char4 cast151 = make_char4(((signed char)(((val859>>0u)&255u))),((signed char)(((val859>>8u)&255u))),((signed char)(((val859>>16u)&255u))),((signed char)(((val859>>24u)&255u))));
    char4 cast152 = make_char4(((signed char)(((val860>>0u)&255u))),((signed char)(((val860>>8u)&255u))),((signed char)(((val860>>16u)&255u))),((signed char)(((val860>>24u)&255u))));
    int4 wmma144 = __WMMA_8_16_16_signed_char_int(alu928, cast145, cast0);
    int4 wmma145 = __WMMA_8_16_16_signed_char_int(alu929, cast146, cast0);
    int4 wmma146 = __WMMA_8_16_16_signed_char_int(alu930, cast147, cast0);
    int4 wmma147 = __WMMA_8_16_16_signed_char_int(alu931, cast148, cast0);
    int4 wmma148 = __WMMA_8_16_16_signed_char_int(alu932, cast149, cast0);
    int4 wmma149 = __WMMA_8_16_16_signed_char_int(alu933, cast150, cast0);
    int4 wmma150 = __WMMA_8_16_16_signed_char_int(alu934, cast151, cast0);
    int4 wmma151 = __WMMA_8_16_16_signed_char_int(alu935, cast152, cast0);
    float cast153 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val865&65535u)))))));
    buf9 = (buf9+(cast153*tg_bitcast<float>((unsigned int)(val861))*((((float)(((signed char)(((val866>>0u)&255u)))))*((float)(wmma144.x)))+(((float)(((signed char)(((val866>>8u)&255u)))))*((float)(wmma145.x)))))+(cast153*tg_bitcast<float>((unsigned int)(val862))*((((float)(((signed char)(((val866>>16u)&255u)))))*((float)(wmma146.x)))+(((float)(((signed char)(((val866>>24u)&255u)))))*((float)(wmma147.x)))))+(cast153*tg_bitcast<float>((unsigned int)(val863))*((((float)(((signed char)(((val867>>0u)&255u)))))*((float)(wmma148.x)))+(((float)(((signed char)(((val867>>8u)&255u)))))*((float)(wmma149.x)))))+(cast153*tg_bitcast<float>((unsigned int)(val864))*((((float)(((signed char)(((val867>>16u)&255u)))))*((float)(wmma150.x)))+(((float)(((signed char)(((val867>>24u)&255u)))))*((float)(wmma151.x))))));
    unsigned int val868 = (*(buf0+alu213));
    unsigned int val869 = (*(buf0+alu214));
    unsigned int val870 = (*(buf0+alu215));
    unsigned int val871 = (*(buf0+alu216));
    unsigned int val872 = (*(buf0+alu180));
    unsigned int val873 = (*(buf0+alu183));
    unsigned int val874 = (*(buf0+alu184));
    if (0) {
      *(data0_5570560+(alu110+17)) = buf10;
    }
    float cast154 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val872&65535u)))))));
    buf10 = (buf10+(cast154*tg_bitcast<float>((unsigned int)(val868))*((((float)(((signed char)(((val873>>0u)&255u)))))*((float)(wmma144.y)))+(((float)(((signed char)(((val873>>8u)&255u)))))*((float)(wmma145.y)))))+(cast154*tg_bitcast<float>((unsigned int)(val869))*((((float)(((signed char)(((val873>>16u)&255u)))))*((float)(wmma146.y)))+(((float)(((signed char)(((val873>>24u)&255u)))))*((float)(wmma147.y)))))+(cast154*tg_bitcast<float>((unsigned int)(val870))*((((float)(((signed char)(((val874>>0u)&255u)))))*((float)(wmma148.y)))+(((float)(((signed char)(((val874>>8u)&255u)))))*((float)(wmma149.y)))))+(cast154*tg_bitcast<float>((unsigned int)(val871))*((((float)(((signed char)(((val874>>16u)&255u)))))*((float)(wmma150.y)))+(((float)(((signed char)(((val874>>24u)&255u)))))*((float)(wmma151.y))))));
    unsigned int val875 = (*(buf0+alu209));
    unsigned int val876 = (*(buf0+alu210));
    unsigned int val877 = (*(buf0+alu211));
    unsigned int val878 = (*(buf0+alu212));
    unsigned int val879 = (*(buf0+alu185));
    unsigned int val880 = (*(buf0+alu188));
    unsigned int val881 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+1040)) = buf11;
    }
    float cast155 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val879&65535u)))))));
    buf11 = (buf11+(cast155*tg_bitcast<float>((unsigned int)(val875))*((((float)(((signed char)(((val880>>0u)&255u)))))*((float)(wmma144.z)))+(((float)(((signed char)(((val880>>8u)&255u)))))*((float)(wmma145.z)))))+(cast155*tg_bitcast<float>((unsigned int)(val876))*((((float)(((signed char)(((val880>>16u)&255u)))))*((float)(wmma146.z)))+(((float)(((signed char)(((val880>>24u)&255u)))))*((float)(wmma147.z)))))+(cast155*tg_bitcast<float>((unsigned int)(val877))*((((float)(((signed char)(((val881>>0u)&255u)))))*((float)(wmma148.z)))+(((float)(((signed char)(((val881>>8u)&255u)))))*((float)(wmma149.z)))))+(cast155*tg_bitcast<float>((unsigned int)(val878))*((((float)(((signed char)(((val881>>16u)&255u)))))*((float)(wmma150.z)))+(((float)(((signed char)(((val881>>24u)&255u)))))*((float)(wmma151.z))))));
    unsigned int val882 = (*(buf0+alu213));
    unsigned int val883 = (*(buf0+alu214));
    unsigned int val884 = (*(buf0+alu215));
    unsigned int val885 = (*(buf0+alu216));
    unsigned int val886 = (*(buf0+alu185));
    unsigned int val887 = (*(buf0+alu188));
    unsigned int val888 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+1041)) = buf12;
    }
    float cast156 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val886&65535u)))))));
    buf12 = (buf12+(cast156*tg_bitcast<float>((unsigned int)(val882))*((((float)(((signed char)(((val887>>0u)&255u)))))*((float)(wmma144.w)))+(((float)(((signed char)(((val887>>8u)&255u)))))*((float)(wmma145.w)))))+(cast156*tg_bitcast<float>((unsigned int)(val883))*((((float)(((signed char)(((val887>>16u)&255u)))))*((float)(wmma146.w)))+(((float)(((signed char)(((val887>>24u)&255u)))))*((float)(wmma147.w)))))+(cast156*tg_bitcast<float>((unsigned int)(val884))*((((float)(((signed char)(((val888>>0u)&255u)))))*((float)(wmma148.w)))+(((float)(((signed char)(((val888>>8u)&255u)))))*((float)(wmma149.w)))))+(cast156*tg_bitcast<float>((unsigned int)(val885))*((((float)(((signed char)(((val888>>16u)&255u)))))*((float)(wmma150.w)))+(((float)(((signed char)(((val888>>24u)&255u)))))*((float)(wmma151.w))))));
    unsigned int val889 = (*(buf0+alu209));
    unsigned int val890 = (*(buf0+alu210));
    unsigned int val891 = (*(buf0+alu211));
    unsigned int val892 = (*(buf0+alu212));
    unsigned int val893 = (*(buf0+alu190));
    unsigned int val894 = (*(buf0+alu193));
    unsigned int val895 = (*(buf0+alu194));
    if (0) {
      *(data0_5570560+(alu110+2064)) = buf13;
    }
    int4 wmma152 = __WMMA_8_16_16_signed_char_int(alu952, cast145, cast0);
    int4 wmma153 = __WMMA_8_16_16_signed_char_int(alu953, cast146, cast0);
    int4 wmma154 = __WMMA_8_16_16_signed_char_int(alu954, cast147, cast0);
    int4 wmma155 = __WMMA_8_16_16_signed_char_int(alu955, cast148, cast0);
    int4 wmma156 = __WMMA_8_16_16_signed_char_int(alu956, cast149, cast0);
    int4 wmma157 = __WMMA_8_16_16_signed_char_int(alu957, cast150, cast0);
    int4 wmma158 = __WMMA_8_16_16_signed_char_int(alu958, cast151, cast0);
    int4 wmma159 = __WMMA_8_16_16_signed_char_int(alu959, cast152, cast0);
    float cast157 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val893&65535u)))))));
    buf13 = (buf13+(cast157*tg_bitcast<float>((unsigned int)(val889))*((((float)(((signed char)(((val894>>0u)&255u)))))*((float)(wmma152.x)))+(((float)(((signed char)(((val894>>8u)&255u)))))*((float)(wmma153.x)))))+(cast157*tg_bitcast<float>((unsigned int)(val890))*((((float)(((signed char)(((val894>>16u)&255u)))))*((float)(wmma154.x)))+(((float)(((signed char)(((val894>>24u)&255u)))))*((float)(wmma155.x)))))+(cast157*tg_bitcast<float>((unsigned int)(val891))*((((float)(((signed char)(((val895>>0u)&255u)))))*((float)(wmma156.x)))+(((float)(((signed char)(((val895>>8u)&255u)))))*((float)(wmma157.x)))))+(cast157*tg_bitcast<float>((unsigned int)(val892))*((((float)(((signed char)(((val895>>16u)&255u)))))*((float)(wmma158.x)))+(((float)(((signed char)(((val895>>24u)&255u)))))*((float)(wmma159.x))))));
    unsigned int val896 = (*(buf0+alu213));
    unsigned int val897 = (*(buf0+alu214));
    unsigned int val898 = (*(buf0+alu215));
    unsigned int val899 = (*(buf0+alu216));
    unsigned int val900 = (*(buf0+alu190));
    unsigned int val901 = (*(buf0+alu193));
    unsigned int val902 = (*(buf0+alu194));
    if (0) {
      *(data0_5570560+(alu110+2065)) = buf14;
    }
    float cast158 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val900&65535u)))))));
    buf14 = (buf14+(cast158*tg_bitcast<float>((unsigned int)(val896))*((((float)(((signed char)(((val901>>0u)&255u)))))*((float)(wmma152.y)))+(((float)(((signed char)(((val901>>8u)&255u)))))*((float)(wmma153.y)))))+(cast158*tg_bitcast<float>((unsigned int)(val897))*((((float)(((signed char)(((val901>>16u)&255u)))))*((float)(wmma154.y)))+(((float)(((signed char)(((val901>>24u)&255u)))))*((float)(wmma155.y)))))+(cast158*tg_bitcast<float>((unsigned int)(val898))*((((float)(((signed char)(((val902>>0u)&255u)))))*((float)(wmma156.y)))+(((float)(((signed char)(((val902>>8u)&255u)))))*((float)(wmma157.y)))))+(cast158*tg_bitcast<float>((unsigned int)(val899))*((((float)(((signed char)(((val902>>16u)&255u)))))*((float)(wmma158.y)))+(((float)(((signed char)(((val902>>24u)&255u)))))*((float)(wmma159.y))))));
    unsigned int val903 = (*(buf0+alu209));
    unsigned int val904 = (*(buf0+alu210));
    unsigned int val905 = (*(buf0+alu211));
    unsigned int val906 = (*(buf0+alu212));
    unsigned int val907 = (*(buf0+alu195));
    unsigned int val908 = (*(buf0+alu198));
    unsigned int val909 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+3088)) = buf15;
    }
    float cast159 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val907&65535u)))))));
    buf15 = (buf15+(cast159*tg_bitcast<float>((unsigned int)(val903))*((((float)(((signed char)(((val908>>0u)&255u)))))*((float)(wmma152.z)))+(((float)(((signed char)(((val908>>8u)&255u)))))*((float)(wmma153.z)))))+(cast159*tg_bitcast<float>((unsigned int)(val904))*((((float)(((signed char)(((val908>>16u)&255u)))))*((float)(wmma154.z)))+(((float)(((signed char)(((val908>>24u)&255u)))))*((float)(wmma155.z)))))+(cast159*tg_bitcast<float>((unsigned int)(val905))*((((float)(((signed char)(((val909>>0u)&255u)))))*((float)(wmma156.z)))+(((float)(((signed char)(((val909>>8u)&255u)))))*((float)(wmma157.z)))))+(cast159*tg_bitcast<float>((unsigned int)(val906))*((((float)(((signed char)(((val909>>16u)&255u)))))*((float)(wmma158.z)))+(((float)(((signed char)(((val909>>24u)&255u)))))*((float)(wmma159.z))))));
    unsigned int val910 = (*(buf0+alu213));
    unsigned int val911 = (*(buf0+alu214));
    unsigned int val912 = (*(buf0+alu215));
    unsigned int val913 = (*(buf0+alu216));
    unsigned int val914 = (*(buf0+alu195));
    unsigned int val915 = (*(buf0+alu198));
    unsigned int val916 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+3089)) = buf16;
    }
    float cast160 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val914&65535u)))))));
    buf16 = (buf16+(cast160*tg_bitcast<float>((unsigned int)(val910))*((((float)(((signed char)(((val915>>0u)&255u)))))*((float)(wmma152.w)))+(((float)(((signed char)(((val915>>8u)&255u)))))*((float)(wmma153.w)))))+(cast160*tg_bitcast<float>((unsigned int)(val911))*((((float)(((signed char)(((val915>>16u)&255u)))))*((float)(wmma154.w)))+(((float)(((signed char)(((val915>>24u)&255u)))))*((float)(wmma155.w)))))+(cast160*tg_bitcast<float>((unsigned int)(val912))*((((float)(((signed char)(((val916>>0u)&255u)))))*((float)(wmma156.w)))+(((float)(((signed char)(((val916>>8u)&255u)))))*((float)(wmma157.w)))))+(cast160*tg_bitcast<float>((unsigned int)(val913))*((((float)(((signed char)(((val916>>16u)&255u)))))*((float)(wmma158.w)))+(((float)(((signed char)(((val916>>24u)&255u)))))*((float)(wmma159.w))))));
    unsigned int val917 = (*(buf0+alu131));
    unsigned int val918 = (*(buf0+alu132));
    unsigned int val919 = (*(buf0+alu133));
    unsigned int val920 = (*(buf0+alu134));
    unsigned int val921 = (*(buf0+alu135));
    unsigned int val922 = (*(buf0+alu136));
    unsigned int val923 = (*(buf0+alu137));
    unsigned int val924 = (*(buf0+alu138));
    unsigned int val925 = (*(buf0+alu217));
    unsigned int val926 = (*(buf0+alu218));
    unsigned int val927 = (*(buf0+alu219));
    unsigned int val928 = (*(buf0+alu220));
    unsigned int val929 = (*(buf0+alu180));
    unsigned int val930 = (*(buf0+alu183));
    unsigned int val931 = (*(buf0+alu184));
    if (0) {
      *(data0_5570560+(alu110+32)) = buf17;
    }
    char4 cast161 = make_char4(((signed char)(((val917>>0u)&255u))),((signed char)(((val917>>8u)&255u))),((signed char)(((val917>>16u)&255u))),((signed char)(((val917>>24u)&255u))));
    char4 cast162 = make_char4(((signed char)(((val918>>0u)&255u))),((signed char)(((val918>>8u)&255u))),((signed char)(((val918>>16u)&255u))),((signed char)(((val918>>24u)&255u))));
    char4 cast163 = make_char4(((signed char)(((val919>>0u)&255u))),((signed char)(((val919>>8u)&255u))),((signed char)(((val919>>16u)&255u))),((signed char)(((val919>>24u)&255u))));
    char4 cast164 = make_char4(((signed char)(((val920>>0u)&255u))),((signed char)(((val920>>8u)&255u))),((signed char)(((val920>>16u)&255u))),((signed char)(((val920>>24u)&255u))));
    char4 cast165 = make_char4(((signed char)(((val921>>0u)&255u))),((signed char)(((val921>>8u)&255u))),((signed char)(((val921>>16u)&255u))),((signed char)(((val921>>24u)&255u))));
    char4 cast166 = make_char4(((signed char)(((val922>>0u)&255u))),((signed char)(((val922>>8u)&255u))),((signed char)(((val922>>16u)&255u))),((signed char)(((val922>>24u)&255u))));
    char4 cast167 = make_char4(((signed char)(((val923>>0u)&255u))),((signed char)(((val923>>8u)&255u))),((signed char)(((val923>>16u)&255u))),((signed char)(((val923>>24u)&255u))));
    char4 cast168 = make_char4(((signed char)(((val924>>0u)&255u))),((signed char)(((val924>>8u)&255u))),((signed char)(((val924>>16u)&255u))),((signed char)(((val924>>24u)&255u))));
    int4 wmma160 = __WMMA_8_16_16_signed_char_int(alu928, cast161, cast0);
    int4 wmma161 = __WMMA_8_16_16_signed_char_int(alu929, cast162, cast0);
    int4 wmma162 = __WMMA_8_16_16_signed_char_int(alu930, cast163, cast0);
    int4 wmma163 = __WMMA_8_16_16_signed_char_int(alu931, cast164, cast0);
    int4 wmma164 = __WMMA_8_16_16_signed_char_int(alu932, cast165, cast0);
    int4 wmma165 = __WMMA_8_16_16_signed_char_int(alu933, cast166, cast0);
    int4 wmma166 = __WMMA_8_16_16_signed_char_int(alu934, cast167, cast0);
    int4 wmma167 = __WMMA_8_16_16_signed_char_int(alu935, cast168, cast0);
    float cast169 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val929&65535u)))))));
    buf17 = (buf17+(cast169*tg_bitcast<float>((unsigned int)(val925))*((((float)(((signed char)(((val930>>0u)&255u)))))*((float)(wmma160.x)))+(((float)(((signed char)(((val930>>8u)&255u)))))*((float)(wmma161.x)))))+(cast169*tg_bitcast<float>((unsigned int)(val926))*((((float)(((signed char)(((val930>>16u)&255u)))))*((float)(wmma162.x)))+(((float)(((signed char)(((val930>>24u)&255u)))))*((float)(wmma163.x)))))+(cast169*tg_bitcast<float>((unsigned int)(val927))*((((float)(((signed char)(((val931>>0u)&255u)))))*((float)(wmma164.x)))+(((float)(((signed char)(((val931>>8u)&255u)))))*((float)(wmma165.x)))))+(cast169*tg_bitcast<float>((unsigned int)(val928))*((((float)(((signed char)(((val931>>16u)&255u)))))*((float)(wmma166.x)))+(((float)(((signed char)(((val931>>24u)&255u)))))*((float)(wmma167.x))))));
    unsigned int val932 = (*(buf0+alu221));
    unsigned int val933 = (*(buf0+alu222));
    unsigned int val934 = (*(buf0+alu223));
    unsigned int val935 = (*(buf0+alu224));
    unsigned int val936 = (*(buf0+alu180));
    unsigned int val937 = (*(buf0+alu183));
    unsigned int val938 = (*(buf0+alu184));
    if (0) {
      *(data0_5570560+(alu110+33)) = buf18;
    }
    float cast170 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val936&65535u)))))));
    buf18 = (buf18+(cast170*tg_bitcast<float>((unsigned int)(val932))*((((float)(((signed char)(((val937>>0u)&255u)))))*((float)(wmma160.y)))+(((float)(((signed char)(((val937>>8u)&255u)))))*((float)(wmma161.y)))))+(cast170*tg_bitcast<float>((unsigned int)(val933))*((((float)(((signed char)(((val937>>16u)&255u)))))*((float)(wmma162.y)))+(((float)(((signed char)(((val937>>24u)&255u)))))*((float)(wmma163.y)))))+(cast170*tg_bitcast<float>((unsigned int)(val934))*((((float)(((signed char)(((val938>>0u)&255u)))))*((float)(wmma164.y)))+(((float)(((signed char)(((val938>>8u)&255u)))))*((float)(wmma165.y)))))+(cast170*tg_bitcast<float>((unsigned int)(val935))*((((float)(((signed char)(((val938>>16u)&255u)))))*((float)(wmma166.y)))+(((float)(((signed char)(((val938>>24u)&255u)))))*((float)(wmma167.y))))));
    unsigned int val939 = (*(buf0+alu217));
    unsigned int val940 = (*(buf0+alu218));
    unsigned int val941 = (*(buf0+alu219));
    unsigned int val942 = (*(buf0+alu220));
    unsigned int val943 = (*(buf0+alu185));
    unsigned int val944 = (*(buf0+alu188));
    unsigned int val945 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+1056)) = buf19;
    }
    float cast171 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val943&65535u)))))));
    buf19 = (buf19+(cast171*tg_bitcast<float>((unsigned int)(val939))*((((float)(((signed char)(((val944>>0u)&255u)))))*((float)(wmma160.z)))+(((float)(((signed char)(((val944>>8u)&255u)))))*((float)(wmma161.z)))))+(cast171*tg_bitcast<float>((unsigned int)(val940))*((((float)(((signed char)(((val944>>16u)&255u)))))*((float)(wmma162.z)))+(((float)(((signed char)(((val944>>24u)&255u)))))*((float)(wmma163.z)))))+(cast171*tg_bitcast<float>((unsigned int)(val941))*((((float)(((signed char)(((val945>>0u)&255u)))))*((float)(wmma164.z)))+(((float)(((signed char)(((val945>>8u)&255u)))))*((float)(wmma165.z)))))+(cast171*tg_bitcast<float>((unsigned int)(val942))*((((float)(((signed char)(((val945>>16u)&255u)))))*((float)(wmma166.z)))+(((float)(((signed char)(((val945>>24u)&255u)))))*((float)(wmma167.z))))));
    unsigned int val946 = (*(buf0+alu221));
    unsigned int val947 = (*(buf0+alu222));
    unsigned int val948 = (*(buf0+alu223));
    unsigned int val949 = (*(buf0+alu224));
    unsigned int val950 = (*(buf0+alu185));
    unsigned int val951 = (*(buf0+alu188));
    unsigned int val952 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+1057)) = buf20;
    }
    float cast172 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val950&65535u)))))));
    buf20 = (buf20+(cast172*tg_bitcast<float>((unsigned int)(val946))*((((float)(((signed char)(((val951>>0u)&255u)))))*((float)(wmma160.w)))+(((float)(((signed char)(((val951>>8u)&255u)))))*((float)(wmma161.w)))))+(cast172*tg_bitcast<float>((unsigned int)(val947))*((((float)(((signed char)(((val951>>16u)&255u)))))*((float)(wmma162.w)))+(((float)(((signed char)(((val951>>24u)&255u)))))*((float)(wmma163.w)))))+(cast172*tg_bitcast<float>((unsigned int)(val948))*((((float)(((signed char)(((val952>>0u)&255u)))))*((float)(wmma164.w)))+(((float)(((signed char)(((val952>>8u)&255u)))))*((float)(wmma165.w)))))+(cast172*tg_bitcast<float>((unsigned int)(val949))*((((float)(((signed char)(((val952>>16u)&255u)))))*((float)(wmma166.w)))+(((float)(((signed char)(((val952>>24u)&255u)))))*((float)(wmma167.w))))));
    unsigned int val953 = (*(buf0+alu217));
    unsigned int val954 = (*(buf0+alu218));
    unsigned int val955 = (*(buf0+alu219));
    unsigned int val956 = (*(buf0+alu220));
    unsigned int val957 = (*(buf0+alu190));
    unsigned int val958 = (*(buf0+alu193));
    unsigned int val959 = (*(buf0+alu194));
    if (0) {
      *(data0_5570560+(alu110+2080)) = buf21;
    }
    int4 wmma168 = __WMMA_8_16_16_signed_char_int(alu952, cast161, cast0);
    int4 wmma169 = __WMMA_8_16_16_signed_char_int(alu953, cast162, cast0);
    int4 wmma170 = __WMMA_8_16_16_signed_char_int(alu954, cast163, cast0);
    int4 wmma171 = __WMMA_8_16_16_signed_char_int(alu955, cast164, cast0);
    int4 wmma172 = __WMMA_8_16_16_signed_char_int(alu956, cast165, cast0);
    int4 wmma173 = __WMMA_8_16_16_signed_char_int(alu957, cast166, cast0);
    int4 wmma174 = __WMMA_8_16_16_signed_char_int(alu958, cast167, cast0);
    int4 wmma175 = __WMMA_8_16_16_signed_char_int(alu959, cast168, cast0);
    float cast173 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val957&65535u)))))));
    buf21 = (buf21+(cast173*tg_bitcast<float>((unsigned int)(val953))*((((float)(((signed char)(((val958>>0u)&255u)))))*((float)(wmma168.x)))+(((float)(((signed char)(((val958>>8u)&255u)))))*((float)(wmma169.x)))))+(cast173*tg_bitcast<float>((unsigned int)(val954))*((((float)(((signed char)(((val958>>16u)&255u)))))*((float)(wmma170.x)))+(((float)(((signed char)(((val958>>24u)&255u)))))*((float)(wmma171.x)))))+(cast173*tg_bitcast<float>((unsigned int)(val955))*((((float)(((signed char)(((val959>>0u)&255u)))))*((float)(wmma172.x)))+(((float)(((signed char)(((val959>>8u)&255u)))))*((float)(wmma173.x)))))+(cast173*tg_bitcast<float>((unsigned int)(val956))*((((float)(((signed char)(((val959>>16u)&255u)))))*((float)(wmma174.x)))+(((float)(((signed char)(((val959>>24u)&255u)))))*((float)(wmma175.x))))));
    unsigned int val960 = (*(buf0+alu221));
    unsigned int val961 = (*(buf0+alu222));
    unsigned int val962 = (*(buf0+alu223));
    unsigned int val963 = (*(buf0+alu224));
    unsigned int val964 = (*(buf0+alu190));
    unsigned int val965 = (*(buf0+alu193));
    unsigned int val966 = (*(buf0+alu194));
    if (0) {
      *(data0_5570560+(alu110+2081)) = buf22;
    }
    float cast174 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val964&65535u)))))));
    buf22 = (buf22+(cast174*tg_bitcast<float>((unsigned int)(val960))*((((float)(((signed char)(((val965>>0u)&255u)))))*((float)(wmma168.y)))+(((float)(((signed char)(((val965>>8u)&255u)))))*((float)(wmma169.y)))))+(cast174*tg_bitcast<float>((unsigned int)(val961))*((((float)(((signed char)(((val965>>16u)&255u)))))*((float)(wmma170.y)))+(((float)(((signed char)(((val965>>24u)&255u)))))*((float)(wmma171.y)))))+(cast174*tg_bitcast<float>((unsigned int)(val962))*((((float)(((signed char)(((val966>>0u)&255u)))))*((float)(wmma172.y)))+(((float)(((signed char)(((val966>>8u)&255u)))))*((float)(wmma173.y)))))+(cast174*tg_bitcast<float>((unsigned int)(val963))*((((float)(((signed char)(((val966>>16u)&255u)))))*((float)(wmma174.y)))+(((float)(((signed char)(((val966>>24u)&255u)))))*((float)(wmma175.y))))));
    unsigned int val967 = (*(buf0+alu217));
    unsigned int val968 = (*(buf0+alu218));
    unsigned int val969 = (*(buf0+alu219));
    unsigned int val970 = (*(buf0+alu220));
    unsigned int val971 = (*(buf0+alu195));
    unsigned int val972 = (*(buf0+alu198));
    unsigned int val973 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+3104)) = buf23;
    }
    float cast175 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val971&65535u)))))));
    buf23 = (buf23+(cast175*tg_bitcast<float>((unsigned int)(val967))*((((float)(((signed char)(((val972>>0u)&255u)))))*((float)(wmma168.z)))+(((float)(((signed char)(((val972>>8u)&255u)))))*((float)(wmma169.z)))))+(cast175*tg_bitcast<float>((unsigned int)(val968))*((((float)(((signed char)(((val972>>16u)&255u)))))*((float)(wmma170.z)))+(((float)(((signed char)(((val972>>24u)&255u)))))*((float)(wmma171.z)))))+(cast175*tg_bitcast<float>((unsigned int)(val969))*((((float)(((signed char)(((val973>>0u)&255u)))))*((float)(wmma172.z)))+(((float)(((signed char)(((val973>>8u)&255u)))))*((float)(wmma173.z)))))+(cast175*tg_bitcast<float>((unsigned int)(val970))*((((float)(((signed char)(((val973>>16u)&255u)))))*((float)(wmma174.z)))+(((float)(((signed char)(((val973>>24u)&255u)))))*((float)(wmma175.z))))));
    unsigned int val974 = (*(buf0+alu221));
    unsigned int val975 = (*(buf0+alu222));
    unsigned int val976 = (*(buf0+alu223));
    unsigned int val977 = (*(buf0+alu224));
    unsigned int val978 = (*(buf0+alu195));
    unsigned int val979 = (*(buf0+alu198));
    unsigned int val980 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+3105)) = buf24;
    }
    float cast176 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val978&65535u)))))));
    buf24 = (buf24+(cast176*tg_bitcast<float>((unsigned int)(val974))*((((float)(((signed char)(((val979>>0u)&255u)))))*((float)(wmma168.w)))+(((float)(((signed char)(((val979>>8u)&255u)))))*((float)(wmma169.w)))))+(cast176*tg_bitcast<float>((unsigned int)(val975))*((((float)(((signed char)(((val979>>16u)&255u)))))*((float)(wmma170.w)))+(((float)(((signed char)(((val979>>24u)&255u)))))*((float)(wmma171.w)))))+(cast176*tg_bitcast<float>((unsigned int)(val976))*((((float)(((signed char)(((val980>>0u)&255u)))))*((float)(wmma172.w)))+(((float)(((signed char)(((val980>>8u)&255u)))))*((float)(wmma173.w)))))+(cast176*tg_bitcast<float>((unsigned int)(val977))*((((float)(((signed char)(((val980>>16u)&255u)))))*((float)(wmma174.w)))+(((float)(((signed char)(((val980>>24u)&255u)))))*((float)(wmma175.w))))));
    unsigned int val981 = (*(buf0+alu139));
    unsigned int val982 = (*(buf0+alu140));
    unsigned int val983 = (*(buf0+alu141));
    unsigned int val984 = (*(buf0+alu142));
    unsigned int val985 = (*(buf0+alu143));
    unsigned int val986 = (*(buf0+alu144));
    unsigned int val987 = (*(buf0+alu145));
    unsigned int val988 = (*(buf0+alu146));
    unsigned int val989 = (*(buf0+alu225));
    unsigned int val990 = (*(buf0+alu226));
    unsigned int val991 = (*(buf0+alu227));
    unsigned int val992 = (*(buf0+alu228));
    unsigned int val993 = (*(buf0+alu180));
    unsigned int val994 = (*(buf0+alu183));
    unsigned int val995 = (*(buf0+alu184));
    if (0) {
      *(data0_5570560+(alu110+48)) = buf25;
    }
    char4 cast177 = make_char4(((signed char)(((val981>>0u)&255u))),((signed char)(((val981>>8u)&255u))),((signed char)(((val981>>16u)&255u))),((signed char)(((val981>>24u)&255u))));
    char4 cast178 = make_char4(((signed char)(((val982>>0u)&255u))),((signed char)(((val982>>8u)&255u))),((signed char)(((val982>>16u)&255u))),((signed char)(((val982>>24u)&255u))));
    char4 cast179 = make_char4(((signed char)(((val983>>0u)&255u))),((signed char)(((val983>>8u)&255u))),((signed char)(((val983>>16u)&255u))),((signed char)(((val983>>24u)&255u))));
    char4 cast180 = make_char4(((signed char)(((val984>>0u)&255u))),((signed char)(((val984>>8u)&255u))),((signed char)(((val984>>16u)&255u))),((signed char)(((val984>>24u)&255u))));
    char4 cast181 = make_char4(((signed char)(((val985>>0u)&255u))),((signed char)(((val985>>8u)&255u))),((signed char)(((val985>>16u)&255u))),((signed char)(((val985>>24u)&255u))));
    char4 cast182 = make_char4(((signed char)(((val986>>0u)&255u))),((signed char)(((val986>>8u)&255u))),((signed char)(((val986>>16u)&255u))),((signed char)(((val986>>24u)&255u))));
    char4 cast183 = make_char4(((signed char)(((val987>>0u)&255u))),((signed char)(((val987>>8u)&255u))),((signed char)(((val987>>16u)&255u))),((signed char)(((val987>>24u)&255u))));
    char4 cast184 = make_char4(((signed char)(((val988>>0u)&255u))),((signed char)(((val988>>8u)&255u))),((signed char)(((val988>>16u)&255u))),((signed char)(((val988>>24u)&255u))));
    int4 wmma176 = __WMMA_8_16_16_signed_char_int(alu928, cast177, cast0);
    int4 wmma177 = __WMMA_8_16_16_signed_char_int(alu929, cast178, cast0);
    int4 wmma178 = __WMMA_8_16_16_signed_char_int(alu930, cast179, cast0);
    int4 wmma179 = __WMMA_8_16_16_signed_char_int(alu931, cast180, cast0);
    int4 wmma180 = __WMMA_8_16_16_signed_char_int(alu932, cast181, cast0);
    int4 wmma181 = __WMMA_8_16_16_signed_char_int(alu933, cast182, cast0);
    int4 wmma182 = __WMMA_8_16_16_signed_char_int(alu934, cast183, cast0);
    int4 wmma183 = __WMMA_8_16_16_signed_char_int(alu935, cast184, cast0);
    float cast185 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val993&65535u)))))));
    buf25 = (buf25+(cast185*tg_bitcast<float>((unsigned int)(val989))*((((float)(((signed char)(((val994>>0u)&255u)))))*((float)(wmma176.x)))+(((float)(((signed char)(((val994>>8u)&255u)))))*((float)(wmma177.x)))))+(cast185*tg_bitcast<float>((unsigned int)(val990))*((((float)(((signed char)(((val994>>16u)&255u)))))*((float)(wmma178.x)))+(((float)(((signed char)(((val994>>24u)&255u)))))*((float)(wmma179.x)))))+(cast185*tg_bitcast<float>((unsigned int)(val991))*((((float)(((signed char)(((val995>>0u)&255u)))))*((float)(wmma180.x)))+(((float)(((signed char)(((val995>>8u)&255u)))))*((float)(wmma181.x)))))+(cast185*tg_bitcast<float>((unsigned int)(val992))*((((float)(((signed char)(((val995>>16u)&255u)))))*((float)(wmma182.x)))+(((float)(((signed char)(((val995>>24u)&255u)))))*((float)(wmma183.x))))));
    unsigned int val996 = (*(buf0+alu229));
    unsigned int val997 = (*(buf0+alu230));
    unsigned int val998 = (*(buf0+alu231));
    unsigned int val999 = (*(buf0+alu232));
    unsigned int val1000 = (*(buf0+alu180));
    unsigned int val1001 = (*(buf0+alu183));
    unsigned int val1002 = (*(buf0+alu184));
    if (0) {
      *(data0_5570560+(alu110+49)) = buf26;
    }
    float cast186 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1000&65535u)))))));
    buf26 = (buf26+(cast186*tg_bitcast<float>((unsigned int)(val996))*((((float)(((signed char)(((val1001>>0u)&255u)))))*((float)(wmma176.y)))+(((float)(((signed char)(((val1001>>8u)&255u)))))*((float)(wmma177.y)))))+(cast186*tg_bitcast<float>((unsigned int)(val997))*((((float)(((signed char)(((val1001>>16u)&255u)))))*((float)(wmma178.y)))+(((float)(((signed char)(((val1001>>24u)&255u)))))*((float)(wmma179.y)))))+(cast186*tg_bitcast<float>((unsigned int)(val998))*((((float)(((signed char)(((val1002>>0u)&255u)))))*((float)(wmma180.y)))+(((float)(((signed char)(((val1002>>8u)&255u)))))*((float)(wmma181.y)))))+(cast186*tg_bitcast<float>((unsigned int)(val999))*((((float)(((signed char)(((val1002>>16u)&255u)))))*((float)(wmma182.y)))+(((float)(((signed char)(((val1002>>24u)&255u)))))*((float)(wmma183.y))))));
    unsigned int val1003 = (*(buf0+alu225));
    unsigned int val1004 = (*(buf0+alu226));
    unsigned int val1005 = (*(buf0+alu227));
    unsigned int val1006 = (*(buf0+alu228));
    unsigned int val1007 = (*(buf0+alu185));
    unsigned int val1008 = (*(buf0+alu188));
    unsigned int val1009 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+1072)) = buf27;
    }
    float cast187 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1007&65535u)))))));
    buf27 = (buf27+(cast187*tg_bitcast<float>((unsigned int)(val1003))*((((float)(((signed char)(((val1008>>0u)&255u)))))*((float)(wmma176.z)))+(((float)(((signed char)(((val1008>>8u)&255u)))))*((float)(wmma177.z)))))+(cast187*tg_bitcast<float>((unsigned int)(val1004))*((((float)(((signed char)(((val1008>>16u)&255u)))))*((float)(wmma178.z)))+(((float)(((signed char)(((val1008>>24u)&255u)))))*((float)(wmma179.z)))))+(cast187*tg_bitcast<float>((unsigned int)(val1005))*((((float)(((signed char)(((val1009>>0u)&255u)))))*((float)(wmma180.z)))+(((float)(((signed char)(((val1009>>8u)&255u)))))*((float)(wmma181.z)))))+(cast187*tg_bitcast<float>((unsigned int)(val1006))*((((float)(((signed char)(((val1009>>16u)&255u)))))*((float)(wmma182.z)))+(((float)(((signed char)(((val1009>>24u)&255u)))))*((float)(wmma183.z))))));
    unsigned int val1010 = (*(buf0+alu229));
    unsigned int val1011 = (*(buf0+alu230));
    unsigned int val1012 = (*(buf0+alu231));
    unsigned int val1013 = (*(buf0+alu232));
    unsigned int val1014 = (*(buf0+alu185));
    unsigned int val1015 = (*(buf0+alu188));
    unsigned int val1016 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+1073)) = buf28;
    }
    float cast188 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1014&65535u)))))));
    buf28 = (buf28+(cast188*tg_bitcast<float>((unsigned int)(val1010))*((((float)(((signed char)(((val1015>>0u)&255u)))))*((float)(wmma176.w)))+(((float)(((signed char)(((val1015>>8u)&255u)))))*((float)(wmma177.w)))))+(cast188*tg_bitcast<float>((unsigned int)(val1011))*((((float)(((signed char)(((val1015>>16u)&255u)))))*((float)(wmma178.w)))+(((float)(((signed char)(((val1015>>24u)&255u)))))*((float)(wmma179.w)))))+(cast188*tg_bitcast<float>((unsigned int)(val1012))*((((float)(((signed char)(((val1016>>0u)&255u)))))*((float)(wmma180.w)))+(((float)(((signed char)(((val1016>>8u)&255u)))))*((float)(wmma181.w)))))+(cast188*tg_bitcast<float>((unsigned int)(val1013))*((((float)(((signed char)(((val1016>>16u)&255u)))))*((float)(wmma182.w)))+(((float)(((signed char)(((val1016>>24u)&255u)))))*((float)(wmma183.w))))));
    unsigned int val1017 = (*(buf0+alu225));
    unsigned int val1018 = (*(buf0+alu226));
    unsigned int val1019 = (*(buf0+alu227));
    unsigned int val1020 = (*(buf0+alu228));
    unsigned int val1021 = (*(buf0+alu190));
    unsigned int val1022 = (*(buf0+alu193));
    unsigned int val1023 = (*(buf0+alu194));
    if (0) {
      *(data0_5570560+(alu110+2096)) = buf29;
    }
    int4 wmma184 = __WMMA_8_16_16_signed_char_int(alu952, cast177, cast0);
    int4 wmma185 = __WMMA_8_16_16_signed_char_int(alu953, cast178, cast0);
    int4 wmma186 = __WMMA_8_16_16_signed_char_int(alu954, cast179, cast0);
    int4 wmma187 = __WMMA_8_16_16_signed_char_int(alu955, cast180, cast0);
    int4 wmma188 = __WMMA_8_16_16_signed_char_int(alu956, cast181, cast0);
    int4 wmma189 = __WMMA_8_16_16_signed_char_int(alu957, cast182, cast0);
    int4 wmma190 = __WMMA_8_16_16_signed_char_int(alu958, cast183, cast0);
    int4 wmma191 = __WMMA_8_16_16_signed_char_int(alu959, cast184, cast0);
    float cast189 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1021&65535u)))))));
    buf29 = (buf29+(cast189*tg_bitcast<float>((unsigned int)(val1017))*((((float)(((signed char)(((val1022>>0u)&255u)))))*((float)(wmma184.x)))+(((float)(((signed char)(((val1022>>8u)&255u)))))*((float)(wmma185.x)))))+(cast189*tg_bitcast<float>((unsigned int)(val1018))*((((float)(((signed char)(((val1022>>16u)&255u)))))*((float)(wmma186.x)))+(((float)(((signed char)(((val1022>>24u)&255u)))))*((float)(wmma187.x)))))+(cast189*tg_bitcast<float>((unsigned int)(val1019))*((((float)(((signed char)(((val1023>>0u)&255u)))))*((float)(wmma188.x)))+(((float)(((signed char)(((val1023>>8u)&255u)))))*((float)(wmma189.x)))))+(cast189*tg_bitcast<float>((unsigned int)(val1020))*((((float)(((signed char)(((val1023>>16u)&255u)))))*((float)(wmma190.x)))+(((float)(((signed char)(((val1023>>24u)&255u)))))*((float)(wmma191.x))))));
    unsigned int val1024 = (*(buf0+alu229));
    unsigned int val1025 = (*(buf0+alu230));
    unsigned int val1026 = (*(buf0+alu231));
    unsigned int val1027 = (*(buf0+alu232));
    unsigned int val1028 = (*(buf0+alu190));
    unsigned int val1029 = (*(buf0+alu193));
    unsigned int val1030 = (*(buf0+alu194));
    if (0) {
      *(data0_5570560+(alu110+2097)) = buf30;
    }
    float cast190 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1028&65535u)))))));
    buf30 = (buf30+(cast190*tg_bitcast<float>((unsigned int)(val1024))*((((float)(((signed char)(((val1029>>0u)&255u)))))*((float)(wmma184.y)))+(((float)(((signed char)(((val1029>>8u)&255u)))))*((float)(wmma185.y)))))+(cast190*tg_bitcast<float>((unsigned int)(val1025))*((((float)(((signed char)(((val1029>>16u)&255u)))))*((float)(wmma186.y)))+(((float)(((signed char)(((val1029>>24u)&255u)))))*((float)(wmma187.y)))))+(cast190*tg_bitcast<float>((unsigned int)(val1026))*((((float)(((signed char)(((val1030>>0u)&255u)))))*((float)(wmma188.y)))+(((float)(((signed char)(((val1030>>8u)&255u)))))*((float)(wmma189.y)))))+(cast190*tg_bitcast<float>((unsigned int)(val1027))*((((float)(((signed char)(((val1030>>16u)&255u)))))*((float)(wmma190.y)))+(((float)(((signed char)(((val1030>>24u)&255u)))))*((float)(wmma191.y))))));
    unsigned int val1031 = (*(buf0+alu225));
    unsigned int val1032 = (*(buf0+alu226));
    unsigned int val1033 = (*(buf0+alu227));
    unsigned int val1034 = (*(buf0+alu228));
    unsigned int val1035 = (*(buf0+alu195));
    unsigned int val1036 = (*(buf0+alu198));
    unsigned int val1037 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+3120)) = buf31;
    }
    float cast191 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1035&65535u)))))));
    buf31 = (buf31+(cast191*tg_bitcast<float>((unsigned int)(val1031))*((((float)(((signed char)(((val1036>>0u)&255u)))))*((float)(wmma184.z)))+(((float)(((signed char)(((val1036>>8u)&255u)))))*((float)(wmma185.z)))))+(cast191*tg_bitcast<float>((unsigned int)(val1032))*((((float)(((signed char)(((val1036>>16u)&255u)))))*((float)(wmma186.z)))+(((float)(((signed char)(((val1036>>24u)&255u)))))*((float)(wmma187.z)))))+(cast191*tg_bitcast<float>((unsigned int)(val1033))*((((float)(((signed char)(((val1037>>0u)&255u)))))*((float)(wmma188.z)))+(((float)(((signed char)(((val1037>>8u)&255u)))))*((float)(wmma189.z)))))+(cast191*tg_bitcast<float>((unsigned int)(val1034))*((((float)(((signed char)(((val1037>>16u)&255u)))))*((float)(wmma190.z)))+(((float)(((signed char)(((val1037>>24u)&255u)))))*((float)(wmma191.z))))));
    unsigned int val1038 = (*(buf0+alu229));
    unsigned int val1039 = (*(buf0+alu230));
    unsigned int val1040 = (*(buf0+alu231));
    unsigned int val1041 = (*(buf0+alu232));
    unsigned int val1042 = (*(buf0+alu195));
    unsigned int val1043 = (*(buf0+alu198));
    unsigned int val1044 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+3121)) = buf32;
    }
    float cast192 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1042&65535u)))))));
    buf32 = (buf32+(cast192*tg_bitcast<float>((unsigned int)(val1038))*((((float)(((signed char)(((val1043>>0u)&255u)))))*((float)(wmma184.w)))+(((float)(((signed char)(((val1043>>8u)&255u)))))*((float)(wmma185.w)))))+(cast192*tg_bitcast<float>((unsigned int)(val1039))*((((float)(((signed char)(((val1043>>16u)&255u)))))*((float)(wmma186.w)))+(((float)(((signed char)(((val1043>>24u)&255u)))))*((float)(wmma187.w)))))+(cast192*tg_bitcast<float>((unsigned int)(val1040))*((((float)(((signed char)(((val1044>>0u)&255u)))))*((float)(wmma188.w)))+(((float)(((signed char)(((val1044>>8u)&255u)))))*((float)(wmma189.w)))))+(cast192*tg_bitcast<float>((unsigned int)(val1041))*((((float)(((signed char)(((val1044>>16u)&255u)))))*((float)(wmma190.w)))+(((float)(((signed char)(((val1044>>24u)&255u)))))*((float)(wmma191.w))))));
    unsigned int val1045 = (*(buf0+alu147));
    unsigned int val1046 = (*(buf0+alu148));
    unsigned int val1047 = (*(buf0+alu149));
    unsigned int val1048 = (*(buf0+alu150));
    unsigned int val1049 = (*(buf0+alu151));
    unsigned int val1050 = (*(buf0+alu152));
    unsigned int val1051 = (*(buf0+alu153));
    unsigned int val1052 = (*(buf0+alu154));
    unsigned int val1053 = (*(buf0+alu233));
    unsigned int val1054 = (*(buf0+alu234));
    unsigned int val1055 = (*(buf0+alu235));
    unsigned int val1056 = (*(buf0+alu236));
    unsigned int val1057 = (*(buf0+alu180));
    unsigned int val1058 = (*(buf0+alu183));
    unsigned int val1059 = (*(buf0+alu184));
    if (0) {
      *(data0_5570560+(alu110+64)) = buf33;
    }
    char4 cast193 = make_char4(((signed char)(((val1045>>0u)&255u))),((signed char)(((val1045>>8u)&255u))),((signed char)(((val1045>>16u)&255u))),((signed char)(((val1045>>24u)&255u))));
    char4 cast194 = make_char4(((signed char)(((val1046>>0u)&255u))),((signed char)(((val1046>>8u)&255u))),((signed char)(((val1046>>16u)&255u))),((signed char)(((val1046>>24u)&255u))));
    char4 cast195 = make_char4(((signed char)(((val1047>>0u)&255u))),((signed char)(((val1047>>8u)&255u))),((signed char)(((val1047>>16u)&255u))),((signed char)(((val1047>>24u)&255u))));
    char4 cast196 = make_char4(((signed char)(((val1048>>0u)&255u))),((signed char)(((val1048>>8u)&255u))),((signed char)(((val1048>>16u)&255u))),((signed char)(((val1048>>24u)&255u))));
    char4 cast197 = make_char4(((signed char)(((val1049>>0u)&255u))),((signed char)(((val1049>>8u)&255u))),((signed char)(((val1049>>16u)&255u))),((signed char)(((val1049>>24u)&255u))));
    char4 cast198 = make_char4(((signed char)(((val1050>>0u)&255u))),((signed char)(((val1050>>8u)&255u))),((signed char)(((val1050>>16u)&255u))),((signed char)(((val1050>>24u)&255u))));
    char4 cast199 = make_char4(((signed char)(((val1051>>0u)&255u))),((signed char)(((val1051>>8u)&255u))),((signed char)(((val1051>>16u)&255u))),((signed char)(((val1051>>24u)&255u))));
    char4 cast200 = make_char4(((signed char)(((val1052>>0u)&255u))),((signed char)(((val1052>>8u)&255u))),((signed char)(((val1052>>16u)&255u))),((signed char)(((val1052>>24u)&255u))));
    int4 wmma192 = __WMMA_8_16_16_signed_char_int(alu928, cast193, cast0);
    int4 wmma193 = __WMMA_8_16_16_signed_char_int(alu929, cast194, cast0);
    int4 wmma194 = __WMMA_8_16_16_signed_char_int(alu930, cast195, cast0);
    int4 wmma195 = __WMMA_8_16_16_signed_char_int(alu931, cast196, cast0);
    int4 wmma196 = __WMMA_8_16_16_signed_char_int(alu932, cast197, cast0);
    int4 wmma197 = __WMMA_8_16_16_signed_char_int(alu933, cast198, cast0);
    int4 wmma198 = __WMMA_8_16_16_signed_char_int(alu934, cast199, cast0);
    int4 wmma199 = __WMMA_8_16_16_signed_char_int(alu935, cast200, cast0);
    float cast201 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1057&65535u)))))));
    buf33 = (buf33+(cast201*tg_bitcast<float>((unsigned int)(val1053))*((((float)(((signed char)(((val1058>>0u)&255u)))))*((float)(wmma192.x)))+(((float)(((signed char)(((val1058>>8u)&255u)))))*((float)(wmma193.x)))))+(cast201*tg_bitcast<float>((unsigned int)(val1054))*((((float)(((signed char)(((val1058>>16u)&255u)))))*((float)(wmma194.x)))+(((float)(((signed char)(((val1058>>24u)&255u)))))*((float)(wmma195.x)))))+(cast201*tg_bitcast<float>((unsigned int)(val1055))*((((float)(((signed char)(((val1059>>0u)&255u)))))*((float)(wmma196.x)))+(((float)(((signed char)(((val1059>>8u)&255u)))))*((float)(wmma197.x)))))+(cast201*tg_bitcast<float>((unsigned int)(val1056))*((((float)(((signed char)(((val1059>>16u)&255u)))))*((float)(wmma198.x)))+(((float)(((signed char)(((val1059>>24u)&255u)))))*((float)(wmma199.x))))));
    unsigned int val1060 = (*(buf0+alu237));
    unsigned int val1061 = (*(buf0+alu238));
    unsigned int val1062 = (*(buf0+alu239));
    unsigned int val1063 = (*(buf0+alu240));
    unsigned int val1064 = (*(buf0+alu180));
    unsigned int val1065 = (*(buf0+alu183));
    unsigned int val1066 = (*(buf0+alu184));
    if (0) {
      *(data0_5570560+(alu110+65)) = buf34;
    }
    float cast202 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1064&65535u)))))));
    buf34 = (buf34+(cast202*tg_bitcast<float>((unsigned int)(val1060))*((((float)(((signed char)(((val1065>>0u)&255u)))))*((float)(wmma192.y)))+(((float)(((signed char)(((val1065>>8u)&255u)))))*((float)(wmma193.y)))))+(cast202*tg_bitcast<float>((unsigned int)(val1061))*((((float)(((signed char)(((val1065>>16u)&255u)))))*((float)(wmma194.y)))+(((float)(((signed char)(((val1065>>24u)&255u)))))*((float)(wmma195.y)))))+(cast202*tg_bitcast<float>((unsigned int)(val1062))*((((float)(((signed char)(((val1066>>0u)&255u)))))*((float)(wmma196.y)))+(((float)(((signed char)(((val1066>>8u)&255u)))))*((float)(wmma197.y)))))+(cast202*tg_bitcast<float>((unsigned int)(val1063))*((((float)(((signed char)(((val1066>>16u)&255u)))))*((float)(wmma198.y)))+(((float)(((signed char)(((val1066>>24u)&255u)))))*((float)(wmma199.y))))));
    unsigned int val1067 = (*(buf0+alu233));
    unsigned int val1068 = (*(buf0+alu234));
    unsigned int val1069 = (*(buf0+alu235));
    unsigned int val1070 = (*(buf0+alu236));
    unsigned int val1071 = (*(buf0+alu185));
    unsigned int val1072 = (*(buf0+alu188));
    unsigned int val1073 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+1088)) = buf35;
    }
    float cast203 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1071&65535u)))))));
    buf35 = (buf35+(cast203*tg_bitcast<float>((unsigned int)(val1067))*((((float)(((signed char)(((val1072>>0u)&255u)))))*((float)(wmma192.z)))+(((float)(((signed char)(((val1072>>8u)&255u)))))*((float)(wmma193.z)))))+(cast203*tg_bitcast<float>((unsigned int)(val1068))*((((float)(((signed char)(((val1072>>16u)&255u)))))*((float)(wmma194.z)))+(((float)(((signed char)(((val1072>>24u)&255u)))))*((float)(wmma195.z)))))+(cast203*tg_bitcast<float>((unsigned int)(val1069))*((((float)(((signed char)(((val1073>>0u)&255u)))))*((float)(wmma196.z)))+(((float)(((signed char)(((val1073>>8u)&255u)))))*((float)(wmma197.z)))))+(cast203*tg_bitcast<float>((unsigned int)(val1070))*((((float)(((signed char)(((val1073>>16u)&255u)))))*((float)(wmma198.z)))+(((float)(((signed char)(((val1073>>24u)&255u)))))*((float)(wmma199.z))))));
    unsigned int val1074 = (*(buf0+alu237));
    unsigned int val1075 = (*(buf0+alu238));
    unsigned int val1076 = (*(buf0+alu239));
    unsigned int val1077 = (*(buf0+alu240));
    unsigned int val1078 = (*(buf0+alu185));
    unsigned int val1079 = (*(buf0+alu188));
    unsigned int val1080 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+1089)) = buf36;
    }
    float cast204 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1078&65535u)))))));
    buf36 = (buf36+(cast204*tg_bitcast<float>((unsigned int)(val1074))*((((float)(((signed char)(((val1079>>0u)&255u)))))*((float)(wmma192.w)))+(((float)(((signed char)(((val1079>>8u)&255u)))))*((float)(wmma193.w)))))+(cast204*tg_bitcast<float>((unsigned int)(val1075))*((((float)(((signed char)(((val1079>>16u)&255u)))))*((float)(wmma194.w)))+(((float)(((signed char)(((val1079>>24u)&255u)))))*((float)(wmma195.w)))))+(cast204*tg_bitcast<float>((unsigned int)(val1076))*((((float)(((signed char)(((val1080>>0u)&255u)))))*((float)(wmma196.w)))+(((float)(((signed char)(((val1080>>8u)&255u)))))*((float)(wmma197.w)))))+(cast204*tg_bitcast<float>((unsigned int)(val1077))*((((float)(((signed char)(((val1080>>16u)&255u)))))*((float)(wmma198.w)))+(((float)(((signed char)(((val1080>>24u)&255u)))))*((float)(wmma199.w))))));
    unsigned int val1081 = (*(buf0+alu233));
    unsigned int val1082 = (*(buf0+alu234));
    unsigned int val1083 = (*(buf0+alu235));
    unsigned int val1084 = (*(buf0+alu236));
    unsigned int val1085 = (*(buf0+alu190));
    unsigned int val1086 = (*(buf0+alu193));
    unsigned int val1087 = (*(buf0+alu194));
    if (0) {
      *(data0_5570560+(alu110+2112)) = buf37;
    }
    int4 wmma200 = __WMMA_8_16_16_signed_char_int(alu952, cast193, cast0);
    int4 wmma201 = __WMMA_8_16_16_signed_char_int(alu953, cast194, cast0);
    int4 wmma202 = __WMMA_8_16_16_signed_char_int(alu954, cast195, cast0);
    int4 wmma203 = __WMMA_8_16_16_signed_char_int(alu955, cast196, cast0);
    int4 wmma204 = __WMMA_8_16_16_signed_char_int(alu956, cast197, cast0);
    int4 wmma205 = __WMMA_8_16_16_signed_char_int(alu957, cast198, cast0);
    int4 wmma206 = __WMMA_8_16_16_signed_char_int(alu958, cast199, cast0);
    int4 wmma207 = __WMMA_8_16_16_signed_char_int(alu959, cast200, cast0);
    float cast205 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1085&65535u)))))));
    buf37 = (buf37+(cast205*tg_bitcast<float>((unsigned int)(val1081))*((((float)(((signed char)(((val1086>>0u)&255u)))))*((float)(wmma200.x)))+(((float)(((signed char)(((val1086>>8u)&255u)))))*((float)(wmma201.x)))))+(cast205*tg_bitcast<float>((unsigned int)(val1082))*((((float)(((signed char)(((val1086>>16u)&255u)))))*((float)(wmma202.x)))+(((float)(((signed char)(((val1086>>24u)&255u)))))*((float)(wmma203.x)))))+(cast205*tg_bitcast<float>((unsigned int)(val1083))*((((float)(((signed char)(((val1087>>0u)&255u)))))*((float)(wmma204.x)))+(((float)(((signed char)(((val1087>>8u)&255u)))))*((float)(wmma205.x)))))+(cast205*tg_bitcast<float>((unsigned int)(val1084))*((((float)(((signed char)(((val1087>>16u)&255u)))))*((float)(wmma206.x)))+(((float)(((signed char)(((val1087>>24u)&255u)))))*((float)(wmma207.x))))));
    unsigned int val1088 = (*(buf0+alu237));
    unsigned int val1089 = (*(buf0+alu238));
    unsigned int val1090 = (*(buf0+alu239));
    unsigned int val1091 = (*(buf0+alu240));
    unsigned int val1092 = (*(buf0+alu190));
    unsigned int val1093 = (*(buf0+alu193));
    unsigned int val1094 = (*(buf0+alu194));
    if (0) {
      *(data0_5570560+(alu110+2113)) = buf38;
    }
    float cast206 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1092&65535u)))))));
    buf38 = (buf38+(cast206*tg_bitcast<float>((unsigned int)(val1088))*((((float)(((signed char)(((val1093>>0u)&255u)))))*((float)(wmma200.y)))+(((float)(((signed char)(((val1093>>8u)&255u)))))*((float)(wmma201.y)))))+(cast206*tg_bitcast<float>((unsigned int)(val1089))*((((float)(((signed char)(((val1093>>16u)&255u)))))*((float)(wmma202.y)))+(((float)(((signed char)(((val1093>>24u)&255u)))))*((float)(wmma203.y)))))+(cast206*tg_bitcast<float>((unsigned int)(val1090))*((((float)(((signed char)(((val1094>>0u)&255u)))))*((float)(wmma204.y)))+(((float)(((signed char)(((val1094>>8u)&255u)))))*((float)(wmma205.y)))))+(cast206*tg_bitcast<float>((unsigned int)(val1091))*((((float)(((signed char)(((val1094>>16u)&255u)))))*((float)(wmma206.y)))+(((float)(((signed char)(((val1094>>24u)&255u)))))*((float)(wmma207.y))))));
    unsigned int val1095 = (*(buf0+alu233));
    unsigned int val1096 = (*(buf0+alu234));
    unsigned int val1097 = (*(buf0+alu235));
    unsigned int val1098 = (*(buf0+alu236));
    unsigned int val1099 = (*(buf0+alu195));
    unsigned int val1100 = (*(buf0+alu198));
    unsigned int val1101 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+3136)) = buf39;
    }
    float cast207 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1099&65535u)))))));
    buf39 = (buf39+(cast207*tg_bitcast<float>((unsigned int)(val1095))*((((float)(((signed char)(((val1100>>0u)&255u)))))*((float)(wmma200.z)))+(((float)(((signed char)(((val1100>>8u)&255u)))))*((float)(wmma201.z)))))+(cast207*tg_bitcast<float>((unsigned int)(val1096))*((((float)(((signed char)(((val1100>>16u)&255u)))))*((float)(wmma202.z)))+(((float)(((signed char)(((val1100>>24u)&255u)))))*((float)(wmma203.z)))))+(cast207*tg_bitcast<float>((unsigned int)(val1097))*((((float)(((signed char)(((val1101>>0u)&255u)))))*((float)(wmma204.z)))+(((float)(((signed char)(((val1101>>8u)&255u)))))*((float)(wmma205.z)))))+(cast207*tg_bitcast<float>((unsigned int)(val1098))*((((float)(((signed char)(((val1101>>16u)&255u)))))*((float)(wmma206.z)))+(((float)(((signed char)(((val1101>>24u)&255u)))))*((float)(wmma207.z))))));
    unsigned int val1102 = (*(buf0+alu237));
    unsigned int val1103 = (*(buf0+alu238));
    unsigned int val1104 = (*(buf0+alu239));
    unsigned int val1105 = (*(buf0+alu240));
    unsigned int val1106 = (*(buf0+alu195));
    unsigned int val1107 = (*(buf0+alu198));
    unsigned int val1108 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+3137)) = buf40;
    }
    float cast208 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1106&65535u)))))));
    buf40 = (buf40+(cast208*tg_bitcast<float>((unsigned int)(val1102))*((((float)(((signed char)(((val1107>>0u)&255u)))))*((float)(wmma200.w)))+(((float)(((signed char)(((val1107>>8u)&255u)))))*((float)(wmma201.w)))))+(cast208*tg_bitcast<float>((unsigned int)(val1103))*((((float)(((signed char)(((val1107>>16u)&255u)))))*((float)(wmma202.w)))+(((float)(((signed char)(((val1107>>24u)&255u)))))*((float)(wmma203.w)))))+(cast208*tg_bitcast<float>((unsigned int)(val1104))*((((float)(((signed char)(((val1108>>0u)&255u)))))*((float)(wmma204.w)))+(((float)(((signed char)(((val1108>>8u)&255u)))))*((float)(wmma205.w)))))+(cast208*tg_bitcast<float>((unsigned int)(val1105))*((((float)(((signed char)(((val1108>>16u)&255u)))))*((float)(wmma206.w)))+(((float)(((signed char)(((val1108>>24u)&255u)))))*((float)(wmma207.w))))));
    unsigned int val1109 = (*(buf0+alu155));
    unsigned int val1110 = (*(buf0+alu156));
    unsigned int val1111 = (*(buf0+alu157));
    unsigned int val1112 = (*(buf0+alu158));
    unsigned int val1113 = (*(buf0+alu159));
    unsigned int val1114 = (*(buf0+alu160));
    unsigned int val1115 = (*(buf0+alu161));
    unsigned int val1116 = (*(buf0+alu162));
    unsigned int val1117 = (*(buf0+alu241));
    unsigned int val1118 = (*(buf0+alu242));
    unsigned int val1119 = (*(buf0+alu243));
    unsigned int val1120 = (*(buf0+alu244));
    unsigned int val1121 = (*(buf0+alu180));
    unsigned int val1122 = (*(buf0+alu183));
    unsigned int val1123 = (*(buf0+alu184));
    if (0) {
      *(data0_5570560+(alu110+80)) = buf41;
    }
    char4 cast209 = make_char4(((signed char)(((val1109>>0u)&255u))),((signed char)(((val1109>>8u)&255u))),((signed char)(((val1109>>16u)&255u))),((signed char)(((val1109>>24u)&255u))));
    char4 cast210 = make_char4(((signed char)(((val1110>>0u)&255u))),((signed char)(((val1110>>8u)&255u))),((signed char)(((val1110>>16u)&255u))),((signed char)(((val1110>>24u)&255u))));
    char4 cast211 = make_char4(((signed char)(((val1111>>0u)&255u))),((signed char)(((val1111>>8u)&255u))),((signed char)(((val1111>>16u)&255u))),((signed char)(((val1111>>24u)&255u))));
    char4 cast212 = make_char4(((signed char)(((val1112>>0u)&255u))),((signed char)(((val1112>>8u)&255u))),((signed char)(((val1112>>16u)&255u))),((signed char)(((val1112>>24u)&255u))));
    char4 cast213 = make_char4(((signed char)(((val1113>>0u)&255u))),((signed char)(((val1113>>8u)&255u))),((signed char)(((val1113>>16u)&255u))),((signed char)(((val1113>>24u)&255u))));
    char4 cast214 = make_char4(((signed char)(((val1114>>0u)&255u))),((signed char)(((val1114>>8u)&255u))),((signed char)(((val1114>>16u)&255u))),((signed char)(((val1114>>24u)&255u))));
    char4 cast215 = make_char4(((signed char)(((val1115>>0u)&255u))),((signed char)(((val1115>>8u)&255u))),((signed char)(((val1115>>16u)&255u))),((signed char)(((val1115>>24u)&255u))));
    char4 cast216 = make_char4(((signed char)(((val1116>>0u)&255u))),((signed char)(((val1116>>8u)&255u))),((signed char)(((val1116>>16u)&255u))),((signed char)(((val1116>>24u)&255u))));
    int4 wmma208 = __WMMA_8_16_16_signed_char_int(alu928, cast209, cast0);
    int4 wmma209 = __WMMA_8_16_16_signed_char_int(alu929, cast210, cast0);
    int4 wmma210 = __WMMA_8_16_16_signed_char_int(alu930, cast211, cast0);
    int4 wmma211 = __WMMA_8_16_16_signed_char_int(alu931, cast212, cast0);
    int4 wmma212 = __WMMA_8_16_16_signed_char_int(alu932, cast213, cast0);
    int4 wmma213 = __WMMA_8_16_16_signed_char_int(alu933, cast214, cast0);
    int4 wmma214 = __WMMA_8_16_16_signed_char_int(alu934, cast215, cast0);
    int4 wmma215 = __WMMA_8_16_16_signed_char_int(alu935, cast216, cast0);
    float cast217 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1121&65535u)))))));
    buf41 = (buf41+(cast217*tg_bitcast<float>((unsigned int)(val1117))*((((float)(((signed char)(((val1122>>0u)&255u)))))*((float)(wmma208.x)))+(((float)(((signed char)(((val1122>>8u)&255u)))))*((float)(wmma209.x)))))+(cast217*tg_bitcast<float>((unsigned int)(val1118))*((((float)(((signed char)(((val1122>>16u)&255u)))))*((float)(wmma210.x)))+(((float)(((signed char)(((val1122>>24u)&255u)))))*((float)(wmma211.x)))))+(cast217*tg_bitcast<float>((unsigned int)(val1119))*((((float)(((signed char)(((val1123>>0u)&255u)))))*((float)(wmma212.x)))+(((float)(((signed char)(((val1123>>8u)&255u)))))*((float)(wmma213.x)))))+(cast217*tg_bitcast<float>((unsigned int)(val1120))*((((float)(((signed char)(((val1123>>16u)&255u)))))*((float)(wmma214.x)))+(((float)(((signed char)(((val1123>>24u)&255u)))))*((float)(wmma215.x))))));
    unsigned int val1124 = (*(buf0+alu245));
    unsigned int val1125 = (*(buf0+alu246));
    unsigned int val1126 = (*(buf0+alu247));
    unsigned int val1127 = (*(buf0+alu248));
    unsigned int val1128 = (*(buf0+alu180));
    unsigned int val1129 = (*(buf0+alu183));
    unsigned int val1130 = (*(buf0+alu184));
    if (0) {
      *(data0_5570560+(alu110+81)) = buf42;
    }
    float cast218 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1128&65535u)))))));
    buf42 = (buf42+(cast218*tg_bitcast<float>((unsigned int)(val1124))*((((float)(((signed char)(((val1129>>0u)&255u)))))*((float)(wmma208.y)))+(((float)(((signed char)(((val1129>>8u)&255u)))))*((float)(wmma209.y)))))+(cast218*tg_bitcast<float>((unsigned int)(val1125))*((((float)(((signed char)(((val1129>>16u)&255u)))))*((float)(wmma210.y)))+(((float)(((signed char)(((val1129>>24u)&255u)))))*((float)(wmma211.y)))))+(cast218*tg_bitcast<float>((unsigned int)(val1126))*((((float)(((signed char)(((val1130>>0u)&255u)))))*((float)(wmma212.y)))+(((float)(((signed char)(((val1130>>8u)&255u)))))*((float)(wmma213.y)))))+(cast218*tg_bitcast<float>((unsigned int)(val1127))*((((float)(((signed char)(((val1130>>16u)&255u)))))*((float)(wmma214.y)))+(((float)(((signed char)(((val1130>>24u)&255u)))))*((float)(wmma215.y))))));
    unsigned int val1131 = (*(buf0+alu241));
    unsigned int val1132 = (*(buf0+alu242));
    unsigned int val1133 = (*(buf0+alu243));
    unsigned int val1134 = (*(buf0+alu244));
    unsigned int val1135 = (*(buf0+alu185));
    unsigned int val1136 = (*(buf0+alu188));
    unsigned int val1137 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+1104)) = buf43;
    }
    float cast219 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1135&65535u)))))));
    buf43 = (buf43+(cast219*tg_bitcast<float>((unsigned int)(val1131))*((((float)(((signed char)(((val1136>>0u)&255u)))))*((float)(wmma208.z)))+(((float)(((signed char)(((val1136>>8u)&255u)))))*((float)(wmma209.z)))))+(cast219*tg_bitcast<float>((unsigned int)(val1132))*((((float)(((signed char)(((val1136>>16u)&255u)))))*((float)(wmma210.z)))+(((float)(((signed char)(((val1136>>24u)&255u)))))*((float)(wmma211.z)))))+(cast219*tg_bitcast<float>((unsigned int)(val1133))*((((float)(((signed char)(((val1137>>0u)&255u)))))*((float)(wmma212.z)))+(((float)(((signed char)(((val1137>>8u)&255u)))))*((float)(wmma213.z)))))+(cast219*tg_bitcast<float>((unsigned int)(val1134))*((((float)(((signed char)(((val1137>>16u)&255u)))))*((float)(wmma214.z)))+(((float)(((signed char)(((val1137>>24u)&255u)))))*((float)(wmma215.z))))));
    unsigned int val1138 = (*(buf0+alu245));
    unsigned int val1139 = (*(buf0+alu246));
    unsigned int val1140 = (*(buf0+alu247));
    unsigned int val1141 = (*(buf0+alu248));
    unsigned int val1142 = (*(buf0+alu185));
    unsigned int val1143 = (*(buf0+alu188));
    unsigned int val1144 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+1105)) = buf44;
    }
    float cast220 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1142&65535u)))))));
    buf44 = (buf44+(cast220*tg_bitcast<float>((unsigned int)(val1138))*((((float)(((signed char)(((val1143>>0u)&255u)))))*((float)(wmma208.w)))+(((float)(((signed char)(((val1143>>8u)&255u)))))*((float)(wmma209.w)))))+(cast220*tg_bitcast<float>((unsigned int)(val1139))*((((float)(((signed char)(((val1143>>16u)&255u)))))*((float)(wmma210.w)))+(((float)(((signed char)(((val1143>>24u)&255u)))))*((float)(wmma211.w)))))+(cast220*tg_bitcast<float>((unsigned int)(val1140))*((((float)(((signed char)(((val1144>>0u)&255u)))))*((float)(wmma212.w)))+(((float)(((signed char)(((val1144>>8u)&255u)))))*((float)(wmma213.w)))))+(cast220*tg_bitcast<float>((unsigned int)(val1141))*((((float)(((signed char)(((val1144>>16u)&255u)))))*((float)(wmma214.w)))+(((float)(((signed char)(((val1144>>24u)&255u)))))*((float)(wmma215.w))))));
    unsigned int val1145 = (*(buf0+alu241));
    unsigned int val1146 = (*(buf0+alu242));
    unsigned int val1147 = (*(buf0+alu243));
    unsigned int val1148 = (*(buf0+alu244));
    unsigned int val1149 = (*(buf0+alu190));
    unsigned int val1150 = (*(buf0+alu193));
    unsigned int val1151 = (*(buf0+alu194));
    if (0) {
      *(data0_5570560+(alu110+2128)) = buf45;
    }
    int4 wmma216 = __WMMA_8_16_16_signed_char_int(alu952, cast209, cast0);
    int4 wmma217 = __WMMA_8_16_16_signed_char_int(alu953, cast210, cast0);
    int4 wmma218 = __WMMA_8_16_16_signed_char_int(alu954, cast211, cast0);
    int4 wmma219 = __WMMA_8_16_16_signed_char_int(alu955, cast212, cast0);
    int4 wmma220 = __WMMA_8_16_16_signed_char_int(alu956, cast213, cast0);
    int4 wmma221 = __WMMA_8_16_16_signed_char_int(alu957, cast214, cast0);
    int4 wmma222 = __WMMA_8_16_16_signed_char_int(alu958, cast215, cast0);
    int4 wmma223 = __WMMA_8_16_16_signed_char_int(alu959, cast216, cast0);
    float cast221 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1149&65535u)))))));
    buf45 = (buf45+(cast221*tg_bitcast<float>((unsigned int)(val1145))*((((float)(((signed char)(((val1150>>0u)&255u)))))*((float)(wmma216.x)))+(((float)(((signed char)(((val1150>>8u)&255u)))))*((float)(wmma217.x)))))+(cast221*tg_bitcast<float>((unsigned int)(val1146))*((((float)(((signed char)(((val1150>>16u)&255u)))))*((float)(wmma218.x)))+(((float)(((signed char)(((val1150>>24u)&255u)))))*((float)(wmma219.x)))))+(cast221*tg_bitcast<float>((unsigned int)(val1147))*((((float)(((signed char)(((val1151>>0u)&255u)))))*((float)(wmma220.x)))+(((float)(((signed char)(((val1151>>8u)&255u)))))*((float)(wmma221.x)))))+(cast221*tg_bitcast<float>((unsigned int)(val1148))*((((float)(((signed char)(((val1151>>16u)&255u)))))*((float)(wmma222.x)))+(((float)(((signed char)(((val1151>>24u)&255u)))))*((float)(wmma223.x))))));
    unsigned int val1152 = (*(buf0+alu245));
    unsigned int val1153 = (*(buf0+alu246));
    unsigned int val1154 = (*(buf0+alu247));
    unsigned int val1155 = (*(buf0+alu248));
    unsigned int val1156 = (*(buf0+alu190));
    unsigned int val1157 = (*(buf0+alu193));
    unsigned int val1158 = (*(buf0+alu194));
    if (0) {
      *(data0_5570560+(alu110+2129)) = buf46;
    }
    float cast222 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1156&65535u)))))));
    buf46 = (buf46+(cast222*tg_bitcast<float>((unsigned int)(val1152))*((((float)(((signed char)(((val1157>>0u)&255u)))))*((float)(wmma216.y)))+(((float)(((signed char)(((val1157>>8u)&255u)))))*((float)(wmma217.y)))))+(cast222*tg_bitcast<float>((unsigned int)(val1153))*((((float)(((signed char)(((val1157>>16u)&255u)))))*((float)(wmma218.y)))+(((float)(((signed char)(((val1157>>24u)&255u)))))*((float)(wmma219.y)))))+(cast222*tg_bitcast<float>((unsigned int)(val1154))*((((float)(((signed char)(((val1158>>0u)&255u)))))*((float)(wmma220.y)))+(((float)(((signed char)(((val1158>>8u)&255u)))))*((float)(wmma221.y)))))+(cast222*tg_bitcast<float>((unsigned int)(val1155))*((((float)(((signed char)(((val1158>>16u)&255u)))))*((float)(wmma222.y)))+(((float)(((signed char)(((val1158>>24u)&255u)))))*((float)(wmma223.y))))));
    unsigned int val1159 = (*(buf0+alu241));
    unsigned int val1160 = (*(buf0+alu242));
    unsigned int val1161 = (*(buf0+alu243));
    unsigned int val1162 = (*(buf0+alu244));
    unsigned int val1163 = (*(buf0+alu195));
    unsigned int val1164 = (*(buf0+alu198));
    unsigned int val1165 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+3152)) = buf47;
    }
    float cast223 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1163&65535u)))))));
    buf47 = (buf47+(cast223*tg_bitcast<float>((unsigned int)(val1159))*((((float)(((signed char)(((val1164>>0u)&255u)))))*((float)(wmma216.z)))+(((float)(((signed char)(((val1164>>8u)&255u)))))*((float)(wmma217.z)))))+(cast223*tg_bitcast<float>((unsigned int)(val1160))*((((float)(((signed char)(((val1164>>16u)&255u)))))*((float)(wmma218.z)))+(((float)(((signed char)(((val1164>>24u)&255u)))))*((float)(wmma219.z)))))+(cast223*tg_bitcast<float>((unsigned int)(val1161))*((((float)(((signed char)(((val1165>>0u)&255u)))))*((float)(wmma220.z)))+(((float)(((signed char)(((val1165>>8u)&255u)))))*((float)(wmma221.z)))))+(cast223*tg_bitcast<float>((unsigned int)(val1162))*((((float)(((signed char)(((val1165>>16u)&255u)))))*((float)(wmma222.z)))+(((float)(((signed char)(((val1165>>24u)&255u)))))*((float)(wmma223.z))))));
    unsigned int val1166 = (*(buf0+alu245));
    unsigned int val1167 = (*(buf0+alu246));
    unsigned int val1168 = (*(buf0+alu247));
    unsigned int val1169 = (*(buf0+alu248));
    unsigned int val1170 = (*(buf0+alu195));
    unsigned int val1171 = (*(buf0+alu198));
    unsigned int val1172 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+3153)) = buf48;
    }
    float cast224 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1170&65535u)))))));
    buf48 = (buf48+(cast224*tg_bitcast<float>((unsigned int)(val1166))*((((float)(((signed char)(((val1171>>0u)&255u)))))*((float)(wmma216.w)))+(((float)(((signed char)(((val1171>>8u)&255u)))))*((float)(wmma217.w)))))+(cast224*tg_bitcast<float>((unsigned int)(val1167))*((((float)(((signed char)(((val1171>>16u)&255u)))))*((float)(wmma218.w)))+(((float)(((signed char)(((val1171>>24u)&255u)))))*((float)(wmma219.w)))))+(cast224*tg_bitcast<float>((unsigned int)(val1168))*((((float)(((signed char)(((val1172>>0u)&255u)))))*((float)(wmma220.w)))+(((float)(((signed char)(((val1172>>8u)&255u)))))*((float)(wmma221.w)))))+(cast224*tg_bitcast<float>((unsigned int)(val1169))*((((float)(((signed char)(((val1172>>16u)&255u)))))*((float)(wmma222.w)))+(((float)(((signed char)(((val1172>>24u)&255u)))))*((float)(wmma223.w))))));
    unsigned int val1173 = (*(buf0+alu163));
    unsigned int val1174 = (*(buf0+alu164));
    unsigned int val1175 = (*(buf0+alu165));
    unsigned int val1176 = (*(buf0+alu166));
    unsigned int val1177 = (*(buf0+alu167));
    unsigned int val1178 = (*(buf0+alu168));
    unsigned int val1179 = (*(buf0+alu169));
    unsigned int val1180 = (*(buf0+alu170));
    unsigned int val1181 = (*(buf0+alu249));
    unsigned int val1182 = (*(buf0+alu250));
    unsigned int val1183 = (*(buf0+alu251));
    unsigned int val1184 = (*(buf0+alu252));
    unsigned int val1185 = (*(buf0+alu180));
    unsigned int val1186 = (*(buf0+alu183));
    unsigned int val1187 = (*(buf0+alu184));
    if (0) {
      *(data0_5570560+(alu110+96)) = buf49;
    }
    char4 cast225 = make_char4(((signed char)(((val1173>>0u)&255u))),((signed char)(((val1173>>8u)&255u))),((signed char)(((val1173>>16u)&255u))),((signed char)(((val1173>>24u)&255u))));
    char4 cast226 = make_char4(((signed char)(((val1174>>0u)&255u))),((signed char)(((val1174>>8u)&255u))),((signed char)(((val1174>>16u)&255u))),((signed char)(((val1174>>24u)&255u))));
    char4 cast227 = make_char4(((signed char)(((val1175>>0u)&255u))),((signed char)(((val1175>>8u)&255u))),((signed char)(((val1175>>16u)&255u))),((signed char)(((val1175>>24u)&255u))));
    char4 cast228 = make_char4(((signed char)(((val1176>>0u)&255u))),((signed char)(((val1176>>8u)&255u))),((signed char)(((val1176>>16u)&255u))),((signed char)(((val1176>>24u)&255u))));
    char4 cast229 = make_char4(((signed char)(((val1177>>0u)&255u))),((signed char)(((val1177>>8u)&255u))),((signed char)(((val1177>>16u)&255u))),((signed char)(((val1177>>24u)&255u))));
    char4 cast230 = make_char4(((signed char)(((val1178>>0u)&255u))),((signed char)(((val1178>>8u)&255u))),((signed char)(((val1178>>16u)&255u))),((signed char)(((val1178>>24u)&255u))));
    char4 cast231 = make_char4(((signed char)(((val1179>>0u)&255u))),((signed char)(((val1179>>8u)&255u))),((signed char)(((val1179>>16u)&255u))),((signed char)(((val1179>>24u)&255u))));
    char4 cast232 = make_char4(((signed char)(((val1180>>0u)&255u))),((signed char)(((val1180>>8u)&255u))),((signed char)(((val1180>>16u)&255u))),((signed char)(((val1180>>24u)&255u))));
    int4 wmma224 = __WMMA_8_16_16_signed_char_int(alu928, cast225, cast0);
    int4 wmma225 = __WMMA_8_16_16_signed_char_int(alu929, cast226, cast0);
    int4 wmma226 = __WMMA_8_16_16_signed_char_int(alu930, cast227, cast0);
    int4 wmma227 = __WMMA_8_16_16_signed_char_int(alu931, cast228, cast0);
    int4 wmma228 = __WMMA_8_16_16_signed_char_int(alu932, cast229, cast0);
    int4 wmma229 = __WMMA_8_16_16_signed_char_int(alu933, cast230, cast0);
    int4 wmma230 = __WMMA_8_16_16_signed_char_int(alu934, cast231, cast0);
    int4 wmma231 = __WMMA_8_16_16_signed_char_int(alu935, cast232, cast0);
    float cast233 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1185&65535u)))))));
    buf49 = (buf49+(cast233*tg_bitcast<float>((unsigned int)(val1181))*((((float)(((signed char)(((val1186>>0u)&255u)))))*((float)(wmma224.x)))+(((float)(((signed char)(((val1186>>8u)&255u)))))*((float)(wmma225.x)))))+(cast233*tg_bitcast<float>((unsigned int)(val1182))*((((float)(((signed char)(((val1186>>16u)&255u)))))*((float)(wmma226.x)))+(((float)(((signed char)(((val1186>>24u)&255u)))))*((float)(wmma227.x)))))+(cast233*tg_bitcast<float>((unsigned int)(val1183))*((((float)(((signed char)(((val1187>>0u)&255u)))))*((float)(wmma228.x)))+(((float)(((signed char)(((val1187>>8u)&255u)))))*((float)(wmma229.x)))))+(cast233*tg_bitcast<float>((unsigned int)(val1184))*((((float)(((signed char)(((val1187>>16u)&255u)))))*((float)(wmma230.x)))+(((float)(((signed char)(((val1187>>24u)&255u)))))*((float)(wmma231.x))))));
    unsigned int val1188 = (*(buf0+alu253));
    unsigned int val1189 = (*(buf0+alu254));
    unsigned int val1190 = (*(buf0+alu255));
    unsigned int val1191 = (*(buf0+alu256));
    unsigned int val1192 = (*(buf0+alu180));
    unsigned int val1193 = (*(buf0+alu183));
    unsigned int val1194 = (*(buf0+alu184));
    if (0) {
      *(data0_5570560+(alu110+97)) = buf50;
    }
    float cast234 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1192&65535u)))))));
    buf50 = (buf50+(cast234*tg_bitcast<float>((unsigned int)(val1188))*((((float)(((signed char)(((val1193>>0u)&255u)))))*((float)(wmma224.y)))+(((float)(((signed char)(((val1193>>8u)&255u)))))*((float)(wmma225.y)))))+(cast234*tg_bitcast<float>((unsigned int)(val1189))*((((float)(((signed char)(((val1193>>16u)&255u)))))*((float)(wmma226.y)))+(((float)(((signed char)(((val1193>>24u)&255u)))))*((float)(wmma227.y)))))+(cast234*tg_bitcast<float>((unsigned int)(val1190))*((((float)(((signed char)(((val1194>>0u)&255u)))))*((float)(wmma228.y)))+(((float)(((signed char)(((val1194>>8u)&255u)))))*((float)(wmma229.y)))))+(cast234*tg_bitcast<float>((unsigned int)(val1191))*((((float)(((signed char)(((val1194>>16u)&255u)))))*((float)(wmma230.y)))+(((float)(((signed char)(((val1194>>24u)&255u)))))*((float)(wmma231.y))))));
    unsigned int val1195 = (*(buf0+alu249));
    unsigned int val1196 = (*(buf0+alu250));
    unsigned int val1197 = (*(buf0+alu251));
    unsigned int val1198 = (*(buf0+alu252));
    unsigned int val1199 = (*(buf0+alu185));
    unsigned int val1200 = (*(buf0+alu188));
    unsigned int val1201 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+1120)) = buf51;
    }
    float cast235 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1199&65535u)))))));
    buf51 = (buf51+(cast235*tg_bitcast<float>((unsigned int)(val1195))*((((float)(((signed char)(((val1200>>0u)&255u)))))*((float)(wmma224.z)))+(((float)(((signed char)(((val1200>>8u)&255u)))))*((float)(wmma225.z)))))+(cast235*tg_bitcast<float>((unsigned int)(val1196))*((((float)(((signed char)(((val1200>>16u)&255u)))))*((float)(wmma226.z)))+(((float)(((signed char)(((val1200>>24u)&255u)))))*((float)(wmma227.z)))))+(cast235*tg_bitcast<float>((unsigned int)(val1197))*((((float)(((signed char)(((val1201>>0u)&255u)))))*((float)(wmma228.z)))+(((float)(((signed char)(((val1201>>8u)&255u)))))*((float)(wmma229.z)))))+(cast235*tg_bitcast<float>((unsigned int)(val1198))*((((float)(((signed char)(((val1201>>16u)&255u)))))*((float)(wmma230.z)))+(((float)(((signed char)(((val1201>>24u)&255u)))))*((float)(wmma231.z))))));
    unsigned int val1202 = (*(buf0+alu253));
    unsigned int val1203 = (*(buf0+alu254));
    unsigned int val1204 = (*(buf0+alu255));
    unsigned int val1205 = (*(buf0+alu256));
    unsigned int val1206 = (*(buf0+alu185));
    unsigned int val1207 = (*(buf0+alu188));
    unsigned int val1208 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+1121)) = buf52;
    }
    float cast236 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1206&65535u)))))));
    buf52 = (buf52+(cast236*tg_bitcast<float>((unsigned int)(val1202))*((((float)(((signed char)(((val1207>>0u)&255u)))))*((float)(wmma224.w)))+(((float)(((signed char)(((val1207>>8u)&255u)))))*((float)(wmma225.w)))))+(cast236*tg_bitcast<float>((unsigned int)(val1203))*((((float)(((signed char)(((val1207>>16u)&255u)))))*((float)(wmma226.w)))+(((float)(((signed char)(((val1207>>24u)&255u)))))*((float)(wmma227.w)))))+(cast236*tg_bitcast<float>((unsigned int)(val1204))*((((float)(((signed char)(((val1208>>0u)&255u)))))*((float)(wmma228.w)))+(((float)(((signed char)(((val1208>>8u)&255u)))))*((float)(wmma229.w)))))+(cast236*tg_bitcast<float>((unsigned int)(val1205))*((((float)(((signed char)(((val1208>>16u)&255u)))))*((float)(wmma230.w)))+(((float)(((signed char)(((val1208>>24u)&255u)))))*((float)(wmma231.w))))));
    unsigned int val1209 = (*(buf0+alu249));
    unsigned int val1210 = (*(buf0+alu250));
    unsigned int val1211 = (*(buf0+alu251));
    unsigned int val1212 = (*(buf0+alu252));
    unsigned int val1213 = (*(buf0+alu190));
    unsigned int val1214 = (*(buf0+alu193));
    unsigned int val1215 = (*(buf0+alu194));
    if (0) {
      *(data0_5570560+(alu110+2144)) = buf53;
    }
    int4 wmma232 = __WMMA_8_16_16_signed_char_int(alu952, cast225, cast0);
    int4 wmma233 = __WMMA_8_16_16_signed_char_int(alu953, cast226, cast0);
    int4 wmma234 = __WMMA_8_16_16_signed_char_int(alu954, cast227, cast0);
    int4 wmma235 = __WMMA_8_16_16_signed_char_int(alu955, cast228, cast0);
    int4 wmma236 = __WMMA_8_16_16_signed_char_int(alu956, cast229, cast0);
    int4 wmma237 = __WMMA_8_16_16_signed_char_int(alu957, cast230, cast0);
    int4 wmma238 = __WMMA_8_16_16_signed_char_int(alu958, cast231, cast0);
    int4 wmma239 = __WMMA_8_16_16_signed_char_int(alu959, cast232, cast0);
    float cast237 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1213&65535u)))))));
    buf53 = (buf53+(cast237*tg_bitcast<float>((unsigned int)(val1209))*((((float)(((signed char)(((val1214>>0u)&255u)))))*((float)(wmma232.x)))+(((float)(((signed char)(((val1214>>8u)&255u)))))*((float)(wmma233.x)))))+(cast237*tg_bitcast<float>((unsigned int)(val1210))*((((float)(((signed char)(((val1214>>16u)&255u)))))*((float)(wmma234.x)))+(((float)(((signed char)(((val1214>>24u)&255u)))))*((float)(wmma235.x)))))+(cast237*tg_bitcast<float>((unsigned int)(val1211))*((((float)(((signed char)(((val1215>>0u)&255u)))))*((float)(wmma236.x)))+(((float)(((signed char)(((val1215>>8u)&255u)))))*((float)(wmma237.x)))))+(cast237*tg_bitcast<float>((unsigned int)(val1212))*((((float)(((signed char)(((val1215>>16u)&255u)))))*((float)(wmma238.x)))+(((float)(((signed char)(((val1215>>24u)&255u)))))*((float)(wmma239.x))))));
    unsigned int val1216 = (*(buf0+alu253));
    unsigned int val1217 = (*(buf0+alu254));
    unsigned int val1218 = (*(buf0+alu255));
    unsigned int val1219 = (*(buf0+alu256));
    unsigned int val1220 = (*(buf0+alu190));
    unsigned int val1221 = (*(buf0+alu193));
    unsigned int val1222 = (*(buf0+alu194));
    if (0) {
      *(data0_5570560+(alu110+2145)) = buf54;
    }
    float cast238 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1220&65535u)))))));
    buf54 = (buf54+(cast238*tg_bitcast<float>((unsigned int)(val1216))*((((float)(((signed char)(((val1221>>0u)&255u)))))*((float)(wmma232.y)))+(((float)(((signed char)(((val1221>>8u)&255u)))))*((float)(wmma233.y)))))+(cast238*tg_bitcast<float>((unsigned int)(val1217))*((((float)(((signed char)(((val1221>>16u)&255u)))))*((float)(wmma234.y)))+(((float)(((signed char)(((val1221>>24u)&255u)))))*((float)(wmma235.y)))))+(cast238*tg_bitcast<float>((unsigned int)(val1218))*((((float)(((signed char)(((val1222>>0u)&255u)))))*((float)(wmma236.y)))+(((float)(((signed char)(((val1222>>8u)&255u)))))*((float)(wmma237.y)))))+(cast238*tg_bitcast<float>((unsigned int)(val1219))*((((float)(((signed char)(((val1222>>16u)&255u)))))*((float)(wmma238.y)))+(((float)(((signed char)(((val1222>>24u)&255u)))))*((float)(wmma239.y))))));
    unsigned int val1223 = (*(buf0+alu249));
    unsigned int val1224 = (*(buf0+alu250));
    unsigned int val1225 = (*(buf0+alu251));
    unsigned int val1226 = (*(buf0+alu252));
    unsigned int val1227 = (*(buf0+alu195));
    unsigned int val1228 = (*(buf0+alu198));
    unsigned int val1229 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+3168)) = buf55;
    }
    float cast239 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1227&65535u)))))));
    buf55 = (buf55+(cast239*tg_bitcast<float>((unsigned int)(val1223))*((((float)(((signed char)(((val1228>>0u)&255u)))))*((float)(wmma232.z)))+(((float)(((signed char)(((val1228>>8u)&255u)))))*((float)(wmma233.z)))))+(cast239*tg_bitcast<float>((unsigned int)(val1224))*((((float)(((signed char)(((val1228>>16u)&255u)))))*((float)(wmma234.z)))+(((float)(((signed char)(((val1228>>24u)&255u)))))*((float)(wmma235.z)))))+(cast239*tg_bitcast<float>((unsigned int)(val1225))*((((float)(((signed char)(((val1229>>0u)&255u)))))*((float)(wmma236.z)))+(((float)(((signed char)(((val1229>>8u)&255u)))))*((float)(wmma237.z)))))+(cast239*tg_bitcast<float>((unsigned int)(val1226))*((((float)(((signed char)(((val1229>>16u)&255u)))))*((float)(wmma238.z)))+(((float)(((signed char)(((val1229>>24u)&255u)))))*((float)(wmma239.z))))));
    unsigned int val1230 = (*(buf0+alu253));
    unsigned int val1231 = (*(buf0+alu254));
    unsigned int val1232 = (*(buf0+alu255));
    unsigned int val1233 = (*(buf0+alu256));
    unsigned int val1234 = (*(buf0+alu195));
    unsigned int val1235 = (*(buf0+alu198));
    unsigned int val1236 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+3169)) = buf56;
    }
    float cast240 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1234&65535u)))))));
    buf56 = (buf56+(cast240*tg_bitcast<float>((unsigned int)(val1230))*((((float)(((signed char)(((val1235>>0u)&255u)))))*((float)(wmma232.w)))+(((float)(((signed char)(((val1235>>8u)&255u)))))*((float)(wmma233.w)))))+(cast240*tg_bitcast<float>((unsigned int)(val1231))*((((float)(((signed char)(((val1235>>16u)&255u)))))*((float)(wmma234.w)))+(((float)(((signed char)(((val1235>>24u)&255u)))))*((float)(wmma235.w)))))+(cast240*tg_bitcast<float>((unsigned int)(val1232))*((((float)(((signed char)(((val1236>>0u)&255u)))))*((float)(wmma236.w)))+(((float)(((signed char)(((val1236>>8u)&255u)))))*((float)(wmma237.w)))))+(cast240*tg_bitcast<float>((unsigned int)(val1233))*((((float)(((signed char)(((val1236>>16u)&255u)))))*((float)(wmma238.w)))+(((float)(((signed char)(((val1236>>24u)&255u)))))*((float)(wmma239.w))))));
    unsigned int val1237 = (*(buf0+alu171));
    unsigned int val1238 = (*(buf0+alu172));
    unsigned int val1239 = (*(buf0+alu173));
    unsigned int val1240 = (*(buf0+alu174));
    unsigned int val1241 = (*(buf0+alu175));
    unsigned int val1242 = (*(buf0+alu176));
    unsigned int val1243 = (*(buf0+alu177));
    unsigned int val1244 = (*(buf0+alu178));
    unsigned int val1245 = (*(buf0+alu257));
    unsigned int val1246 = (*(buf0+alu258));
    unsigned int val1247 = (*(buf0+alu259));
    unsigned int val1248 = (*(buf0+alu260));
    unsigned int val1249 = (*(buf0+alu180));
    unsigned int val1250 = (*(buf0+alu183));
    unsigned int val1251 = (*(buf0+alu184));
    if (0) {
      *(data0_5570560+(alu110+112)) = buf57;
    }
    char4 cast241 = make_char4(((signed char)(((val1237>>0u)&255u))),((signed char)(((val1237>>8u)&255u))),((signed char)(((val1237>>16u)&255u))),((signed char)(((val1237>>24u)&255u))));
    char4 cast242 = make_char4(((signed char)(((val1238>>0u)&255u))),((signed char)(((val1238>>8u)&255u))),((signed char)(((val1238>>16u)&255u))),((signed char)(((val1238>>24u)&255u))));
    char4 cast243 = make_char4(((signed char)(((val1239>>0u)&255u))),((signed char)(((val1239>>8u)&255u))),((signed char)(((val1239>>16u)&255u))),((signed char)(((val1239>>24u)&255u))));
    char4 cast244 = make_char4(((signed char)(((val1240>>0u)&255u))),((signed char)(((val1240>>8u)&255u))),((signed char)(((val1240>>16u)&255u))),((signed char)(((val1240>>24u)&255u))));
    char4 cast245 = make_char4(((signed char)(((val1241>>0u)&255u))),((signed char)(((val1241>>8u)&255u))),((signed char)(((val1241>>16u)&255u))),((signed char)(((val1241>>24u)&255u))));
    char4 cast246 = make_char4(((signed char)(((val1242>>0u)&255u))),((signed char)(((val1242>>8u)&255u))),((signed char)(((val1242>>16u)&255u))),((signed char)(((val1242>>24u)&255u))));
    char4 cast247 = make_char4(((signed char)(((val1243>>0u)&255u))),((signed char)(((val1243>>8u)&255u))),((signed char)(((val1243>>16u)&255u))),((signed char)(((val1243>>24u)&255u))));
    char4 cast248 = make_char4(((signed char)(((val1244>>0u)&255u))),((signed char)(((val1244>>8u)&255u))),((signed char)(((val1244>>16u)&255u))),((signed char)(((val1244>>24u)&255u))));
    int4 wmma240 = __WMMA_8_16_16_signed_char_int(alu928, cast241, cast0);
    int4 wmma241 = __WMMA_8_16_16_signed_char_int(alu929, cast242, cast0);
    int4 wmma242 = __WMMA_8_16_16_signed_char_int(alu930, cast243, cast0);
    int4 wmma243 = __WMMA_8_16_16_signed_char_int(alu931, cast244, cast0);
    int4 wmma244 = __WMMA_8_16_16_signed_char_int(alu932, cast245, cast0);
    int4 wmma245 = __WMMA_8_16_16_signed_char_int(alu933, cast246, cast0);
    int4 wmma246 = __WMMA_8_16_16_signed_char_int(alu934, cast247, cast0);
    int4 wmma247 = __WMMA_8_16_16_signed_char_int(alu935, cast248, cast0);
    float cast249 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1249&65535u)))))));
    buf57 = (buf57+(cast249*tg_bitcast<float>((unsigned int)(val1245))*((((float)(((signed char)(((val1250>>0u)&255u)))))*((float)(wmma240.x)))+(((float)(((signed char)(((val1250>>8u)&255u)))))*((float)(wmma241.x)))))+(cast249*tg_bitcast<float>((unsigned int)(val1246))*((((float)(((signed char)(((val1250>>16u)&255u)))))*((float)(wmma242.x)))+(((float)(((signed char)(((val1250>>24u)&255u)))))*((float)(wmma243.x)))))+(cast249*tg_bitcast<float>((unsigned int)(val1247))*((((float)(((signed char)(((val1251>>0u)&255u)))))*((float)(wmma244.x)))+(((float)(((signed char)(((val1251>>8u)&255u)))))*((float)(wmma245.x)))))+(cast249*tg_bitcast<float>((unsigned int)(val1248))*((((float)(((signed char)(((val1251>>16u)&255u)))))*((float)(wmma246.x)))+(((float)(((signed char)(((val1251>>24u)&255u)))))*((float)(wmma247.x))))));
    unsigned int val1252 = (*(buf0+alu261));
    unsigned int val1253 = (*(buf0+alu262));
    unsigned int val1254 = (*(buf0+alu263));
    unsigned int val1255 = (*(buf0+alu264));
    unsigned int val1256 = (*(buf0+alu180));
    unsigned int val1257 = (*(buf0+alu183));
    unsigned int val1258 = (*(buf0+alu184));
    if (0) {
      *(data0_5570560+(alu110+113)) = buf58;
    }
    float cast250 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1256&65535u)))))));
    buf58 = (buf58+(cast250*tg_bitcast<float>((unsigned int)(val1252))*((((float)(((signed char)(((val1257>>0u)&255u)))))*((float)(wmma240.y)))+(((float)(((signed char)(((val1257>>8u)&255u)))))*((float)(wmma241.y)))))+(cast250*tg_bitcast<float>((unsigned int)(val1253))*((((float)(((signed char)(((val1257>>16u)&255u)))))*((float)(wmma242.y)))+(((float)(((signed char)(((val1257>>24u)&255u)))))*((float)(wmma243.y)))))+(cast250*tg_bitcast<float>((unsigned int)(val1254))*((((float)(((signed char)(((val1258>>0u)&255u)))))*((float)(wmma244.y)))+(((float)(((signed char)(((val1258>>8u)&255u)))))*((float)(wmma245.y)))))+(cast250*tg_bitcast<float>((unsigned int)(val1255))*((((float)(((signed char)(((val1258>>16u)&255u)))))*((float)(wmma246.y)))+(((float)(((signed char)(((val1258>>24u)&255u)))))*((float)(wmma247.y))))));
    unsigned int val1259 = (*(buf0+alu257));
    unsigned int val1260 = (*(buf0+alu258));
    unsigned int val1261 = (*(buf0+alu259));
    unsigned int val1262 = (*(buf0+alu260));
    unsigned int val1263 = (*(buf0+alu185));
    unsigned int val1264 = (*(buf0+alu188));
    unsigned int val1265 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+1136)) = buf59;
    }
    float cast251 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1263&65535u)))))));
    buf59 = (buf59+(cast251*tg_bitcast<float>((unsigned int)(val1259))*((((float)(((signed char)(((val1264>>0u)&255u)))))*((float)(wmma240.z)))+(((float)(((signed char)(((val1264>>8u)&255u)))))*((float)(wmma241.z)))))+(cast251*tg_bitcast<float>((unsigned int)(val1260))*((((float)(((signed char)(((val1264>>16u)&255u)))))*((float)(wmma242.z)))+(((float)(((signed char)(((val1264>>24u)&255u)))))*((float)(wmma243.z)))))+(cast251*tg_bitcast<float>((unsigned int)(val1261))*((((float)(((signed char)(((val1265>>0u)&255u)))))*((float)(wmma244.z)))+(((float)(((signed char)(((val1265>>8u)&255u)))))*((float)(wmma245.z)))))+(cast251*tg_bitcast<float>((unsigned int)(val1262))*((((float)(((signed char)(((val1265>>16u)&255u)))))*((float)(wmma246.z)))+(((float)(((signed char)(((val1265>>24u)&255u)))))*((float)(wmma247.z))))));
    unsigned int val1266 = (*(buf0+alu261));
    unsigned int val1267 = (*(buf0+alu262));
    unsigned int val1268 = (*(buf0+alu263));
    unsigned int val1269 = (*(buf0+alu264));
    unsigned int val1270 = (*(buf0+alu185));
    unsigned int val1271 = (*(buf0+alu188));
    unsigned int val1272 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+1137)) = buf60;
    }
    float cast252 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1270&65535u)))))));
    buf60 = (buf60+(cast252*tg_bitcast<float>((unsigned int)(val1266))*((((float)(((signed char)(((val1271>>0u)&255u)))))*((float)(wmma240.w)))+(((float)(((signed char)(((val1271>>8u)&255u)))))*((float)(wmma241.w)))))+(cast252*tg_bitcast<float>((unsigned int)(val1267))*((((float)(((signed char)(((val1271>>16u)&255u)))))*((float)(wmma242.w)))+(((float)(((signed char)(((val1271>>24u)&255u)))))*((float)(wmma243.w)))))+(cast252*tg_bitcast<float>((unsigned int)(val1268))*((((float)(((signed char)(((val1272>>0u)&255u)))))*((float)(wmma244.w)))+(((float)(((signed char)(((val1272>>8u)&255u)))))*((float)(wmma245.w)))))+(cast252*tg_bitcast<float>((unsigned int)(val1269))*((((float)(((signed char)(((val1272>>16u)&255u)))))*((float)(wmma246.w)))+(((float)(((signed char)(((val1272>>24u)&255u)))))*((float)(wmma247.w))))));
    unsigned int val1273 = (*(buf0+alu257));
    unsigned int val1274 = (*(buf0+alu258));
    unsigned int val1275 = (*(buf0+alu259));
    unsigned int val1276 = (*(buf0+alu260));
    unsigned int val1277 = (*(buf0+alu190));
    unsigned int val1278 = (*(buf0+alu193));
    unsigned int val1279 = (*(buf0+alu194));
    if (0) {
      *(data0_5570560+(alu110+2160)) = buf61;
    }
    int4 wmma248 = __WMMA_8_16_16_signed_char_int(alu952, cast241, cast0);
    int4 wmma249 = __WMMA_8_16_16_signed_char_int(alu953, cast242, cast0);
    int4 wmma250 = __WMMA_8_16_16_signed_char_int(alu954, cast243, cast0);
    int4 wmma251 = __WMMA_8_16_16_signed_char_int(alu955, cast244, cast0);
    int4 wmma252 = __WMMA_8_16_16_signed_char_int(alu956, cast245, cast0);
    int4 wmma253 = __WMMA_8_16_16_signed_char_int(alu957, cast246, cast0);
    int4 wmma254 = __WMMA_8_16_16_signed_char_int(alu958, cast247, cast0);
    int4 wmma255 = __WMMA_8_16_16_signed_char_int(alu959, cast248, cast0);
    float cast253 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1277&65535u)))))));
    buf61 = (buf61+(cast253*tg_bitcast<float>((unsigned int)(val1273))*((((float)(((signed char)(((val1278>>0u)&255u)))))*((float)(wmma248.x)))+(((float)(((signed char)(((val1278>>8u)&255u)))))*((float)(wmma249.x)))))+(cast253*tg_bitcast<float>((unsigned int)(val1274))*((((float)(((signed char)(((val1278>>16u)&255u)))))*((float)(wmma250.x)))+(((float)(((signed char)(((val1278>>24u)&255u)))))*((float)(wmma251.x)))))+(cast253*tg_bitcast<float>((unsigned int)(val1275))*((((float)(((signed char)(((val1279>>0u)&255u)))))*((float)(wmma252.x)))+(((float)(((signed char)(((val1279>>8u)&255u)))))*((float)(wmma253.x)))))+(cast253*tg_bitcast<float>((unsigned int)(val1276))*((((float)(((signed char)(((val1279>>16u)&255u)))))*((float)(wmma254.x)))+(((float)(((signed char)(((val1279>>24u)&255u)))))*((float)(wmma255.x))))));
    unsigned int val1280 = (*(buf0+alu261));
    unsigned int val1281 = (*(buf0+alu262));
    unsigned int val1282 = (*(buf0+alu263));
    unsigned int val1283 = (*(buf0+alu264));
    unsigned int val1284 = (*(buf0+alu190));
    unsigned int val1285 = (*(buf0+alu193));
    unsigned int val1286 = (*(buf0+alu194));
    if (0) {
      *(data0_5570560+(alu110+2161)) = buf62;
    }
    float cast254 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1284&65535u)))))));
    buf62 = (buf62+(cast254*tg_bitcast<float>((unsigned int)(val1280))*((((float)(((signed char)(((val1285>>0u)&255u)))))*((float)(wmma248.y)))+(((float)(((signed char)(((val1285>>8u)&255u)))))*((float)(wmma249.y)))))+(cast254*tg_bitcast<float>((unsigned int)(val1281))*((((float)(((signed char)(((val1285>>16u)&255u)))))*((float)(wmma250.y)))+(((float)(((signed char)(((val1285>>24u)&255u)))))*((float)(wmma251.y)))))+(cast254*tg_bitcast<float>((unsigned int)(val1282))*((((float)(((signed char)(((val1286>>0u)&255u)))))*((float)(wmma252.y)))+(((float)(((signed char)(((val1286>>8u)&255u)))))*((float)(wmma253.y)))))+(cast254*tg_bitcast<float>((unsigned int)(val1283))*((((float)(((signed char)(((val1286>>16u)&255u)))))*((float)(wmma254.y)))+(((float)(((signed char)(((val1286>>24u)&255u)))))*((float)(wmma255.y))))));
    unsigned int val1287 = (*(buf0+alu257));
    unsigned int val1288 = (*(buf0+alu258));
    unsigned int val1289 = (*(buf0+alu259));
    unsigned int val1290 = (*(buf0+alu260));
    unsigned int val1291 = (*(buf0+alu195));
    unsigned int val1292 = (*(buf0+alu198));
    unsigned int val1293 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+3184)) = buf63;
    }
    float cast255 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1291&65535u)))))));
    buf63 = (buf63+(cast255*tg_bitcast<float>((unsigned int)(val1287))*((((float)(((signed char)(((val1292>>0u)&255u)))))*((float)(wmma248.z)))+(((float)(((signed char)(((val1292>>8u)&255u)))))*((float)(wmma249.z)))))+(cast255*tg_bitcast<float>((unsigned int)(val1288))*((((float)(((signed char)(((val1292>>16u)&255u)))))*((float)(wmma250.z)))+(((float)(((signed char)(((val1292>>24u)&255u)))))*((float)(wmma251.z)))))+(cast255*tg_bitcast<float>((unsigned int)(val1289))*((((float)(((signed char)(((val1293>>0u)&255u)))))*((float)(wmma252.z)))+(((float)(((signed char)(((val1293>>8u)&255u)))))*((float)(wmma253.z)))))+(cast255*tg_bitcast<float>((unsigned int)(val1290))*((((float)(((signed char)(((val1293>>16u)&255u)))))*((float)(wmma254.z)))+(((float)(((signed char)(((val1293>>24u)&255u)))))*((float)(wmma255.z))))));
    unsigned int val1294 = (*(buf0+alu261));
    unsigned int val1295 = (*(buf0+alu262));
    unsigned int val1296 = (*(buf0+alu263));
    unsigned int val1297 = (*(buf0+alu264));
    unsigned int val1298 = (*(buf0+alu195));
    unsigned int val1299 = (*(buf0+alu198));
    unsigned int val1300 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+3185)) = buf64;
    }
    float cast256 = ((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val1298&65535u)))))));
    buf64 = (buf64+(cast256*tg_bitcast<float>((unsigned int)(val1294))*((((float)(((signed char)(((val1299>>0u)&255u)))))*((float)(wmma248.w)))+(((float)(((signed char)(((val1299>>8u)&255u)))))*((float)(wmma249.w)))))+(cast256*tg_bitcast<float>((unsigned int)(val1295))*((((float)(((signed char)(((val1299>>16u)&255u)))))*((float)(wmma250.w)))+(((float)(((signed char)(((val1299>>24u)&255u)))))*((float)(wmma251.w)))))+(cast256*tg_bitcast<float>((unsigned int)(val1296))*((((float)(((signed char)(((val1300>>0u)&255u)))))*((float)(wmma252.w)))+(((float)(((signed char)(((val1300>>8u)&255u)))))*((float)(wmma253.w)))))+(cast256*tg_bitcast<float>((unsigned int)(val1297))*((((float)(((signed char)(((val1300>>16u)&255u)))))*((float)(wmma254.w)))+(((float)(((signed char)(((val1300>>24u)&255u)))))*((float)(wmma255.w))))));
  }
  int alu1198 = (gidx0<<1);
  int alu1199 = ((alu267+-170)/8160);
  int alu1200 = (alu266/8160);
  bool alu1201 = (alu1200!=alu1199);
  int alu1202 = ((alu1198+((int)(alu1201)))<<14);
  int alu1203 = (alu106+alu1202);
  int alu1204 = (alu1203+alu109);
  int alu1205 = (alu109+alu1203);
  int alu1206 = (alu1202+alu106);
  int alu1207 = (alu109+alu1206);
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
  *(data0_5570560+(alu1206+alu109)) = buf1;
}