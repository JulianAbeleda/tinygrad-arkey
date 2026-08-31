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
extern "C" __global__ void __launch_bounds__(256) nv_generated_q6k_streamk_owner_partials(float* data0_5570560, int* data1_340, unsigned short* data2_20643840, unsigned int* data3_1774080) {
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
    int alu300 = ((alu272/1536)<<7);
    int alu301 = (((((alu274<<1)+1)<<9)+alu300+alu265)*36);
    unsigned int val13 = (*(data3_1774080+(alu301+1)));
    unsigned int val14 = (*(data3_1774080+(alu301+2)));
    unsigned int val15 = (*(data3_1774080+(alu301+3)));
    unsigned int val16 = (*(data3_1774080+(alu301+4)));
    unsigned int val17 = (*(data3_1774080+(alu301+5)));
    unsigned int val18 = (*(data3_1774080+(alu301+6)));
    unsigned int val19 = (*(data3_1774080+(alu301+7)));
    unsigned int val20 = (*(data3_1774080+(alu301+8)));
    unsigned int val21 = (*(data3_1774080+(alu301+9)));
    unsigned int val22 = (*(data3_1774080+(alu301+10)));
    unsigned int val23 = (*(data3_1774080+(alu301+11)));
    unsigned int val24 = (*(data3_1774080+(alu301+12)));
    unsigned int val25 = (*(data3_1774080+(alu301+13)));
    unsigned int val26 = (*(data3_1774080+(alu301+14)));
    unsigned int val27 = (*(data3_1774080+(alu301+15)));
    unsigned int val28 = (*(data3_1774080+(alu301+16)));
    unsigned int val29 = (*(data3_1774080+(alu301+17)));
    unsigned int val30 = (*(data3_1774080+(alu301+18)));
    unsigned int val31 = (*(data3_1774080+(alu301+19)));
    unsigned int val32 = (*(data3_1774080+(alu301+20)));
    unsigned int val33 = (*(data3_1774080+(alu301+21)));
    unsigned int val34 = (*(data3_1774080+(alu301+22)));
    unsigned int val35 = (*(data3_1774080+(alu301+23)));
    unsigned int val36 = (*(data3_1774080+(alu301+24)));
    unsigned int val37 = (*(data3_1774080+(alu301+25)));
    unsigned int val38 = (*(data3_1774080+(alu301+26)));
    unsigned int val39 = (*(data3_1774080+(alu301+27)));
    unsigned int val40 = (*(data3_1774080+(alu301+28)));
    unsigned int val41 = (*(data3_1774080+(alu301+29)));
    unsigned int val42 = (*(data3_1774080+(alu301+30)));
    unsigned int val43 = (*(data3_1774080+(alu301+31)));
    unsigned int val44 = (*(data3_1774080+(alu301+32)));
    unsigned int val45 = (*(data3_1774080+(alu301+33)));
    unsigned int val46 = (*(data3_1774080+(alu301+34)));
    unsigned int val47 = (*(data3_1774080+(alu301+35)));
    int alu302 = (((alu274<<10)+alu300+alu265)*36);
    unsigned int val48 = (*(data3_1774080+(alu302+1)));
    unsigned int val49 = (*(data3_1774080+(alu302+2)));
    unsigned int val50 = (*(data3_1774080+(alu302+3)));
    unsigned int val51 = (*(data3_1774080+(alu302+4)));
    unsigned int val52 = (*(data3_1774080+(alu302+5)));
    unsigned int val53 = (*(data3_1774080+(alu302+6)));
    unsigned int val54 = (*(data3_1774080+(alu302+7)));
    unsigned int val55 = (*(data3_1774080+(alu302+8)));
    unsigned int val56 = (*(data3_1774080+(alu302+9)));
    unsigned int val57 = (*(data3_1774080+(alu302+10)));
    unsigned int val58 = (*(data3_1774080+(alu302+11)));
    unsigned int val59 = (*(data3_1774080+(alu302+12)));
    unsigned int val60 = (*(data3_1774080+(alu302+13)));
    unsigned int val61 = (*(data3_1774080+(alu302+14)));
    unsigned int val62 = (*(data3_1774080+(alu302+15)));
    unsigned int val63 = (*(data3_1774080+(alu302+16)));
    unsigned int val64 = (*(data3_1774080+(alu302+17)));
    unsigned int val65 = (*(data3_1774080+(alu302+18)));
    unsigned int val66 = (*(data3_1774080+(alu302+19)));
    unsigned int val67 = (*(data3_1774080+(alu302+20)));
    unsigned int val68 = (*(data3_1774080+(alu302+21)));
    unsigned int val69 = (*(data3_1774080+(alu302+22)));
    unsigned int val70 = (*(data3_1774080+(alu302+23)));
    unsigned int val71 = (*(data3_1774080+(alu302+24)));
    unsigned int val72 = (*(data3_1774080+(alu302+25)));
    unsigned int val73 = (*(data3_1774080+(alu302+26)));
    unsigned int val74 = (*(data3_1774080+(alu302+27)));
    unsigned int val75 = (*(data3_1774080+(alu302+28)));
    unsigned int val76 = (*(data3_1774080+(alu302+29)));
    unsigned int val77 = (*(data3_1774080+(alu302+30)));
    unsigned int val78 = (*(data3_1774080+(alu302+31)));
    unsigned int val79 = (*(data3_1774080+(alu302+32)));
    unsigned int val80 = (*(data3_1774080+(alu302+33)));
    unsigned int val81 = (*(data3_1774080+(alu302+34)));
    unsigned int val82 = (*(data3_1774080+(alu302+35)));
    unsigned int val83 = (*(data3_1774080+alu301));
    unsigned int val84 = (*(data3_1774080+alu302));
    if (alu271) {
      *(buf0+alu68) = val84;
    }
    if (alu271) {
      *(buf0+alu69) = val48;
    }
    if (alu271) {
      *(buf0+alu70) = val49;
    }
    if (alu271) {
      *(buf0+alu71) = val50;
    }
    if (alu271) {
      *(buf0+alu72) = val51;
    }
    if (alu271) {
      *(buf0+alu73) = val52;
    }
    if (alu271) {
      *(buf0+alu74) = val53;
    }
    if (alu271) {
      *(buf0+alu75) = val54;
    }
    if (alu271) {
      *(buf0+alu76) = val55;
    }
    if (alu271) {
      *(buf0+alu77) = val56;
    }
    if (alu271) {
      *(buf0+alu78) = val57;
    }
    if (alu271) {
      *(buf0+alu79) = val58;
    }
    if (alu271) {
      *(buf0+alu80) = val59;
    }
    if (alu271) {
      *(buf0+alu81) = val60;
    }
    if (alu271) {
      *(buf0+alu82) = val61;
    }
    if (alu271) {
      *(buf0+alu83) = val62;
    }
    if (alu271) {
      *(buf0+alu84) = val63;
    }
    if (alu271) {
      *(buf0+alu85) = val64;
    }
    if (alu271) {
      *(buf0+alu86) = val65;
    }
    if (alu271) {
      *(buf0+alu87) = val66;
    }
    if (alu271) {
      *(buf0+alu88) = val67;
    }
    if (alu271) {
      *(buf0+alu89) = val68;
    }
    if (alu271) {
      *(buf0+alu90) = val69;
    }
    if (alu271) {
      *(buf0+alu91) = val70;
    }
    if (alu271) {
      *(buf0+alu92) = val71;
    }
    if (alu271) {
      *(buf0+alu93) = val72;
    }
    if (alu271) {
      *(buf0+alu94) = val73;
    }
    if (alu271) {
      *(buf0+alu95) = val74;
    }
    if (alu271) {
      *(buf0+alu96) = val75;
    }
    if (alu271) {
      *(buf0+alu97) = val76;
    }
    if (alu271) {
      *(buf0+alu98) = val77;
    }
    if (alu271) {
      *(buf0+alu99) = val78;
    }
    if (alu271) {
      *(buf0+alu100) = val79;
    }
    if (alu271) {
      *(buf0+alu101) = val80;
    }
    if (alu271) {
      *(buf0+alu102) = val81;
    }
    if (alu271) {
      *(buf0+alu103) = val82;
    }
    __syncthreads();
    unsigned int val85 = (*(buf0+alu115));
    unsigned int val86 = (*(buf0+alu116));
    unsigned int val87 = (*(buf0+alu117));
    unsigned int val88 = (*(buf0+alu118));
    unsigned int val89 = (*(buf0+alu119));
    unsigned int val90 = (*(buf0+alu120));
    unsigned int val91 = (*(buf0+alu121));
    unsigned int val92 = (*(buf0+alu122));
    unsigned int val93 = (*(buf0+alu201));
    unsigned int val94 = (*(buf0+alu202));
    unsigned int val95 = (*(buf0+alu203));
    unsigned int val96 = (*(buf0+alu204));
    unsigned int val97 = (*(buf0+alu205));
    unsigned int val98 = (*(buf0+alu206));
    unsigned int val99 = (*(buf0+alu207));
    unsigned int val100 = (*(buf0+alu208));
    __syncthreads();
    unsigned int val101 = (*(buf0+alu180));
    unsigned int val102 = (*(buf0+alu181));
    unsigned int val103 = (*(buf0+alu182));
    unsigned int val104 = (*(buf0+alu186));
    unsigned int val105 = (*(buf0+alu187));
    int alu413 = (alu272+-1);
    bool alu414 = ((0<Ridx50)&(alu273!=((alu413/48)-((int)((((alu413%48)!=0)&((alu413<0)!=0)))))));
    if (alu414) {
      *(data0_5570560+alu110) = buf1;
    }
    char4 cast1 = make_char4(((signed char)(((val85>>0u)&255u))),((signed char)(((val85>>8u)&255u))),((signed char)(((val85>>16u)&255u))),((signed char)(((val85>>24u)&255u))));
    char4 cast2 = make_char4(((signed char)(((val86>>0u)&255u))),((signed char)(((val86>>8u)&255u))),((signed char)(((val86>>16u)&255u))),((signed char)(((val86>>24u)&255u))));
    char4 cast3 = make_char4(((signed char)(((val87>>0u)&255u))),((signed char)(((val87>>8u)&255u))),((signed char)(((val87>>16u)&255u))),((signed char)(((val87>>24u)&255u))));
    char4 cast4 = make_char4(((signed char)(((val88>>0u)&255u))),((signed char)(((val88>>8u)&255u))),((signed char)(((val88>>16u)&255u))),((signed char)(((val88>>24u)&255u))));
    char4 cast5 = make_char4(((signed char)(((val89>>0u)&255u))),((signed char)(((val89>>8u)&255u))),((signed char)(((val89>>16u)&255u))),((signed char)(((val89>>24u)&255u))));
    char4 cast6 = make_char4(((signed char)(((val90>>0u)&255u))),((signed char)(((val90>>8u)&255u))),((signed char)(((val90>>16u)&255u))),((signed char)(((val90>>24u)&255u))));
    char4 cast7 = make_char4(((signed char)(((val91>>0u)&255u))),((signed char)(((val91>>8u)&255u))),((signed char)(((val91>>16u)&255u))),((signed char)(((val91>>24u)&255u))));
    char4 cast8 = make_char4(((signed char)(((val92>>0u)&255u))),((signed char)(((val92>>8u)&255u))),((signed char)(((val92>>16u)&255u))),((signed char)(((val92>>24u)&255u))));
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
    int cast9 = ((int)(((signed char)(((val102>>0u)&255u)))));
    int cast10 = ((int)(((signed char)(((val102>>8u)&255u)))));
    int cast11 = ((int)(((signed char)(((val102>>16u)&255u)))));
    int cast12 = ((int)(((signed char)(((val102>>24u)&255u)))));
    int cast13 = ((int)(((signed char)(((val103>>0u)&255u)))));
    int cast14 = ((int)(((signed char)(((val103>>8u)&255u)))));
    int cast15 = ((int)(((signed char)(((val103>>16u)&255u)))));
    int cast16 = ((int)(((signed char)(((val103>>24u)&255u)))));
    float cast17 = tg_bitcast<float>((unsigned int)(val93));
    float cast18 = tg_bitcast<float>((unsigned int)(val94));
    float cast19 = tg_bitcast<float>((unsigned int)(val95));
    float cast20 = tg_bitcast<float>((unsigned int)(val96));
    float alu426 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val101&65535u)))))))*((cast17*((float)(((cast9*wmma7.x)+(cast10*wmma0.x)))))+(cast18*((float)(((cast11*wmma1.x)+(cast12*wmma2.x)))))+(cast19*((float)(((cast13*wmma3.x)+(cast14*wmma4.x)))))+(cast20*((float)(((cast15*wmma5.x)+(cast16*wmma6.x)))))));
    float alu427 = (alu414?alu426:(buf1+alu426));
    buf1 = alu427;
    unsigned int val106 = (*(buf0+alu180));
    if (alu414) {
      *(data0_5570560+(alu110+1)) = buf2;
    }
    float cast21 = tg_bitcast<float>((unsigned int)(val97));
    float cast22 = tg_bitcast<float>((unsigned int)(val98));
    float cast23 = tg_bitcast<float>((unsigned int)(val99));
    float cast24 = tg_bitcast<float>((unsigned int)(val100));
    float alu432 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val106&65535u)))))))*((cast21*((float)(((cast9*wmma7.y)+(cast10*wmma0.y)))))+(cast22*((float)(((cast11*wmma1.y)+(cast12*wmma2.y)))))+(cast23*((float)(((cast13*wmma3.y)+(cast14*wmma4.y)))))+(cast24*((float)(((cast15*wmma5.y)+(cast16*wmma6.y)))))));
    float alu433 = (alu414?alu432:(buf2+alu432));
    buf2 = alu433;
    unsigned int val107 = (*(buf0+alu185));
    if (alu414) {
      *(data0_5570560+(alu110+1024)) = buf3;
    }
    int cast25 = ((int)(((signed char)(((val104>>0u)&255u)))));
    int cast26 = ((int)(((signed char)(((val104>>8u)&255u)))));
    int cast27 = ((int)(((signed char)(((val104>>16u)&255u)))));
    int cast28 = ((int)(((signed char)(((val104>>24u)&255u)))));
    int cast29 = ((int)(((signed char)(((val105>>0u)&255u)))));
    int cast30 = ((int)(((signed char)(((val105>>8u)&255u)))));
    int cast31 = ((int)(((signed char)(((val105>>16u)&255u)))));
    int cast32 = ((int)(((signed char)(((val105>>24u)&255u)))));
    float alu438 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val107&65535u)))))))*((cast17*((float)(((cast25*wmma7.z)+(cast26*wmma0.z)))))+(cast18*((float)(((cast27*wmma1.z)+(cast28*wmma2.z)))))+(cast19*((float)(((cast29*wmma3.z)+(cast30*wmma4.z)))))+(cast20*((float)(((cast31*wmma5.z)+(cast32*wmma6.z)))))));
    float alu439 = (alu414?alu438:(buf3+alu438));
    buf3 = alu439;
    unsigned int val108 = (*(buf0+alu185));
    if (alu414) {
      *(data0_5570560+(alu110+1025)) = buf4;
    }
    float alu444 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val108&65535u)))))))*((cast21*((float)(((cast25*wmma7.w)+(cast26*wmma0.w)))))+(cast22*((float)(((cast27*wmma1.w)+(cast28*wmma2.w)))))+(cast23*((float)(((cast29*wmma3.w)+(cast30*wmma4.w)))))+(cast24*((float)(((cast31*wmma5.w)+(cast32*wmma6.w)))))));
    float alu445 = (alu414?alu444:(buf4+alu444));
    buf4 = alu445;
    unsigned int val109 = (*(buf0+alu201));
    unsigned int val110 = (*(buf0+alu202));
    unsigned int val111 = (*(buf0+alu203));
    unsigned int val112 = (*(buf0+alu204));
    unsigned int val113 = (*(buf0+alu205));
    unsigned int val114 = (*(buf0+alu206));
    unsigned int val115 = (*(buf0+alu207));
    unsigned int val116 = (*(buf0+alu208));
    unsigned int val117 = (*(buf0+alu190));
    unsigned int val118 = (*(buf0+alu191));
    unsigned int val119 = (*(buf0+alu192));
    unsigned int val120 = (*(buf0+alu196));
    unsigned int val121 = (*(buf0+alu197));
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
    int cast33 = ((int)(((signed char)(((val118>>0u)&255u)))));
    int cast34 = ((int)(((signed char)(((val118>>8u)&255u)))));
    int cast35 = ((int)(((signed char)(((val118>>16u)&255u)))));
    int cast36 = ((int)(((signed char)(((val118>>24u)&255u)))));
    int cast37 = ((int)(((signed char)(((val119>>0u)&255u)))));
    int cast38 = ((int)(((signed char)(((val119>>8u)&255u)))));
    int cast39 = ((int)(((signed char)(((val119>>16u)&255u)))));
    int cast40 = ((int)(((signed char)(((val119>>24u)&255u)))));
    float cast41 = tg_bitcast<float>((unsigned int)(val109));
    float cast42 = tg_bitcast<float>((unsigned int)(val110));
    float cast43 = tg_bitcast<float>((unsigned int)(val111));
    float cast44 = tg_bitcast<float>((unsigned int)(val112));
    float alu458 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val117&65535u)))))))*((cast41*((float)(((cast33*wmma8.x)+(cast34*wmma9.x)))))+(cast42*((float)(((cast35*wmma10.x)+(cast36*wmma11.x)))))+(cast43*((float)(((cast37*wmma12.x)+(cast38*wmma13.x)))))+(cast44*((float)(((cast39*wmma14.x)+(cast40*wmma15.x)))))));
    float alu459 = (alu414?alu458:(buf5+alu458));
    buf5 = alu459;
    unsigned int val122 = (*(buf0+alu190));
    if (alu414) {
      *(data0_5570560+(alu110+2049)) = buf6;
    }
    float cast45 = tg_bitcast<float>((unsigned int)(val113));
    float cast46 = tg_bitcast<float>((unsigned int)(val114));
    float cast47 = tg_bitcast<float>((unsigned int)(val115));
    float cast48 = tg_bitcast<float>((unsigned int)(val116));
    float alu464 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val122&65535u)))))))*((cast45*((float)(((cast33*wmma8.y)+(cast34*wmma9.y)))))+(cast46*((float)(((cast35*wmma10.y)+(cast36*wmma11.y)))))+(cast47*((float)(((cast37*wmma12.y)+(cast38*wmma13.y)))))+(cast48*((float)(((cast39*wmma14.y)+(cast40*wmma15.y)))))));
    float alu465 = (alu414?alu464:(buf6+alu464));
    buf6 = alu465;
    unsigned int val123 = (*(buf0+alu195));
    if (alu414) {
      *(data0_5570560+(alu110+3072)) = buf7;
    }
    int cast49 = ((int)(((signed char)(((val120>>0u)&255u)))));
    int cast50 = ((int)(((signed char)(((val120>>8u)&255u)))));
    int cast51 = ((int)(((signed char)(((val120>>16u)&255u)))));
    int cast52 = ((int)(((signed char)(((val120>>24u)&255u)))));
    int cast53 = ((int)(((signed char)(((val121>>0u)&255u)))));
    int cast54 = ((int)(((signed char)(((val121>>8u)&255u)))));
    int cast55 = ((int)(((signed char)(((val121>>16u)&255u)))));
    int cast56 = ((int)(((signed char)(((val121>>24u)&255u)))));
    float alu470 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val123&65535u)))))))*((cast41*((float)(((cast49*wmma8.z)+(cast50*wmma9.z)))))+(cast42*((float)(((cast51*wmma10.z)+(cast52*wmma11.z)))))+(cast43*((float)(((cast53*wmma12.z)+(cast54*wmma13.z)))))+(cast44*((float)(((cast55*wmma14.z)+(cast56*wmma15.z)))))));
    float alu471 = (alu414?alu470:(buf7+alu470));
    buf7 = alu471;
    unsigned int val124 = (*(buf0+alu195));
    if (alu414) {
      *(data0_5570560+(alu110+3073)) = buf8;
    }
    float alu476 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val124&65535u)))))))*((cast45*((float)(((cast49*wmma8.w)+(cast50*wmma9.w)))))+(cast46*((float)(((cast51*wmma10.w)+(cast52*wmma11.w)))))+(cast47*((float)(((cast53*wmma12.w)+(cast54*wmma13.w)))))+(cast48*((float)(((cast55*wmma14.w)+(cast56*wmma15.w)))))));
    float alu477 = (alu414?alu476:(buf8+alu476));
    buf8 = alu477;
    unsigned int val125 = (*(buf0+alu123));
    unsigned int val126 = (*(buf0+alu124));
    unsigned int val127 = (*(buf0+alu125));
    unsigned int val128 = (*(buf0+alu126));
    unsigned int val129 = (*(buf0+alu127));
    unsigned int val130 = (*(buf0+alu128));
    unsigned int val131 = (*(buf0+alu129));
    unsigned int val132 = (*(buf0+alu130));
    unsigned int val133 = (*(buf0+alu209));
    unsigned int val134 = (*(buf0+alu210));
    unsigned int val135 = (*(buf0+alu211));
    unsigned int val136 = (*(buf0+alu212));
    unsigned int val137 = (*(buf0+alu213));
    unsigned int val138 = (*(buf0+alu214));
    unsigned int val139 = (*(buf0+alu215));
    unsigned int val140 = (*(buf0+alu216));
    unsigned int val141 = (*(buf0+alu180));
    unsigned int val142 = (*(buf0+alu181));
    unsigned int val143 = (*(buf0+alu182));
    unsigned int val144 = (*(buf0+alu186));
    unsigned int val145 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+16)) = buf9;
    }
    char4 cast57 = make_char4(((signed char)(((val125>>0u)&255u))),((signed char)(((val125>>8u)&255u))),((signed char)(((val125>>16u)&255u))),((signed char)(((val125>>24u)&255u))));
    char4 cast58 = make_char4(((signed char)(((val126>>0u)&255u))),((signed char)(((val126>>8u)&255u))),((signed char)(((val126>>16u)&255u))),((signed char)(((val126>>24u)&255u))));
    char4 cast59 = make_char4(((signed char)(((val127>>0u)&255u))),((signed char)(((val127>>8u)&255u))),((signed char)(((val127>>16u)&255u))),((signed char)(((val127>>24u)&255u))));
    char4 cast60 = make_char4(((signed char)(((val128>>0u)&255u))),((signed char)(((val128>>8u)&255u))),((signed char)(((val128>>16u)&255u))),((signed char)(((val128>>24u)&255u))));
    char4 cast61 = make_char4(((signed char)(((val129>>0u)&255u))),((signed char)(((val129>>8u)&255u))),((signed char)(((val129>>16u)&255u))),((signed char)(((val129>>24u)&255u))));
    char4 cast62 = make_char4(((signed char)(((val130>>0u)&255u))),((signed char)(((val130>>8u)&255u))),((signed char)(((val130>>16u)&255u))),((signed char)(((val130>>24u)&255u))));
    char4 cast63 = make_char4(((signed char)(((val131>>0u)&255u))),((signed char)(((val131>>8u)&255u))),((signed char)(((val131>>16u)&255u))),((signed char)(((val131>>24u)&255u))));
    char4 cast64 = make_char4(((signed char)(((val132>>0u)&255u))),((signed char)(((val132>>8u)&255u))),((signed char)(((val132>>16u)&255u))),((signed char)(((val132>>24u)&255u))));
    int4 wmma16 = __WMMA_8_16_16_signed_char_int(alu418, cast58, cast0);
    int4 wmma17 = __WMMA_8_16_16_signed_char_int(alu419, cast59, cast0);
    int4 wmma18 = __WMMA_8_16_16_signed_char_int(alu420, cast60, cast0);
    int4 wmma19 = __WMMA_8_16_16_signed_char_int(alu421, cast61, cast0);
    int4 wmma20 = __WMMA_8_16_16_signed_char_int(alu422, cast62, cast0);
    int4 wmma21 = __WMMA_8_16_16_signed_char_int(alu423, cast63, cast0);
    int4 wmma22 = __WMMA_8_16_16_signed_char_int(alu424, cast64, cast0);
    int4 wmma23 = __WMMA_8_16_16_signed_char_int(alu425, cast57, cast0);
    int cast65 = ((int)(((signed char)(((val142>>0u)&255u)))));
    int cast66 = ((int)(((signed char)(((val142>>8u)&255u)))));
    int cast67 = ((int)(((signed char)(((val142>>16u)&255u)))));
    int cast68 = ((int)(((signed char)(((val142>>24u)&255u)))));
    int cast69 = ((int)(((signed char)(((val143>>0u)&255u)))));
    int cast70 = ((int)(((signed char)(((val143>>8u)&255u)))));
    int cast71 = ((int)(((signed char)(((val143>>16u)&255u)))));
    int cast72 = ((int)(((signed char)(((val143>>24u)&255u)))));
    float cast73 = tg_bitcast<float>((unsigned int)(val133));
    float cast74 = tg_bitcast<float>((unsigned int)(val134));
    float cast75 = tg_bitcast<float>((unsigned int)(val135));
    float cast76 = tg_bitcast<float>((unsigned int)(val136));
    float alu482 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val141&65535u)))))))*((cast73*((float)(((cast65*wmma23.x)+(cast66*wmma16.x)))))+(cast74*((float)(((cast67*wmma17.x)+(cast68*wmma18.x)))))+(cast75*((float)(((cast69*wmma19.x)+(cast70*wmma20.x)))))+(cast76*((float)(((cast71*wmma21.x)+(cast72*wmma22.x)))))));
    float alu483 = (alu414?alu482:(buf9+alu482));
    buf9 = alu483;
    unsigned int val146 = (*(buf0+alu180));
    if (alu414) {
      *(data0_5570560+(alu110+17)) = buf10;
    }
    float cast77 = tg_bitcast<float>((unsigned int)(val137));
    float cast78 = tg_bitcast<float>((unsigned int)(val138));
    float cast79 = tg_bitcast<float>((unsigned int)(val139));
    float cast80 = tg_bitcast<float>((unsigned int)(val140));
    float alu488 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val146&65535u)))))))*((cast77*((float)(((cast65*wmma23.y)+(cast66*wmma16.y)))))+(cast78*((float)(((cast67*wmma17.y)+(cast68*wmma18.y)))))+(cast79*((float)(((cast69*wmma19.y)+(cast70*wmma20.y)))))+(cast80*((float)(((cast71*wmma21.y)+(cast72*wmma22.y)))))));
    float alu489 = (alu414?alu488:(buf10+alu488));
    buf10 = alu489;
    unsigned int val147 = (*(buf0+alu185));
    if (alu414) {
      *(data0_5570560+(alu110+1040)) = buf11;
    }
    int cast81 = ((int)(((signed char)(((val144>>0u)&255u)))));
    int cast82 = ((int)(((signed char)(((val144>>8u)&255u)))));
    int cast83 = ((int)(((signed char)(((val144>>16u)&255u)))));
    int cast84 = ((int)(((signed char)(((val144>>24u)&255u)))));
    int cast85 = ((int)(((signed char)(((val145>>0u)&255u)))));
    int cast86 = ((int)(((signed char)(((val145>>8u)&255u)))));
    int cast87 = ((int)(((signed char)(((val145>>16u)&255u)))));
    int cast88 = ((int)(((signed char)(((val145>>24u)&255u)))));
    float alu494 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val147&65535u)))))))*((cast73*((float)(((cast81*wmma23.z)+(cast82*wmma16.z)))))+(cast74*((float)(((cast83*wmma17.z)+(cast84*wmma18.z)))))+(cast75*((float)(((cast85*wmma19.z)+(cast86*wmma20.z)))))+(cast76*((float)(((cast87*wmma21.z)+(cast88*wmma22.z)))))));
    float alu495 = (alu414?alu494:(buf11+alu494));
    buf11 = alu495;
    unsigned int val148 = (*(buf0+alu185));
    if (alu414) {
      *(data0_5570560+(alu110+1041)) = buf12;
    }
    float alu500 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val148&65535u)))))))*((cast77*((float)(((cast81*wmma23.w)+(cast82*wmma16.w)))))+(cast78*((float)(((cast83*wmma17.w)+(cast84*wmma18.w)))))+(cast79*((float)(((cast85*wmma19.w)+(cast86*wmma20.w)))))+(cast80*((float)(((cast87*wmma21.w)+(cast88*wmma22.w)))))));
    float alu501 = (alu414?alu500:(buf12+alu500));
    buf12 = alu501;
    unsigned int val149 = (*(buf0+alu209));
    unsigned int val150 = (*(buf0+alu210));
    unsigned int val151 = (*(buf0+alu211));
    unsigned int val152 = (*(buf0+alu212));
    unsigned int val153 = (*(buf0+alu213));
    unsigned int val154 = (*(buf0+alu214));
    unsigned int val155 = (*(buf0+alu215));
    unsigned int val156 = (*(buf0+alu216));
    unsigned int val157 = (*(buf0+alu190));
    unsigned int val158 = (*(buf0+alu191));
    unsigned int val159 = (*(buf0+alu192));
    unsigned int val160 = (*(buf0+alu196));
    unsigned int val161 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+2064)) = buf13;
    }
    int4 wmma24 = __WMMA_8_16_16_signed_char_int(alu450, cast57, cast0);
    int4 wmma25 = __WMMA_8_16_16_signed_char_int(alu451, cast58, cast0);
    int4 wmma26 = __WMMA_8_16_16_signed_char_int(alu452, cast59, cast0);
    int4 wmma27 = __WMMA_8_16_16_signed_char_int(alu453, cast60, cast0);
    int4 wmma28 = __WMMA_8_16_16_signed_char_int(alu454, cast61, cast0);
    int4 wmma29 = __WMMA_8_16_16_signed_char_int(alu455, cast62, cast0);
    int4 wmma30 = __WMMA_8_16_16_signed_char_int(alu456, cast63, cast0);
    int4 wmma31 = __WMMA_8_16_16_signed_char_int(alu457, cast64, cast0);
    int cast89 = ((int)(((signed char)(((val158>>0u)&255u)))));
    int cast90 = ((int)(((signed char)(((val158>>8u)&255u)))));
    int cast91 = ((int)(((signed char)(((val158>>16u)&255u)))));
    int cast92 = ((int)(((signed char)(((val158>>24u)&255u)))));
    int cast93 = ((int)(((signed char)(((val159>>0u)&255u)))));
    int cast94 = ((int)(((signed char)(((val159>>8u)&255u)))));
    int cast95 = ((int)(((signed char)(((val159>>16u)&255u)))));
    int cast96 = ((int)(((signed char)(((val159>>24u)&255u)))));
    float cast97 = tg_bitcast<float>((unsigned int)(val149));
    float cast98 = tg_bitcast<float>((unsigned int)(val150));
    float cast99 = tg_bitcast<float>((unsigned int)(val151));
    float cast100 = tg_bitcast<float>((unsigned int)(val152));
    float alu506 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val157&65535u)))))))*((cast97*((float)(((cast89*wmma24.x)+(cast90*wmma25.x)))))+(cast98*((float)(((cast91*wmma26.x)+(cast92*wmma27.x)))))+(cast99*((float)(((cast93*wmma28.x)+(cast94*wmma29.x)))))+(cast100*((float)(((cast95*wmma30.x)+(cast96*wmma31.x)))))));
    float alu507 = (alu414?alu506:(buf13+alu506));
    buf13 = alu507;
    unsigned int val162 = (*(buf0+alu190));
    if (alu414) {
      *(data0_5570560+(alu110+2065)) = buf14;
    }
    float cast101 = tg_bitcast<float>((unsigned int)(val153));
    float cast102 = tg_bitcast<float>((unsigned int)(val154));
    float cast103 = tg_bitcast<float>((unsigned int)(val155));
    float cast104 = tg_bitcast<float>((unsigned int)(val156));
    float alu512 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val162&65535u)))))))*((cast101*((float)(((cast89*wmma24.y)+(cast90*wmma25.y)))))+(cast102*((float)(((cast91*wmma26.y)+(cast92*wmma27.y)))))+(cast103*((float)(((cast93*wmma28.y)+(cast94*wmma29.y)))))+(cast104*((float)(((cast95*wmma30.y)+(cast96*wmma31.y)))))));
    float alu513 = (alu414?alu512:(buf14+alu512));
    buf14 = alu513;
    unsigned int val163 = (*(buf0+alu195));
    if (alu414) {
      *(data0_5570560+(alu110+3088)) = buf15;
    }
    int cast105 = ((int)(((signed char)(((val160>>0u)&255u)))));
    int cast106 = ((int)(((signed char)(((val160>>8u)&255u)))));
    int cast107 = ((int)(((signed char)(((val160>>16u)&255u)))));
    int cast108 = ((int)(((signed char)(((val160>>24u)&255u)))));
    int cast109 = ((int)(((signed char)(((val161>>0u)&255u)))));
    int cast110 = ((int)(((signed char)(((val161>>8u)&255u)))));
    int cast111 = ((int)(((signed char)(((val161>>16u)&255u)))));
    int cast112 = ((int)(((signed char)(((val161>>24u)&255u)))));
    float alu518 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val163&65535u)))))))*((cast97*((float)(((cast105*wmma24.z)+(cast106*wmma25.z)))))+(cast98*((float)(((cast107*wmma26.z)+(cast108*wmma27.z)))))+(cast99*((float)(((cast109*wmma28.z)+(cast110*wmma29.z)))))+(cast100*((float)(((cast111*wmma30.z)+(cast112*wmma31.z)))))));
    float alu519 = (alu414?alu518:(buf15+alu518));
    buf15 = alu519;
    unsigned int val164 = (*(buf0+alu195));
    if (alu414) {
      *(data0_5570560+(alu110+3089)) = buf16;
    }
    float alu524 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val164&65535u)))))))*((cast101*((float)(((cast105*wmma24.w)+(cast106*wmma25.w)))))+(cast102*((float)(((cast107*wmma26.w)+(cast108*wmma27.w)))))+(cast103*((float)(((cast109*wmma28.w)+(cast110*wmma29.w)))))+(cast104*((float)(((cast111*wmma30.w)+(cast112*wmma31.w)))))));
    float alu525 = (alu414?alu524:(buf16+alu524));
    buf16 = alu525;
    unsigned int val165 = (*(buf0+alu131));
    unsigned int val166 = (*(buf0+alu132));
    unsigned int val167 = (*(buf0+alu133));
    unsigned int val168 = (*(buf0+alu134));
    unsigned int val169 = (*(buf0+alu135));
    unsigned int val170 = (*(buf0+alu136));
    unsigned int val171 = (*(buf0+alu137));
    unsigned int val172 = (*(buf0+alu138));
    unsigned int val173 = (*(buf0+alu217));
    unsigned int val174 = (*(buf0+alu218));
    unsigned int val175 = (*(buf0+alu219));
    unsigned int val176 = (*(buf0+alu220));
    unsigned int val177 = (*(buf0+alu221));
    unsigned int val178 = (*(buf0+alu222));
    unsigned int val179 = (*(buf0+alu223));
    unsigned int val180 = (*(buf0+alu224));
    unsigned int val181 = (*(buf0+alu180));
    unsigned int val182 = (*(buf0+alu181));
    unsigned int val183 = (*(buf0+alu182));
    unsigned int val184 = (*(buf0+alu186));
    unsigned int val185 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+32)) = buf17;
    }
    char4 cast113 = make_char4(((signed char)(((val165>>0u)&255u))),((signed char)(((val165>>8u)&255u))),((signed char)(((val165>>16u)&255u))),((signed char)(((val165>>24u)&255u))));
    char4 cast114 = make_char4(((signed char)(((val166>>0u)&255u))),((signed char)(((val166>>8u)&255u))),((signed char)(((val166>>16u)&255u))),((signed char)(((val166>>24u)&255u))));
    char4 cast115 = make_char4(((signed char)(((val167>>0u)&255u))),((signed char)(((val167>>8u)&255u))),((signed char)(((val167>>16u)&255u))),((signed char)(((val167>>24u)&255u))));
    char4 cast116 = make_char4(((signed char)(((val168>>0u)&255u))),((signed char)(((val168>>8u)&255u))),((signed char)(((val168>>16u)&255u))),((signed char)(((val168>>24u)&255u))));
    char4 cast117 = make_char4(((signed char)(((val169>>0u)&255u))),((signed char)(((val169>>8u)&255u))),((signed char)(((val169>>16u)&255u))),((signed char)(((val169>>24u)&255u))));
    char4 cast118 = make_char4(((signed char)(((val170>>0u)&255u))),((signed char)(((val170>>8u)&255u))),((signed char)(((val170>>16u)&255u))),((signed char)(((val170>>24u)&255u))));
    char4 cast119 = make_char4(((signed char)(((val171>>0u)&255u))),((signed char)(((val171>>8u)&255u))),((signed char)(((val171>>16u)&255u))),((signed char)(((val171>>24u)&255u))));
    char4 cast120 = make_char4(((signed char)(((val172>>0u)&255u))),((signed char)(((val172>>8u)&255u))),((signed char)(((val172>>16u)&255u))),((signed char)(((val172>>24u)&255u))));
    int4 wmma32 = __WMMA_8_16_16_signed_char_int(alu418, cast114, cast0);
    int4 wmma33 = __WMMA_8_16_16_signed_char_int(alu419, cast115, cast0);
    int4 wmma34 = __WMMA_8_16_16_signed_char_int(alu420, cast116, cast0);
    int4 wmma35 = __WMMA_8_16_16_signed_char_int(alu421, cast117, cast0);
    int4 wmma36 = __WMMA_8_16_16_signed_char_int(alu422, cast118, cast0);
    int4 wmma37 = __WMMA_8_16_16_signed_char_int(alu423, cast119, cast0);
    int4 wmma38 = __WMMA_8_16_16_signed_char_int(alu424, cast120, cast0);
    int4 wmma39 = __WMMA_8_16_16_signed_char_int(alu425, cast113, cast0);
    int cast121 = ((int)(((signed char)(((val182>>0u)&255u)))));
    int cast122 = ((int)(((signed char)(((val182>>8u)&255u)))));
    int cast123 = ((int)(((signed char)(((val182>>16u)&255u)))));
    int cast124 = ((int)(((signed char)(((val182>>24u)&255u)))));
    int cast125 = ((int)(((signed char)(((val183>>0u)&255u)))));
    int cast126 = ((int)(((signed char)(((val183>>8u)&255u)))));
    int cast127 = ((int)(((signed char)(((val183>>16u)&255u)))));
    int cast128 = ((int)(((signed char)(((val183>>24u)&255u)))));
    float cast129 = tg_bitcast<float>((unsigned int)(val173));
    float cast130 = tg_bitcast<float>((unsigned int)(val174));
    float cast131 = tg_bitcast<float>((unsigned int)(val175));
    float cast132 = tg_bitcast<float>((unsigned int)(val176));
    float alu530 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val181&65535u)))))))*((cast129*((float)(((cast121*wmma39.x)+(cast122*wmma32.x)))))+(cast130*((float)(((cast123*wmma33.x)+(cast124*wmma34.x)))))+(cast131*((float)(((cast125*wmma35.x)+(cast126*wmma36.x)))))+(cast132*((float)(((cast127*wmma37.x)+(cast128*wmma38.x)))))));
    float alu531 = (alu414?alu530:(buf17+alu530));
    buf17 = alu531;
    unsigned int val186 = (*(buf0+alu180));
    if (alu414) {
      *(data0_5570560+(alu110+33)) = buf18;
    }
    float cast133 = tg_bitcast<float>((unsigned int)(val177));
    float cast134 = tg_bitcast<float>((unsigned int)(val178));
    float cast135 = tg_bitcast<float>((unsigned int)(val179));
    float cast136 = tg_bitcast<float>((unsigned int)(val180));
    float alu536 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val186&65535u)))))))*((cast133*((float)(((cast121*wmma39.y)+(cast122*wmma32.y)))))+(cast134*((float)(((cast123*wmma33.y)+(cast124*wmma34.y)))))+(cast135*((float)(((cast125*wmma35.y)+(cast126*wmma36.y)))))+(cast136*((float)(((cast127*wmma37.y)+(cast128*wmma38.y)))))));
    float alu537 = (alu414?alu536:(buf18+alu536));
    buf18 = alu537;
    unsigned int val187 = (*(buf0+alu185));
    if (alu414) {
      *(data0_5570560+(alu110+1056)) = buf19;
    }
    int cast137 = ((int)(((signed char)(((val184>>0u)&255u)))));
    int cast138 = ((int)(((signed char)(((val184>>8u)&255u)))));
    int cast139 = ((int)(((signed char)(((val184>>16u)&255u)))));
    int cast140 = ((int)(((signed char)(((val184>>24u)&255u)))));
    int cast141 = ((int)(((signed char)(((val185>>0u)&255u)))));
    int cast142 = ((int)(((signed char)(((val185>>8u)&255u)))));
    int cast143 = ((int)(((signed char)(((val185>>16u)&255u)))));
    int cast144 = ((int)(((signed char)(((val185>>24u)&255u)))));
    float alu542 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val187&65535u)))))))*((cast129*((float)(((cast137*wmma39.z)+(cast138*wmma32.z)))))+(cast130*((float)(((cast139*wmma33.z)+(cast140*wmma34.z)))))+(cast131*((float)(((cast141*wmma35.z)+(cast142*wmma36.z)))))+(cast132*((float)(((cast143*wmma37.z)+(cast144*wmma38.z)))))));
    float alu543 = (alu414?alu542:(buf19+alu542));
    buf19 = alu543;
    unsigned int val188 = (*(buf0+alu185));
    if (alu414) {
      *(data0_5570560+(alu110+1057)) = buf20;
    }
    float alu548 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val188&65535u)))))))*((cast133*((float)(((cast137*wmma39.w)+(cast138*wmma32.w)))))+(cast134*((float)(((cast139*wmma33.w)+(cast140*wmma34.w)))))+(cast135*((float)(((cast141*wmma35.w)+(cast142*wmma36.w)))))+(cast136*((float)(((cast143*wmma37.w)+(cast144*wmma38.w)))))));
    float alu549 = (alu414?alu548:(buf20+alu548));
    buf20 = alu549;
    unsigned int val189 = (*(buf0+alu217));
    unsigned int val190 = (*(buf0+alu218));
    unsigned int val191 = (*(buf0+alu219));
    unsigned int val192 = (*(buf0+alu220));
    unsigned int val193 = (*(buf0+alu221));
    unsigned int val194 = (*(buf0+alu222));
    unsigned int val195 = (*(buf0+alu223));
    unsigned int val196 = (*(buf0+alu224));
    unsigned int val197 = (*(buf0+alu190));
    unsigned int val198 = (*(buf0+alu191));
    unsigned int val199 = (*(buf0+alu192));
    unsigned int val200 = (*(buf0+alu196));
    unsigned int val201 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+2080)) = buf21;
    }
    int4 wmma40 = __WMMA_8_16_16_signed_char_int(alu450, cast113, cast0);
    int4 wmma41 = __WMMA_8_16_16_signed_char_int(alu451, cast114, cast0);
    int4 wmma42 = __WMMA_8_16_16_signed_char_int(alu452, cast115, cast0);
    int4 wmma43 = __WMMA_8_16_16_signed_char_int(alu453, cast116, cast0);
    int4 wmma44 = __WMMA_8_16_16_signed_char_int(alu454, cast117, cast0);
    int4 wmma45 = __WMMA_8_16_16_signed_char_int(alu455, cast118, cast0);
    int4 wmma46 = __WMMA_8_16_16_signed_char_int(alu456, cast119, cast0);
    int4 wmma47 = __WMMA_8_16_16_signed_char_int(alu457, cast120, cast0);
    int cast145 = ((int)(((signed char)(((val198>>0u)&255u)))));
    int cast146 = ((int)(((signed char)(((val198>>8u)&255u)))));
    int cast147 = ((int)(((signed char)(((val198>>16u)&255u)))));
    int cast148 = ((int)(((signed char)(((val198>>24u)&255u)))));
    int cast149 = ((int)(((signed char)(((val199>>0u)&255u)))));
    int cast150 = ((int)(((signed char)(((val199>>8u)&255u)))));
    int cast151 = ((int)(((signed char)(((val199>>16u)&255u)))));
    int cast152 = ((int)(((signed char)(((val199>>24u)&255u)))));
    float cast153 = tg_bitcast<float>((unsigned int)(val189));
    float cast154 = tg_bitcast<float>((unsigned int)(val190));
    float cast155 = tg_bitcast<float>((unsigned int)(val191));
    float cast156 = tg_bitcast<float>((unsigned int)(val192));
    float alu554 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val197&65535u)))))))*((cast153*((float)(((cast145*wmma40.x)+(cast146*wmma41.x)))))+(cast154*((float)(((cast147*wmma42.x)+(cast148*wmma43.x)))))+(cast155*((float)(((cast149*wmma44.x)+(cast150*wmma45.x)))))+(cast156*((float)(((cast151*wmma46.x)+(cast152*wmma47.x)))))));
    float alu555 = (alu414?alu554:(buf21+alu554));
    buf21 = alu555;
    unsigned int val202 = (*(buf0+alu190));
    if (alu414) {
      *(data0_5570560+(alu110+2081)) = buf22;
    }
    float cast157 = tg_bitcast<float>((unsigned int)(val193));
    float cast158 = tg_bitcast<float>((unsigned int)(val194));
    float cast159 = tg_bitcast<float>((unsigned int)(val195));
    float cast160 = tg_bitcast<float>((unsigned int)(val196));
    float alu560 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val202&65535u)))))))*((cast157*((float)(((cast145*wmma40.y)+(cast146*wmma41.y)))))+(cast158*((float)(((cast147*wmma42.y)+(cast148*wmma43.y)))))+(cast159*((float)(((cast149*wmma44.y)+(cast150*wmma45.y)))))+(cast160*((float)(((cast151*wmma46.y)+(cast152*wmma47.y)))))));
    float alu561 = (alu414?alu560:(buf22+alu560));
    buf22 = alu561;
    unsigned int val203 = (*(buf0+alu195));
    if (alu414) {
      *(data0_5570560+(alu110+3104)) = buf23;
    }
    int cast161 = ((int)(((signed char)(((val200>>0u)&255u)))));
    int cast162 = ((int)(((signed char)(((val200>>8u)&255u)))));
    int cast163 = ((int)(((signed char)(((val200>>16u)&255u)))));
    int cast164 = ((int)(((signed char)(((val200>>24u)&255u)))));
    int cast165 = ((int)(((signed char)(((val201>>0u)&255u)))));
    int cast166 = ((int)(((signed char)(((val201>>8u)&255u)))));
    int cast167 = ((int)(((signed char)(((val201>>16u)&255u)))));
    int cast168 = ((int)(((signed char)(((val201>>24u)&255u)))));
    float alu566 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val203&65535u)))))))*((cast153*((float)(((cast161*wmma40.z)+(cast162*wmma41.z)))))+(cast154*((float)(((cast163*wmma42.z)+(cast164*wmma43.z)))))+(cast155*((float)(((cast165*wmma44.z)+(cast166*wmma45.z)))))+(cast156*((float)(((cast167*wmma46.z)+(cast168*wmma47.z)))))));
    float alu567 = (alu414?alu566:(buf23+alu566));
    buf23 = alu567;
    unsigned int val204 = (*(buf0+alu195));
    if (alu414) {
      *(data0_5570560+(alu110+3105)) = buf24;
    }
    float alu572 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val204&65535u)))))))*((cast157*((float)(((cast161*wmma40.w)+(cast162*wmma41.w)))))+(cast158*((float)(((cast163*wmma42.w)+(cast164*wmma43.w)))))+(cast159*((float)(((cast165*wmma44.w)+(cast166*wmma45.w)))))+(cast160*((float)(((cast167*wmma46.w)+(cast168*wmma47.w)))))));
    float alu573 = (alu414?alu572:(buf24+alu572));
    buf24 = alu573;
    unsigned int val205 = (*(buf0+alu139));
    unsigned int val206 = (*(buf0+alu140));
    unsigned int val207 = (*(buf0+alu141));
    unsigned int val208 = (*(buf0+alu142));
    unsigned int val209 = (*(buf0+alu143));
    unsigned int val210 = (*(buf0+alu144));
    unsigned int val211 = (*(buf0+alu145));
    unsigned int val212 = (*(buf0+alu146));
    unsigned int val213 = (*(buf0+alu225));
    unsigned int val214 = (*(buf0+alu226));
    unsigned int val215 = (*(buf0+alu227));
    unsigned int val216 = (*(buf0+alu228));
    unsigned int val217 = (*(buf0+alu229));
    unsigned int val218 = (*(buf0+alu230));
    unsigned int val219 = (*(buf0+alu231));
    unsigned int val220 = (*(buf0+alu232));
    unsigned int val221 = (*(buf0+alu180));
    unsigned int val222 = (*(buf0+alu181));
    unsigned int val223 = (*(buf0+alu182));
    unsigned int val224 = (*(buf0+alu186));
    unsigned int val225 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+48)) = buf25;
    }
    char4 cast169 = make_char4(((signed char)(((val205>>0u)&255u))),((signed char)(((val205>>8u)&255u))),((signed char)(((val205>>16u)&255u))),((signed char)(((val205>>24u)&255u))));
    char4 cast170 = make_char4(((signed char)(((val206>>0u)&255u))),((signed char)(((val206>>8u)&255u))),((signed char)(((val206>>16u)&255u))),((signed char)(((val206>>24u)&255u))));
    char4 cast171 = make_char4(((signed char)(((val207>>0u)&255u))),((signed char)(((val207>>8u)&255u))),((signed char)(((val207>>16u)&255u))),((signed char)(((val207>>24u)&255u))));
    char4 cast172 = make_char4(((signed char)(((val208>>0u)&255u))),((signed char)(((val208>>8u)&255u))),((signed char)(((val208>>16u)&255u))),((signed char)(((val208>>24u)&255u))));
    char4 cast173 = make_char4(((signed char)(((val209>>0u)&255u))),((signed char)(((val209>>8u)&255u))),((signed char)(((val209>>16u)&255u))),((signed char)(((val209>>24u)&255u))));
    char4 cast174 = make_char4(((signed char)(((val210>>0u)&255u))),((signed char)(((val210>>8u)&255u))),((signed char)(((val210>>16u)&255u))),((signed char)(((val210>>24u)&255u))));
    char4 cast175 = make_char4(((signed char)(((val211>>0u)&255u))),((signed char)(((val211>>8u)&255u))),((signed char)(((val211>>16u)&255u))),((signed char)(((val211>>24u)&255u))));
    char4 cast176 = make_char4(((signed char)(((val212>>0u)&255u))),((signed char)(((val212>>8u)&255u))),((signed char)(((val212>>16u)&255u))),((signed char)(((val212>>24u)&255u))));
    int4 wmma48 = __WMMA_8_16_16_signed_char_int(alu418, cast170, cast0);
    int4 wmma49 = __WMMA_8_16_16_signed_char_int(alu419, cast171, cast0);
    int4 wmma50 = __WMMA_8_16_16_signed_char_int(alu420, cast172, cast0);
    int4 wmma51 = __WMMA_8_16_16_signed_char_int(alu421, cast173, cast0);
    int4 wmma52 = __WMMA_8_16_16_signed_char_int(alu422, cast174, cast0);
    int4 wmma53 = __WMMA_8_16_16_signed_char_int(alu423, cast175, cast0);
    int4 wmma54 = __WMMA_8_16_16_signed_char_int(alu424, cast176, cast0);
    int4 wmma55 = __WMMA_8_16_16_signed_char_int(alu425, cast169, cast0);
    int cast177 = ((int)(((signed char)(((val222>>0u)&255u)))));
    int cast178 = ((int)(((signed char)(((val222>>8u)&255u)))));
    int cast179 = ((int)(((signed char)(((val222>>16u)&255u)))));
    int cast180 = ((int)(((signed char)(((val222>>24u)&255u)))));
    int cast181 = ((int)(((signed char)(((val223>>0u)&255u)))));
    int cast182 = ((int)(((signed char)(((val223>>8u)&255u)))));
    int cast183 = ((int)(((signed char)(((val223>>16u)&255u)))));
    int cast184 = ((int)(((signed char)(((val223>>24u)&255u)))));
    float cast185 = tg_bitcast<float>((unsigned int)(val213));
    float cast186 = tg_bitcast<float>((unsigned int)(val214));
    float cast187 = tg_bitcast<float>((unsigned int)(val215));
    float cast188 = tg_bitcast<float>((unsigned int)(val216));
    float alu578 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val221&65535u)))))))*((cast185*((float)(((cast177*wmma55.x)+(cast178*wmma48.x)))))+(cast186*((float)(((cast179*wmma49.x)+(cast180*wmma50.x)))))+(cast187*((float)(((cast181*wmma51.x)+(cast182*wmma52.x)))))+(cast188*((float)(((cast183*wmma53.x)+(cast184*wmma54.x)))))));
    float alu579 = (alu414?alu578:(buf25+alu578));
    buf25 = alu579;
    unsigned int val226 = (*(buf0+alu180));
    if (alu414) {
      *(data0_5570560+(alu110+49)) = buf26;
    }
    float cast189 = tg_bitcast<float>((unsigned int)(val217));
    float cast190 = tg_bitcast<float>((unsigned int)(val218));
    float cast191 = tg_bitcast<float>((unsigned int)(val219));
    float cast192 = tg_bitcast<float>((unsigned int)(val220));
    float alu584 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val226&65535u)))))))*((cast189*((float)(((cast177*wmma55.y)+(cast178*wmma48.y)))))+(cast190*((float)(((cast179*wmma49.y)+(cast180*wmma50.y)))))+(cast191*((float)(((cast181*wmma51.y)+(cast182*wmma52.y)))))+(cast192*((float)(((cast183*wmma53.y)+(cast184*wmma54.y)))))));
    float alu585 = (alu414?alu584:(buf26+alu584));
    buf26 = alu585;
    unsigned int val227 = (*(buf0+alu185));
    if (alu414) {
      *(data0_5570560+(alu110+1072)) = buf27;
    }
    int cast193 = ((int)(((signed char)(((val224>>0u)&255u)))));
    int cast194 = ((int)(((signed char)(((val224>>8u)&255u)))));
    int cast195 = ((int)(((signed char)(((val224>>16u)&255u)))));
    int cast196 = ((int)(((signed char)(((val224>>24u)&255u)))));
    int cast197 = ((int)(((signed char)(((val225>>0u)&255u)))));
    int cast198 = ((int)(((signed char)(((val225>>8u)&255u)))));
    int cast199 = ((int)(((signed char)(((val225>>16u)&255u)))));
    int cast200 = ((int)(((signed char)(((val225>>24u)&255u)))));
    float alu590 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val227&65535u)))))))*((cast185*((float)(((cast193*wmma55.z)+(cast194*wmma48.z)))))+(cast186*((float)(((cast195*wmma49.z)+(cast196*wmma50.z)))))+(cast187*((float)(((cast197*wmma51.z)+(cast198*wmma52.z)))))+(cast188*((float)(((cast199*wmma53.z)+(cast200*wmma54.z)))))));
    float alu591 = (alu414?alu590:(buf27+alu590));
    buf27 = alu591;
    unsigned int val228 = (*(buf0+alu185));
    if (alu414) {
      *(data0_5570560+(alu110+1073)) = buf28;
    }
    float alu596 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val228&65535u)))))))*((cast189*((float)(((cast193*wmma55.w)+(cast194*wmma48.w)))))+(cast190*((float)(((cast195*wmma49.w)+(cast196*wmma50.w)))))+(cast191*((float)(((cast197*wmma51.w)+(cast198*wmma52.w)))))+(cast192*((float)(((cast199*wmma53.w)+(cast200*wmma54.w)))))));
    float alu597 = (alu414?alu596:(buf28+alu596));
    buf28 = alu597;
    unsigned int val229 = (*(buf0+alu225));
    unsigned int val230 = (*(buf0+alu226));
    unsigned int val231 = (*(buf0+alu227));
    unsigned int val232 = (*(buf0+alu228));
    unsigned int val233 = (*(buf0+alu229));
    unsigned int val234 = (*(buf0+alu230));
    unsigned int val235 = (*(buf0+alu231));
    unsigned int val236 = (*(buf0+alu232));
    unsigned int val237 = (*(buf0+alu190));
    unsigned int val238 = (*(buf0+alu191));
    unsigned int val239 = (*(buf0+alu192));
    unsigned int val240 = (*(buf0+alu196));
    unsigned int val241 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+2096)) = buf29;
    }
    int4 wmma56 = __WMMA_8_16_16_signed_char_int(alu450, cast169, cast0);
    int4 wmma57 = __WMMA_8_16_16_signed_char_int(alu451, cast170, cast0);
    int4 wmma58 = __WMMA_8_16_16_signed_char_int(alu452, cast171, cast0);
    int4 wmma59 = __WMMA_8_16_16_signed_char_int(alu453, cast172, cast0);
    int4 wmma60 = __WMMA_8_16_16_signed_char_int(alu454, cast173, cast0);
    int4 wmma61 = __WMMA_8_16_16_signed_char_int(alu455, cast174, cast0);
    int4 wmma62 = __WMMA_8_16_16_signed_char_int(alu456, cast175, cast0);
    int4 wmma63 = __WMMA_8_16_16_signed_char_int(alu457, cast176, cast0);
    int cast201 = ((int)(((signed char)(((val238>>0u)&255u)))));
    int cast202 = ((int)(((signed char)(((val238>>8u)&255u)))));
    int cast203 = ((int)(((signed char)(((val238>>16u)&255u)))));
    int cast204 = ((int)(((signed char)(((val238>>24u)&255u)))));
    int cast205 = ((int)(((signed char)(((val239>>0u)&255u)))));
    int cast206 = ((int)(((signed char)(((val239>>8u)&255u)))));
    int cast207 = ((int)(((signed char)(((val239>>16u)&255u)))));
    int cast208 = ((int)(((signed char)(((val239>>24u)&255u)))));
    float cast209 = tg_bitcast<float>((unsigned int)(val229));
    float cast210 = tg_bitcast<float>((unsigned int)(val230));
    float cast211 = tg_bitcast<float>((unsigned int)(val231));
    float cast212 = tg_bitcast<float>((unsigned int)(val232));
    float alu602 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val237&65535u)))))))*((cast209*((float)(((cast201*wmma56.x)+(cast202*wmma57.x)))))+(cast210*((float)(((cast203*wmma58.x)+(cast204*wmma59.x)))))+(cast211*((float)(((cast205*wmma60.x)+(cast206*wmma61.x)))))+(cast212*((float)(((cast207*wmma62.x)+(cast208*wmma63.x)))))));
    float alu603 = (alu414?alu602:(buf29+alu602));
    buf29 = alu603;
    unsigned int val242 = (*(buf0+alu190));
    if (alu414) {
      *(data0_5570560+(alu110+2097)) = buf30;
    }
    float cast213 = tg_bitcast<float>((unsigned int)(val233));
    float cast214 = tg_bitcast<float>((unsigned int)(val234));
    float cast215 = tg_bitcast<float>((unsigned int)(val235));
    float cast216 = tg_bitcast<float>((unsigned int)(val236));
    float alu608 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val242&65535u)))))))*((cast213*((float)(((cast201*wmma56.y)+(cast202*wmma57.y)))))+(cast214*((float)(((cast203*wmma58.y)+(cast204*wmma59.y)))))+(cast215*((float)(((cast205*wmma60.y)+(cast206*wmma61.y)))))+(cast216*((float)(((cast207*wmma62.y)+(cast208*wmma63.y)))))));
    float alu609 = (alu414?alu608:(buf30+alu608));
    buf30 = alu609;
    unsigned int val243 = (*(buf0+alu195));
    if (alu414) {
      *(data0_5570560+(alu110+3120)) = buf31;
    }
    int cast217 = ((int)(((signed char)(((val240>>0u)&255u)))));
    int cast218 = ((int)(((signed char)(((val240>>8u)&255u)))));
    int cast219 = ((int)(((signed char)(((val240>>16u)&255u)))));
    int cast220 = ((int)(((signed char)(((val240>>24u)&255u)))));
    int cast221 = ((int)(((signed char)(((val241>>0u)&255u)))));
    int cast222 = ((int)(((signed char)(((val241>>8u)&255u)))));
    int cast223 = ((int)(((signed char)(((val241>>16u)&255u)))));
    int cast224 = ((int)(((signed char)(((val241>>24u)&255u)))));
    float alu614 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val243&65535u)))))))*((cast209*((float)(((cast217*wmma56.z)+(cast218*wmma57.z)))))+(cast210*((float)(((cast219*wmma58.z)+(cast220*wmma59.z)))))+(cast211*((float)(((cast221*wmma60.z)+(cast222*wmma61.z)))))+(cast212*((float)(((cast223*wmma62.z)+(cast224*wmma63.z)))))));
    float alu615 = (alu414?alu614:(buf31+alu614));
    buf31 = alu615;
    unsigned int val244 = (*(buf0+alu195));
    if (alu414) {
      *(data0_5570560+(alu110+3121)) = buf32;
    }
    float alu620 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val244&65535u)))))))*((cast213*((float)(((cast217*wmma56.w)+(cast218*wmma57.w)))))+(cast214*((float)(((cast219*wmma58.w)+(cast220*wmma59.w)))))+(cast215*((float)(((cast221*wmma60.w)+(cast222*wmma61.w)))))+(cast216*((float)(((cast223*wmma62.w)+(cast224*wmma63.w)))))));
    float alu621 = (alu414?alu620:(buf32+alu620));
    buf32 = alu621;
    unsigned int val245 = (*(buf0+alu147));
    unsigned int val246 = (*(buf0+alu148));
    unsigned int val247 = (*(buf0+alu149));
    unsigned int val248 = (*(buf0+alu150));
    unsigned int val249 = (*(buf0+alu151));
    unsigned int val250 = (*(buf0+alu152));
    unsigned int val251 = (*(buf0+alu153));
    unsigned int val252 = (*(buf0+alu154));
    unsigned int val253 = (*(buf0+alu233));
    unsigned int val254 = (*(buf0+alu234));
    unsigned int val255 = (*(buf0+alu235));
    unsigned int val256 = (*(buf0+alu236));
    unsigned int val257 = (*(buf0+alu237));
    unsigned int val258 = (*(buf0+alu238));
    unsigned int val259 = (*(buf0+alu239));
    unsigned int val260 = (*(buf0+alu240));
    unsigned int val261 = (*(buf0+alu180));
    unsigned int val262 = (*(buf0+alu181));
    unsigned int val263 = (*(buf0+alu182));
    unsigned int val264 = (*(buf0+alu186));
    unsigned int val265 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+64)) = buf33;
    }
    char4 cast225 = make_char4(((signed char)(((val245>>0u)&255u))),((signed char)(((val245>>8u)&255u))),((signed char)(((val245>>16u)&255u))),((signed char)(((val245>>24u)&255u))));
    char4 cast226 = make_char4(((signed char)(((val246>>0u)&255u))),((signed char)(((val246>>8u)&255u))),((signed char)(((val246>>16u)&255u))),((signed char)(((val246>>24u)&255u))));
    char4 cast227 = make_char4(((signed char)(((val247>>0u)&255u))),((signed char)(((val247>>8u)&255u))),((signed char)(((val247>>16u)&255u))),((signed char)(((val247>>24u)&255u))));
    char4 cast228 = make_char4(((signed char)(((val248>>0u)&255u))),((signed char)(((val248>>8u)&255u))),((signed char)(((val248>>16u)&255u))),((signed char)(((val248>>24u)&255u))));
    char4 cast229 = make_char4(((signed char)(((val249>>0u)&255u))),((signed char)(((val249>>8u)&255u))),((signed char)(((val249>>16u)&255u))),((signed char)(((val249>>24u)&255u))));
    char4 cast230 = make_char4(((signed char)(((val250>>0u)&255u))),((signed char)(((val250>>8u)&255u))),((signed char)(((val250>>16u)&255u))),((signed char)(((val250>>24u)&255u))));
    char4 cast231 = make_char4(((signed char)(((val251>>0u)&255u))),((signed char)(((val251>>8u)&255u))),((signed char)(((val251>>16u)&255u))),((signed char)(((val251>>24u)&255u))));
    char4 cast232 = make_char4(((signed char)(((val252>>0u)&255u))),((signed char)(((val252>>8u)&255u))),((signed char)(((val252>>16u)&255u))),((signed char)(((val252>>24u)&255u))));
    int4 wmma64 = __WMMA_8_16_16_signed_char_int(alu418, cast226, cast0);
    int4 wmma65 = __WMMA_8_16_16_signed_char_int(alu419, cast227, cast0);
    int4 wmma66 = __WMMA_8_16_16_signed_char_int(alu420, cast228, cast0);
    int4 wmma67 = __WMMA_8_16_16_signed_char_int(alu421, cast229, cast0);
    int4 wmma68 = __WMMA_8_16_16_signed_char_int(alu422, cast230, cast0);
    int4 wmma69 = __WMMA_8_16_16_signed_char_int(alu423, cast231, cast0);
    int4 wmma70 = __WMMA_8_16_16_signed_char_int(alu424, cast232, cast0);
    int4 wmma71 = __WMMA_8_16_16_signed_char_int(alu425, cast225, cast0);
    int cast233 = ((int)(((signed char)(((val262>>0u)&255u)))));
    int cast234 = ((int)(((signed char)(((val262>>8u)&255u)))));
    int cast235 = ((int)(((signed char)(((val262>>16u)&255u)))));
    int cast236 = ((int)(((signed char)(((val262>>24u)&255u)))));
    int cast237 = ((int)(((signed char)(((val263>>0u)&255u)))));
    int cast238 = ((int)(((signed char)(((val263>>8u)&255u)))));
    int cast239 = ((int)(((signed char)(((val263>>16u)&255u)))));
    int cast240 = ((int)(((signed char)(((val263>>24u)&255u)))));
    float cast241 = tg_bitcast<float>((unsigned int)(val253));
    float cast242 = tg_bitcast<float>((unsigned int)(val254));
    float cast243 = tg_bitcast<float>((unsigned int)(val255));
    float cast244 = tg_bitcast<float>((unsigned int)(val256));
    float alu626 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val261&65535u)))))))*((cast241*((float)(((cast233*wmma71.x)+(cast234*wmma64.x)))))+(cast242*((float)(((cast235*wmma65.x)+(cast236*wmma66.x)))))+(cast243*((float)(((cast237*wmma67.x)+(cast238*wmma68.x)))))+(cast244*((float)(((cast239*wmma69.x)+(cast240*wmma70.x)))))));
    float alu627 = (alu414?alu626:(buf33+alu626));
    buf33 = alu627;
    unsigned int val266 = (*(buf0+alu180));
    if (alu414) {
      *(data0_5570560+(alu110+65)) = buf34;
    }
    float cast245 = tg_bitcast<float>((unsigned int)(val257));
    float cast246 = tg_bitcast<float>((unsigned int)(val258));
    float cast247 = tg_bitcast<float>((unsigned int)(val259));
    float cast248 = tg_bitcast<float>((unsigned int)(val260));
    float alu632 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val266&65535u)))))))*((cast245*((float)(((cast233*wmma71.y)+(cast234*wmma64.y)))))+(cast246*((float)(((cast235*wmma65.y)+(cast236*wmma66.y)))))+(cast247*((float)(((cast237*wmma67.y)+(cast238*wmma68.y)))))+(cast248*((float)(((cast239*wmma69.y)+(cast240*wmma70.y)))))));
    float alu633 = (alu414?alu632:(buf34+alu632));
    buf34 = alu633;
    unsigned int val267 = (*(buf0+alu185));
    if (alu414) {
      *(data0_5570560+(alu110+1088)) = buf35;
    }
    int cast249 = ((int)(((signed char)(((val264>>0u)&255u)))));
    int cast250 = ((int)(((signed char)(((val264>>8u)&255u)))));
    int cast251 = ((int)(((signed char)(((val264>>16u)&255u)))));
    int cast252 = ((int)(((signed char)(((val264>>24u)&255u)))));
    int cast253 = ((int)(((signed char)(((val265>>0u)&255u)))));
    int cast254 = ((int)(((signed char)(((val265>>8u)&255u)))));
    int cast255 = ((int)(((signed char)(((val265>>16u)&255u)))));
    int cast256 = ((int)(((signed char)(((val265>>24u)&255u)))));
    float alu638 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val267&65535u)))))))*((cast241*((float)(((cast249*wmma71.z)+(cast250*wmma64.z)))))+(cast242*((float)(((cast251*wmma65.z)+(cast252*wmma66.z)))))+(cast243*((float)(((cast253*wmma67.z)+(cast254*wmma68.z)))))+(cast244*((float)(((cast255*wmma69.z)+(cast256*wmma70.z)))))));
    float alu639 = (alu414?alu638:(buf35+alu638));
    buf35 = alu639;
    unsigned int val268 = (*(buf0+alu185));
    if (alu414) {
      *(data0_5570560+(alu110+1089)) = buf36;
    }
    float alu644 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val268&65535u)))))))*((cast245*((float)(((cast249*wmma71.w)+(cast250*wmma64.w)))))+(cast246*((float)(((cast251*wmma65.w)+(cast252*wmma66.w)))))+(cast247*((float)(((cast253*wmma67.w)+(cast254*wmma68.w)))))+(cast248*((float)(((cast255*wmma69.w)+(cast256*wmma70.w)))))));
    float alu645 = (alu414?alu644:(buf36+alu644));
    buf36 = alu645;
    unsigned int val269 = (*(buf0+alu233));
    unsigned int val270 = (*(buf0+alu234));
    unsigned int val271 = (*(buf0+alu235));
    unsigned int val272 = (*(buf0+alu236));
    unsigned int val273 = (*(buf0+alu237));
    unsigned int val274 = (*(buf0+alu238));
    unsigned int val275 = (*(buf0+alu239));
    unsigned int val276 = (*(buf0+alu240));
    unsigned int val277 = (*(buf0+alu190));
    unsigned int val278 = (*(buf0+alu191));
    unsigned int val279 = (*(buf0+alu192));
    unsigned int val280 = (*(buf0+alu196));
    unsigned int val281 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+2112)) = buf37;
    }
    int4 wmma72 = __WMMA_8_16_16_signed_char_int(alu450, cast225, cast0);
    int4 wmma73 = __WMMA_8_16_16_signed_char_int(alu451, cast226, cast0);
    int4 wmma74 = __WMMA_8_16_16_signed_char_int(alu452, cast227, cast0);
    int4 wmma75 = __WMMA_8_16_16_signed_char_int(alu453, cast228, cast0);
    int4 wmma76 = __WMMA_8_16_16_signed_char_int(alu454, cast229, cast0);
    int4 wmma77 = __WMMA_8_16_16_signed_char_int(alu455, cast230, cast0);
    int4 wmma78 = __WMMA_8_16_16_signed_char_int(alu456, cast231, cast0);
    int4 wmma79 = __WMMA_8_16_16_signed_char_int(alu457, cast232, cast0);
    int cast257 = ((int)(((signed char)(((val278>>0u)&255u)))));
    int cast258 = ((int)(((signed char)(((val278>>8u)&255u)))));
    int cast259 = ((int)(((signed char)(((val278>>16u)&255u)))));
    int cast260 = ((int)(((signed char)(((val278>>24u)&255u)))));
    int cast261 = ((int)(((signed char)(((val279>>0u)&255u)))));
    int cast262 = ((int)(((signed char)(((val279>>8u)&255u)))));
    int cast263 = ((int)(((signed char)(((val279>>16u)&255u)))));
    int cast264 = ((int)(((signed char)(((val279>>24u)&255u)))));
    float cast265 = tg_bitcast<float>((unsigned int)(val269));
    float cast266 = tg_bitcast<float>((unsigned int)(val270));
    float cast267 = tg_bitcast<float>((unsigned int)(val271));
    float cast268 = tg_bitcast<float>((unsigned int)(val272));
    float alu650 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val277&65535u)))))))*((cast265*((float)(((cast257*wmma72.x)+(cast258*wmma73.x)))))+(cast266*((float)(((cast259*wmma74.x)+(cast260*wmma75.x)))))+(cast267*((float)(((cast261*wmma76.x)+(cast262*wmma77.x)))))+(cast268*((float)(((cast263*wmma78.x)+(cast264*wmma79.x)))))));
    float alu651 = (alu414?alu650:(buf37+alu650));
    buf37 = alu651;
    unsigned int val282 = (*(buf0+alu190));
    if (alu414) {
      *(data0_5570560+(alu110+2113)) = buf38;
    }
    float cast269 = tg_bitcast<float>((unsigned int)(val273));
    float cast270 = tg_bitcast<float>((unsigned int)(val274));
    float cast271 = tg_bitcast<float>((unsigned int)(val275));
    float cast272 = tg_bitcast<float>((unsigned int)(val276));
    float alu656 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val282&65535u)))))))*((cast269*((float)(((cast257*wmma72.y)+(cast258*wmma73.y)))))+(cast270*((float)(((cast259*wmma74.y)+(cast260*wmma75.y)))))+(cast271*((float)(((cast261*wmma76.y)+(cast262*wmma77.y)))))+(cast272*((float)(((cast263*wmma78.y)+(cast264*wmma79.y)))))));
    float alu657 = (alu414?alu656:(buf38+alu656));
    buf38 = alu657;
    unsigned int val283 = (*(buf0+alu195));
    if (alu414) {
      *(data0_5570560+(alu110+3136)) = buf39;
    }
    int cast273 = ((int)(((signed char)(((val280>>0u)&255u)))));
    int cast274 = ((int)(((signed char)(((val280>>8u)&255u)))));
    int cast275 = ((int)(((signed char)(((val280>>16u)&255u)))));
    int cast276 = ((int)(((signed char)(((val280>>24u)&255u)))));
    int cast277 = ((int)(((signed char)(((val281>>0u)&255u)))));
    int cast278 = ((int)(((signed char)(((val281>>8u)&255u)))));
    int cast279 = ((int)(((signed char)(((val281>>16u)&255u)))));
    int cast280 = ((int)(((signed char)(((val281>>24u)&255u)))));
    float alu662 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val283&65535u)))))))*((cast265*((float)(((cast273*wmma72.z)+(cast274*wmma73.z)))))+(cast266*((float)(((cast275*wmma74.z)+(cast276*wmma75.z)))))+(cast267*((float)(((cast277*wmma76.z)+(cast278*wmma77.z)))))+(cast268*((float)(((cast279*wmma78.z)+(cast280*wmma79.z)))))));
    float alu663 = (alu414?alu662:(buf39+alu662));
    buf39 = alu663;
    unsigned int val284 = (*(buf0+alu195));
    if (alu414) {
      *(data0_5570560+(alu110+3137)) = buf40;
    }
    float alu668 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val284&65535u)))))))*((cast269*((float)(((cast273*wmma72.w)+(cast274*wmma73.w)))))+(cast270*((float)(((cast275*wmma74.w)+(cast276*wmma75.w)))))+(cast271*((float)(((cast277*wmma76.w)+(cast278*wmma77.w)))))+(cast272*((float)(((cast279*wmma78.w)+(cast280*wmma79.w)))))));
    float alu669 = (alu414?alu668:(buf40+alu668));
    buf40 = alu669;
    unsigned int val285 = (*(buf0+alu155));
    unsigned int val286 = (*(buf0+alu156));
    unsigned int val287 = (*(buf0+alu157));
    unsigned int val288 = (*(buf0+alu158));
    unsigned int val289 = (*(buf0+alu159));
    unsigned int val290 = (*(buf0+alu160));
    unsigned int val291 = (*(buf0+alu161));
    unsigned int val292 = (*(buf0+alu162));
    unsigned int val293 = (*(buf0+alu241));
    unsigned int val294 = (*(buf0+alu242));
    unsigned int val295 = (*(buf0+alu243));
    unsigned int val296 = (*(buf0+alu244));
    unsigned int val297 = (*(buf0+alu245));
    unsigned int val298 = (*(buf0+alu246));
    unsigned int val299 = (*(buf0+alu247));
    unsigned int val300 = (*(buf0+alu248));
    unsigned int val301 = (*(buf0+alu180));
    unsigned int val302 = (*(buf0+alu181));
    unsigned int val303 = (*(buf0+alu182));
    unsigned int val304 = (*(buf0+alu186));
    unsigned int val305 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+80)) = buf41;
    }
    char4 cast281 = make_char4(((signed char)(((val285>>0u)&255u))),((signed char)(((val285>>8u)&255u))),((signed char)(((val285>>16u)&255u))),((signed char)(((val285>>24u)&255u))));
    char4 cast282 = make_char4(((signed char)(((val286>>0u)&255u))),((signed char)(((val286>>8u)&255u))),((signed char)(((val286>>16u)&255u))),((signed char)(((val286>>24u)&255u))));
    char4 cast283 = make_char4(((signed char)(((val287>>0u)&255u))),((signed char)(((val287>>8u)&255u))),((signed char)(((val287>>16u)&255u))),((signed char)(((val287>>24u)&255u))));
    char4 cast284 = make_char4(((signed char)(((val288>>0u)&255u))),((signed char)(((val288>>8u)&255u))),((signed char)(((val288>>16u)&255u))),((signed char)(((val288>>24u)&255u))));
    char4 cast285 = make_char4(((signed char)(((val289>>0u)&255u))),((signed char)(((val289>>8u)&255u))),((signed char)(((val289>>16u)&255u))),((signed char)(((val289>>24u)&255u))));
    char4 cast286 = make_char4(((signed char)(((val290>>0u)&255u))),((signed char)(((val290>>8u)&255u))),((signed char)(((val290>>16u)&255u))),((signed char)(((val290>>24u)&255u))));
    char4 cast287 = make_char4(((signed char)(((val291>>0u)&255u))),((signed char)(((val291>>8u)&255u))),((signed char)(((val291>>16u)&255u))),((signed char)(((val291>>24u)&255u))));
    char4 cast288 = make_char4(((signed char)(((val292>>0u)&255u))),((signed char)(((val292>>8u)&255u))),((signed char)(((val292>>16u)&255u))),((signed char)(((val292>>24u)&255u))));
    int4 wmma80 = __WMMA_8_16_16_signed_char_int(alu418, cast282, cast0);
    int4 wmma81 = __WMMA_8_16_16_signed_char_int(alu419, cast283, cast0);
    int4 wmma82 = __WMMA_8_16_16_signed_char_int(alu420, cast284, cast0);
    int4 wmma83 = __WMMA_8_16_16_signed_char_int(alu421, cast285, cast0);
    int4 wmma84 = __WMMA_8_16_16_signed_char_int(alu422, cast286, cast0);
    int4 wmma85 = __WMMA_8_16_16_signed_char_int(alu423, cast287, cast0);
    int4 wmma86 = __WMMA_8_16_16_signed_char_int(alu424, cast288, cast0);
    int4 wmma87 = __WMMA_8_16_16_signed_char_int(alu425, cast281, cast0);
    int cast289 = ((int)(((signed char)(((val302>>0u)&255u)))));
    int cast290 = ((int)(((signed char)(((val302>>8u)&255u)))));
    int cast291 = ((int)(((signed char)(((val302>>16u)&255u)))));
    int cast292 = ((int)(((signed char)(((val302>>24u)&255u)))));
    int cast293 = ((int)(((signed char)(((val303>>0u)&255u)))));
    int cast294 = ((int)(((signed char)(((val303>>8u)&255u)))));
    int cast295 = ((int)(((signed char)(((val303>>16u)&255u)))));
    int cast296 = ((int)(((signed char)(((val303>>24u)&255u)))));
    float cast297 = tg_bitcast<float>((unsigned int)(val293));
    float cast298 = tg_bitcast<float>((unsigned int)(val294));
    float cast299 = tg_bitcast<float>((unsigned int)(val295));
    float cast300 = tg_bitcast<float>((unsigned int)(val296));
    float alu674 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val301&65535u)))))))*((cast297*((float)(((cast289*wmma87.x)+(cast290*wmma80.x)))))+(cast298*((float)(((cast291*wmma81.x)+(cast292*wmma82.x)))))+(cast299*((float)(((cast293*wmma83.x)+(cast294*wmma84.x)))))+(cast300*((float)(((cast295*wmma85.x)+(cast296*wmma86.x)))))));
    float alu675 = (alu414?alu674:(buf41+alu674));
    buf41 = alu675;
    unsigned int val306 = (*(buf0+alu180));
    if (alu414) {
      *(data0_5570560+(alu110+81)) = buf42;
    }
    float cast301 = tg_bitcast<float>((unsigned int)(val297));
    float cast302 = tg_bitcast<float>((unsigned int)(val298));
    float cast303 = tg_bitcast<float>((unsigned int)(val299));
    float cast304 = tg_bitcast<float>((unsigned int)(val300));
    float alu680 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val306&65535u)))))))*((cast301*((float)(((cast289*wmma87.y)+(cast290*wmma80.y)))))+(cast302*((float)(((cast291*wmma81.y)+(cast292*wmma82.y)))))+(cast303*((float)(((cast293*wmma83.y)+(cast294*wmma84.y)))))+(cast304*((float)(((cast295*wmma85.y)+(cast296*wmma86.y)))))));
    float alu681 = (alu414?alu680:(buf42+alu680));
    buf42 = alu681;
    unsigned int val307 = (*(buf0+alu185));
    if (alu414) {
      *(data0_5570560+(alu110+1104)) = buf43;
    }
    int cast305 = ((int)(((signed char)(((val304>>0u)&255u)))));
    int cast306 = ((int)(((signed char)(((val304>>8u)&255u)))));
    int cast307 = ((int)(((signed char)(((val304>>16u)&255u)))));
    int cast308 = ((int)(((signed char)(((val304>>24u)&255u)))));
    int cast309 = ((int)(((signed char)(((val305>>0u)&255u)))));
    int cast310 = ((int)(((signed char)(((val305>>8u)&255u)))));
    int cast311 = ((int)(((signed char)(((val305>>16u)&255u)))));
    int cast312 = ((int)(((signed char)(((val305>>24u)&255u)))));
    float alu686 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val307&65535u)))))))*((cast297*((float)(((cast305*wmma87.z)+(cast306*wmma80.z)))))+(cast298*((float)(((cast307*wmma81.z)+(cast308*wmma82.z)))))+(cast299*((float)(((cast309*wmma83.z)+(cast310*wmma84.z)))))+(cast300*((float)(((cast311*wmma85.z)+(cast312*wmma86.z)))))));
    float alu687 = (alu414?alu686:(buf43+alu686));
    buf43 = alu687;
    unsigned int val308 = (*(buf0+alu185));
    if (alu414) {
      *(data0_5570560+(alu110+1105)) = buf44;
    }
    float alu692 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val308&65535u)))))))*((cast301*((float)(((cast305*wmma87.w)+(cast306*wmma80.w)))))+(cast302*((float)(((cast307*wmma81.w)+(cast308*wmma82.w)))))+(cast303*((float)(((cast309*wmma83.w)+(cast310*wmma84.w)))))+(cast304*((float)(((cast311*wmma85.w)+(cast312*wmma86.w)))))));
    float alu693 = (alu414?alu692:(buf44+alu692));
    buf44 = alu693;
    unsigned int val309 = (*(buf0+alu241));
    unsigned int val310 = (*(buf0+alu242));
    unsigned int val311 = (*(buf0+alu243));
    unsigned int val312 = (*(buf0+alu244));
    unsigned int val313 = (*(buf0+alu245));
    unsigned int val314 = (*(buf0+alu246));
    unsigned int val315 = (*(buf0+alu247));
    unsigned int val316 = (*(buf0+alu248));
    unsigned int val317 = (*(buf0+alu190));
    unsigned int val318 = (*(buf0+alu191));
    unsigned int val319 = (*(buf0+alu192));
    unsigned int val320 = (*(buf0+alu196));
    unsigned int val321 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+2128)) = buf45;
    }
    int4 wmma88 = __WMMA_8_16_16_signed_char_int(alu450, cast281, cast0);
    int4 wmma89 = __WMMA_8_16_16_signed_char_int(alu451, cast282, cast0);
    int4 wmma90 = __WMMA_8_16_16_signed_char_int(alu452, cast283, cast0);
    int4 wmma91 = __WMMA_8_16_16_signed_char_int(alu453, cast284, cast0);
    int4 wmma92 = __WMMA_8_16_16_signed_char_int(alu454, cast285, cast0);
    int4 wmma93 = __WMMA_8_16_16_signed_char_int(alu455, cast286, cast0);
    int4 wmma94 = __WMMA_8_16_16_signed_char_int(alu456, cast287, cast0);
    int4 wmma95 = __WMMA_8_16_16_signed_char_int(alu457, cast288, cast0);
    int cast313 = ((int)(((signed char)(((val318>>0u)&255u)))));
    int cast314 = ((int)(((signed char)(((val318>>8u)&255u)))));
    int cast315 = ((int)(((signed char)(((val318>>16u)&255u)))));
    int cast316 = ((int)(((signed char)(((val318>>24u)&255u)))));
    int cast317 = ((int)(((signed char)(((val319>>0u)&255u)))));
    int cast318 = ((int)(((signed char)(((val319>>8u)&255u)))));
    int cast319 = ((int)(((signed char)(((val319>>16u)&255u)))));
    int cast320 = ((int)(((signed char)(((val319>>24u)&255u)))));
    float cast321 = tg_bitcast<float>((unsigned int)(val309));
    float cast322 = tg_bitcast<float>((unsigned int)(val310));
    float cast323 = tg_bitcast<float>((unsigned int)(val311));
    float cast324 = tg_bitcast<float>((unsigned int)(val312));
    float alu698 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val317&65535u)))))))*((cast321*((float)(((cast313*wmma88.x)+(cast314*wmma89.x)))))+(cast322*((float)(((cast315*wmma90.x)+(cast316*wmma91.x)))))+(cast323*((float)(((cast317*wmma92.x)+(cast318*wmma93.x)))))+(cast324*((float)(((cast319*wmma94.x)+(cast320*wmma95.x)))))));
    float alu699 = (alu414?alu698:(buf45+alu698));
    buf45 = alu699;
    unsigned int val322 = (*(buf0+alu190));
    if (alu414) {
      *(data0_5570560+(alu110+2129)) = buf46;
    }
    float cast325 = tg_bitcast<float>((unsigned int)(val313));
    float cast326 = tg_bitcast<float>((unsigned int)(val314));
    float cast327 = tg_bitcast<float>((unsigned int)(val315));
    float cast328 = tg_bitcast<float>((unsigned int)(val316));
    float alu704 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val322&65535u)))))))*((cast325*((float)(((cast313*wmma88.y)+(cast314*wmma89.y)))))+(cast326*((float)(((cast315*wmma90.y)+(cast316*wmma91.y)))))+(cast327*((float)(((cast317*wmma92.y)+(cast318*wmma93.y)))))+(cast328*((float)(((cast319*wmma94.y)+(cast320*wmma95.y)))))));
    float alu705 = (alu414?alu704:(buf46+alu704));
    buf46 = alu705;
    unsigned int val323 = (*(buf0+alu195));
    if (alu414) {
      *(data0_5570560+(alu110+3152)) = buf47;
    }
    int cast329 = ((int)(((signed char)(((val320>>0u)&255u)))));
    int cast330 = ((int)(((signed char)(((val320>>8u)&255u)))));
    int cast331 = ((int)(((signed char)(((val320>>16u)&255u)))));
    int cast332 = ((int)(((signed char)(((val320>>24u)&255u)))));
    int cast333 = ((int)(((signed char)(((val321>>0u)&255u)))));
    int cast334 = ((int)(((signed char)(((val321>>8u)&255u)))));
    int cast335 = ((int)(((signed char)(((val321>>16u)&255u)))));
    int cast336 = ((int)(((signed char)(((val321>>24u)&255u)))));
    float alu710 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val323&65535u)))))))*((cast321*((float)(((cast329*wmma88.z)+(cast330*wmma89.z)))))+(cast322*((float)(((cast331*wmma90.z)+(cast332*wmma91.z)))))+(cast323*((float)(((cast333*wmma92.z)+(cast334*wmma93.z)))))+(cast324*((float)(((cast335*wmma94.z)+(cast336*wmma95.z)))))));
    float alu711 = (alu414?alu710:(buf47+alu710));
    buf47 = alu711;
    unsigned int val324 = (*(buf0+alu195));
    if (alu414) {
      *(data0_5570560+(alu110+3153)) = buf48;
    }
    float alu716 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val324&65535u)))))))*((cast325*((float)(((cast329*wmma88.w)+(cast330*wmma89.w)))))+(cast326*((float)(((cast331*wmma90.w)+(cast332*wmma91.w)))))+(cast327*((float)(((cast333*wmma92.w)+(cast334*wmma93.w)))))+(cast328*((float)(((cast335*wmma94.w)+(cast336*wmma95.w)))))));
    float alu717 = (alu414?alu716:(buf48+alu716));
    buf48 = alu717;
    unsigned int val325 = (*(buf0+alu163));
    unsigned int val326 = (*(buf0+alu164));
    unsigned int val327 = (*(buf0+alu165));
    unsigned int val328 = (*(buf0+alu166));
    unsigned int val329 = (*(buf0+alu167));
    unsigned int val330 = (*(buf0+alu168));
    unsigned int val331 = (*(buf0+alu169));
    unsigned int val332 = (*(buf0+alu170));
    unsigned int val333 = (*(buf0+alu249));
    unsigned int val334 = (*(buf0+alu250));
    unsigned int val335 = (*(buf0+alu251));
    unsigned int val336 = (*(buf0+alu252));
    unsigned int val337 = (*(buf0+alu253));
    unsigned int val338 = (*(buf0+alu254));
    unsigned int val339 = (*(buf0+alu255));
    unsigned int val340 = (*(buf0+alu256));
    unsigned int val341 = (*(buf0+alu180));
    unsigned int val342 = (*(buf0+alu181));
    unsigned int val343 = (*(buf0+alu182));
    unsigned int val344 = (*(buf0+alu186));
    unsigned int val345 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+96)) = buf49;
    }
    char4 cast337 = make_char4(((signed char)(((val325>>0u)&255u))),((signed char)(((val325>>8u)&255u))),((signed char)(((val325>>16u)&255u))),((signed char)(((val325>>24u)&255u))));
    char4 cast338 = make_char4(((signed char)(((val326>>0u)&255u))),((signed char)(((val326>>8u)&255u))),((signed char)(((val326>>16u)&255u))),((signed char)(((val326>>24u)&255u))));
    char4 cast339 = make_char4(((signed char)(((val327>>0u)&255u))),((signed char)(((val327>>8u)&255u))),((signed char)(((val327>>16u)&255u))),((signed char)(((val327>>24u)&255u))));
    char4 cast340 = make_char4(((signed char)(((val328>>0u)&255u))),((signed char)(((val328>>8u)&255u))),((signed char)(((val328>>16u)&255u))),((signed char)(((val328>>24u)&255u))));
    char4 cast341 = make_char4(((signed char)(((val329>>0u)&255u))),((signed char)(((val329>>8u)&255u))),((signed char)(((val329>>16u)&255u))),((signed char)(((val329>>24u)&255u))));
    char4 cast342 = make_char4(((signed char)(((val330>>0u)&255u))),((signed char)(((val330>>8u)&255u))),((signed char)(((val330>>16u)&255u))),((signed char)(((val330>>24u)&255u))));
    char4 cast343 = make_char4(((signed char)(((val331>>0u)&255u))),((signed char)(((val331>>8u)&255u))),((signed char)(((val331>>16u)&255u))),((signed char)(((val331>>24u)&255u))));
    char4 cast344 = make_char4(((signed char)(((val332>>0u)&255u))),((signed char)(((val332>>8u)&255u))),((signed char)(((val332>>16u)&255u))),((signed char)(((val332>>24u)&255u))));
    int4 wmma96 = __WMMA_8_16_16_signed_char_int(alu418, cast338, cast0);
    int4 wmma97 = __WMMA_8_16_16_signed_char_int(alu419, cast339, cast0);
    int4 wmma98 = __WMMA_8_16_16_signed_char_int(alu420, cast340, cast0);
    int4 wmma99 = __WMMA_8_16_16_signed_char_int(alu421, cast341, cast0);
    int4 wmma100 = __WMMA_8_16_16_signed_char_int(alu422, cast342, cast0);
    int4 wmma101 = __WMMA_8_16_16_signed_char_int(alu423, cast343, cast0);
    int4 wmma102 = __WMMA_8_16_16_signed_char_int(alu424, cast344, cast0);
    int4 wmma103 = __WMMA_8_16_16_signed_char_int(alu425, cast337, cast0);
    int cast345 = ((int)(((signed char)(((val342>>0u)&255u)))));
    int cast346 = ((int)(((signed char)(((val342>>8u)&255u)))));
    int cast347 = ((int)(((signed char)(((val342>>16u)&255u)))));
    int cast348 = ((int)(((signed char)(((val342>>24u)&255u)))));
    int cast349 = ((int)(((signed char)(((val343>>0u)&255u)))));
    int cast350 = ((int)(((signed char)(((val343>>8u)&255u)))));
    int cast351 = ((int)(((signed char)(((val343>>16u)&255u)))));
    int cast352 = ((int)(((signed char)(((val343>>24u)&255u)))));
    float cast353 = tg_bitcast<float>((unsigned int)(val333));
    float cast354 = tg_bitcast<float>((unsigned int)(val334));
    float cast355 = tg_bitcast<float>((unsigned int)(val335));
    float cast356 = tg_bitcast<float>((unsigned int)(val336));
    float alu722 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val341&65535u)))))))*((cast353*((float)(((cast345*wmma103.x)+(cast346*wmma96.x)))))+(cast354*((float)(((cast347*wmma97.x)+(cast348*wmma98.x)))))+(cast355*((float)(((cast349*wmma99.x)+(cast350*wmma100.x)))))+(cast356*((float)(((cast351*wmma101.x)+(cast352*wmma102.x)))))));
    float alu723 = (alu414?alu722:(buf49+alu722));
    buf49 = alu723;
    unsigned int val346 = (*(buf0+alu180));
    if (alu414) {
      *(data0_5570560+(alu110+97)) = buf50;
    }
    float cast357 = tg_bitcast<float>((unsigned int)(val337));
    float cast358 = tg_bitcast<float>((unsigned int)(val338));
    float cast359 = tg_bitcast<float>((unsigned int)(val339));
    float cast360 = tg_bitcast<float>((unsigned int)(val340));
    float alu728 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val346&65535u)))))))*((cast357*((float)(((cast345*wmma103.y)+(cast346*wmma96.y)))))+(cast358*((float)(((cast347*wmma97.y)+(cast348*wmma98.y)))))+(cast359*((float)(((cast349*wmma99.y)+(cast350*wmma100.y)))))+(cast360*((float)(((cast351*wmma101.y)+(cast352*wmma102.y)))))));
    float alu729 = (alu414?alu728:(buf50+alu728));
    buf50 = alu729;
    unsigned int val347 = (*(buf0+alu185));
    if (alu414) {
      *(data0_5570560+(alu110+1120)) = buf51;
    }
    int cast361 = ((int)(((signed char)(((val344>>0u)&255u)))));
    int cast362 = ((int)(((signed char)(((val344>>8u)&255u)))));
    int cast363 = ((int)(((signed char)(((val344>>16u)&255u)))));
    int cast364 = ((int)(((signed char)(((val344>>24u)&255u)))));
    int cast365 = ((int)(((signed char)(((val345>>0u)&255u)))));
    int cast366 = ((int)(((signed char)(((val345>>8u)&255u)))));
    int cast367 = ((int)(((signed char)(((val345>>16u)&255u)))));
    int cast368 = ((int)(((signed char)(((val345>>24u)&255u)))));
    float alu734 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val347&65535u)))))))*((cast353*((float)(((cast361*wmma103.z)+(cast362*wmma96.z)))))+(cast354*((float)(((cast363*wmma97.z)+(cast364*wmma98.z)))))+(cast355*((float)(((cast365*wmma99.z)+(cast366*wmma100.z)))))+(cast356*((float)(((cast367*wmma101.z)+(cast368*wmma102.z)))))));
    float alu735 = (alu414?alu734:(buf51+alu734));
    buf51 = alu735;
    unsigned int val348 = (*(buf0+alu185));
    if (alu414) {
      *(data0_5570560+(alu110+1121)) = buf52;
    }
    float alu740 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val348&65535u)))))))*((cast357*((float)(((cast361*wmma103.w)+(cast362*wmma96.w)))))+(cast358*((float)(((cast363*wmma97.w)+(cast364*wmma98.w)))))+(cast359*((float)(((cast365*wmma99.w)+(cast366*wmma100.w)))))+(cast360*((float)(((cast367*wmma101.w)+(cast368*wmma102.w)))))));
    float alu741 = (alu414?alu740:(buf52+alu740));
    buf52 = alu741;
    unsigned int val349 = (*(buf0+alu249));
    unsigned int val350 = (*(buf0+alu250));
    unsigned int val351 = (*(buf0+alu251));
    unsigned int val352 = (*(buf0+alu252));
    unsigned int val353 = (*(buf0+alu253));
    unsigned int val354 = (*(buf0+alu254));
    unsigned int val355 = (*(buf0+alu255));
    unsigned int val356 = (*(buf0+alu256));
    unsigned int val357 = (*(buf0+alu190));
    unsigned int val358 = (*(buf0+alu191));
    unsigned int val359 = (*(buf0+alu192));
    unsigned int val360 = (*(buf0+alu196));
    unsigned int val361 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+2144)) = buf53;
    }
    int4 wmma104 = __WMMA_8_16_16_signed_char_int(alu450, cast337, cast0);
    int4 wmma105 = __WMMA_8_16_16_signed_char_int(alu451, cast338, cast0);
    int4 wmma106 = __WMMA_8_16_16_signed_char_int(alu452, cast339, cast0);
    int4 wmma107 = __WMMA_8_16_16_signed_char_int(alu453, cast340, cast0);
    int4 wmma108 = __WMMA_8_16_16_signed_char_int(alu454, cast341, cast0);
    int4 wmma109 = __WMMA_8_16_16_signed_char_int(alu455, cast342, cast0);
    int4 wmma110 = __WMMA_8_16_16_signed_char_int(alu456, cast343, cast0);
    int4 wmma111 = __WMMA_8_16_16_signed_char_int(alu457, cast344, cast0);
    int cast369 = ((int)(((signed char)(((val358>>0u)&255u)))));
    int cast370 = ((int)(((signed char)(((val358>>8u)&255u)))));
    int cast371 = ((int)(((signed char)(((val358>>16u)&255u)))));
    int cast372 = ((int)(((signed char)(((val358>>24u)&255u)))));
    int cast373 = ((int)(((signed char)(((val359>>0u)&255u)))));
    int cast374 = ((int)(((signed char)(((val359>>8u)&255u)))));
    int cast375 = ((int)(((signed char)(((val359>>16u)&255u)))));
    int cast376 = ((int)(((signed char)(((val359>>24u)&255u)))));
    float cast377 = tg_bitcast<float>((unsigned int)(val349));
    float cast378 = tg_bitcast<float>((unsigned int)(val350));
    float cast379 = tg_bitcast<float>((unsigned int)(val351));
    float cast380 = tg_bitcast<float>((unsigned int)(val352));
    float alu746 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val357&65535u)))))))*((cast377*((float)(((cast369*wmma104.x)+(cast370*wmma105.x)))))+(cast378*((float)(((cast371*wmma106.x)+(cast372*wmma107.x)))))+(cast379*((float)(((cast373*wmma108.x)+(cast374*wmma109.x)))))+(cast380*((float)(((cast375*wmma110.x)+(cast376*wmma111.x)))))));
    float alu747 = (alu414?alu746:(buf53+alu746));
    buf53 = alu747;
    unsigned int val362 = (*(buf0+alu190));
    if (alu414) {
      *(data0_5570560+(alu110+2145)) = buf54;
    }
    float cast381 = tg_bitcast<float>((unsigned int)(val353));
    float cast382 = tg_bitcast<float>((unsigned int)(val354));
    float cast383 = tg_bitcast<float>((unsigned int)(val355));
    float cast384 = tg_bitcast<float>((unsigned int)(val356));
    float alu752 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val362&65535u)))))))*((cast381*((float)(((cast369*wmma104.y)+(cast370*wmma105.y)))))+(cast382*((float)(((cast371*wmma106.y)+(cast372*wmma107.y)))))+(cast383*((float)(((cast373*wmma108.y)+(cast374*wmma109.y)))))+(cast384*((float)(((cast375*wmma110.y)+(cast376*wmma111.y)))))));
    float alu753 = (alu414?alu752:(buf54+alu752));
    buf54 = alu753;
    unsigned int val363 = (*(buf0+alu195));
    if (alu414) {
      *(data0_5570560+(alu110+3168)) = buf55;
    }
    int cast385 = ((int)(((signed char)(((val360>>0u)&255u)))));
    int cast386 = ((int)(((signed char)(((val360>>8u)&255u)))));
    int cast387 = ((int)(((signed char)(((val360>>16u)&255u)))));
    int cast388 = ((int)(((signed char)(((val360>>24u)&255u)))));
    int cast389 = ((int)(((signed char)(((val361>>0u)&255u)))));
    int cast390 = ((int)(((signed char)(((val361>>8u)&255u)))));
    int cast391 = ((int)(((signed char)(((val361>>16u)&255u)))));
    int cast392 = ((int)(((signed char)(((val361>>24u)&255u)))));
    float alu758 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val363&65535u)))))))*((cast377*((float)(((cast385*wmma104.z)+(cast386*wmma105.z)))))+(cast378*((float)(((cast387*wmma106.z)+(cast388*wmma107.z)))))+(cast379*((float)(((cast389*wmma108.z)+(cast390*wmma109.z)))))+(cast380*((float)(((cast391*wmma110.z)+(cast392*wmma111.z)))))));
    float alu759 = (alu414?alu758:(buf55+alu758));
    buf55 = alu759;
    unsigned int val364 = (*(buf0+alu195));
    if (alu414) {
      *(data0_5570560+(alu110+3169)) = buf56;
    }
    float alu764 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val364&65535u)))))))*((cast381*((float)(((cast385*wmma104.w)+(cast386*wmma105.w)))))+(cast382*((float)(((cast387*wmma106.w)+(cast388*wmma107.w)))))+(cast383*((float)(((cast389*wmma108.w)+(cast390*wmma109.w)))))+(cast384*((float)(((cast391*wmma110.w)+(cast392*wmma111.w)))))));
    float alu765 = (alu414?alu764:(buf56+alu764));
    buf56 = alu765;
    unsigned int val365 = (*(buf0+alu171));
    unsigned int val366 = (*(buf0+alu172));
    unsigned int val367 = (*(buf0+alu173));
    unsigned int val368 = (*(buf0+alu174));
    unsigned int val369 = (*(buf0+alu175));
    unsigned int val370 = (*(buf0+alu176));
    unsigned int val371 = (*(buf0+alu177));
    unsigned int val372 = (*(buf0+alu178));
    unsigned int val373 = (*(buf0+alu257));
    unsigned int val374 = (*(buf0+alu258));
    unsigned int val375 = (*(buf0+alu259));
    unsigned int val376 = (*(buf0+alu260));
    unsigned int val377 = (*(buf0+alu261));
    unsigned int val378 = (*(buf0+alu262));
    unsigned int val379 = (*(buf0+alu263));
    unsigned int val380 = (*(buf0+alu264));
    unsigned int val381 = (*(buf0+alu180));
    unsigned int val382 = (*(buf0+alu181));
    unsigned int val383 = (*(buf0+alu182));
    unsigned int val384 = (*(buf0+alu186));
    unsigned int val385 = (*(buf0+alu187));
    if (alu414) {
      *(data0_5570560+(alu110+112)) = buf57;
    }
    char4 cast393 = make_char4(((signed char)(((val365>>0u)&255u))),((signed char)(((val365>>8u)&255u))),((signed char)(((val365>>16u)&255u))),((signed char)(((val365>>24u)&255u))));
    char4 cast394 = make_char4(((signed char)(((val366>>0u)&255u))),((signed char)(((val366>>8u)&255u))),((signed char)(((val366>>16u)&255u))),((signed char)(((val366>>24u)&255u))));
    char4 cast395 = make_char4(((signed char)(((val367>>0u)&255u))),((signed char)(((val367>>8u)&255u))),((signed char)(((val367>>16u)&255u))),((signed char)(((val367>>24u)&255u))));
    char4 cast396 = make_char4(((signed char)(((val368>>0u)&255u))),((signed char)(((val368>>8u)&255u))),((signed char)(((val368>>16u)&255u))),((signed char)(((val368>>24u)&255u))));
    char4 cast397 = make_char4(((signed char)(((val369>>0u)&255u))),((signed char)(((val369>>8u)&255u))),((signed char)(((val369>>16u)&255u))),((signed char)(((val369>>24u)&255u))));
    char4 cast398 = make_char4(((signed char)(((val370>>0u)&255u))),((signed char)(((val370>>8u)&255u))),((signed char)(((val370>>16u)&255u))),((signed char)(((val370>>24u)&255u))));
    char4 cast399 = make_char4(((signed char)(((val371>>0u)&255u))),((signed char)(((val371>>8u)&255u))),((signed char)(((val371>>16u)&255u))),((signed char)(((val371>>24u)&255u))));
    char4 cast400 = make_char4(((signed char)(((val372>>0u)&255u))),((signed char)(((val372>>8u)&255u))),((signed char)(((val372>>16u)&255u))),((signed char)(((val372>>24u)&255u))));
    int4 wmma112 = __WMMA_8_16_16_signed_char_int(alu418, cast394, cast0);
    int4 wmma113 = __WMMA_8_16_16_signed_char_int(alu419, cast395, cast0);
    int4 wmma114 = __WMMA_8_16_16_signed_char_int(alu420, cast396, cast0);
    int4 wmma115 = __WMMA_8_16_16_signed_char_int(alu421, cast397, cast0);
    int4 wmma116 = __WMMA_8_16_16_signed_char_int(alu422, cast398, cast0);
    int4 wmma117 = __WMMA_8_16_16_signed_char_int(alu423, cast399, cast0);
    int4 wmma118 = __WMMA_8_16_16_signed_char_int(alu424, cast400, cast0);
    int4 wmma119 = __WMMA_8_16_16_signed_char_int(alu425, cast393, cast0);
    int cast401 = ((int)(((signed char)(((val382>>0u)&255u)))));
    int cast402 = ((int)(((signed char)(((val382>>8u)&255u)))));
    int cast403 = ((int)(((signed char)(((val382>>16u)&255u)))));
    int cast404 = ((int)(((signed char)(((val382>>24u)&255u)))));
    int cast405 = ((int)(((signed char)(((val383>>0u)&255u)))));
    int cast406 = ((int)(((signed char)(((val383>>8u)&255u)))));
    int cast407 = ((int)(((signed char)(((val383>>16u)&255u)))));
    int cast408 = ((int)(((signed char)(((val383>>24u)&255u)))));
    float cast409 = tg_bitcast<float>((unsigned int)(val373));
    float cast410 = tg_bitcast<float>((unsigned int)(val374));
    float cast411 = tg_bitcast<float>((unsigned int)(val375));
    float cast412 = tg_bitcast<float>((unsigned int)(val376));
    float alu770 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val381&65535u)))))))*((cast409*((float)(((cast401*wmma119.x)+(cast402*wmma112.x)))))+(cast410*((float)(((cast403*wmma113.x)+(cast404*wmma114.x)))))+(cast411*((float)(((cast405*wmma115.x)+(cast406*wmma116.x)))))+(cast412*((float)(((cast407*wmma117.x)+(cast408*wmma118.x)))))));
    float alu771 = (alu414?alu770:(buf57+alu770));
    buf57 = alu771;
    unsigned int val386 = (*(buf0+alu180));
    if (alu414) {
      *(data0_5570560+(alu110+113)) = buf58;
    }
    float cast413 = tg_bitcast<float>((unsigned int)(val377));
    float cast414 = tg_bitcast<float>((unsigned int)(val378));
    float cast415 = tg_bitcast<float>((unsigned int)(val379));
    float cast416 = tg_bitcast<float>((unsigned int)(val380));
    float alu776 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val386&65535u)))))))*((cast413*((float)(((cast401*wmma119.y)+(cast402*wmma112.y)))))+(cast414*((float)(((cast403*wmma113.y)+(cast404*wmma114.y)))))+(cast415*((float)(((cast405*wmma115.y)+(cast406*wmma116.y)))))+(cast416*((float)(((cast407*wmma117.y)+(cast408*wmma118.y)))))));
    float alu777 = (alu414?alu776:(buf58+alu776));
    buf58 = alu777;
    unsigned int val387 = (*(buf0+alu185));
    if (alu414) {
      *(data0_5570560+(alu110+1136)) = buf59;
    }
    int cast417 = ((int)(((signed char)(((val384>>0u)&255u)))));
    int cast418 = ((int)(((signed char)(((val384>>8u)&255u)))));
    int cast419 = ((int)(((signed char)(((val384>>16u)&255u)))));
    int cast420 = ((int)(((signed char)(((val384>>24u)&255u)))));
    int cast421 = ((int)(((signed char)(((val385>>0u)&255u)))));
    int cast422 = ((int)(((signed char)(((val385>>8u)&255u)))));
    int cast423 = ((int)(((signed char)(((val385>>16u)&255u)))));
    int cast424 = ((int)(((signed char)(((val385>>24u)&255u)))));
    float alu782 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val387&65535u)))))))*((cast409*((float)(((cast417*wmma119.z)+(cast418*wmma112.z)))))+(cast410*((float)(((cast419*wmma113.z)+(cast420*wmma114.z)))))+(cast411*((float)(((cast421*wmma115.z)+(cast422*wmma116.z)))))+(cast412*((float)(((cast423*wmma117.z)+(cast424*wmma118.z)))))));
    float alu783 = (alu414?alu782:(buf59+alu782));
    buf59 = alu783;
    unsigned int val388 = (*(buf0+alu185));
    if (alu414) {
      *(data0_5570560+(alu110+1137)) = buf60;
    }
    float alu788 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val388&65535u)))))))*((cast413*((float)(((cast417*wmma119.w)+(cast418*wmma112.w)))))+(cast414*((float)(((cast419*wmma113.w)+(cast420*wmma114.w)))))+(cast415*((float)(((cast421*wmma115.w)+(cast422*wmma116.w)))))+(cast416*((float)(((cast423*wmma117.w)+(cast424*wmma118.w)))))));
    float alu789 = (alu414?alu788:(buf60+alu788));
    buf60 = alu789;
    unsigned int val389 = (*(buf0+alu257));
    unsigned int val390 = (*(buf0+alu258));
    unsigned int val391 = (*(buf0+alu259));
    unsigned int val392 = (*(buf0+alu260));
    unsigned int val393 = (*(buf0+alu261));
    unsigned int val394 = (*(buf0+alu262));
    unsigned int val395 = (*(buf0+alu263));
    unsigned int val396 = (*(buf0+alu264));
    unsigned int val397 = (*(buf0+alu190));
    unsigned int val398 = (*(buf0+alu191));
    unsigned int val399 = (*(buf0+alu192));
    unsigned int val400 = (*(buf0+alu196));
    unsigned int val401 = (*(buf0+alu197));
    if (alu414) {
      *(data0_5570560+(alu110+2160)) = buf61;
    }
    int4 wmma120 = __WMMA_8_16_16_signed_char_int(alu450, cast393, cast0);
    int4 wmma121 = __WMMA_8_16_16_signed_char_int(alu451, cast394, cast0);
    int4 wmma122 = __WMMA_8_16_16_signed_char_int(alu452, cast395, cast0);
    int4 wmma123 = __WMMA_8_16_16_signed_char_int(alu453, cast396, cast0);
    int4 wmma124 = __WMMA_8_16_16_signed_char_int(alu454, cast397, cast0);
    int4 wmma125 = __WMMA_8_16_16_signed_char_int(alu455, cast398, cast0);
    int4 wmma126 = __WMMA_8_16_16_signed_char_int(alu456, cast399, cast0);
    int4 wmma127 = __WMMA_8_16_16_signed_char_int(alu457, cast400, cast0);
    int cast425 = ((int)(((signed char)(((val398>>0u)&255u)))));
    int cast426 = ((int)(((signed char)(((val398>>8u)&255u)))));
    int cast427 = ((int)(((signed char)(((val398>>16u)&255u)))));
    int cast428 = ((int)(((signed char)(((val398>>24u)&255u)))));
    int cast429 = ((int)(((signed char)(((val399>>0u)&255u)))));
    int cast430 = ((int)(((signed char)(((val399>>8u)&255u)))));
    int cast431 = ((int)(((signed char)(((val399>>16u)&255u)))));
    int cast432 = ((int)(((signed char)(((val399>>24u)&255u)))));
    float cast433 = tg_bitcast<float>((unsigned int)(val389));
    float cast434 = tg_bitcast<float>((unsigned int)(val390));
    float cast435 = tg_bitcast<float>((unsigned int)(val391));
    float cast436 = tg_bitcast<float>((unsigned int)(val392));
    float alu794 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val397&65535u)))))))*((cast433*((float)(((cast425*wmma120.x)+(cast426*wmma121.x)))))+(cast434*((float)(((cast427*wmma122.x)+(cast428*wmma123.x)))))+(cast435*((float)(((cast429*wmma124.x)+(cast430*wmma125.x)))))+(cast436*((float)(((cast431*wmma126.x)+(cast432*wmma127.x)))))));
    float alu795 = (alu414?alu794:(buf61+alu794));
    buf61 = alu795;
    unsigned int val402 = (*(buf0+alu190));
    if (alu414) {
      *(data0_5570560+(alu110+2161)) = buf62;
    }
    float cast437 = tg_bitcast<float>((unsigned int)(val393));
    float cast438 = tg_bitcast<float>((unsigned int)(val394));
    float cast439 = tg_bitcast<float>((unsigned int)(val395));
    float cast440 = tg_bitcast<float>((unsigned int)(val396));
    float alu800 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val402&65535u)))))))*((cast437*((float)(((cast425*wmma120.y)+(cast426*wmma121.y)))))+(cast438*((float)(((cast427*wmma122.y)+(cast428*wmma123.y)))))+(cast439*((float)(((cast429*wmma124.y)+(cast430*wmma125.y)))))+(cast440*((float)(((cast431*wmma126.y)+(cast432*wmma127.y)))))));
    float alu801 = (alu414?alu800:(buf62+alu800));
    buf62 = alu801;
    unsigned int val403 = (*(buf0+alu195));
    if (alu414) {
      *(data0_5570560+(alu110+3184)) = buf63;
    }
    int cast441 = ((int)(((signed char)(((val400>>0u)&255u)))));
    int cast442 = ((int)(((signed char)(((val400>>8u)&255u)))));
    int cast443 = ((int)(((signed char)(((val400>>16u)&255u)))));
    int cast444 = ((int)(((signed char)(((val400>>24u)&255u)))));
    int cast445 = ((int)(((signed char)(((val401>>0u)&255u)))));
    int cast446 = ((int)(((signed char)(((val401>>8u)&255u)))));
    int cast447 = ((int)(((signed char)(((val401>>16u)&255u)))));
    int cast448 = ((int)(((signed char)(((val401>>24u)&255u)))));
    float alu806 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val403&65535u)))))))*((cast433*((float)(((cast441*wmma120.z)+(cast442*wmma121.z)))))+(cast434*((float)(((cast443*wmma122.z)+(cast444*wmma123.z)))))+(cast435*((float)(((cast445*wmma124.z)+(cast446*wmma125.z)))))+(cast436*((float)(((cast447*wmma126.z)+(cast448*wmma127.z)))))));
    float alu807 = (alu414?alu806:(buf63+alu806));
    buf63 = alu807;
    unsigned int val404 = (*(buf0+alu195));
    if (alu414) {
      *(data0_5570560+(alu110+3185)) = buf64;
    }
    float alu812 = (((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val404&65535u)))))))*((cast437*((float)(((cast441*wmma120.w)+(cast442*wmma121.w)))))+(cast438*((float)(((cast443*wmma122.w)+(cast444*wmma123.w)))))+(cast439*((float)(((cast445*wmma124.w)+(cast446*wmma125.w)))))+(cast440*((float)(((cast447*wmma126.w)+(cast448*wmma127.w)))))));
    float alu813 = (alu414?alu812:(buf64+alu812));
    buf64 = alu813;
    __syncthreads();
    if (alu271) {
      *(buf0+alu68) = val83;
    }
    if (alu271) {
      *(buf0+alu69) = val13;
    }
    if (alu271) {
      *(buf0+alu70) = val14;
    }
    if (alu271) {
      *(buf0+alu71) = val15;
    }
    if (alu271) {
      *(buf0+alu72) = val16;
    }
    if (alu271) {
      *(buf0+alu73) = val17;
    }
    if (alu271) {
      *(buf0+alu74) = val18;
    }
    if (alu271) {
      *(buf0+alu75) = val19;
    }
    if (alu271) {
      *(buf0+alu76) = val20;
    }
    if (alu271) {
      *(buf0+alu77) = val21;
    }
    if (alu271) {
      *(buf0+alu78) = val22;
    }
    if (alu271) {
      *(buf0+alu79) = val23;
    }
    if (alu271) {
      *(buf0+alu80) = val24;
    }
    if (alu271) {
      *(buf0+alu81) = val25;
    }
    if (alu271) {
      *(buf0+alu82) = val26;
    }
    if (alu271) {
      *(buf0+alu83) = val27;
    }
    if (alu271) {
      *(buf0+alu84) = val28;
    }
    if (alu271) {
      *(buf0+alu85) = val29;
    }
    if (alu271) {
      *(buf0+alu86) = val30;
    }
    if (alu271) {
      *(buf0+alu87) = val31;
    }
    if (alu271) {
      *(buf0+alu88) = val32;
    }
    if (alu271) {
      *(buf0+alu89) = val33;
    }
    if (alu271) {
      *(buf0+alu90) = val34;
    }
    if (alu271) {
      *(buf0+alu91) = val35;
    }
    if (alu271) {
      *(buf0+alu92) = val36;
    }
    if (alu271) {
      *(buf0+alu93) = val37;
    }
    if (alu271) {
      *(buf0+alu94) = val38;
    }
    if (alu271) {
      *(buf0+alu95) = val39;
    }
    if (alu271) {
      *(buf0+alu96) = val40;
    }
    if (alu271) {
      *(buf0+alu97) = val41;
    }
    if (alu271) {
      *(buf0+alu98) = val42;
    }
    if (alu271) {
      *(buf0+alu99) = val43;
    }
    if (alu271) {
      *(buf0+alu100) = val44;
    }
    if (alu271) {
      *(buf0+alu101) = val45;
    }
    if (alu271) {
      *(buf0+alu102) = val46;
    }
    if (alu271) {
      *(buf0+alu103) = val47;
    }
    __syncthreads();
    unsigned int val405 = (*(buf0+alu115));
    unsigned int val406 = (*(buf0+alu116));
    unsigned int val407 = (*(buf0+alu117));
    unsigned int val408 = (*(buf0+alu118));
    unsigned int val409 = (*(buf0+alu119));
    unsigned int val410 = (*(buf0+alu120));
    unsigned int val411 = (*(buf0+alu121));
    unsigned int val412 = (*(buf0+alu122));
    unsigned int val413 = (*(buf0+alu201));
    unsigned int val414 = (*(buf0+alu202));
    unsigned int val415 = (*(buf0+alu203));
    unsigned int val416 = (*(buf0+alu204));
    unsigned int val417 = (*(buf0+alu205));
    unsigned int val418 = (*(buf0+alu206));
    unsigned int val419 = (*(buf0+alu207));
    unsigned int val420 = (*(buf0+alu208));
    unsigned int val421 = (*(buf0+alu180));
    unsigned int val422 = (*(buf0+alu183));
    unsigned int val423 = (*(buf0+alu184));
    if (0) {
      *(data0_5570560+alu110) = buf1;
    }
    char4 cast449 = make_char4(((signed char)(((val405>>0u)&255u))),((signed char)(((val405>>8u)&255u))),((signed char)(((val405>>16u)&255u))),((signed char)(((val405>>24u)&255u))));
    char4 cast450 = make_char4(((signed char)(((val406>>0u)&255u))),((signed char)(((val406>>8u)&255u))),((signed char)(((val406>>16u)&255u))),((signed char)(((val406>>24u)&255u))));
    char4 cast451 = make_char4(((signed char)(((val407>>0u)&255u))),((signed char)(((val407>>8u)&255u))),((signed char)(((val407>>16u)&255u))),((signed char)(((val407>>24u)&255u))));
    char4 cast452 = make_char4(((signed char)(((val408>>0u)&255u))),((signed char)(((val408>>8u)&255u))),((signed char)(((val408>>16u)&255u))),((signed char)(((val408>>24u)&255u))));
    char4 cast453 = make_char4(((signed char)(((val409>>0u)&255u))),((signed char)(((val409>>8u)&255u))),((signed char)(((val409>>16u)&255u))),((signed char)(((val409>>24u)&255u))));
    char4 cast454 = make_char4(((signed char)(((val410>>0u)&255u))),((signed char)(((val410>>8u)&255u))),((signed char)(((val410>>16u)&255u))),((signed char)(((val410>>24u)&255u))));
    char4 cast455 = make_char4(((signed char)(((val411>>0u)&255u))),((signed char)(((val411>>8u)&255u))),((signed char)(((val411>>16u)&255u))),((signed char)(((val411>>24u)&255u))));
    char4 cast456 = make_char4(((signed char)(((val412>>0u)&255u))),((signed char)(((val412>>8u)&255u))),((signed char)(((val412>>16u)&255u))),((signed char)(((val412>>24u)&255u))));
    signed_char8 alu928 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+32))))*4)));
    int4 wmma128 = __WMMA_8_16_16_signed_char_int(alu928, cast449, cast0);
    signed_char8 alu929 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+36))))*4)));
    int4 wmma129 = __WMMA_8_16_16_signed_char_int(alu929, cast450, cast0);
    signed_char8 alu930 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+40))))*4)));
    int4 wmma130 = __WMMA_8_16_16_signed_char_int(alu930, cast451, cast0);
    signed_char8 alu931 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+44))))*4)));
    int4 wmma131 = __WMMA_8_16_16_signed_char_int(alu931, cast452, cast0);
    signed_char8 alu932 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+48))))*4)));
    int4 wmma132 = __WMMA_8_16_16_signed_char_int(alu932, cast453, cast0);
    signed_char8 alu933 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+52))))*4)));
    int4 wmma133 = __WMMA_8_16_16_signed_char_int(alu933, cast454, cast0);
    signed_char8 alu934 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+56))))*4)));
    int4 wmma134 = __WMMA_8_16_16_signed_char_int(alu934, cast455, cast0);
    signed_char8 alu935 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+60))))*4)));
    int4 wmma135 = __WMMA_8_16_16_signed_char_int(alu935, cast456, cast0);
    int cast457 = ((int)(((signed char)(((val422>>0u)&255u)))));
    int cast458 = ((int)(((signed char)(((val422>>8u)&255u)))));
    int cast459 = ((int)(((signed char)(((val422>>16u)&255u)))));
    int cast460 = ((int)(((signed char)(((val422>>24u)&255u)))));
    int cast461 = ((int)(((signed char)(((val423>>0u)&255u)))));
    int cast462 = ((int)(((signed char)(((val423>>8u)&255u)))));
    int cast463 = ((int)(((signed char)(((val423>>16u)&255u)))));
    int cast464 = ((int)(((signed char)(((val423>>24u)&255u)))));
    float cast465 = tg_bitcast<float>((unsigned int)(val413));
    float cast466 = tg_bitcast<float>((unsigned int)(val414));
    float cast467 = tg_bitcast<float>((unsigned int)(val415));
    float cast468 = tg_bitcast<float>((unsigned int)(val416));
    buf1 = (buf1+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val421&65535u)))))))*((cast465*((float)(((cast457*wmma128.x)+(cast458*wmma129.x)))))+(cast466*((float)(((cast459*wmma130.x)+(cast460*wmma131.x)))))+(cast467*((float)(((cast461*wmma132.x)+(cast462*wmma133.x)))))+(cast468*((float)(((cast463*wmma134.x)+(cast464*wmma135.x))))))));
    unsigned int val424 = (*(buf0+alu180));
    if (0) {
      *(data0_5570560+(alu110+1)) = buf2;
    }
    float cast469 = tg_bitcast<float>((unsigned int)(val417));
    float cast470 = tg_bitcast<float>((unsigned int)(val418));
    float cast471 = tg_bitcast<float>((unsigned int)(val419));
    float cast472 = tg_bitcast<float>((unsigned int)(val420));
    buf2 = (buf2+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val424&65535u)))))))*((cast469*((float)(((cast457*wmma128.y)+(cast458*wmma129.y)))))+(cast470*((float)(((cast459*wmma130.y)+(cast460*wmma131.y)))))+(cast471*((float)(((cast461*wmma132.y)+(cast462*wmma133.y)))))+(cast472*((float)(((cast463*wmma134.y)+(cast464*wmma135.y))))))));
    unsigned int val425 = (*(buf0+alu185));
    unsigned int val426 = (*(buf0+alu188));
    unsigned int val427 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+1024)) = buf3;
    }
    int cast473 = ((int)(((signed char)(((val426>>0u)&255u)))));
    int cast474 = ((int)(((signed char)(((val426>>8u)&255u)))));
    int cast475 = ((int)(((signed char)(((val426>>16u)&255u)))));
    int cast476 = ((int)(((signed char)(((val426>>24u)&255u)))));
    int cast477 = ((int)(((signed char)(((val427>>0u)&255u)))));
    int cast478 = ((int)(((signed char)(((val427>>8u)&255u)))));
    int cast479 = ((int)(((signed char)(((val427>>16u)&255u)))));
    int cast480 = ((int)(((signed char)(((val427>>24u)&255u)))));
    buf3 = (buf3+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val425&65535u)))))))*((cast465*((float)(((cast473*wmma128.z)+(cast474*wmma129.z)))))+(cast466*((float)(((cast475*wmma130.z)+(cast476*wmma131.z)))))+(cast467*((float)(((cast477*wmma132.z)+(cast478*wmma133.z)))))+(cast468*((float)(((cast479*wmma134.z)+(cast480*wmma135.z))))))));
    unsigned int val428 = (*(buf0+alu185));
    if (0) {
      *(data0_5570560+(alu110+1025)) = buf4;
    }
    buf4 = (buf4+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val428&65535u)))))))*((cast469*((float)(((cast473*wmma128.w)+(cast474*wmma129.w)))))+(cast470*((float)(((cast475*wmma130.w)+(cast476*wmma131.w)))))+(cast471*((float)(((cast477*wmma132.w)+(cast478*wmma133.w)))))+(cast472*((float)(((cast479*wmma134.w)+(cast480*wmma135.w))))))));
    unsigned int val429 = (*(buf0+alu201));
    unsigned int val430 = (*(buf0+alu202));
    unsigned int val431 = (*(buf0+alu203));
    unsigned int val432 = (*(buf0+alu204));
    unsigned int val433 = (*(buf0+alu205));
    unsigned int val434 = (*(buf0+alu206));
    unsigned int val435 = (*(buf0+alu207));
    unsigned int val436 = (*(buf0+alu208));
    unsigned int val437 = (*(buf0+alu190));
    unsigned int val438 = (*(buf0+alu193));
    unsigned int val439 = (*(buf0+alu194));
    unsigned int val440 = (*(buf0+alu198));
    unsigned int val441 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+2048)) = buf5;
    }
    signed_char8 alu952 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1248))))*4)));
    int4 wmma136 = __WMMA_8_16_16_signed_char_int(alu952, cast449, cast0);
    signed_char8 alu953 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1252))))*4)));
    int4 wmma137 = __WMMA_8_16_16_signed_char_int(alu953, cast450, cast0);
    signed_char8 alu954 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1256))))*4)));
    int4 wmma138 = __WMMA_8_16_16_signed_char_int(alu954, cast451, cast0);
    signed_char8 alu955 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1260))))*4)));
    int4 wmma139 = __WMMA_8_16_16_signed_char_int(alu955, cast452, cast0);
    signed_char8 alu956 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1264))))*4)));
    int4 wmma140 = __WMMA_8_16_16_signed_char_int(alu956, cast453, cast0);
    signed_char8 alu957 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1268))))*4)));
    int4 wmma141 = __WMMA_8_16_16_signed_char_int(alu957, cast454, cast0);
    signed_char8 alu958 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1272))))*4)));
    int4 wmma142 = __WMMA_8_16_16_signed_char_int(alu958, cast455, cast0);
    signed_char8 alu959 = tg_bitcast<signed_char8>(tg_ldmatrix_x2((const void*)((const char*)(buf0)+(((int)((alu112+1276))))*4)));
    int4 wmma143 = __WMMA_8_16_16_signed_char_int(alu959, cast456, cast0);
    int cast481 = ((int)(((signed char)(((val438>>0u)&255u)))));
    int cast482 = ((int)(((signed char)(((val438>>8u)&255u)))));
    int cast483 = ((int)(((signed char)(((val438>>16u)&255u)))));
    int cast484 = ((int)(((signed char)(((val438>>24u)&255u)))));
    int cast485 = ((int)(((signed char)(((val439>>0u)&255u)))));
    int cast486 = ((int)(((signed char)(((val439>>8u)&255u)))));
    int cast487 = ((int)(((signed char)(((val439>>16u)&255u)))));
    int cast488 = ((int)(((signed char)(((val439>>24u)&255u)))));
    float cast489 = tg_bitcast<float>((unsigned int)(val429));
    float cast490 = tg_bitcast<float>((unsigned int)(val430));
    float cast491 = tg_bitcast<float>((unsigned int)(val431));
    float cast492 = tg_bitcast<float>((unsigned int)(val432));
    buf5 = (buf5+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val437&65535u)))))))*((cast489*((float)(((cast481*wmma136.x)+(cast482*wmma137.x)))))+(cast490*((float)(((cast483*wmma138.x)+(cast484*wmma139.x)))))+(cast491*((float)(((cast485*wmma140.x)+(cast486*wmma141.x)))))+(cast492*((float)(((cast487*wmma142.x)+(cast488*wmma143.x))))))));
    unsigned int val442 = (*(buf0+alu190));
    if (0) {
      *(data0_5570560+(alu110+2049)) = buf6;
    }
    float cast493 = tg_bitcast<float>((unsigned int)(val433));
    float cast494 = tg_bitcast<float>((unsigned int)(val434));
    float cast495 = tg_bitcast<float>((unsigned int)(val435));
    float cast496 = tg_bitcast<float>((unsigned int)(val436));
    buf6 = (buf6+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val442&65535u)))))))*((cast493*((float)(((cast481*wmma136.y)+(cast482*wmma137.y)))))+(cast494*((float)(((cast483*wmma138.y)+(cast484*wmma139.y)))))+(cast495*((float)(((cast485*wmma140.y)+(cast486*wmma141.y)))))+(cast496*((float)(((cast487*wmma142.y)+(cast488*wmma143.y))))))));
    unsigned int val443 = (*(buf0+alu195));
    if (0) {
      *(data0_5570560+(alu110+3072)) = buf7;
    }
    int cast497 = ((int)(((signed char)(((val440>>0u)&255u)))));
    int cast498 = ((int)(((signed char)(((val440>>8u)&255u)))));
    int cast499 = ((int)(((signed char)(((val440>>16u)&255u)))));
    int cast500 = ((int)(((signed char)(((val440>>24u)&255u)))));
    int cast501 = ((int)(((signed char)(((val441>>0u)&255u)))));
    int cast502 = ((int)(((signed char)(((val441>>8u)&255u)))));
    int cast503 = ((int)(((signed char)(((val441>>16u)&255u)))));
    int cast504 = ((int)(((signed char)(((val441>>24u)&255u)))));
    buf7 = (buf7+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val443&65535u)))))))*((cast489*((float)(((cast497*wmma136.z)+(cast498*wmma137.z)))))+(cast490*((float)(((cast499*wmma138.z)+(cast500*wmma139.z)))))+(cast491*((float)(((cast501*wmma140.z)+(cast502*wmma141.z)))))+(cast492*((float)(((cast503*wmma142.z)+(cast504*wmma143.z))))))));
    unsigned int val444 = (*(buf0+alu195));
    if (0) {
      *(data0_5570560+(alu110+3073)) = buf8;
    }
    buf8 = (buf8+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val444&65535u)))))))*((cast493*((float)(((cast497*wmma136.w)+(cast498*wmma137.w)))))+(cast494*((float)(((cast499*wmma138.w)+(cast500*wmma139.w)))))+(cast495*((float)(((cast501*wmma140.w)+(cast502*wmma141.w)))))+(cast496*((float)(((cast503*wmma142.w)+(cast504*wmma143.w))))))));
    unsigned int val445 = (*(buf0+alu123));
    unsigned int val446 = (*(buf0+alu124));
    unsigned int val447 = (*(buf0+alu125));
    unsigned int val448 = (*(buf0+alu126));
    unsigned int val449 = (*(buf0+alu127));
    unsigned int val450 = (*(buf0+alu128));
    unsigned int val451 = (*(buf0+alu129));
    unsigned int val452 = (*(buf0+alu130));
    unsigned int val453 = (*(buf0+alu209));
    unsigned int val454 = (*(buf0+alu210));
    unsigned int val455 = (*(buf0+alu211));
    unsigned int val456 = (*(buf0+alu212));
    unsigned int val457 = (*(buf0+alu213));
    unsigned int val458 = (*(buf0+alu214));
    unsigned int val459 = (*(buf0+alu215));
    unsigned int val460 = (*(buf0+alu216));
    unsigned int val461 = (*(buf0+alu180));
    unsigned int val462 = (*(buf0+alu183));
    unsigned int val463 = (*(buf0+alu184));
    unsigned int val464 = (*(buf0+alu188));
    unsigned int val465 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+16)) = buf9;
    }
    char4 cast505 = make_char4(((signed char)(((val445>>0u)&255u))),((signed char)(((val445>>8u)&255u))),((signed char)(((val445>>16u)&255u))),((signed char)(((val445>>24u)&255u))));
    char4 cast506 = make_char4(((signed char)(((val446>>0u)&255u))),((signed char)(((val446>>8u)&255u))),((signed char)(((val446>>16u)&255u))),((signed char)(((val446>>24u)&255u))));
    char4 cast507 = make_char4(((signed char)(((val447>>0u)&255u))),((signed char)(((val447>>8u)&255u))),((signed char)(((val447>>16u)&255u))),((signed char)(((val447>>24u)&255u))));
    char4 cast508 = make_char4(((signed char)(((val448>>0u)&255u))),((signed char)(((val448>>8u)&255u))),((signed char)(((val448>>16u)&255u))),((signed char)(((val448>>24u)&255u))));
    char4 cast509 = make_char4(((signed char)(((val449>>0u)&255u))),((signed char)(((val449>>8u)&255u))),((signed char)(((val449>>16u)&255u))),((signed char)(((val449>>24u)&255u))));
    char4 cast510 = make_char4(((signed char)(((val450>>0u)&255u))),((signed char)(((val450>>8u)&255u))),((signed char)(((val450>>16u)&255u))),((signed char)(((val450>>24u)&255u))));
    char4 cast511 = make_char4(((signed char)(((val451>>0u)&255u))),((signed char)(((val451>>8u)&255u))),((signed char)(((val451>>16u)&255u))),((signed char)(((val451>>24u)&255u))));
    char4 cast512 = make_char4(((signed char)(((val452>>0u)&255u))),((signed char)(((val452>>8u)&255u))),((signed char)(((val452>>16u)&255u))),((signed char)(((val452>>24u)&255u))));
    int4 wmma144 = __WMMA_8_16_16_signed_char_int(alu928, cast505, cast0);
    int4 wmma145 = __WMMA_8_16_16_signed_char_int(alu929, cast506, cast0);
    int4 wmma146 = __WMMA_8_16_16_signed_char_int(alu930, cast507, cast0);
    int4 wmma147 = __WMMA_8_16_16_signed_char_int(alu931, cast508, cast0);
    int4 wmma148 = __WMMA_8_16_16_signed_char_int(alu932, cast509, cast0);
    int4 wmma149 = __WMMA_8_16_16_signed_char_int(alu933, cast510, cast0);
    int4 wmma150 = __WMMA_8_16_16_signed_char_int(alu934, cast511, cast0);
    int4 wmma151 = __WMMA_8_16_16_signed_char_int(alu935, cast512, cast0);
    int cast513 = ((int)(((signed char)(((val462>>0u)&255u)))));
    int cast514 = ((int)(((signed char)(((val462>>8u)&255u)))));
    int cast515 = ((int)(((signed char)(((val462>>16u)&255u)))));
    int cast516 = ((int)(((signed char)(((val462>>24u)&255u)))));
    int cast517 = ((int)(((signed char)(((val463>>0u)&255u)))));
    int cast518 = ((int)(((signed char)(((val463>>8u)&255u)))));
    int cast519 = ((int)(((signed char)(((val463>>16u)&255u)))));
    int cast520 = ((int)(((signed char)(((val463>>24u)&255u)))));
    float cast521 = tg_bitcast<float>((unsigned int)(val453));
    float cast522 = tg_bitcast<float>((unsigned int)(val454));
    float cast523 = tg_bitcast<float>((unsigned int)(val455));
    float cast524 = tg_bitcast<float>((unsigned int)(val456));
    buf9 = (buf9+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val461&65535u)))))))*((cast521*((float)(((cast513*wmma144.x)+(cast514*wmma145.x)))))+(cast522*((float)(((cast515*wmma146.x)+(cast516*wmma147.x)))))+(cast523*((float)(((cast517*wmma148.x)+(cast518*wmma149.x)))))+(cast524*((float)(((cast519*wmma150.x)+(cast520*wmma151.x))))))));
    unsigned int val466 = (*(buf0+alu180));
    if (0) {
      *(data0_5570560+(alu110+17)) = buf10;
    }
    float cast525 = tg_bitcast<float>((unsigned int)(val457));
    float cast526 = tg_bitcast<float>((unsigned int)(val458));
    float cast527 = tg_bitcast<float>((unsigned int)(val459));
    float cast528 = tg_bitcast<float>((unsigned int)(val460));
    buf10 = (buf10+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val466&65535u)))))))*((cast525*((float)(((cast513*wmma144.y)+(cast514*wmma145.y)))))+(cast526*((float)(((cast515*wmma146.y)+(cast516*wmma147.y)))))+(cast527*((float)(((cast517*wmma148.y)+(cast518*wmma149.y)))))+(cast528*((float)(((cast519*wmma150.y)+(cast520*wmma151.y))))))));
    unsigned int val467 = (*(buf0+alu185));
    if (0) {
      *(data0_5570560+(alu110+1040)) = buf11;
    }
    int cast529 = ((int)(((signed char)(((val464>>0u)&255u)))));
    int cast530 = ((int)(((signed char)(((val464>>8u)&255u)))));
    int cast531 = ((int)(((signed char)(((val464>>16u)&255u)))));
    int cast532 = ((int)(((signed char)(((val464>>24u)&255u)))));
    int cast533 = ((int)(((signed char)(((val465>>0u)&255u)))));
    int cast534 = ((int)(((signed char)(((val465>>8u)&255u)))));
    int cast535 = ((int)(((signed char)(((val465>>16u)&255u)))));
    int cast536 = ((int)(((signed char)(((val465>>24u)&255u)))));
    buf11 = (buf11+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val467&65535u)))))))*((cast521*((float)(((cast529*wmma144.z)+(cast530*wmma145.z)))))+(cast522*((float)(((cast531*wmma146.z)+(cast532*wmma147.z)))))+(cast523*((float)(((cast533*wmma148.z)+(cast534*wmma149.z)))))+(cast524*((float)(((cast535*wmma150.z)+(cast536*wmma151.z))))))));
    unsigned int val468 = (*(buf0+alu185));
    if (0) {
      *(data0_5570560+(alu110+1041)) = buf12;
    }
    buf12 = (buf12+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val468&65535u)))))))*((cast525*((float)(((cast529*wmma144.w)+(cast530*wmma145.w)))))+(cast526*((float)(((cast531*wmma146.w)+(cast532*wmma147.w)))))+(cast527*((float)(((cast533*wmma148.w)+(cast534*wmma149.w)))))+(cast528*((float)(((cast535*wmma150.w)+(cast536*wmma151.w))))))));
    unsigned int val469 = (*(buf0+alu209));
    unsigned int val470 = (*(buf0+alu210));
    unsigned int val471 = (*(buf0+alu211));
    unsigned int val472 = (*(buf0+alu212));
    unsigned int val473 = (*(buf0+alu213));
    unsigned int val474 = (*(buf0+alu214));
    unsigned int val475 = (*(buf0+alu215));
    unsigned int val476 = (*(buf0+alu216));
    unsigned int val477 = (*(buf0+alu190));
    unsigned int val478 = (*(buf0+alu193));
    unsigned int val479 = (*(buf0+alu194));
    unsigned int val480 = (*(buf0+alu198));
    unsigned int val481 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+2064)) = buf13;
    }
    int4 wmma152 = __WMMA_8_16_16_signed_char_int(alu952, cast505, cast0);
    int4 wmma153 = __WMMA_8_16_16_signed_char_int(alu953, cast506, cast0);
    int4 wmma154 = __WMMA_8_16_16_signed_char_int(alu954, cast507, cast0);
    int4 wmma155 = __WMMA_8_16_16_signed_char_int(alu955, cast508, cast0);
    int4 wmma156 = __WMMA_8_16_16_signed_char_int(alu956, cast509, cast0);
    int4 wmma157 = __WMMA_8_16_16_signed_char_int(alu957, cast510, cast0);
    int4 wmma158 = __WMMA_8_16_16_signed_char_int(alu958, cast511, cast0);
    int4 wmma159 = __WMMA_8_16_16_signed_char_int(alu959, cast512, cast0);
    int cast537 = ((int)(((signed char)(((val478>>0u)&255u)))));
    int cast538 = ((int)(((signed char)(((val478>>8u)&255u)))));
    int cast539 = ((int)(((signed char)(((val478>>16u)&255u)))));
    int cast540 = ((int)(((signed char)(((val478>>24u)&255u)))));
    int cast541 = ((int)(((signed char)(((val479>>0u)&255u)))));
    int cast542 = ((int)(((signed char)(((val479>>8u)&255u)))));
    int cast543 = ((int)(((signed char)(((val479>>16u)&255u)))));
    int cast544 = ((int)(((signed char)(((val479>>24u)&255u)))));
    float cast545 = tg_bitcast<float>((unsigned int)(val469));
    float cast546 = tg_bitcast<float>((unsigned int)(val470));
    float cast547 = tg_bitcast<float>((unsigned int)(val471));
    float cast548 = tg_bitcast<float>((unsigned int)(val472));
    buf13 = (buf13+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val477&65535u)))))))*((cast545*((float)(((cast537*wmma152.x)+(cast538*wmma153.x)))))+(cast546*((float)(((cast539*wmma154.x)+(cast540*wmma155.x)))))+(cast547*((float)(((cast541*wmma156.x)+(cast542*wmma157.x)))))+(cast548*((float)(((cast543*wmma158.x)+(cast544*wmma159.x))))))));
    unsigned int val482 = (*(buf0+alu190));
    if (0) {
      *(data0_5570560+(alu110+2065)) = buf14;
    }
    float cast549 = tg_bitcast<float>((unsigned int)(val473));
    float cast550 = tg_bitcast<float>((unsigned int)(val474));
    float cast551 = tg_bitcast<float>((unsigned int)(val475));
    float cast552 = tg_bitcast<float>((unsigned int)(val476));
    buf14 = (buf14+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val482&65535u)))))))*((cast549*((float)(((cast537*wmma152.y)+(cast538*wmma153.y)))))+(cast550*((float)(((cast539*wmma154.y)+(cast540*wmma155.y)))))+(cast551*((float)(((cast541*wmma156.y)+(cast542*wmma157.y)))))+(cast552*((float)(((cast543*wmma158.y)+(cast544*wmma159.y))))))));
    unsigned int val483 = (*(buf0+alu195));
    if (0) {
      *(data0_5570560+(alu110+3088)) = buf15;
    }
    int cast553 = ((int)(((signed char)(((val480>>0u)&255u)))));
    int cast554 = ((int)(((signed char)(((val480>>8u)&255u)))));
    int cast555 = ((int)(((signed char)(((val480>>16u)&255u)))));
    int cast556 = ((int)(((signed char)(((val480>>24u)&255u)))));
    int cast557 = ((int)(((signed char)(((val481>>0u)&255u)))));
    int cast558 = ((int)(((signed char)(((val481>>8u)&255u)))));
    int cast559 = ((int)(((signed char)(((val481>>16u)&255u)))));
    int cast560 = ((int)(((signed char)(((val481>>24u)&255u)))));
    buf15 = (buf15+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val483&65535u)))))))*((cast545*((float)(((cast553*wmma152.z)+(cast554*wmma153.z)))))+(cast546*((float)(((cast555*wmma154.z)+(cast556*wmma155.z)))))+(cast547*((float)(((cast557*wmma156.z)+(cast558*wmma157.z)))))+(cast548*((float)(((cast559*wmma158.z)+(cast560*wmma159.z))))))));
    unsigned int val484 = (*(buf0+alu195));
    if (0) {
      *(data0_5570560+(alu110+3089)) = buf16;
    }
    buf16 = (buf16+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val484&65535u)))))))*((cast549*((float)(((cast553*wmma152.w)+(cast554*wmma153.w)))))+(cast550*((float)(((cast555*wmma154.w)+(cast556*wmma155.w)))))+(cast551*((float)(((cast557*wmma156.w)+(cast558*wmma157.w)))))+(cast552*((float)(((cast559*wmma158.w)+(cast560*wmma159.w))))))));
    unsigned int val485 = (*(buf0+alu131));
    unsigned int val486 = (*(buf0+alu132));
    unsigned int val487 = (*(buf0+alu133));
    unsigned int val488 = (*(buf0+alu134));
    unsigned int val489 = (*(buf0+alu135));
    unsigned int val490 = (*(buf0+alu136));
    unsigned int val491 = (*(buf0+alu137));
    unsigned int val492 = (*(buf0+alu138));
    unsigned int val493 = (*(buf0+alu217));
    unsigned int val494 = (*(buf0+alu218));
    unsigned int val495 = (*(buf0+alu219));
    unsigned int val496 = (*(buf0+alu220));
    unsigned int val497 = (*(buf0+alu221));
    unsigned int val498 = (*(buf0+alu222));
    unsigned int val499 = (*(buf0+alu223));
    unsigned int val500 = (*(buf0+alu224));
    unsigned int val501 = (*(buf0+alu180));
    unsigned int val502 = (*(buf0+alu183));
    unsigned int val503 = (*(buf0+alu184));
    unsigned int val504 = (*(buf0+alu188));
    unsigned int val505 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+32)) = buf17;
    }
    char4 cast561 = make_char4(((signed char)(((val485>>0u)&255u))),((signed char)(((val485>>8u)&255u))),((signed char)(((val485>>16u)&255u))),((signed char)(((val485>>24u)&255u))));
    char4 cast562 = make_char4(((signed char)(((val486>>0u)&255u))),((signed char)(((val486>>8u)&255u))),((signed char)(((val486>>16u)&255u))),((signed char)(((val486>>24u)&255u))));
    char4 cast563 = make_char4(((signed char)(((val487>>0u)&255u))),((signed char)(((val487>>8u)&255u))),((signed char)(((val487>>16u)&255u))),((signed char)(((val487>>24u)&255u))));
    char4 cast564 = make_char4(((signed char)(((val488>>0u)&255u))),((signed char)(((val488>>8u)&255u))),((signed char)(((val488>>16u)&255u))),((signed char)(((val488>>24u)&255u))));
    char4 cast565 = make_char4(((signed char)(((val489>>0u)&255u))),((signed char)(((val489>>8u)&255u))),((signed char)(((val489>>16u)&255u))),((signed char)(((val489>>24u)&255u))));
    char4 cast566 = make_char4(((signed char)(((val490>>0u)&255u))),((signed char)(((val490>>8u)&255u))),((signed char)(((val490>>16u)&255u))),((signed char)(((val490>>24u)&255u))));
    char4 cast567 = make_char4(((signed char)(((val491>>0u)&255u))),((signed char)(((val491>>8u)&255u))),((signed char)(((val491>>16u)&255u))),((signed char)(((val491>>24u)&255u))));
    char4 cast568 = make_char4(((signed char)(((val492>>0u)&255u))),((signed char)(((val492>>8u)&255u))),((signed char)(((val492>>16u)&255u))),((signed char)(((val492>>24u)&255u))));
    int4 wmma160 = __WMMA_8_16_16_signed_char_int(alu928, cast561, cast0);
    int4 wmma161 = __WMMA_8_16_16_signed_char_int(alu929, cast562, cast0);
    int4 wmma162 = __WMMA_8_16_16_signed_char_int(alu930, cast563, cast0);
    int4 wmma163 = __WMMA_8_16_16_signed_char_int(alu931, cast564, cast0);
    int4 wmma164 = __WMMA_8_16_16_signed_char_int(alu932, cast565, cast0);
    int4 wmma165 = __WMMA_8_16_16_signed_char_int(alu933, cast566, cast0);
    int4 wmma166 = __WMMA_8_16_16_signed_char_int(alu934, cast567, cast0);
    int4 wmma167 = __WMMA_8_16_16_signed_char_int(alu935, cast568, cast0);
    int cast569 = ((int)(((signed char)(((val502>>0u)&255u)))));
    int cast570 = ((int)(((signed char)(((val502>>8u)&255u)))));
    int cast571 = ((int)(((signed char)(((val502>>16u)&255u)))));
    int cast572 = ((int)(((signed char)(((val502>>24u)&255u)))));
    int cast573 = ((int)(((signed char)(((val503>>0u)&255u)))));
    int cast574 = ((int)(((signed char)(((val503>>8u)&255u)))));
    int cast575 = ((int)(((signed char)(((val503>>16u)&255u)))));
    int cast576 = ((int)(((signed char)(((val503>>24u)&255u)))));
    float cast577 = tg_bitcast<float>((unsigned int)(val493));
    float cast578 = tg_bitcast<float>((unsigned int)(val494));
    float cast579 = tg_bitcast<float>((unsigned int)(val495));
    float cast580 = tg_bitcast<float>((unsigned int)(val496));
    buf17 = (buf17+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val501&65535u)))))))*((cast577*((float)(((cast569*wmma160.x)+(cast570*wmma161.x)))))+(cast578*((float)(((cast571*wmma162.x)+(cast572*wmma163.x)))))+(cast579*((float)(((cast573*wmma164.x)+(cast574*wmma165.x)))))+(cast580*((float)(((cast575*wmma166.x)+(cast576*wmma167.x))))))));
    unsigned int val506 = (*(buf0+alu180));
    if (0) {
      *(data0_5570560+(alu110+33)) = buf18;
    }
    float cast581 = tg_bitcast<float>((unsigned int)(val497));
    float cast582 = tg_bitcast<float>((unsigned int)(val498));
    float cast583 = tg_bitcast<float>((unsigned int)(val499));
    float cast584 = tg_bitcast<float>((unsigned int)(val500));
    buf18 = (buf18+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val506&65535u)))))))*((cast581*((float)(((cast569*wmma160.y)+(cast570*wmma161.y)))))+(cast582*((float)(((cast571*wmma162.y)+(cast572*wmma163.y)))))+(cast583*((float)(((cast573*wmma164.y)+(cast574*wmma165.y)))))+(cast584*((float)(((cast575*wmma166.y)+(cast576*wmma167.y))))))));
    unsigned int val507 = (*(buf0+alu185));
    if (0) {
      *(data0_5570560+(alu110+1056)) = buf19;
    }
    int cast585 = ((int)(((signed char)(((val504>>0u)&255u)))));
    int cast586 = ((int)(((signed char)(((val504>>8u)&255u)))));
    int cast587 = ((int)(((signed char)(((val504>>16u)&255u)))));
    int cast588 = ((int)(((signed char)(((val504>>24u)&255u)))));
    int cast589 = ((int)(((signed char)(((val505>>0u)&255u)))));
    int cast590 = ((int)(((signed char)(((val505>>8u)&255u)))));
    int cast591 = ((int)(((signed char)(((val505>>16u)&255u)))));
    int cast592 = ((int)(((signed char)(((val505>>24u)&255u)))));
    buf19 = (buf19+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val507&65535u)))))))*((cast577*((float)(((cast585*wmma160.z)+(cast586*wmma161.z)))))+(cast578*((float)(((cast587*wmma162.z)+(cast588*wmma163.z)))))+(cast579*((float)(((cast589*wmma164.z)+(cast590*wmma165.z)))))+(cast580*((float)(((cast591*wmma166.z)+(cast592*wmma167.z))))))));
    unsigned int val508 = (*(buf0+alu185));
    if (0) {
      *(data0_5570560+(alu110+1057)) = buf20;
    }
    buf20 = (buf20+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val508&65535u)))))))*((cast581*((float)(((cast585*wmma160.w)+(cast586*wmma161.w)))))+(cast582*((float)(((cast587*wmma162.w)+(cast588*wmma163.w)))))+(cast583*((float)(((cast589*wmma164.w)+(cast590*wmma165.w)))))+(cast584*((float)(((cast591*wmma166.w)+(cast592*wmma167.w))))))));
    unsigned int val509 = (*(buf0+alu217));
    unsigned int val510 = (*(buf0+alu218));
    unsigned int val511 = (*(buf0+alu219));
    unsigned int val512 = (*(buf0+alu220));
    unsigned int val513 = (*(buf0+alu221));
    unsigned int val514 = (*(buf0+alu222));
    unsigned int val515 = (*(buf0+alu223));
    unsigned int val516 = (*(buf0+alu224));
    unsigned int val517 = (*(buf0+alu190));
    unsigned int val518 = (*(buf0+alu193));
    unsigned int val519 = (*(buf0+alu194));
    unsigned int val520 = (*(buf0+alu198));
    unsigned int val521 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+2080)) = buf21;
    }
    int4 wmma168 = __WMMA_8_16_16_signed_char_int(alu952, cast561, cast0);
    int4 wmma169 = __WMMA_8_16_16_signed_char_int(alu953, cast562, cast0);
    int4 wmma170 = __WMMA_8_16_16_signed_char_int(alu954, cast563, cast0);
    int4 wmma171 = __WMMA_8_16_16_signed_char_int(alu955, cast564, cast0);
    int4 wmma172 = __WMMA_8_16_16_signed_char_int(alu956, cast565, cast0);
    int4 wmma173 = __WMMA_8_16_16_signed_char_int(alu957, cast566, cast0);
    int4 wmma174 = __WMMA_8_16_16_signed_char_int(alu958, cast567, cast0);
    int4 wmma175 = __WMMA_8_16_16_signed_char_int(alu959, cast568, cast0);
    int cast593 = ((int)(((signed char)(((val518>>0u)&255u)))));
    int cast594 = ((int)(((signed char)(((val518>>8u)&255u)))));
    int cast595 = ((int)(((signed char)(((val518>>16u)&255u)))));
    int cast596 = ((int)(((signed char)(((val518>>24u)&255u)))));
    int cast597 = ((int)(((signed char)(((val519>>0u)&255u)))));
    int cast598 = ((int)(((signed char)(((val519>>8u)&255u)))));
    int cast599 = ((int)(((signed char)(((val519>>16u)&255u)))));
    int cast600 = ((int)(((signed char)(((val519>>24u)&255u)))));
    float cast601 = tg_bitcast<float>((unsigned int)(val509));
    float cast602 = tg_bitcast<float>((unsigned int)(val510));
    float cast603 = tg_bitcast<float>((unsigned int)(val511));
    float cast604 = tg_bitcast<float>((unsigned int)(val512));
    buf21 = (buf21+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val517&65535u)))))))*((cast601*((float)(((cast593*wmma168.x)+(cast594*wmma169.x)))))+(cast602*((float)(((cast595*wmma170.x)+(cast596*wmma171.x)))))+(cast603*((float)(((cast597*wmma172.x)+(cast598*wmma173.x)))))+(cast604*((float)(((cast599*wmma174.x)+(cast600*wmma175.x))))))));
    unsigned int val522 = (*(buf0+alu190));
    if (0) {
      *(data0_5570560+(alu110+2081)) = buf22;
    }
    float cast605 = tg_bitcast<float>((unsigned int)(val513));
    float cast606 = tg_bitcast<float>((unsigned int)(val514));
    float cast607 = tg_bitcast<float>((unsigned int)(val515));
    float cast608 = tg_bitcast<float>((unsigned int)(val516));
    buf22 = (buf22+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val522&65535u)))))))*((cast605*((float)(((cast593*wmma168.y)+(cast594*wmma169.y)))))+(cast606*((float)(((cast595*wmma170.y)+(cast596*wmma171.y)))))+(cast607*((float)(((cast597*wmma172.y)+(cast598*wmma173.y)))))+(cast608*((float)(((cast599*wmma174.y)+(cast600*wmma175.y))))))));
    unsigned int val523 = (*(buf0+alu195));
    if (0) {
      *(data0_5570560+(alu110+3104)) = buf23;
    }
    int cast609 = ((int)(((signed char)(((val520>>0u)&255u)))));
    int cast610 = ((int)(((signed char)(((val520>>8u)&255u)))));
    int cast611 = ((int)(((signed char)(((val520>>16u)&255u)))));
    int cast612 = ((int)(((signed char)(((val520>>24u)&255u)))));
    int cast613 = ((int)(((signed char)(((val521>>0u)&255u)))));
    int cast614 = ((int)(((signed char)(((val521>>8u)&255u)))));
    int cast615 = ((int)(((signed char)(((val521>>16u)&255u)))));
    int cast616 = ((int)(((signed char)(((val521>>24u)&255u)))));
    buf23 = (buf23+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val523&65535u)))))))*((cast601*((float)(((cast609*wmma168.z)+(cast610*wmma169.z)))))+(cast602*((float)(((cast611*wmma170.z)+(cast612*wmma171.z)))))+(cast603*((float)(((cast613*wmma172.z)+(cast614*wmma173.z)))))+(cast604*((float)(((cast615*wmma174.z)+(cast616*wmma175.z))))))));
    unsigned int val524 = (*(buf0+alu195));
    if (0) {
      *(data0_5570560+(alu110+3105)) = buf24;
    }
    buf24 = (buf24+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val524&65535u)))))))*((cast605*((float)(((cast609*wmma168.w)+(cast610*wmma169.w)))))+(cast606*((float)(((cast611*wmma170.w)+(cast612*wmma171.w)))))+(cast607*((float)(((cast613*wmma172.w)+(cast614*wmma173.w)))))+(cast608*((float)(((cast615*wmma174.w)+(cast616*wmma175.w))))))));
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
    unsigned int val541 = (*(buf0+alu180));
    unsigned int val542 = (*(buf0+alu183));
    unsigned int val543 = (*(buf0+alu184));
    unsigned int val544 = (*(buf0+alu188));
    unsigned int val545 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+48)) = buf25;
    }
    char4 cast617 = make_char4(((signed char)(((val525>>0u)&255u))),((signed char)(((val525>>8u)&255u))),((signed char)(((val525>>16u)&255u))),((signed char)(((val525>>24u)&255u))));
    char4 cast618 = make_char4(((signed char)(((val526>>0u)&255u))),((signed char)(((val526>>8u)&255u))),((signed char)(((val526>>16u)&255u))),((signed char)(((val526>>24u)&255u))));
    char4 cast619 = make_char4(((signed char)(((val527>>0u)&255u))),((signed char)(((val527>>8u)&255u))),((signed char)(((val527>>16u)&255u))),((signed char)(((val527>>24u)&255u))));
    char4 cast620 = make_char4(((signed char)(((val528>>0u)&255u))),((signed char)(((val528>>8u)&255u))),((signed char)(((val528>>16u)&255u))),((signed char)(((val528>>24u)&255u))));
    char4 cast621 = make_char4(((signed char)(((val529>>0u)&255u))),((signed char)(((val529>>8u)&255u))),((signed char)(((val529>>16u)&255u))),((signed char)(((val529>>24u)&255u))));
    char4 cast622 = make_char4(((signed char)(((val530>>0u)&255u))),((signed char)(((val530>>8u)&255u))),((signed char)(((val530>>16u)&255u))),((signed char)(((val530>>24u)&255u))));
    char4 cast623 = make_char4(((signed char)(((val531>>0u)&255u))),((signed char)(((val531>>8u)&255u))),((signed char)(((val531>>16u)&255u))),((signed char)(((val531>>24u)&255u))));
    char4 cast624 = make_char4(((signed char)(((val532>>0u)&255u))),((signed char)(((val532>>8u)&255u))),((signed char)(((val532>>16u)&255u))),((signed char)(((val532>>24u)&255u))));
    int4 wmma176 = __WMMA_8_16_16_signed_char_int(alu928, cast617, cast0);
    int4 wmma177 = __WMMA_8_16_16_signed_char_int(alu929, cast618, cast0);
    int4 wmma178 = __WMMA_8_16_16_signed_char_int(alu930, cast619, cast0);
    int4 wmma179 = __WMMA_8_16_16_signed_char_int(alu931, cast620, cast0);
    int4 wmma180 = __WMMA_8_16_16_signed_char_int(alu932, cast621, cast0);
    int4 wmma181 = __WMMA_8_16_16_signed_char_int(alu933, cast622, cast0);
    int4 wmma182 = __WMMA_8_16_16_signed_char_int(alu934, cast623, cast0);
    int4 wmma183 = __WMMA_8_16_16_signed_char_int(alu935, cast624, cast0);
    int cast625 = ((int)(((signed char)(((val542>>0u)&255u)))));
    int cast626 = ((int)(((signed char)(((val542>>8u)&255u)))));
    int cast627 = ((int)(((signed char)(((val542>>16u)&255u)))));
    int cast628 = ((int)(((signed char)(((val542>>24u)&255u)))));
    int cast629 = ((int)(((signed char)(((val543>>0u)&255u)))));
    int cast630 = ((int)(((signed char)(((val543>>8u)&255u)))));
    int cast631 = ((int)(((signed char)(((val543>>16u)&255u)))));
    int cast632 = ((int)(((signed char)(((val543>>24u)&255u)))));
    float cast633 = tg_bitcast<float>((unsigned int)(val533));
    float cast634 = tg_bitcast<float>((unsigned int)(val534));
    float cast635 = tg_bitcast<float>((unsigned int)(val535));
    float cast636 = tg_bitcast<float>((unsigned int)(val536));
    buf25 = (buf25+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val541&65535u)))))))*((cast633*((float)(((cast625*wmma176.x)+(cast626*wmma177.x)))))+(cast634*((float)(((cast627*wmma178.x)+(cast628*wmma179.x)))))+(cast635*((float)(((cast629*wmma180.x)+(cast630*wmma181.x)))))+(cast636*((float)(((cast631*wmma182.x)+(cast632*wmma183.x))))))));
    unsigned int val546 = (*(buf0+alu180));
    if (0) {
      *(data0_5570560+(alu110+49)) = buf26;
    }
    float cast637 = tg_bitcast<float>((unsigned int)(val537));
    float cast638 = tg_bitcast<float>((unsigned int)(val538));
    float cast639 = tg_bitcast<float>((unsigned int)(val539));
    float cast640 = tg_bitcast<float>((unsigned int)(val540));
    buf26 = (buf26+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val546&65535u)))))))*((cast637*((float)(((cast625*wmma176.y)+(cast626*wmma177.y)))))+(cast638*((float)(((cast627*wmma178.y)+(cast628*wmma179.y)))))+(cast639*((float)(((cast629*wmma180.y)+(cast630*wmma181.y)))))+(cast640*((float)(((cast631*wmma182.y)+(cast632*wmma183.y))))))));
    unsigned int val547 = (*(buf0+alu185));
    if (0) {
      *(data0_5570560+(alu110+1072)) = buf27;
    }
    int cast641 = ((int)(((signed char)(((val544>>0u)&255u)))));
    int cast642 = ((int)(((signed char)(((val544>>8u)&255u)))));
    int cast643 = ((int)(((signed char)(((val544>>16u)&255u)))));
    int cast644 = ((int)(((signed char)(((val544>>24u)&255u)))));
    int cast645 = ((int)(((signed char)(((val545>>0u)&255u)))));
    int cast646 = ((int)(((signed char)(((val545>>8u)&255u)))));
    int cast647 = ((int)(((signed char)(((val545>>16u)&255u)))));
    int cast648 = ((int)(((signed char)(((val545>>24u)&255u)))));
    buf27 = (buf27+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val547&65535u)))))))*((cast633*((float)(((cast641*wmma176.z)+(cast642*wmma177.z)))))+(cast634*((float)(((cast643*wmma178.z)+(cast644*wmma179.z)))))+(cast635*((float)(((cast645*wmma180.z)+(cast646*wmma181.z)))))+(cast636*((float)(((cast647*wmma182.z)+(cast648*wmma183.z))))))));
    unsigned int val548 = (*(buf0+alu185));
    if (0) {
      *(data0_5570560+(alu110+1073)) = buf28;
    }
    buf28 = (buf28+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val548&65535u)))))))*((cast637*((float)(((cast641*wmma176.w)+(cast642*wmma177.w)))))+(cast638*((float)(((cast643*wmma178.w)+(cast644*wmma179.w)))))+(cast639*((float)(((cast645*wmma180.w)+(cast646*wmma181.w)))))+(cast640*((float)(((cast647*wmma182.w)+(cast648*wmma183.w))))))));
    unsigned int val549 = (*(buf0+alu225));
    unsigned int val550 = (*(buf0+alu226));
    unsigned int val551 = (*(buf0+alu227));
    unsigned int val552 = (*(buf0+alu228));
    unsigned int val553 = (*(buf0+alu229));
    unsigned int val554 = (*(buf0+alu230));
    unsigned int val555 = (*(buf0+alu231));
    unsigned int val556 = (*(buf0+alu232));
    unsigned int val557 = (*(buf0+alu190));
    unsigned int val558 = (*(buf0+alu193));
    unsigned int val559 = (*(buf0+alu194));
    unsigned int val560 = (*(buf0+alu198));
    unsigned int val561 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+2096)) = buf29;
    }
    int4 wmma184 = __WMMA_8_16_16_signed_char_int(alu952, cast617, cast0);
    int4 wmma185 = __WMMA_8_16_16_signed_char_int(alu953, cast618, cast0);
    int4 wmma186 = __WMMA_8_16_16_signed_char_int(alu954, cast619, cast0);
    int4 wmma187 = __WMMA_8_16_16_signed_char_int(alu955, cast620, cast0);
    int4 wmma188 = __WMMA_8_16_16_signed_char_int(alu956, cast621, cast0);
    int4 wmma189 = __WMMA_8_16_16_signed_char_int(alu957, cast622, cast0);
    int4 wmma190 = __WMMA_8_16_16_signed_char_int(alu958, cast623, cast0);
    int4 wmma191 = __WMMA_8_16_16_signed_char_int(alu959, cast624, cast0);
    int cast649 = ((int)(((signed char)(((val558>>0u)&255u)))));
    int cast650 = ((int)(((signed char)(((val558>>8u)&255u)))));
    int cast651 = ((int)(((signed char)(((val558>>16u)&255u)))));
    int cast652 = ((int)(((signed char)(((val558>>24u)&255u)))));
    int cast653 = ((int)(((signed char)(((val559>>0u)&255u)))));
    int cast654 = ((int)(((signed char)(((val559>>8u)&255u)))));
    int cast655 = ((int)(((signed char)(((val559>>16u)&255u)))));
    int cast656 = ((int)(((signed char)(((val559>>24u)&255u)))));
    float cast657 = tg_bitcast<float>((unsigned int)(val549));
    float cast658 = tg_bitcast<float>((unsigned int)(val550));
    float cast659 = tg_bitcast<float>((unsigned int)(val551));
    float cast660 = tg_bitcast<float>((unsigned int)(val552));
    buf29 = (buf29+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val557&65535u)))))))*((cast657*((float)(((cast649*wmma184.x)+(cast650*wmma185.x)))))+(cast658*((float)(((cast651*wmma186.x)+(cast652*wmma187.x)))))+(cast659*((float)(((cast653*wmma188.x)+(cast654*wmma189.x)))))+(cast660*((float)(((cast655*wmma190.x)+(cast656*wmma191.x))))))));
    unsigned int val562 = (*(buf0+alu190));
    if (0) {
      *(data0_5570560+(alu110+2097)) = buf30;
    }
    float cast661 = tg_bitcast<float>((unsigned int)(val553));
    float cast662 = tg_bitcast<float>((unsigned int)(val554));
    float cast663 = tg_bitcast<float>((unsigned int)(val555));
    float cast664 = tg_bitcast<float>((unsigned int)(val556));
    buf30 = (buf30+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val562&65535u)))))))*((cast661*((float)(((cast649*wmma184.y)+(cast650*wmma185.y)))))+(cast662*((float)(((cast651*wmma186.y)+(cast652*wmma187.y)))))+(cast663*((float)(((cast653*wmma188.y)+(cast654*wmma189.y)))))+(cast664*((float)(((cast655*wmma190.y)+(cast656*wmma191.y))))))));
    unsigned int val563 = (*(buf0+alu195));
    if (0) {
      *(data0_5570560+(alu110+3120)) = buf31;
    }
    int cast665 = ((int)(((signed char)(((val560>>0u)&255u)))));
    int cast666 = ((int)(((signed char)(((val560>>8u)&255u)))));
    int cast667 = ((int)(((signed char)(((val560>>16u)&255u)))));
    int cast668 = ((int)(((signed char)(((val560>>24u)&255u)))));
    int cast669 = ((int)(((signed char)(((val561>>0u)&255u)))));
    int cast670 = ((int)(((signed char)(((val561>>8u)&255u)))));
    int cast671 = ((int)(((signed char)(((val561>>16u)&255u)))));
    int cast672 = ((int)(((signed char)(((val561>>24u)&255u)))));
    buf31 = (buf31+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val563&65535u)))))))*((cast657*((float)(((cast665*wmma184.z)+(cast666*wmma185.z)))))+(cast658*((float)(((cast667*wmma186.z)+(cast668*wmma187.z)))))+(cast659*((float)(((cast669*wmma188.z)+(cast670*wmma189.z)))))+(cast660*((float)(((cast671*wmma190.z)+(cast672*wmma191.z))))))));
    unsigned int val564 = (*(buf0+alu195));
    if (0) {
      *(data0_5570560+(alu110+3121)) = buf32;
    }
    buf32 = (buf32+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val564&65535u)))))))*((cast661*((float)(((cast665*wmma184.w)+(cast666*wmma185.w)))))+(cast662*((float)(((cast667*wmma186.w)+(cast668*wmma187.w)))))+(cast663*((float)(((cast669*wmma188.w)+(cast670*wmma189.w)))))+(cast664*((float)(((cast671*wmma190.w)+(cast672*wmma191.w))))))));
    unsigned int val565 = (*(buf0+alu147));
    unsigned int val566 = (*(buf0+alu148));
    unsigned int val567 = (*(buf0+alu149));
    unsigned int val568 = (*(buf0+alu150));
    unsigned int val569 = (*(buf0+alu151));
    unsigned int val570 = (*(buf0+alu152));
    unsigned int val571 = (*(buf0+alu153));
    unsigned int val572 = (*(buf0+alu154));
    unsigned int val573 = (*(buf0+alu233));
    unsigned int val574 = (*(buf0+alu234));
    unsigned int val575 = (*(buf0+alu235));
    unsigned int val576 = (*(buf0+alu236));
    unsigned int val577 = (*(buf0+alu237));
    unsigned int val578 = (*(buf0+alu238));
    unsigned int val579 = (*(buf0+alu239));
    unsigned int val580 = (*(buf0+alu240));
    unsigned int val581 = (*(buf0+alu180));
    unsigned int val582 = (*(buf0+alu183));
    unsigned int val583 = (*(buf0+alu184));
    unsigned int val584 = (*(buf0+alu188));
    unsigned int val585 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+64)) = buf33;
    }
    char4 cast673 = make_char4(((signed char)(((val565>>0u)&255u))),((signed char)(((val565>>8u)&255u))),((signed char)(((val565>>16u)&255u))),((signed char)(((val565>>24u)&255u))));
    char4 cast674 = make_char4(((signed char)(((val566>>0u)&255u))),((signed char)(((val566>>8u)&255u))),((signed char)(((val566>>16u)&255u))),((signed char)(((val566>>24u)&255u))));
    char4 cast675 = make_char4(((signed char)(((val567>>0u)&255u))),((signed char)(((val567>>8u)&255u))),((signed char)(((val567>>16u)&255u))),((signed char)(((val567>>24u)&255u))));
    char4 cast676 = make_char4(((signed char)(((val568>>0u)&255u))),((signed char)(((val568>>8u)&255u))),((signed char)(((val568>>16u)&255u))),((signed char)(((val568>>24u)&255u))));
    char4 cast677 = make_char4(((signed char)(((val569>>0u)&255u))),((signed char)(((val569>>8u)&255u))),((signed char)(((val569>>16u)&255u))),((signed char)(((val569>>24u)&255u))));
    char4 cast678 = make_char4(((signed char)(((val570>>0u)&255u))),((signed char)(((val570>>8u)&255u))),((signed char)(((val570>>16u)&255u))),((signed char)(((val570>>24u)&255u))));
    char4 cast679 = make_char4(((signed char)(((val571>>0u)&255u))),((signed char)(((val571>>8u)&255u))),((signed char)(((val571>>16u)&255u))),((signed char)(((val571>>24u)&255u))));
    char4 cast680 = make_char4(((signed char)(((val572>>0u)&255u))),((signed char)(((val572>>8u)&255u))),((signed char)(((val572>>16u)&255u))),((signed char)(((val572>>24u)&255u))));
    int4 wmma192 = __WMMA_8_16_16_signed_char_int(alu928, cast673, cast0);
    int4 wmma193 = __WMMA_8_16_16_signed_char_int(alu929, cast674, cast0);
    int4 wmma194 = __WMMA_8_16_16_signed_char_int(alu930, cast675, cast0);
    int4 wmma195 = __WMMA_8_16_16_signed_char_int(alu931, cast676, cast0);
    int4 wmma196 = __WMMA_8_16_16_signed_char_int(alu932, cast677, cast0);
    int4 wmma197 = __WMMA_8_16_16_signed_char_int(alu933, cast678, cast0);
    int4 wmma198 = __WMMA_8_16_16_signed_char_int(alu934, cast679, cast0);
    int4 wmma199 = __WMMA_8_16_16_signed_char_int(alu935, cast680, cast0);
    int cast681 = ((int)(((signed char)(((val582>>0u)&255u)))));
    int cast682 = ((int)(((signed char)(((val582>>8u)&255u)))));
    int cast683 = ((int)(((signed char)(((val582>>16u)&255u)))));
    int cast684 = ((int)(((signed char)(((val582>>24u)&255u)))));
    int cast685 = ((int)(((signed char)(((val583>>0u)&255u)))));
    int cast686 = ((int)(((signed char)(((val583>>8u)&255u)))));
    int cast687 = ((int)(((signed char)(((val583>>16u)&255u)))));
    int cast688 = ((int)(((signed char)(((val583>>24u)&255u)))));
    float cast689 = tg_bitcast<float>((unsigned int)(val573));
    float cast690 = tg_bitcast<float>((unsigned int)(val574));
    float cast691 = tg_bitcast<float>((unsigned int)(val575));
    float cast692 = tg_bitcast<float>((unsigned int)(val576));
    buf33 = (buf33+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val581&65535u)))))))*((cast689*((float)(((cast681*wmma192.x)+(cast682*wmma193.x)))))+(cast690*((float)(((cast683*wmma194.x)+(cast684*wmma195.x)))))+(cast691*((float)(((cast685*wmma196.x)+(cast686*wmma197.x)))))+(cast692*((float)(((cast687*wmma198.x)+(cast688*wmma199.x))))))));
    unsigned int val586 = (*(buf0+alu180));
    if (0) {
      *(data0_5570560+(alu110+65)) = buf34;
    }
    float cast693 = tg_bitcast<float>((unsigned int)(val577));
    float cast694 = tg_bitcast<float>((unsigned int)(val578));
    float cast695 = tg_bitcast<float>((unsigned int)(val579));
    float cast696 = tg_bitcast<float>((unsigned int)(val580));
    buf34 = (buf34+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val586&65535u)))))))*((cast693*((float)(((cast681*wmma192.y)+(cast682*wmma193.y)))))+(cast694*((float)(((cast683*wmma194.y)+(cast684*wmma195.y)))))+(cast695*((float)(((cast685*wmma196.y)+(cast686*wmma197.y)))))+(cast696*((float)(((cast687*wmma198.y)+(cast688*wmma199.y))))))));
    unsigned int val587 = (*(buf0+alu185));
    if (0) {
      *(data0_5570560+(alu110+1088)) = buf35;
    }
    int cast697 = ((int)(((signed char)(((val584>>0u)&255u)))));
    int cast698 = ((int)(((signed char)(((val584>>8u)&255u)))));
    int cast699 = ((int)(((signed char)(((val584>>16u)&255u)))));
    int cast700 = ((int)(((signed char)(((val584>>24u)&255u)))));
    int cast701 = ((int)(((signed char)(((val585>>0u)&255u)))));
    int cast702 = ((int)(((signed char)(((val585>>8u)&255u)))));
    int cast703 = ((int)(((signed char)(((val585>>16u)&255u)))));
    int cast704 = ((int)(((signed char)(((val585>>24u)&255u)))));
    buf35 = (buf35+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val587&65535u)))))))*((cast689*((float)(((cast697*wmma192.z)+(cast698*wmma193.z)))))+(cast690*((float)(((cast699*wmma194.z)+(cast700*wmma195.z)))))+(cast691*((float)(((cast701*wmma196.z)+(cast702*wmma197.z)))))+(cast692*((float)(((cast703*wmma198.z)+(cast704*wmma199.z))))))));
    unsigned int val588 = (*(buf0+alu185));
    if (0) {
      *(data0_5570560+(alu110+1089)) = buf36;
    }
    buf36 = (buf36+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val588&65535u)))))))*((cast693*((float)(((cast697*wmma192.w)+(cast698*wmma193.w)))))+(cast694*((float)(((cast699*wmma194.w)+(cast700*wmma195.w)))))+(cast695*((float)(((cast701*wmma196.w)+(cast702*wmma197.w)))))+(cast696*((float)(((cast703*wmma198.w)+(cast704*wmma199.w))))))));
    unsigned int val589 = (*(buf0+alu233));
    unsigned int val590 = (*(buf0+alu234));
    unsigned int val591 = (*(buf0+alu235));
    unsigned int val592 = (*(buf0+alu236));
    unsigned int val593 = (*(buf0+alu237));
    unsigned int val594 = (*(buf0+alu238));
    unsigned int val595 = (*(buf0+alu239));
    unsigned int val596 = (*(buf0+alu240));
    unsigned int val597 = (*(buf0+alu190));
    unsigned int val598 = (*(buf0+alu193));
    unsigned int val599 = (*(buf0+alu194));
    unsigned int val600 = (*(buf0+alu198));
    unsigned int val601 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+2112)) = buf37;
    }
    int4 wmma200 = __WMMA_8_16_16_signed_char_int(alu952, cast673, cast0);
    int4 wmma201 = __WMMA_8_16_16_signed_char_int(alu953, cast674, cast0);
    int4 wmma202 = __WMMA_8_16_16_signed_char_int(alu954, cast675, cast0);
    int4 wmma203 = __WMMA_8_16_16_signed_char_int(alu955, cast676, cast0);
    int4 wmma204 = __WMMA_8_16_16_signed_char_int(alu956, cast677, cast0);
    int4 wmma205 = __WMMA_8_16_16_signed_char_int(alu957, cast678, cast0);
    int4 wmma206 = __WMMA_8_16_16_signed_char_int(alu958, cast679, cast0);
    int4 wmma207 = __WMMA_8_16_16_signed_char_int(alu959, cast680, cast0);
    int cast705 = ((int)(((signed char)(((val598>>0u)&255u)))));
    int cast706 = ((int)(((signed char)(((val598>>8u)&255u)))));
    int cast707 = ((int)(((signed char)(((val598>>16u)&255u)))));
    int cast708 = ((int)(((signed char)(((val598>>24u)&255u)))));
    int cast709 = ((int)(((signed char)(((val599>>0u)&255u)))));
    int cast710 = ((int)(((signed char)(((val599>>8u)&255u)))));
    int cast711 = ((int)(((signed char)(((val599>>16u)&255u)))));
    int cast712 = ((int)(((signed char)(((val599>>24u)&255u)))));
    float cast713 = tg_bitcast<float>((unsigned int)(val589));
    float cast714 = tg_bitcast<float>((unsigned int)(val590));
    float cast715 = tg_bitcast<float>((unsigned int)(val591));
    float cast716 = tg_bitcast<float>((unsigned int)(val592));
    buf37 = (buf37+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val597&65535u)))))))*((cast713*((float)(((cast705*wmma200.x)+(cast706*wmma201.x)))))+(cast714*((float)(((cast707*wmma202.x)+(cast708*wmma203.x)))))+(cast715*((float)(((cast709*wmma204.x)+(cast710*wmma205.x)))))+(cast716*((float)(((cast711*wmma206.x)+(cast712*wmma207.x))))))));
    unsigned int val602 = (*(buf0+alu190));
    if (0) {
      *(data0_5570560+(alu110+2113)) = buf38;
    }
    float cast717 = tg_bitcast<float>((unsigned int)(val593));
    float cast718 = tg_bitcast<float>((unsigned int)(val594));
    float cast719 = tg_bitcast<float>((unsigned int)(val595));
    float cast720 = tg_bitcast<float>((unsigned int)(val596));
    buf38 = (buf38+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val602&65535u)))))))*((cast717*((float)(((cast705*wmma200.y)+(cast706*wmma201.y)))))+(cast718*((float)(((cast707*wmma202.y)+(cast708*wmma203.y)))))+(cast719*((float)(((cast709*wmma204.y)+(cast710*wmma205.y)))))+(cast720*((float)(((cast711*wmma206.y)+(cast712*wmma207.y))))))));
    unsigned int val603 = (*(buf0+alu195));
    if (0) {
      *(data0_5570560+(alu110+3136)) = buf39;
    }
    int cast721 = ((int)(((signed char)(((val600>>0u)&255u)))));
    int cast722 = ((int)(((signed char)(((val600>>8u)&255u)))));
    int cast723 = ((int)(((signed char)(((val600>>16u)&255u)))));
    int cast724 = ((int)(((signed char)(((val600>>24u)&255u)))));
    int cast725 = ((int)(((signed char)(((val601>>0u)&255u)))));
    int cast726 = ((int)(((signed char)(((val601>>8u)&255u)))));
    int cast727 = ((int)(((signed char)(((val601>>16u)&255u)))));
    int cast728 = ((int)(((signed char)(((val601>>24u)&255u)))));
    buf39 = (buf39+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val603&65535u)))))))*((cast713*((float)(((cast721*wmma200.z)+(cast722*wmma201.z)))))+(cast714*((float)(((cast723*wmma202.z)+(cast724*wmma203.z)))))+(cast715*((float)(((cast725*wmma204.z)+(cast726*wmma205.z)))))+(cast716*((float)(((cast727*wmma206.z)+(cast728*wmma207.z))))))));
    unsigned int val604 = (*(buf0+alu195));
    if (0) {
      *(data0_5570560+(alu110+3137)) = buf40;
    }
    buf40 = (buf40+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val604&65535u)))))))*((cast717*((float)(((cast721*wmma200.w)+(cast722*wmma201.w)))))+(cast718*((float)(((cast723*wmma202.w)+(cast724*wmma203.w)))))+(cast719*((float)(((cast725*wmma204.w)+(cast726*wmma205.w)))))+(cast720*((float)(((cast727*wmma206.w)+(cast728*wmma207.w))))))));
    unsigned int val605 = (*(buf0+alu155));
    unsigned int val606 = (*(buf0+alu156));
    unsigned int val607 = (*(buf0+alu157));
    unsigned int val608 = (*(buf0+alu158));
    unsigned int val609 = (*(buf0+alu159));
    unsigned int val610 = (*(buf0+alu160));
    unsigned int val611 = (*(buf0+alu161));
    unsigned int val612 = (*(buf0+alu162));
    unsigned int val613 = (*(buf0+alu241));
    unsigned int val614 = (*(buf0+alu242));
    unsigned int val615 = (*(buf0+alu243));
    unsigned int val616 = (*(buf0+alu244));
    unsigned int val617 = (*(buf0+alu245));
    unsigned int val618 = (*(buf0+alu246));
    unsigned int val619 = (*(buf0+alu247));
    unsigned int val620 = (*(buf0+alu248));
    unsigned int val621 = (*(buf0+alu180));
    unsigned int val622 = (*(buf0+alu183));
    unsigned int val623 = (*(buf0+alu184));
    unsigned int val624 = (*(buf0+alu188));
    unsigned int val625 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+80)) = buf41;
    }
    char4 cast729 = make_char4(((signed char)(((val605>>0u)&255u))),((signed char)(((val605>>8u)&255u))),((signed char)(((val605>>16u)&255u))),((signed char)(((val605>>24u)&255u))));
    char4 cast730 = make_char4(((signed char)(((val606>>0u)&255u))),((signed char)(((val606>>8u)&255u))),((signed char)(((val606>>16u)&255u))),((signed char)(((val606>>24u)&255u))));
    char4 cast731 = make_char4(((signed char)(((val607>>0u)&255u))),((signed char)(((val607>>8u)&255u))),((signed char)(((val607>>16u)&255u))),((signed char)(((val607>>24u)&255u))));
    char4 cast732 = make_char4(((signed char)(((val608>>0u)&255u))),((signed char)(((val608>>8u)&255u))),((signed char)(((val608>>16u)&255u))),((signed char)(((val608>>24u)&255u))));
    char4 cast733 = make_char4(((signed char)(((val609>>0u)&255u))),((signed char)(((val609>>8u)&255u))),((signed char)(((val609>>16u)&255u))),((signed char)(((val609>>24u)&255u))));
    char4 cast734 = make_char4(((signed char)(((val610>>0u)&255u))),((signed char)(((val610>>8u)&255u))),((signed char)(((val610>>16u)&255u))),((signed char)(((val610>>24u)&255u))));
    char4 cast735 = make_char4(((signed char)(((val611>>0u)&255u))),((signed char)(((val611>>8u)&255u))),((signed char)(((val611>>16u)&255u))),((signed char)(((val611>>24u)&255u))));
    char4 cast736 = make_char4(((signed char)(((val612>>0u)&255u))),((signed char)(((val612>>8u)&255u))),((signed char)(((val612>>16u)&255u))),((signed char)(((val612>>24u)&255u))));
    int4 wmma208 = __WMMA_8_16_16_signed_char_int(alu928, cast729, cast0);
    int4 wmma209 = __WMMA_8_16_16_signed_char_int(alu929, cast730, cast0);
    int4 wmma210 = __WMMA_8_16_16_signed_char_int(alu930, cast731, cast0);
    int4 wmma211 = __WMMA_8_16_16_signed_char_int(alu931, cast732, cast0);
    int4 wmma212 = __WMMA_8_16_16_signed_char_int(alu932, cast733, cast0);
    int4 wmma213 = __WMMA_8_16_16_signed_char_int(alu933, cast734, cast0);
    int4 wmma214 = __WMMA_8_16_16_signed_char_int(alu934, cast735, cast0);
    int4 wmma215 = __WMMA_8_16_16_signed_char_int(alu935, cast736, cast0);
    int cast737 = ((int)(((signed char)(((val622>>0u)&255u)))));
    int cast738 = ((int)(((signed char)(((val622>>8u)&255u)))));
    int cast739 = ((int)(((signed char)(((val622>>16u)&255u)))));
    int cast740 = ((int)(((signed char)(((val622>>24u)&255u)))));
    int cast741 = ((int)(((signed char)(((val623>>0u)&255u)))));
    int cast742 = ((int)(((signed char)(((val623>>8u)&255u)))));
    int cast743 = ((int)(((signed char)(((val623>>16u)&255u)))));
    int cast744 = ((int)(((signed char)(((val623>>24u)&255u)))));
    float cast745 = tg_bitcast<float>((unsigned int)(val613));
    float cast746 = tg_bitcast<float>((unsigned int)(val614));
    float cast747 = tg_bitcast<float>((unsigned int)(val615));
    float cast748 = tg_bitcast<float>((unsigned int)(val616));
    buf41 = (buf41+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val621&65535u)))))))*((cast745*((float)(((cast737*wmma208.x)+(cast738*wmma209.x)))))+(cast746*((float)(((cast739*wmma210.x)+(cast740*wmma211.x)))))+(cast747*((float)(((cast741*wmma212.x)+(cast742*wmma213.x)))))+(cast748*((float)(((cast743*wmma214.x)+(cast744*wmma215.x))))))));
    unsigned int val626 = (*(buf0+alu180));
    if (0) {
      *(data0_5570560+(alu110+81)) = buf42;
    }
    float cast749 = tg_bitcast<float>((unsigned int)(val617));
    float cast750 = tg_bitcast<float>((unsigned int)(val618));
    float cast751 = tg_bitcast<float>((unsigned int)(val619));
    float cast752 = tg_bitcast<float>((unsigned int)(val620));
    buf42 = (buf42+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val626&65535u)))))))*((cast749*((float)(((cast737*wmma208.y)+(cast738*wmma209.y)))))+(cast750*((float)(((cast739*wmma210.y)+(cast740*wmma211.y)))))+(cast751*((float)(((cast741*wmma212.y)+(cast742*wmma213.y)))))+(cast752*((float)(((cast743*wmma214.y)+(cast744*wmma215.y))))))));
    unsigned int val627 = (*(buf0+alu185));
    if (0) {
      *(data0_5570560+(alu110+1104)) = buf43;
    }
    int cast753 = ((int)(((signed char)(((val624>>0u)&255u)))));
    int cast754 = ((int)(((signed char)(((val624>>8u)&255u)))));
    int cast755 = ((int)(((signed char)(((val624>>16u)&255u)))));
    int cast756 = ((int)(((signed char)(((val624>>24u)&255u)))));
    int cast757 = ((int)(((signed char)(((val625>>0u)&255u)))));
    int cast758 = ((int)(((signed char)(((val625>>8u)&255u)))));
    int cast759 = ((int)(((signed char)(((val625>>16u)&255u)))));
    int cast760 = ((int)(((signed char)(((val625>>24u)&255u)))));
    buf43 = (buf43+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val627&65535u)))))))*((cast745*((float)(((cast753*wmma208.z)+(cast754*wmma209.z)))))+(cast746*((float)(((cast755*wmma210.z)+(cast756*wmma211.z)))))+(cast747*((float)(((cast757*wmma212.z)+(cast758*wmma213.z)))))+(cast748*((float)(((cast759*wmma214.z)+(cast760*wmma215.z))))))));
    unsigned int val628 = (*(buf0+alu185));
    if (0) {
      *(data0_5570560+(alu110+1105)) = buf44;
    }
    buf44 = (buf44+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val628&65535u)))))))*((cast749*((float)(((cast753*wmma208.w)+(cast754*wmma209.w)))))+(cast750*((float)(((cast755*wmma210.w)+(cast756*wmma211.w)))))+(cast751*((float)(((cast757*wmma212.w)+(cast758*wmma213.w)))))+(cast752*((float)(((cast759*wmma214.w)+(cast760*wmma215.w))))))));
    unsigned int val629 = (*(buf0+alu241));
    unsigned int val630 = (*(buf0+alu242));
    unsigned int val631 = (*(buf0+alu243));
    unsigned int val632 = (*(buf0+alu244));
    unsigned int val633 = (*(buf0+alu245));
    unsigned int val634 = (*(buf0+alu246));
    unsigned int val635 = (*(buf0+alu247));
    unsigned int val636 = (*(buf0+alu248));
    unsigned int val637 = (*(buf0+alu190));
    unsigned int val638 = (*(buf0+alu193));
    unsigned int val639 = (*(buf0+alu194));
    unsigned int val640 = (*(buf0+alu198));
    unsigned int val641 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+2128)) = buf45;
    }
    int4 wmma216 = __WMMA_8_16_16_signed_char_int(alu952, cast729, cast0);
    int4 wmma217 = __WMMA_8_16_16_signed_char_int(alu953, cast730, cast0);
    int4 wmma218 = __WMMA_8_16_16_signed_char_int(alu954, cast731, cast0);
    int4 wmma219 = __WMMA_8_16_16_signed_char_int(alu955, cast732, cast0);
    int4 wmma220 = __WMMA_8_16_16_signed_char_int(alu956, cast733, cast0);
    int4 wmma221 = __WMMA_8_16_16_signed_char_int(alu957, cast734, cast0);
    int4 wmma222 = __WMMA_8_16_16_signed_char_int(alu958, cast735, cast0);
    int4 wmma223 = __WMMA_8_16_16_signed_char_int(alu959, cast736, cast0);
    int cast761 = ((int)(((signed char)(((val638>>0u)&255u)))));
    int cast762 = ((int)(((signed char)(((val638>>8u)&255u)))));
    int cast763 = ((int)(((signed char)(((val638>>16u)&255u)))));
    int cast764 = ((int)(((signed char)(((val638>>24u)&255u)))));
    int cast765 = ((int)(((signed char)(((val639>>0u)&255u)))));
    int cast766 = ((int)(((signed char)(((val639>>8u)&255u)))));
    int cast767 = ((int)(((signed char)(((val639>>16u)&255u)))));
    int cast768 = ((int)(((signed char)(((val639>>24u)&255u)))));
    float cast769 = tg_bitcast<float>((unsigned int)(val629));
    float cast770 = tg_bitcast<float>((unsigned int)(val630));
    float cast771 = tg_bitcast<float>((unsigned int)(val631));
    float cast772 = tg_bitcast<float>((unsigned int)(val632));
    buf45 = (buf45+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val637&65535u)))))))*((cast769*((float)(((cast761*wmma216.x)+(cast762*wmma217.x)))))+(cast770*((float)(((cast763*wmma218.x)+(cast764*wmma219.x)))))+(cast771*((float)(((cast765*wmma220.x)+(cast766*wmma221.x)))))+(cast772*((float)(((cast767*wmma222.x)+(cast768*wmma223.x))))))));
    unsigned int val642 = (*(buf0+alu190));
    if (0) {
      *(data0_5570560+(alu110+2129)) = buf46;
    }
    float cast773 = tg_bitcast<float>((unsigned int)(val633));
    float cast774 = tg_bitcast<float>((unsigned int)(val634));
    float cast775 = tg_bitcast<float>((unsigned int)(val635));
    float cast776 = tg_bitcast<float>((unsigned int)(val636));
    buf46 = (buf46+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val642&65535u)))))))*((cast773*((float)(((cast761*wmma216.y)+(cast762*wmma217.y)))))+(cast774*((float)(((cast763*wmma218.y)+(cast764*wmma219.y)))))+(cast775*((float)(((cast765*wmma220.y)+(cast766*wmma221.y)))))+(cast776*((float)(((cast767*wmma222.y)+(cast768*wmma223.y))))))));
    unsigned int val643 = (*(buf0+alu195));
    if (0) {
      *(data0_5570560+(alu110+3152)) = buf47;
    }
    int cast777 = ((int)(((signed char)(((val640>>0u)&255u)))));
    int cast778 = ((int)(((signed char)(((val640>>8u)&255u)))));
    int cast779 = ((int)(((signed char)(((val640>>16u)&255u)))));
    int cast780 = ((int)(((signed char)(((val640>>24u)&255u)))));
    int cast781 = ((int)(((signed char)(((val641>>0u)&255u)))));
    int cast782 = ((int)(((signed char)(((val641>>8u)&255u)))));
    int cast783 = ((int)(((signed char)(((val641>>16u)&255u)))));
    int cast784 = ((int)(((signed char)(((val641>>24u)&255u)))));
    buf47 = (buf47+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val643&65535u)))))))*((cast769*((float)(((cast777*wmma216.z)+(cast778*wmma217.z)))))+(cast770*((float)(((cast779*wmma218.z)+(cast780*wmma219.z)))))+(cast771*((float)(((cast781*wmma220.z)+(cast782*wmma221.z)))))+(cast772*((float)(((cast783*wmma222.z)+(cast784*wmma223.z))))))));
    unsigned int val644 = (*(buf0+alu195));
    if (0) {
      *(data0_5570560+(alu110+3153)) = buf48;
    }
    buf48 = (buf48+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val644&65535u)))))))*((cast773*((float)(((cast777*wmma216.w)+(cast778*wmma217.w)))))+(cast774*((float)(((cast779*wmma218.w)+(cast780*wmma219.w)))))+(cast775*((float)(((cast781*wmma220.w)+(cast782*wmma221.w)))))+(cast776*((float)(((cast783*wmma222.w)+(cast784*wmma223.w))))))));
    unsigned int val645 = (*(buf0+alu163));
    unsigned int val646 = (*(buf0+alu164));
    unsigned int val647 = (*(buf0+alu165));
    unsigned int val648 = (*(buf0+alu166));
    unsigned int val649 = (*(buf0+alu167));
    unsigned int val650 = (*(buf0+alu168));
    unsigned int val651 = (*(buf0+alu169));
    unsigned int val652 = (*(buf0+alu170));
    unsigned int val653 = (*(buf0+alu249));
    unsigned int val654 = (*(buf0+alu250));
    unsigned int val655 = (*(buf0+alu251));
    unsigned int val656 = (*(buf0+alu252));
    unsigned int val657 = (*(buf0+alu253));
    unsigned int val658 = (*(buf0+alu254));
    unsigned int val659 = (*(buf0+alu255));
    unsigned int val660 = (*(buf0+alu256));
    unsigned int val661 = (*(buf0+alu180));
    unsigned int val662 = (*(buf0+alu183));
    unsigned int val663 = (*(buf0+alu184));
    unsigned int val664 = (*(buf0+alu188));
    unsigned int val665 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+96)) = buf49;
    }
    char4 cast785 = make_char4(((signed char)(((val645>>0u)&255u))),((signed char)(((val645>>8u)&255u))),((signed char)(((val645>>16u)&255u))),((signed char)(((val645>>24u)&255u))));
    char4 cast786 = make_char4(((signed char)(((val646>>0u)&255u))),((signed char)(((val646>>8u)&255u))),((signed char)(((val646>>16u)&255u))),((signed char)(((val646>>24u)&255u))));
    char4 cast787 = make_char4(((signed char)(((val647>>0u)&255u))),((signed char)(((val647>>8u)&255u))),((signed char)(((val647>>16u)&255u))),((signed char)(((val647>>24u)&255u))));
    char4 cast788 = make_char4(((signed char)(((val648>>0u)&255u))),((signed char)(((val648>>8u)&255u))),((signed char)(((val648>>16u)&255u))),((signed char)(((val648>>24u)&255u))));
    char4 cast789 = make_char4(((signed char)(((val649>>0u)&255u))),((signed char)(((val649>>8u)&255u))),((signed char)(((val649>>16u)&255u))),((signed char)(((val649>>24u)&255u))));
    char4 cast790 = make_char4(((signed char)(((val650>>0u)&255u))),((signed char)(((val650>>8u)&255u))),((signed char)(((val650>>16u)&255u))),((signed char)(((val650>>24u)&255u))));
    char4 cast791 = make_char4(((signed char)(((val651>>0u)&255u))),((signed char)(((val651>>8u)&255u))),((signed char)(((val651>>16u)&255u))),((signed char)(((val651>>24u)&255u))));
    char4 cast792 = make_char4(((signed char)(((val652>>0u)&255u))),((signed char)(((val652>>8u)&255u))),((signed char)(((val652>>16u)&255u))),((signed char)(((val652>>24u)&255u))));
    int4 wmma224 = __WMMA_8_16_16_signed_char_int(alu928, cast785, cast0);
    int4 wmma225 = __WMMA_8_16_16_signed_char_int(alu929, cast786, cast0);
    int4 wmma226 = __WMMA_8_16_16_signed_char_int(alu930, cast787, cast0);
    int4 wmma227 = __WMMA_8_16_16_signed_char_int(alu931, cast788, cast0);
    int4 wmma228 = __WMMA_8_16_16_signed_char_int(alu932, cast789, cast0);
    int4 wmma229 = __WMMA_8_16_16_signed_char_int(alu933, cast790, cast0);
    int4 wmma230 = __WMMA_8_16_16_signed_char_int(alu934, cast791, cast0);
    int4 wmma231 = __WMMA_8_16_16_signed_char_int(alu935, cast792, cast0);
    int cast793 = ((int)(((signed char)(((val662>>0u)&255u)))));
    int cast794 = ((int)(((signed char)(((val662>>8u)&255u)))));
    int cast795 = ((int)(((signed char)(((val662>>16u)&255u)))));
    int cast796 = ((int)(((signed char)(((val662>>24u)&255u)))));
    int cast797 = ((int)(((signed char)(((val663>>0u)&255u)))));
    int cast798 = ((int)(((signed char)(((val663>>8u)&255u)))));
    int cast799 = ((int)(((signed char)(((val663>>16u)&255u)))));
    int cast800 = ((int)(((signed char)(((val663>>24u)&255u)))));
    float cast801 = tg_bitcast<float>((unsigned int)(val653));
    float cast802 = tg_bitcast<float>((unsigned int)(val654));
    float cast803 = tg_bitcast<float>((unsigned int)(val655));
    float cast804 = tg_bitcast<float>((unsigned int)(val656));
    buf49 = (buf49+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val661&65535u)))))))*((cast801*((float)(((cast793*wmma224.x)+(cast794*wmma225.x)))))+(cast802*((float)(((cast795*wmma226.x)+(cast796*wmma227.x)))))+(cast803*((float)(((cast797*wmma228.x)+(cast798*wmma229.x)))))+(cast804*((float)(((cast799*wmma230.x)+(cast800*wmma231.x))))))));
    unsigned int val666 = (*(buf0+alu180));
    if (0) {
      *(data0_5570560+(alu110+97)) = buf50;
    }
    float cast805 = tg_bitcast<float>((unsigned int)(val657));
    float cast806 = tg_bitcast<float>((unsigned int)(val658));
    float cast807 = tg_bitcast<float>((unsigned int)(val659));
    float cast808 = tg_bitcast<float>((unsigned int)(val660));
    buf50 = (buf50+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val666&65535u)))))))*((cast805*((float)(((cast793*wmma224.y)+(cast794*wmma225.y)))))+(cast806*((float)(((cast795*wmma226.y)+(cast796*wmma227.y)))))+(cast807*((float)(((cast797*wmma228.y)+(cast798*wmma229.y)))))+(cast808*((float)(((cast799*wmma230.y)+(cast800*wmma231.y))))))));
    unsigned int val667 = (*(buf0+alu185));
    if (0) {
      *(data0_5570560+(alu110+1120)) = buf51;
    }
    int cast809 = ((int)(((signed char)(((val664>>0u)&255u)))));
    int cast810 = ((int)(((signed char)(((val664>>8u)&255u)))));
    int cast811 = ((int)(((signed char)(((val664>>16u)&255u)))));
    int cast812 = ((int)(((signed char)(((val664>>24u)&255u)))));
    int cast813 = ((int)(((signed char)(((val665>>0u)&255u)))));
    int cast814 = ((int)(((signed char)(((val665>>8u)&255u)))));
    int cast815 = ((int)(((signed char)(((val665>>16u)&255u)))));
    int cast816 = ((int)(((signed char)(((val665>>24u)&255u)))));
    buf51 = (buf51+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val667&65535u)))))))*((cast801*((float)(((cast809*wmma224.z)+(cast810*wmma225.z)))))+(cast802*((float)(((cast811*wmma226.z)+(cast812*wmma227.z)))))+(cast803*((float)(((cast813*wmma228.z)+(cast814*wmma229.z)))))+(cast804*((float)(((cast815*wmma230.z)+(cast816*wmma231.z))))))));
    unsigned int val668 = (*(buf0+alu185));
    if (0) {
      *(data0_5570560+(alu110+1121)) = buf52;
    }
    buf52 = (buf52+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val668&65535u)))))))*((cast805*((float)(((cast809*wmma224.w)+(cast810*wmma225.w)))))+(cast806*((float)(((cast811*wmma226.w)+(cast812*wmma227.w)))))+(cast807*((float)(((cast813*wmma228.w)+(cast814*wmma229.w)))))+(cast808*((float)(((cast815*wmma230.w)+(cast816*wmma231.w))))))));
    unsigned int val669 = (*(buf0+alu249));
    unsigned int val670 = (*(buf0+alu250));
    unsigned int val671 = (*(buf0+alu251));
    unsigned int val672 = (*(buf0+alu252));
    unsigned int val673 = (*(buf0+alu253));
    unsigned int val674 = (*(buf0+alu254));
    unsigned int val675 = (*(buf0+alu255));
    unsigned int val676 = (*(buf0+alu256));
    unsigned int val677 = (*(buf0+alu190));
    unsigned int val678 = (*(buf0+alu193));
    unsigned int val679 = (*(buf0+alu194));
    unsigned int val680 = (*(buf0+alu198));
    unsigned int val681 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+2144)) = buf53;
    }
    int4 wmma232 = __WMMA_8_16_16_signed_char_int(alu952, cast785, cast0);
    int4 wmma233 = __WMMA_8_16_16_signed_char_int(alu953, cast786, cast0);
    int4 wmma234 = __WMMA_8_16_16_signed_char_int(alu954, cast787, cast0);
    int4 wmma235 = __WMMA_8_16_16_signed_char_int(alu955, cast788, cast0);
    int4 wmma236 = __WMMA_8_16_16_signed_char_int(alu956, cast789, cast0);
    int4 wmma237 = __WMMA_8_16_16_signed_char_int(alu957, cast790, cast0);
    int4 wmma238 = __WMMA_8_16_16_signed_char_int(alu958, cast791, cast0);
    int4 wmma239 = __WMMA_8_16_16_signed_char_int(alu959, cast792, cast0);
    int cast817 = ((int)(((signed char)(((val678>>0u)&255u)))));
    int cast818 = ((int)(((signed char)(((val678>>8u)&255u)))));
    int cast819 = ((int)(((signed char)(((val678>>16u)&255u)))));
    int cast820 = ((int)(((signed char)(((val678>>24u)&255u)))));
    int cast821 = ((int)(((signed char)(((val679>>0u)&255u)))));
    int cast822 = ((int)(((signed char)(((val679>>8u)&255u)))));
    int cast823 = ((int)(((signed char)(((val679>>16u)&255u)))));
    int cast824 = ((int)(((signed char)(((val679>>24u)&255u)))));
    float cast825 = tg_bitcast<float>((unsigned int)(val669));
    float cast826 = tg_bitcast<float>((unsigned int)(val670));
    float cast827 = tg_bitcast<float>((unsigned int)(val671));
    float cast828 = tg_bitcast<float>((unsigned int)(val672));
    buf53 = (buf53+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val677&65535u)))))))*((cast825*((float)(((cast817*wmma232.x)+(cast818*wmma233.x)))))+(cast826*((float)(((cast819*wmma234.x)+(cast820*wmma235.x)))))+(cast827*((float)(((cast821*wmma236.x)+(cast822*wmma237.x)))))+(cast828*((float)(((cast823*wmma238.x)+(cast824*wmma239.x))))))));
    unsigned int val682 = (*(buf0+alu190));
    if (0) {
      *(data0_5570560+(alu110+2145)) = buf54;
    }
    float cast829 = tg_bitcast<float>((unsigned int)(val673));
    float cast830 = tg_bitcast<float>((unsigned int)(val674));
    float cast831 = tg_bitcast<float>((unsigned int)(val675));
    float cast832 = tg_bitcast<float>((unsigned int)(val676));
    buf54 = (buf54+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val682&65535u)))))))*((cast829*((float)(((cast817*wmma232.y)+(cast818*wmma233.y)))))+(cast830*((float)(((cast819*wmma234.y)+(cast820*wmma235.y)))))+(cast831*((float)(((cast821*wmma236.y)+(cast822*wmma237.y)))))+(cast832*((float)(((cast823*wmma238.y)+(cast824*wmma239.y))))))));
    unsigned int val683 = (*(buf0+alu195));
    if (0) {
      *(data0_5570560+(alu110+3168)) = buf55;
    }
    int cast833 = ((int)(((signed char)(((val680>>0u)&255u)))));
    int cast834 = ((int)(((signed char)(((val680>>8u)&255u)))));
    int cast835 = ((int)(((signed char)(((val680>>16u)&255u)))));
    int cast836 = ((int)(((signed char)(((val680>>24u)&255u)))));
    int cast837 = ((int)(((signed char)(((val681>>0u)&255u)))));
    int cast838 = ((int)(((signed char)(((val681>>8u)&255u)))));
    int cast839 = ((int)(((signed char)(((val681>>16u)&255u)))));
    int cast840 = ((int)(((signed char)(((val681>>24u)&255u)))));
    buf55 = (buf55+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val683&65535u)))))))*((cast825*((float)(((cast833*wmma232.z)+(cast834*wmma233.z)))))+(cast826*((float)(((cast835*wmma234.z)+(cast836*wmma235.z)))))+(cast827*((float)(((cast837*wmma236.z)+(cast838*wmma237.z)))))+(cast828*((float)(((cast839*wmma238.z)+(cast840*wmma239.z))))))));
    unsigned int val684 = (*(buf0+alu195));
    if (0) {
      *(data0_5570560+(alu110+3169)) = buf56;
    }
    buf56 = (buf56+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val684&65535u)))))))*((cast829*((float)(((cast833*wmma232.w)+(cast834*wmma233.w)))))+(cast830*((float)(((cast835*wmma234.w)+(cast836*wmma235.w)))))+(cast831*((float)(((cast837*wmma236.w)+(cast838*wmma237.w)))))+(cast832*((float)(((cast839*wmma238.w)+(cast840*wmma239.w))))))));
    unsigned int val685 = (*(buf0+alu171));
    unsigned int val686 = (*(buf0+alu172));
    unsigned int val687 = (*(buf0+alu173));
    unsigned int val688 = (*(buf0+alu174));
    unsigned int val689 = (*(buf0+alu175));
    unsigned int val690 = (*(buf0+alu176));
    unsigned int val691 = (*(buf0+alu177));
    unsigned int val692 = (*(buf0+alu178));
    unsigned int val693 = (*(buf0+alu257));
    unsigned int val694 = (*(buf0+alu258));
    unsigned int val695 = (*(buf0+alu259));
    unsigned int val696 = (*(buf0+alu260));
    unsigned int val697 = (*(buf0+alu261));
    unsigned int val698 = (*(buf0+alu262));
    unsigned int val699 = (*(buf0+alu263));
    unsigned int val700 = (*(buf0+alu264));
    unsigned int val701 = (*(buf0+alu180));
    unsigned int val702 = (*(buf0+alu183));
    unsigned int val703 = (*(buf0+alu184));
    unsigned int val704 = (*(buf0+alu188));
    unsigned int val705 = (*(buf0+alu189));
    if (0) {
      *(data0_5570560+(alu110+112)) = buf57;
    }
    char4 cast841 = make_char4(((signed char)(((val685>>0u)&255u))),((signed char)(((val685>>8u)&255u))),((signed char)(((val685>>16u)&255u))),((signed char)(((val685>>24u)&255u))));
    char4 cast842 = make_char4(((signed char)(((val686>>0u)&255u))),((signed char)(((val686>>8u)&255u))),((signed char)(((val686>>16u)&255u))),((signed char)(((val686>>24u)&255u))));
    char4 cast843 = make_char4(((signed char)(((val687>>0u)&255u))),((signed char)(((val687>>8u)&255u))),((signed char)(((val687>>16u)&255u))),((signed char)(((val687>>24u)&255u))));
    char4 cast844 = make_char4(((signed char)(((val688>>0u)&255u))),((signed char)(((val688>>8u)&255u))),((signed char)(((val688>>16u)&255u))),((signed char)(((val688>>24u)&255u))));
    char4 cast845 = make_char4(((signed char)(((val689>>0u)&255u))),((signed char)(((val689>>8u)&255u))),((signed char)(((val689>>16u)&255u))),((signed char)(((val689>>24u)&255u))));
    char4 cast846 = make_char4(((signed char)(((val690>>0u)&255u))),((signed char)(((val690>>8u)&255u))),((signed char)(((val690>>16u)&255u))),((signed char)(((val690>>24u)&255u))));
    char4 cast847 = make_char4(((signed char)(((val691>>0u)&255u))),((signed char)(((val691>>8u)&255u))),((signed char)(((val691>>16u)&255u))),((signed char)(((val691>>24u)&255u))));
    char4 cast848 = make_char4(((signed char)(((val692>>0u)&255u))),((signed char)(((val692>>8u)&255u))),((signed char)(((val692>>16u)&255u))),((signed char)(((val692>>24u)&255u))));
    int4 wmma240 = __WMMA_8_16_16_signed_char_int(alu928, cast841, cast0);
    int4 wmma241 = __WMMA_8_16_16_signed_char_int(alu929, cast842, cast0);
    int4 wmma242 = __WMMA_8_16_16_signed_char_int(alu930, cast843, cast0);
    int4 wmma243 = __WMMA_8_16_16_signed_char_int(alu931, cast844, cast0);
    int4 wmma244 = __WMMA_8_16_16_signed_char_int(alu932, cast845, cast0);
    int4 wmma245 = __WMMA_8_16_16_signed_char_int(alu933, cast846, cast0);
    int4 wmma246 = __WMMA_8_16_16_signed_char_int(alu934, cast847, cast0);
    int4 wmma247 = __WMMA_8_16_16_signed_char_int(alu935, cast848, cast0);
    int cast849 = ((int)(((signed char)(((val702>>0u)&255u)))));
    int cast850 = ((int)(((signed char)(((val702>>8u)&255u)))));
    int cast851 = ((int)(((signed char)(((val702>>16u)&255u)))));
    int cast852 = ((int)(((signed char)(((val702>>24u)&255u)))));
    int cast853 = ((int)(((signed char)(((val703>>0u)&255u)))));
    int cast854 = ((int)(((signed char)(((val703>>8u)&255u)))));
    int cast855 = ((int)(((signed char)(((val703>>16u)&255u)))));
    int cast856 = ((int)(((signed char)(((val703>>24u)&255u)))));
    float cast857 = tg_bitcast<float>((unsigned int)(val693));
    float cast858 = tg_bitcast<float>((unsigned int)(val694));
    float cast859 = tg_bitcast<float>((unsigned int)(val695));
    float cast860 = tg_bitcast<float>((unsigned int)(val696));
    buf57 = (buf57+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val701&65535u)))))))*((cast857*((float)(((cast849*wmma240.x)+(cast850*wmma241.x)))))+(cast858*((float)(((cast851*wmma242.x)+(cast852*wmma243.x)))))+(cast859*((float)(((cast853*wmma244.x)+(cast854*wmma245.x)))))+(cast860*((float)(((cast855*wmma246.x)+(cast856*wmma247.x))))))));
    unsigned int val706 = (*(buf0+alu180));
    if (0) {
      *(data0_5570560+(alu110+113)) = buf58;
    }
    float cast861 = tg_bitcast<float>((unsigned int)(val697));
    float cast862 = tg_bitcast<float>((unsigned int)(val698));
    float cast863 = tg_bitcast<float>((unsigned int)(val699));
    float cast864 = tg_bitcast<float>((unsigned int)(val700));
    buf58 = (buf58+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val706&65535u)))))))*((cast861*((float)(((cast849*wmma240.y)+(cast850*wmma241.y)))))+(cast862*((float)(((cast851*wmma242.y)+(cast852*wmma243.y)))))+(cast863*((float)(((cast853*wmma244.y)+(cast854*wmma245.y)))))+(cast864*((float)(((cast855*wmma246.y)+(cast856*wmma247.y))))))));
    unsigned int val707 = (*(buf0+alu185));
    if (0) {
      *(data0_5570560+(alu110+1136)) = buf59;
    }
    int cast865 = ((int)(((signed char)(((val704>>0u)&255u)))));
    int cast866 = ((int)(((signed char)(((val704>>8u)&255u)))));
    int cast867 = ((int)(((signed char)(((val704>>16u)&255u)))));
    int cast868 = ((int)(((signed char)(((val704>>24u)&255u)))));
    int cast869 = ((int)(((signed char)(((val705>>0u)&255u)))));
    int cast870 = ((int)(((signed char)(((val705>>8u)&255u)))));
    int cast871 = ((int)(((signed char)(((val705>>16u)&255u)))));
    int cast872 = ((int)(((signed char)(((val705>>24u)&255u)))));
    buf59 = (buf59+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val707&65535u)))))))*((cast857*((float)(((cast865*wmma240.z)+(cast866*wmma241.z)))))+(cast858*((float)(((cast867*wmma242.z)+(cast868*wmma243.z)))))+(cast859*((float)(((cast869*wmma244.z)+(cast870*wmma245.z)))))+(cast860*((float)(((cast871*wmma246.z)+(cast872*wmma247.z))))))));
    unsigned int val708 = (*(buf0+alu185));
    if (0) {
      *(data0_5570560+(alu110+1137)) = buf60;
    }
    buf60 = (buf60+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val708&65535u)))))))*((cast861*((float)(((cast865*wmma240.w)+(cast866*wmma241.w)))))+(cast862*((float)(((cast867*wmma242.w)+(cast868*wmma243.w)))))+(cast863*((float)(((cast869*wmma244.w)+(cast870*wmma245.w)))))+(cast864*((float)(((cast871*wmma246.w)+(cast872*wmma247.w))))))));
    unsigned int val709 = (*(buf0+alu257));
    unsigned int val710 = (*(buf0+alu258));
    unsigned int val711 = (*(buf0+alu259));
    unsigned int val712 = (*(buf0+alu260));
    unsigned int val713 = (*(buf0+alu261));
    unsigned int val714 = (*(buf0+alu262));
    unsigned int val715 = (*(buf0+alu263));
    unsigned int val716 = (*(buf0+alu264));
    unsigned int val717 = (*(buf0+alu190));
    unsigned int val718 = (*(buf0+alu193));
    unsigned int val719 = (*(buf0+alu194));
    unsigned int val720 = (*(buf0+alu198));
    unsigned int val721 = (*(buf0+alu199));
    if (0) {
      *(data0_5570560+(alu110+2160)) = buf61;
    }
    int4 wmma248 = __WMMA_8_16_16_signed_char_int(alu952, cast841, cast0);
    int4 wmma249 = __WMMA_8_16_16_signed_char_int(alu953, cast842, cast0);
    int4 wmma250 = __WMMA_8_16_16_signed_char_int(alu954, cast843, cast0);
    int4 wmma251 = __WMMA_8_16_16_signed_char_int(alu955, cast844, cast0);
    int4 wmma252 = __WMMA_8_16_16_signed_char_int(alu956, cast845, cast0);
    int4 wmma253 = __WMMA_8_16_16_signed_char_int(alu957, cast846, cast0);
    int4 wmma254 = __WMMA_8_16_16_signed_char_int(alu958, cast847, cast0);
    int4 wmma255 = __WMMA_8_16_16_signed_char_int(alu959, cast848, cast0);
    int cast873 = ((int)(((signed char)(((val718>>0u)&255u)))));
    int cast874 = ((int)(((signed char)(((val718>>8u)&255u)))));
    int cast875 = ((int)(((signed char)(((val718>>16u)&255u)))));
    int cast876 = ((int)(((signed char)(((val718>>24u)&255u)))));
    int cast877 = ((int)(((signed char)(((val719>>0u)&255u)))));
    int cast878 = ((int)(((signed char)(((val719>>8u)&255u)))));
    int cast879 = ((int)(((signed char)(((val719>>16u)&255u)))));
    int cast880 = ((int)(((signed char)(((val719>>24u)&255u)))));
    float cast881 = tg_bitcast<float>((unsigned int)(val709));
    float cast882 = tg_bitcast<float>((unsigned int)(val710));
    float cast883 = tg_bitcast<float>((unsigned int)(val711));
    float cast884 = tg_bitcast<float>((unsigned int)(val712));
    buf61 = (buf61+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val717&65535u)))))))*((cast881*((float)(((cast873*wmma248.x)+(cast874*wmma249.x)))))+(cast882*((float)(((cast875*wmma250.x)+(cast876*wmma251.x)))))+(cast883*((float)(((cast877*wmma252.x)+(cast878*wmma253.x)))))+(cast884*((float)(((cast879*wmma254.x)+(cast880*wmma255.x))))))));
    unsigned int val722 = (*(buf0+alu190));
    if (0) {
      *(data0_5570560+(alu110+2161)) = buf62;
    }
    float cast885 = tg_bitcast<float>((unsigned int)(val713));
    float cast886 = tg_bitcast<float>((unsigned int)(val714));
    float cast887 = tg_bitcast<float>((unsigned int)(val715));
    float cast888 = tg_bitcast<float>((unsigned int)(val716));
    buf62 = (buf62+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val722&65535u)))))))*((cast885*((float)(((cast873*wmma248.y)+(cast874*wmma249.y)))))+(cast886*((float)(((cast875*wmma250.y)+(cast876*wmma251.y)))))+(cast887*((float)(((cast877*wmma252.y)+(cast878*wmma253.y)))))+(cast888*((float)(((cast879*wmma254.y)+(cast880*wmma255.y))))))));
    unsigned int val723 = (*(buf0+alu195));
    if (0) {
      *(data0_5570560+(alu110+3184)) = buf63;
    }
    int cast889 = ((int)(((signed char)(((val720>>0u)&255u)))));
    int cast890 = ((int)(((signed char)(((val720>>8u)&255u)))));
    int cast891 = ((int)(((signed char)(((val720>>16u)&255u)))));
    int cast892 = ((int)(((signed char)(((val720>>24u)&255u)))));
    int cast893 = ((int)(((signed char)(((val721>>0u)&255u)))));
    int cast894 = ((int)(((signed char)(((val721>>8u)&255u)))));
    int cast895 = ((int)(((signed char)(((val721>>16u)&255u)))));
    int cast896 = ((int)(((signed char)(((val721>>24u)&255u)))));
    buf63 = (buf63+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val723&65535u)))))))*((cast881*((float)(((cast889*wmma248.z)+(cast890*wmma249.z)))))+(cast882*((float)(((cast891*wmma250.z)+(cast892*wmma251.z)))))+(cast883*((float)(((cast893*wmma252.z)+(cast894*wmma253.z)))))+(cast884*((float)(((cast895*wmma254.z)+(cast896*wmma255.z))))))));
    unsigned int val724 = (*(buf0+alu195));
    if (0) {
      *(data0_5570560+(alu110+3185)) = buf64;
    }
    buf64 = (buf64+(((float)(tg_bitcast<half>((unsigned short)(((unsigned short)((val724&65535u)))))))*((cast885*((float)(((cast889*wmma248.w)+(cast890*wmma249.w)))))+(cast886*((float)(((cast891*wmma250.w)+(cast892*wmma251.w)))))+(cast887*((float)(((cast893*wmma252.w)+(cast894*wmma253.w)))))+(cast888*((float)(((cast895*wmma254.w)+(cast896*wmma255.w))))))));
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