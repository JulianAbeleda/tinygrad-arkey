extern "C" __global__ void q6k_imma_fixup_active(float *out,const float *partials,const int *map,const int *active,int M,int N) {
    int tile=active[blockIdx.x],s0=map[3*tile],s1=map[3*tile+1],s2=map[3*tile+2],nb=(tile%(N/128))*128,mb=(tile/(N/128))*128;
    for (int z=threadIdx.x;z<16384;z+=256) { int r=z/128,c=z%128;
      float v=partials[s0*16384+z]; if(s1>=0)v+=partials[s1*16384+z]; if(s2>=0)v+=partials[s2*16384+z];
      out[(mb+r)*N+nb+c]=v; }
  }