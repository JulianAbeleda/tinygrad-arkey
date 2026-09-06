extern "C" __global__ void q4k_imma_fixup_active(float *out,const float *partials,const int *map,const int *active,int M,int N) {
    int tile=active[blockIdx.x],z=blockIdx.y*4096+threadIdx.x; int s0=map[3*tile+0],s1=map[3*tile+1],s2=map[3*tile+2],nb=(tile%(N/128))*128,mb=(tile/(N/128))*128;
    if(s0<0)return;
    for (;z<((blockIdx.y+1)*4096);z+=128) { int r=z/128,c=z%128;
      out[(mb+r)*N+nb+c]=partials[s0*16384+z]+(s1>=0?partials[s1*16384+z]:0)+(s2>=0?partials[s2*16384+z]:0); }
  }