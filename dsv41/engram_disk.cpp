#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstring>
#include <omp.h>
// Lifetime and buffers are owned by Python; each graph shape owns an immutable job.
struct Job {
 const uint8_t *weight, *scale;
 const int64_t *indices;
 uint8_t *out_weight, *out_scale;
 int64_t n, vocab_rows, dim, block;
};
static void gather(void* opaque){
 auto& j=*(Job*)opaque; const int64_t sd=j.dim/j.block;
 #pragma omp parallel for if(j.n>256) schedule(static)
 for(int64_t i=0;i<j.n;i++){
  int64_t row=j.indices[i];
  if(row>=0 && row<j.vocab_rows){
   memcpy(j.out_weight+i*j.dim,j.weight+row*j.dim,j.dim);
   memcpy(j.out_scale+i*sd,j.scale+row*sd,sd);
  }else{
   memset(j.out_weight+i*j.dim,0,j.dim);
   memset(j.out_scale+i*sd,0,sd);
  }
 }
}
extern "C" int engram_enqueue(uintptr_t stream,Job* job){
 return (int)cudaLaunchHostFunc((cudaStream_t)stream,gather,job);
}
