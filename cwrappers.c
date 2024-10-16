// rm *.mod *.o *.i
// CC -xhip -munsafe-fp-atomics -g -ggdb  --offload-arch=gfx90a -O3 -c cwrappers.c
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <cmath>

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <assert.h>
#include <hip/hip_complex.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


extern "C" void cuda_malloc_all(void **a_d, size_t Np, hipStream_t *stream )
{
  

  gpuErrchk(hipMallocAsync((void **) a_d,  Np ,stream[0]));
   return;
}


extern "C" void cuda_memset_async(void *a_d, int value,  size_t Np, hipStream_t *stream )
{
  hipMemsetAsync( a_d, value , Np ,stream[0]);
}

extern "C" void cuda_device_reset(){
  hipDeviceReset();
}

extern "C" void cuda_free_async(void **a_d, hipStream_t *stream )
{
  gpuErrchk(hipFreeAsync(*a_d, stream[0]));
   return;
}

extern "C" void cuda_cpy_htod(void *a, void *a_d, size_t N, hipStream_t *stream )
{
  gpuErrchk(hipMemcpyAsync(a_d, a, N, hipMemcpyHostToDevice,stream[0] ));
  //gpuErrchk(hipMemcpy(a_d, a, N, hipMemcpyHostToDevice));
   return;
}


extern "C" void cuda_cpy_dtod(void *b_d, void *a_d,size_t N, hipStream_t* stream )
{
  gpuErrchk(hipMemcpyAsync( a_d, b_d, N, hipMemcpyDeviceToDevice,stream[0]));
  //gpuErrchk(hipMemcpy( a_d, b_d, N, hipMemcpyDeviceToDevice));
   return;
}

extern "C" void cuda_cpy_dtoh(void *a_d, void *a, size_t N, hipStream_t *stream )
{
  gpuErrchk(hipMemcpyAsync(a, a_d,  N, hipMemcpyDeviceToHost,stream[0]));
  //gpuErrchk(hipMemcpy(a, a_d,  N, hipMemcpyDeviceToHost));
   return;
}



extern "C" void create_cublas_handle(hipblasHandle_t *handle,hipStream_t *stream )
{
 	  hipblasCreate(handle);
    hipStreamCreate(stream);
    hipblasSetStream(*handle, *stream);
   return;
}

extern "C" void destroy_cublas_handle(hipblasHandle_t *handle,hipStream_t *stream )
{
 	 // Destroy the handle
   hipblasDestroy(*handle);
   hipStreamDestroy(*stream);
   //printf("\n cublas handle destroyed. \n The End? \n");
   return;
}

extern "C" void cuda_set_device( int my_rank)
{

  int  num_gpus=0;
  int mygpuid;
  gpuErrchk(hipGetDeviceCount(&num_gpus));
  gpuErrchk(hipSetDevice(my_rank%num_gpus));
  gpuErrchk(hipGetDevice(&mygpuid));
  /*gpuErrchk(hipSetDevice(0));*/
  printf("\n Seta Aset at %d %d %d %d\n", num_gpus, my_rank%num_gpus,my_rank, mygpuid);
  //exit(0);
  return;
}

extern "C" void gpu_device_sync()
{
  gpuErrchk( hipDeviceSynchronize() );
}
