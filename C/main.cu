//module load LUMI/24.03
//module load partition/G
//module load rocm
// hipcc -munsafe-fp-atomics --offload-arch=gfx90a -O3 main.cu -o hip.out
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
// #include <mpi.h>

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


#define NX 256
#define NY 256

void random_fill(double *array, int nx, int ny) {
    for (int i = 0; i < nx * ny; i++) {
        array[i] = (double)rand() / RAND_MAX;
    }
}

int main(int argc, char *argv[]) {
    double *A_h, *B_h, *C_h, *D_h;
    double *A_d, *B_d;
    int nx = NX, ny = NY;
    int rank, ntasks;
    size_t size = nx * ny * sizeof(double);

    // MPI Initialization
    // MPI_Init(&argc, &argv);
    // MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Allocate host memory
    A_h = (double *)malloc(size);
    B_h = (double *)malloc(size);
    C_h = (double *)malloc(size);
    D_h = (double *)malloc(size);

    // Fill A_h and B_h with random numbers
    random_fill(A_h, nx, ny);
    random_fill(B_h, nx, ny);

    // Set GPU device based on rank
    hipSetDevice(rank);

    // Create GPU stream
    hipStream_t gpu_stream;
    hipStreamCreate(&gpu_stream);

    for (int n_ii = 1; n_ii <= 10000000; n_ii++) {
        // Allocate device memory
        hipMallocAsync(&A_d, size, gpu_stream);
        hipMallocAsync(&B_d, size, gpu_stream);

        // Copy data from host to device
        hipMemcpyAsync(A_d, A_h, size, hipMemcpyHostToDevice, gpu_stream);
        hipMemcpyAsync(B_d, B_h, size, hipMemcpyHostToDevice, gpu_stream);

        // Copy data from device to host (for simplicity, copying back the same data)
        hipMemcpyAsync(D_h, B_d, size, hipMemcpyDeviceToHost, gpu_stream);
        hipMemcpyAsync(C_h, A_d, size, hipMemcpyDeviceToHost, gpu_stream);

        // Free device memory
        hipFreeAsync(A_d, gpu_stream);
        hipFreeAsync(B_d, gpu_stream);

        // Synchronize device
        gpuErrchk( hipDeviceSynchronize() ); //hipStreamSynchronize(gpu_stream);

        // Print the iteration number and difference sums
        double sum_A_diff = 0.0, sum_B_diff = 0.0;
        for (int i = 0; i < nx * ny; i++) {
            sum_A_diff += abs(A_h[i] - C_h[i]);
            sum_B_diff += abs(B_h[i] - D_h[i]);
        }
        printf("Iteration: %d, Sum_A_Diff: %f, Sum_B_Diff: %f\n", n_ii, sum_A_diff, sum_B_diff);
    }

    // Free host memory
    free(A_h);
    free(B_h);
    free(C_h);
    free(D_h);

    // Destroy GPU stream
    hipStreamDestroy(gpu_stream);

    // Finalize MPI
    // MPI_Finalize();

    return 0;
}
