// System includes
#include <stdio.h>
#include <assert.h>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>

#include <chrono>
using namespace std::chrono;


__global__ void gpu_add_fun(int *a, int *b, int *c) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] + b[index];
}


int local_add(int N, int *a, int *b, int *c) {

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  for (int i=0; i<N; i++) c[i] = a[i] + b[i];
  high_resolution_clock::time_point t2 = high_resolution_clock::now();

  auto duration = duration_cast<microseconds>( t2 - t1 ).count();
  printf("\nLocal:            Elapsed time %f: msec.   ", duration/1000.0f);

  long s = 0;
  for (int i=0; i<N; i++) s += (c[i]);

  return s;
}


int random_ints(int *x, int N) {

  srand (time(NULL));
  for (int i = 0; i<N; i++) x[i] = (rand() % 3 - 1);

  return 0;
}


int gpu_add(int N, int m, int *a, int *b, int *c, int *c_gpu) {

  // for measuring execution time
  cudaError_t error;
  cudaEvent_t start;
  error = cudaEventCreate(&start);
  cudaEvent_t stop;
  error = cudaEventCreate(&stop);

  int *d_a, *d_b, *d_c;					// gpu variables
  int size = N * sizeof(int);				// allocation space size

  // Allocate space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  error = cudaEventRecord(start, 0);

  // Launch add() kernel on GPU with N blocks 
  gpu_add_fun<<<N/m,m>>>(d_a, d_b, d_c);
  
  error = cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); 
  
  // calculate execution time
  float msecTotal = 0.0f;
  error = cudaEventElapsedTime(&msecTotal, start, stop);
  printf("Elapsed time %f: msec.   ", msecTotal);

  // Copy result back to host
  cudaMemcpy(c_gpu, d_c, size, cudaMemcpyDeviceToHost);

  // Variable cleanup
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

  long s = 0;
  for (int i=0; i<N; i++) s += (c_gpu[i]);

  return s;
}



int main(void)
{
  // timer
  int N = 1<<15; //2^15
  int size = N * sizeof(int);

  int *a, *b, *c, *c_gpu;				// local variables
  long sum = 0;
  
  // Alloc space for host copies of a, b, c and setup input values 
  a = new int[size]; random_ints(a, N);
  b = new int[size]; random_ints(b, N);
  c = new int[size];
  c_gpu = new int[size];


  sum = local_add(N, a, b, c);
  printf("Sum diff: %ld\n", sum);

  
  for (int i=1; i<=(1<<15); i*=2) {
    printf("GPU i = %5d.    ", i);
    sum = gpu_add(N, i, a, b, c, c_gpu); 
    printf("Sum diff: %ld\n", sum);
  }

  // local variable cleanup
  delete [] a; delete [] b; delete [] c;

  return 0;
}











