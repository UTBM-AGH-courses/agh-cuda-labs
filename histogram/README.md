# Histogram algorithm - CUDA implementation

## Introduction

This algorithm is entended to create an histogram by taking in input an array of integer with a length given by the user (by default set at 256). 

The kernel used to create the histogram is run two times :
* First time : Running on `MULTIPROCESSOR_COUNT*128` block of `min(INPUT_ARRAY_LENGTH*WARP_SIZE, 1024)` threads
* Second time : Running on `1` block of `1` thread

To finish, we compare the both histograms, the process time and the performances.

## Memory movements

1) In order to acheive computation of the two histograms, we need three arrays hosted on the **HOST** :

   * `data` : Will contain the random number with a size of `sizeof(unsigned int) * DATA_SIZE` and a max value of `MAX_BINS - 1`
   * `histogram_t` : Will contain the histogram with a size of `sizeof(unsigned int) * MAX_BINS` computed with paralel processes
   * `histogram_one` : Will contain the histogram with a size of `sizeof(unsigned int) * MAX_BINS` computed with only one thread

2) Then we will copy `data` into the **DEVICE** memory to be accessed during the computation and request a memory space sized `sizeof(unsigned int) * MAX_BINS` for `d_histogram` intended to store the *histogram* during the computation time.

3) When launched, the first kernel will read the *data* and build the *histogram*.

4) When the first kernel is completed, we have to copy the result on the `histogram_t` array in order to be check with the second kernel result at the end of the program.

5) Then, `d_histogram` is "clean" in order to not interfere with the data computed by the launch of the second kernel. So a kernel is launched to fill `d_histogram` with *0*.

6) When done, the second kernel with only one thread is launch and the result is stored brougth back to the **HOST** in `histogram_one`

7) After comparing the two histogram, we free all memory spaces on the **HOST** (`data`,`histogram_t`, `histogram_one`) and on the device (`d_histogram`)


## Shared memory esential step

When using shared memory we have to be sure to have all our threads synchronize thank to calling `__syncthreads()`. If we don't call this, it may result to concurent access to a given address or don't wait enought time to all threads load the shared memory data.


## Speed up the code execution

// TO DO


## Comment on shared memory

`staticReverse(int *d, int n)` and `void dynamicReverse(int *d, int n)` reverse array but they differ in the shared memory assignation.

In the *dynamic* one, we une the `extern` key word which imply that we don't now the size of the shared memory at the compilation time (like for the histogram algorithm, we pass `MAX_BINS` bin argument to the kernel). So to pass the chosen size, we use the third argument in the kernel call : `<<<BLOCK_COUNT, THREAD_COUNT, SHARED_MEMORY_SIZE>>>`

```cpp
#include 

__global__ void staticReverse(int *d, int n)
{
  __shared__ int s[64];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

__global__ void dynamicReverse(int *d, int n)
{
  extern __shared__ int s[];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

int main(void)
{
  const int n = 64;
  int a[n], r[n], d[n];

  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1;
    d[i] = 0;
  }

  int *d_d;
  cudaMalloc(&d_d, n * sizeof(int)); 

  // run version with static shared memory
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  staticReverse<<<1,n>>>(d_d, n);
  cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);

  // run dynamic shared memory version
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  dynamicReverse<<<1,n,n*sizeof(int)>>>(d_d, n);
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);
}
```



