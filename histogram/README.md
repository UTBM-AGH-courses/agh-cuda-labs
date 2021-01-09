# Histogram algorithm - CUDA implementation

## Introduction

This algorithm is entended to create an histogram by taking in input an array of integer with a length given by the user (by default set at 256). 

The kernel used to create the histogram is run two times :
* First time : Running on `MULTIPROCESSOR_COUNT*128` block of `min(INPUT_ARRAY_LENGTH*WARP_SIZE, 1024)` threads
* Second time : Running on `1` block of `1` thread

At the end, we compare both histograms, the process time and the performances.

## Memory movements

1) In order to acheive computation of the two histograms, we need three arrays hosted on the **HOST** :

   * `data` : Will contain the random number with a size of `sizeof(unsigned int) * DATA_SIZE` and a max value of `MAX_BINS - 1`
   * `histogram_t` : Will contain the histogram with a size of `sizeof(unsigned int) * MAX_BINS` computed with paralel processes
   * `histogram_one` : Will contain the histogram with a size of `sizeof(unsigned int) * MAX_BINS` computed with only one thread

2) Then we will copy `data` into the **DEVICE** memory to be accessed during the computation by using `cudaMemCopy(cudaMemcpyHostToDevice)`. This will take the same memory size than `data` stored on the **HOST** (`sizeof(unsigned int) * DATA_SIZE`). Then request a memory space sized `sizeof(unsigned int) * MAX_BINS` for `d_histogram` intended to store the *histogram* during the computation time.

3) When launched, the first kernel will read the *data* and build the *histogram*.

4) When the first kernel is completed, we have to copy the result stored on `d_histogram` (**DEVICE**) on `histogram_t` (**HOST**) array in order to be check with the second kernel result at the end of the program.

5) Then, `d_histogram` is "clean" in order to not interfere with the data computed by the launch of the second kernel. So a kernel is launched to fill `d_histogram` with *0*.

6) When done, the second kernel with only one thread is launched. When done, the result is copied from `d_histogram` (**DEVICE**) to `histogram_one` (**HOST**)

7) After comparing the two histogram, we free all memory spaces on the **HOST** (`data`,`histogram_t`, `histogram_one`) and on the **DEVICE** (`d_histogram`)


## Shared memory esential step

When using shared memory we have to be sure to have all our threads synchronize thank to calling `__syncthreads()`. If we don't call this, it may result to concurent access to a given address or don't wait enought time to all threads load the shared memory data. In our case this could bring incoherent values into our final histogram


## Speed up the code execution

To speed up the code execution, we need to take care to use a numer of `threads` which is a mutiple of the *warp size* of our **DEVICE**. This indeed allow the threads to be more *groupebale* when executed by the hardware. The second important aspect is to maximize the occupency. In other words trying to avoid too much threads (1024T on 2B for example) or too much blocks (64T on 512B for example)


## Comment on shared memory

`staticReverse(int *d, int n)` and `void dynamicReverse(int *d, int n)` reverse array but they differ in the shared memory assignation.

They both both reverse the array using the same technic. In a first place an array containing values from **0** to **63** is generated. Each thread will pick on value of the array `d` and storing it into a shared array.
When all threads reached `__syncthreads`, `d` and `s` are equal. From now, the array will be reversed using the data stored into `s`

However, in the *dynamic* approach, we use the `extern` key word and the empty brackets `s[]` which imply that we don't now the size of the shared memory at the compilation time (like for the histogram algorithm, we pass `MAX_BINS` bin argument to the kernel). So to pass the chosen size, we use the third argument in the kernel call : `<<<BLOCK_COUNT, THREAD_COUNT, SHARED_MEMORY_SIZE>>>`
In the *static* approach, the size of the shared memory which will be use is directly put into the kernel definitio : `s[64]`

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



