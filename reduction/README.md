# Reduction algorithm - CUDA implementation

## Introduction

This algorithm is entended to compute the sum of each elements of a randomly generated array.

The kernel used to create this sum is launched with the following arguments :
* *Block count* : `DATA_SIZE/32`
* *Thread count* : `1024`
* *Shared memory size* : `2 * (DATA_SIZE/32) * sizeof(unsiged int)`

Before lauching the kernel, we compute this sum on the host to compare the result witht the kernel output.

You can directly run the program on the *lhcbgpu2* machine : `/home/valrev_cuda/labs/reduction/reduction.o -dSize=1024`

## Why the reduction algorithm is so vital, start from naming a few problems that represent this type of processing ?

This type of algorithm is intended to be launch with very huge set of data. Running this using "conventional" method like on one thread composed of one *for* loop may take a huge amount of CPU clock. This type of algorithm has the advantage of drasticly reduce the number of iteration to compute the result. The fist thread will handle the most computation but the greater the index, the lower the intensity of workload will be necessary. For an array with 2^N elements, there will be only N steps to get the sum of all elements againts 2^N for a "conventional" *for* loop.

However, running this kind of algorithm may be sometimes quite tricky because the datasize has to fit the grid dimension of our kernel, which may induce some pre process in our code.


## Analyse the workload of each thread. Estimate the min, max and averagenumber  of  real operations that a thread will perform(i.e., when a thread calculate a number contributing to the final result). How many times a single  block will synchronise to before finishing its calculations (until a single partial sum is obtained)

Let's assume our array have 2^N elements. We have seen above that the computation need N steps to be performed. The first thread (index 0) will work the most by having N operations to execute and the last one will have 0 operaion to execute. The first iteration require 2^N/2 working thread (all thread with an idex divisible by 2). The second iteration will only require 2^N/4 (all thread with an idex divisible by 4) on so on until the last operaion hich require only one thread to be executed.

For example, the average operations per thread for 32 (=2⁵) values will be given by : 
                            `(1/16 + 2/8 + 3/4 + 4/2 + 8/1) / 5 = 2.21 operations/thread`

With a sing threaded application, all the work will be done on one thread : 32 operations/thread

In term of synchronisation, each blocks will synchronize N+1 times

## Describe the possible optimisations to the code

My current implemenation is not the optimized because I use blocks of 1024 for every datasize. I only increase the number of working block by divided it by the data size. This may result in a huge loss of computational density. For example, an array of 1025 elements will end up generate a grid of two blocks of 1024 threads but the `local_sum` of the second block will only contains the 1025th elements followed by 1023 "0".

My block count assignation may be a little bit tweaked too because I don't use the SM count to determine it.

I think for huge array, the implementation works well but for little data set (< 10000 values I think) and I get pretty good results (11.1 GFlop/s for 100 millions values), 

## Is the use of atomic operations advisable in this case ?

I actually ended up using the atomic operations `atomicAdd()` for the last operation where every first thread of each blocks add their value to the first element of the array in the global mememory. This operation may end up with access confilcts and errors in the final value

The use `__syncthreads()` help us quite well because exept for the last operation mentionned above, each thread assing a value which is never accessed by another one. They just need to be synchronis in order to have the correct values for the next operation.

## Think in a more abstract way about the reduction algorithm. Does the operation being parallelised need to be commutative?

The operations have to be commutatives (+ or x). In fact, if we try to substracts every elements of an array, the result will be wrong :

 3   4   5   7
-1      -2
 1
 
 Here the last operation gave us **1** but the correct resilt is *3-4-5-7=-13*
