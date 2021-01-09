#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <iostream>
#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#define MAX_BINS 256

int singleThreadedSum (float tab[], int len)
{
    int res = 0;
    for (int i = 0 ; i < len; i++)
    {
        res += tab[i];
    }
    return res;
}



__global__ 
static void reductionKernal(const float *input, float *output)
{
    extern __shared__ float partSum[];
    unsigned int th = threadIdx.x;
    partSum[th] = input[th];
    partSum[th + blockDim.x] = input[th + blockDim.x];
    for (int stride = blockDim.x; stride > 0 ; stride /= 2)
    {
        __syncthreads();
        partSum[th] += partSum[th+stride];
    }
    __syncthreads();
    if (th == 0){
        output[0] = partSum[0];
    }
}


int main(int argc, char **argv)
{
    int lengthTab;

    //Doesn't allow more than 2048 length as 1024 is max threads/block
    if (lengthTab > 2048){printf("LengthTab is > to the possible number of threads \n");  exit(EXIT_FAILURE);}


    std::cout<<std::endl;

}


void reductionWrapper(int dataSize, int display, int threadCount, int blockCount)
{
    float *input = NULL;
    float *dinput = NULL;
    float *doutput = NULL;
    float *output = NULL;
    cudaEvent_t start;
    cudaEvent_t stop;

    // Generate the structures
    output = (float *)malloc(threadCount * 2 * sizeof(float));
    input = (float *)malloc(threadCount * 2 * sizeof(float));

    // Generate data
    for (int i = 0; i < threadCount * 2; i++)
    {
        input[i] = rand() % MAX_BINS;
    }

    // Assing memory on device
    checkCudaErrors(cudaMalloc((void **)&dinput, sizeof(float) * threadCount * 2));
    checkCudaErrors(cudaMalloc((void **)&doutput, sizeof(float) * threadCount * 2));
    checkCudaErrors(cudaMemcpy(dinput, input, sizeof(float) * threadCount * 2, cudaMemcpyHostToDevice));


    // Allocating CUDA events that we'll use for timing
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, NULL));

    // Launch the kernel
    reductionSum<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) *  NUM_THREADS * 2>>>(dinput, doutput);
    cudaDeviceSynchronize();

    // Record stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Fetch the data
    checkCudaErrors(cudaMemcpy(output, doutput, sizeof(float) * NUM_THREADS * 2, cudaMemcpyDeviceToHost));

    // Compute results
    float msecTotal = 0.0f;
    printf("The result of the multithreaded function is: %f \n", output[0]);
    printf("Elapsed Time for reduction function to complete is : %f msec \n", msecTotal);

    // Run on single thread
    int singleThreadRes = singleThreadedSum(input, lengthTab);

    printf("The result on the single thread function is: %d \n", singleThreadRes);

    free(input);
    free(output);
    checkCudaErrors(cudaFree(dinput));
    checkCudaErrors(cudaFree(doutput));

    return EXIT_SUCCESS;
}


int main(int argc, char **argv)
{
    int display = 0;
    int smCount;
    int sharedMemoryPerSm;
    int warpSize;
    int dataSize = 256;
    cudaDeviceProp prop;

    system("clear");
    
    // Get the device    
    int dev = findCudaDevice(argc, (const char **)argv);
    cudaGetDeviceProperties(&prop, dev);
    sharedMemoryPerSm = prop.sharedMemPerMultiprocessor;
    smCount = prop.multiProcessorCount;
    warpSize = prop.warpSize;

    // Get the inputs
    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
        printf("Usage :\n");
        printf("      -dSize=DATA_SIZE [256] (Length of the vector containing the data)\n");
        printf("      -v (Display the data)\n");

        exit(EXIT_SUCCESS);
    }
    printf("CUDA - Sum reduction algorithm\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "nb")) 
    {
        dataSize = getCmdLineArgumentInt(argc, (const char **)argv, "nb");
        if (dataSize > 2048)
        {
            printf("LengthTab is > to the possible number of threads \n");  
            exit(EXIT_FAILURE);
        }
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "verbose"))
    {
        display = 1;
    }


    printf("Data length : %lu\n", u_dataSize);
    if (u_dataSize >= 4294967296 || u_dataSize == 0) {
        printf("Error: Data size must be < 4,294,967,296. Actual: %lu\n", u_dataSize);
        exit(EXIT_FAILURE);
    }

    int threadCount = dataSize/2;
    int blockCount = 1;
    reductionWrapper(dataSize, display, threadCount, blockCount);
    
    return EXIT_SUCCESS;
}