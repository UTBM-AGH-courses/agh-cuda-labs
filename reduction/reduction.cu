#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <numeric> 

#define MAX_BINS 4096


cudaError_t customCudaError(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

void printData(unsigned int *data, unsigned int dataSize)
{
    printf("Data to be process : [");
    for (int i = 0; i < dataSize; i++)
    {
        printf("%d", data[i]);
        if (i != dataSize - 1)
        {
            printf("-");
        }
        if (i == dataSize - 1)
        {
            printf("]\n");
        }
    }
}

__global__
void reductionKernel(unsigned int *data, unsigned int dataSize, unsigned int* globalData)
{

    extern __shared__ unsigned int local_sum[];
    unsigned int th = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Fill with 0 the outbound values 
    local_sum[th] = (i < dataSize ? data[i] : 0);
    local_sum[th + blockDim.x] = (i + blockDim.x*gridDim.x < dataSize ? data[i + blockDim.x* gridDim.x] : 0);

    __syncthreads();

    // Reduction Loop , Interleaved Addressing
    for (unsigned int stride = 1; stride < blockDim.x*2; stride *= 2)
    {
        int index = 2 * stride * th;

        if (index < blockDim.x*2)
            local_sum[index] += local_sum[index + stride];

        __syncthreads();
    }

    // Commit to vram 
    if (th == 0)
    {
        globalData[blockIdx.x] = local_sum[0];
    }
}


unsigned int* reductionWrapper(unsigned int* data, unsigned int dataSize, int threadCount, int blockCount)
{
    unsigned int* finalSum = NULL;
    unsigned int* d_finalSum;
    unsigned int* d_data;
    cudaEvent_t start;
    cudaEvent_t stop;

    // Create structures
    finalSum = (unsigned int *)malloc(sizeof(unsigned int)*blockCount);

    // Assign data into the device
    customCudaError(cudaMalloc((void**)&d_finalSum, blockCount*sizeof(unsigned int)));
    customCudaError(cudaMalloc((void**)&d_data, dataSize*sizeof(unsigned int)));

    // Copy the data
    customCudaError(cudaMemcpy(d_data, data, sizeof(unsigned int) * dataSize, cudaMemcpyHostToDevice));
        
    // Record the start event for the first kernel
    customCudaError(cudaEventCreate(&start));
    customCudaError(cudaEventCreate(&stop));
    customCudaError(cudaEventRecord(start, NULL));

    // Run the kernel
    printf("Lauching kernel on %d threads / %d blocks...\n", threadCount, blockCount);
    reductionKernel<<<blockCount, threadCount, 2*threadCount*sizeof(unsigned int)>>>(d_data, dataSize, d_finalSum);
    customCudaError(cudaDeviceSynchronize());
    printf("Kernel ended\n");

    // Fetch the results
    customCudaError(cudaMemcpy(finalSum, d_finalSum, sizeof(unsigned int) * blockCount, cudaMemcpyDeviceToHost));     

    // Record the stop event for the first event
    customCudaError(cudaEventRecord(stop, NULL)); 
    customCudaError(cudaEventSynchronize(stop));

    printf("################\n");
    float msecTotal = 0.0f;
    customCudaError(cudaEventElapsedTime(&msecTotal, start, stop));
    double gigaFlops = (dataSize * 1.0e-9f) / (msecTotal / 1000.0f);
    printf("Cuda processing time = %.3fms, Performance = %.3f GFlop/s\n",threadCount, msecTotal, gigaFlops);

    // Free the memory
    customCudaError(cudaFree(d_finalSum));
    customCudaError(cudaFree(d_data));

    return finalSum;
}

int main(int argc, char** argv)
{
    unsigned int* data = NULL;
    int smCount;
    int sharedMemoryPerSm;
    int warpSize;
    unsigned int dataSize = 61;
    int display = 0;
    unsigned int hostResult = 0;
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
        printf("      -dSize=DATA_SIZE [256] (Length of the vector containing the data < 10^8)\n");
        printf("      -verbose (Display the data and the histogram)\n");

        exit(EXIT_SUCCESS);
    }
    printf("CUDA - Redution algorithm\n");

    // Init Data Size 
    if (checkCmdLineFlag(argc, (const char**)argv, "dSize")) 
    {
        dataSize = getCmdLineArgumentInt(argc, (const char**)argv, "dSize");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "verbose"))
    {
        display = 1;
    }

    // Allocating memory space for data
    data = (unsigned int *)malloc(sizeof(unsigned int)*dataSize);

    // Generate the data
    printf("Generating data...\n");
    srand(time(NULL));
    for (int i = 0; i < dataSize; ++i)
    {
        data[i] = rand() % MAX_BINS;
        hostResult += data[i];
    }
    printf("Generation done\n");

    // Print the input
    if (display == 1)
    {
	    printData(data, dataSize);
    }

    unsigned int threadCount = 16; 
    unsigned int blockCount  = 2;
    
    unsigned int* finalSum = reductionWrapper(data, dataSize, threadCount, blockCount);

    for (int i = 0; i < blockCount; ++i)
    {
        printf("%d\n", finalSum[i]);
    }

    unsigned int deviceResult = std::accumulate(data, data + dataSize, (unsigned int)0);

    // Compare the results
    printf("################\n");
    if (hostResult == deviceResult)
    {
        printf("OK : Both histogram match\n");
    }
    else
    {
        printf("NOK : Both histogram don't match\n");
    }

    // Cuda free
    free(data);
    free(finalSum);

    exit(EXIT_SUCCESS);
}