#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#define MAX_BINS 1024
#define DATA_SIZE 256
#define WARP_SIZE 32
#define MAX_BLOCKS 1024

__global__ 
static void histogramKernel(unsigned int *inputArray, unsigned int *histogram, int unsigned dataSize, int unsigned binSize)
{
    int th = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int local_histogram[];

    for (int bin = threadIdx.x; bin < binSize; bin += blockDim.x)
    {
        local_histogram[bin] = 0;
    }
    __syncthreads();

    for (int i = th; i < dataSize; i += blockDim.x * gridDim.x)
    {
        atomicAdd(&local_histogram[inputArray[i]], 1);
    }

    __syncthreads();

    for (int bin = th; bin < binSize; bin += blockDim.x)
    {
        atomicAdd(&histogram[bin], local_histogram[bin]);
    }
}

__global__ 
static void cleanHistogram(unsigned int *histogram, int unsigned binSize)
{
    for (int bin = threadIdx.x; bin < binSize; bin += blockDim.x)
    {
        histogram[bin] = 0;
    }
    __syncthreads();

}




void printResult(unsigned int *result, unsigned int resultSize, int threadCount)
{
    printf("Result for %d threads: [", threadCount);
    for (int i = 0; i < resultSize; i++)
    {
        printf("%d", result[i]);
        if (i != resultSize - 1)
        {
            printf("-");
        }
        if (i == resultSize - 1)
        {
            printf("]\n");
        }
    }
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

bool compareResults(unsigned int *array1, unsigned int *array2, int size)
{
    for(int i = 0; i<size; i++)
    {
	if (array1[i] != array2[i])
	{
	    return false;
	}
    }
    return true;
}


void histogramWrapper(unsigned int dataSize, unsigned int binSize, int display)
{
    unsigned int *histogram_t = NULL;
    unsigned int *histogram_one = NULL;
    unsigned int *d_histogram = NULL;
    unsigned int *data = NULL;
    unsigned int *d_data = NULL;
    int threadCount = (dataSize/WARP_SIZE);
    cudaEvent_t start_t;
    cudaEvent_t start_one;
    cudaEvent_t stop_t;
    cudaEvent_t stop_one;

    // Generate the structures
    data = (unsigned int *)malloc(dataSize * sizeof(unsigned int));
    histogram_t = (unsigned int *)malloc(binSize * sizeof(unsigned int));
    histogram_one = (unsigned int *)malloc(binSize * sizeof(unsigned int));

    if (threadCount > 1024)
    { 
 	threadCount = 1024;
    }

    // Generate the data    
    printf("Generating data...\n");
    srand(time(NULL));
    for (int i = 0; i < dataSize; i++)
    {
        data[i] = rand() % binSize;
    }
    printf("Generation done\n");
    
    // Print the input
    if (display == 1)
    {
	printData(data, dataSize);
    }

    // Assing memory on device
    checkCudaErrors(cudaMalloc((void **)&d_histogram, sizeof(unsigned int) * binSize));
    checkCudaErrors(cudaMalloc((void **)&d_data, sizeof(unsigned int) * dataSize));

    // Copy the data
    checkCudaErrors(cudaMemcpy(d_data, data, sizeof(unsigned int) * dataSize, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaEventCreate(&start_t));
    checkCudaErrors(cudaEventCreate(&stop_t));
    checkCudaErrors(cudaEventRecord(start_t, NULL));

    // Record the start event
    printf("Lauching kernel on %d threads...\n", threadCount);
    // Launch the kernel
    histogramKernel<<<1, threadCount,sizeof(unsigned int) * binSize>>>(d_data, d_histogram, dataSize, binSize);
    printf("%d\n", sizeof(unsigned int) * binSize);
    cudaDeviceSynchronize();

    // Fetch the result
    printf("Kernel ended\n");
    checkCudaErrors(cudaMemcpy(histogram_t, d_histogram, sizeof(unsigned int) * binSize, cudaMemcpyDeviceToHost));

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop_t, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop_t));

    checkCudaErrors(cudaEventCreate(&start_one));
    checkCudaErrors(cudaEventCreate(&stop_one));
    checkCudaErrors(cudaEventRecord(start_one, NULL));

    // Clean d_histogram
    printf("Cleaning GPU's histogram...\n");
    cleanHistogram<<<1, threadCount>>>(d_histogram, binSize);
    cudaDeviceSynchronize();
    printf("Cleaning done\n");


    // Record the start event
    printf("Lauching kernel on 1 thread...\n");
    // Launch the kernel
    histogramKernel<<<1, 1,sizeof(unsigned int) * binSize>>>(d_data, d_histogram, dataSize, binSize);
    cudaDeviceSynchronize();

    // Fetch the result
    printf("Kernel ended\n");
    checkCudaErrors(cudaMemcpy(histogram_one, d_histogram, sizeof(unsigned int) * binSize, cudaMemcpyDeviceToHost));

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop_one, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop_one));

    float msecTotal_t = 0.0f;
    float msecTotal_one = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal_t, start_t, stop_t));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal_one, start_one, stop_one));
    double gigaFlops_t = (dataSize * 1.0e-9f) / (msecTotal_t / 1000.0f);
    double gigaFlops_one = (dataSize * 1.0e-9f) / (msecTotal_one / 1000.0f);

    // Print the output
    if (display == 1)
    {
	printResult(histogram_t, binSize, threadCount); 
	printResult(histogram_one, binSize, 1); 
    }
    // Compare the results
    printf("################\n");
    if (compareResults(histogram_t, histogram_one, binSize))
    {
	printf("OK : Both histogram match\n");
    }
    else
    {
	printf("NOK : Both histogram don't match\n");
    }
    // Print time enlapsed
    printf("################\n");
    printf("For %d threads :\nCuda processing time = %.3fms, Performance = %.3f GFLOPS\n",threadCount, msecTotal_t, gigaFlops_t);
    printf("For 1 thread :\nCuda processing time = %.3fms, Performance = %.3f GFLOPS\n", msecTotal_one, gigaFlops_one);
    
    free(histogram_t);
    free(histogram_one);
    free(data);
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_histogram));
}

int main(int argc, char **argv)
{
    unsigned int binSize = MAX_BINS;
    unsigned long long u_dataSize = DATA_SIZE;
    int display = 0;
    char *dataSize = NULL;
    cudaDeviceProp prop;

    system("clear");
    
    // Get the device    
    int dev = findCudaDevice(argc, (const char **)argv);
    cudaGetDeviceProperties(&prop, dev);
    printf("%d\n", prop.sharedMemPerMultiprocessor);

    // Get the inputs
    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
        printf("Usage :\n");
        printf("      -dSize=DATA_SIZE [256] (Length of the vector containing the data)\n");
        printf("      -v (Display the data and the histogram)\n");

        exit(EXIT_SUCCESS);
    }
    printf("CUDA - Histogramming algorithm\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "dSize"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "dSize", &dataSize);
	u_dataSize = atoll(dataSize);

    }

    if (checkCmdLineFlag(argc, (const char **)argv, "verbose"))
    {
        display = 1;
    }


    printf("Length of the data : %lu\n", u_dataSize);
    if (u_dataSize >= 4294967296 || u_dataSize == 0) {
        printf("Error: Data size must be < 4,294,967,296. Actual: %lu\n", u_dataSize);
        exit(EXIT_FAILURE);
    }


    histogramWrapper(u_dataSize, binSize, display);
    
    return EXIT_SUCCESS;
}

