
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size)	y[i] += scale * x[i];
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	// Initialize host and device varibales
	float * x, * y, * y_dup;	// host (cpu) variables
	float * x_d, * y_d;			// device (gpu) variables

	// Memory allocation for host variables
	x = (float *) malloc(vectorSize * sizeof(float));
	y = (float *) malloc(vectorSize * sizeof(float));
	y_dup = (float *) malloc(vectorSize * sizeof(float));

	if (x == NULL || y == NULL || y_dup == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	// Assign random values to x and y
	for (int i = 0; i < vectorSize; i++) {
		x[i] = (float)rand() / (float)rand();
		y[i] = (float)rand() / (float)rand();
	}
	//	y_dup = y
	std::memcpy(y_dup, y, vectorSize * sizeof(float));
	float scale = (float)rand() / (float)rand();

	// Allocate memory for device variables and copy values from host variables
	cudaMalloc((void **) &x_d, vectorSize * sizeof(float));
	cudaMemcpy(x_d, x, vectorSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void **) &y_d, vectorSize * sizeof(float));
	cudaMemcpy(y_d, y, vectorSize * sizeof(float), cudaMemcpyHostToDevice);

	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" x = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", x[i]);
		}
		printf(" ... }\n");
		printf(" y = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", y[i]);
		}
		printf(" ... }\n");
	#endif

	// Run device code
	saxpy_gpu<<<ceil(vectorSize/256.0), 256>>>(x_d, y_d, scale, vectorSize);
	cudaMemcpy(y, y_d, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);

	#ifndef DEBUG_PRINT_DISABLE 
		printf("\nAfter SAXPY, y = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", y[i]);
		}
		printf(" ... }\n");
	#endif

	// Verify saxpy operation using cpu
	int errorCount = 0;
	for (int i = 0; i < vectorSize; i++) {
		if (y[i] - scale * x[i] + y_dup[i] < 1e-5) {
			errorCount++;
			#ifndef DEBUG_PRINT_DISABLE
				std::cout << "Idx " << i << " expected " << scale * x[i] + y_dup[i] 
					<< " found " << y[i] << " = " << scale << " * " << x[i] << " + " << y_dup[i] << "\n";
			#endif
		}
	}
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	// Free memory
	cudaFree(x_d);
	cudaFree(y_d);
	free(x);
	free(y);
	free(y_dup);


	std::cout << "\nLazy, you are!\n";
	std::cout << "Write code, you must\n";
	std::cout << ":( Sorry. Done, it is\n";

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	float x, y;

	// Setup RNG
	curandState_t rng;
	curand_init(clock64(), idx, 0, &rng);

    // Get points
	if (idx < pSumSize) {
		for (uint64_t i = 0; i < sampleSize; i++) {
			x = curand_uniform(&rng);
			y = curand_uniform(&rng);
			pSums[idx] += (uint64_t) 1 - (uint64_t) (x*x + y*y);	// hit if distance from origin to (x, y) is less than 1
		}
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//  Insert code here
	// Initialize variables
	uint64_t *pSums;
	uint64_t *pSums_d;
	double hits = 0;

	pSums = (uint64_t *) malloc(generateThreadCount * sizeof(uint64_t));
	cudaMalloc((void **) &pSums_d, generateThreadCount * sizeof(uint64_t));

	if (pSums == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	// Generate points
	generatePoints<<<ceil(generateThreadCount/256.0), 256>>>(pSums_d, generateThreadCount, sampleSize);
	cudaMemcpy(pSums, pSums_d, generateThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	// Calculate hits
	for (uint64_t i = 0; i < generateThreadCount; i++) {
		hits += pSums[i];
	}

	// Calculate pi
	approxPi = 4.0 * hits / ((double)sampleSize *(double)generateThreadCount);

	free(pSums);
	cudaFree(pSums_d);

	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n\n";
	return approxPi;
}
