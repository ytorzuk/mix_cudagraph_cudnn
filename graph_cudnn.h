#include <cuda_runtime.h>

cudaError_t buildAndRunCudaGraph(float* c, const float* input, const float* bias, unsigned int size);
