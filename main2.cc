#include <iostream>
#include "graph_cudnn.h"
#include "cudnn_activation.h"
#include "errors.h"

constexpr int kInputSize = 10;

//float ref[kInputSize] = {0.00000000f,0.46318550f,0.57272375f,0.57411450f,1.24199620f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f};

int main(int argc, const char ** argv)
{
    const float input[kInputSize] = {-0.24822682f,0.46318550f,0.57272375f,0.57411451f,1.24199617f,-1.25960900f,-0.36175220f,-0.58040797f,-0.40433765f,-1.21488504f};;
    float output[kInputSize];

    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    float *dev_input = nullptr;
    float *dev_output = nullptr;
    //allocate arrays on GPU
    cudaMalloc(&dev_input, kInputSize * sizeof(float));
    cudaMalloc(&dev_output, kInputSize * sizeof(float));
    //copy input data to GPU array
    cudaMemcpy(dev_input, input, kInputSize * sizeof(float), cudaMemcpyHostToDevice);
    //initize output data on GPU
    cudaMemset(dev_output, 0, kInputSize * sizeof(float));

    ActivationParams act_params { dev_input, dev_output, kInputSize, cudnn };
    activation(&act_params);

    cudaMemcpy(output, dev_output, kInputSize * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < kInputSize; i++)
        std::cout << "c[" << i << "]=" << output[i] << '\n';

    cudnnDestroy(cudnn);
    //free GPU arrays
    cudaFree(dev_input);
    cudaFree(dev_output);

    return 0;
}
