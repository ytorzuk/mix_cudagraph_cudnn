#include "graph_cudnn.h"
#include "cudnn_activation.h"
#include "errors.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define VERBOSE true

__global__ void addKernel(float* c, const float* a, const float* b, unsigned int size)
{
    int i = (1 + blockIdx.x) * threadIdx.x;
    c[i] = a[i] + b[i];
}


cudaError_t buildAndRunCudaGraph(float* output, const float* input, const float* bias, unsigned int size)
{
    cudaStream_t streamForGraph;
    cudnnHandle_t cudnn;
    checkCudaErrors(cudaStreamCreateWithFlags(&streamForGraph, cudaStreamNonBlocking));
    checkCUDNN(cudnnCreate(&cudnn));
    checkCUDNN(cudnnSetStream(cudnn, streamForGraph));

    // Original
    float* dev_input = nullptr;
    float* dev_activation = nullptr;
    float* dev_bias = nullptr;
    float* dev_output = nullptr;
    cudaError_t cudaStatus;
    int threads = std::min(256u, size);
    int blocks = (size + threads - 1) / threads;
    // For Graph
    cudaGraph_t graph;
    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t memcpyInputNode, memcpyBiasNode, memcpyOutputNode, kernelNode, cudnnNode;
    cudaKernelNodeParams kernelNodeParams = { 0 };
    cudaHostNodeParams cudnnNodeParams = { 0 };
    cudaMemcpy3DParms memcpyParams = { 0 };
    // Choose which GPU to run on, change this on a multi-GPU system. Then allocate GPU memory.
    {
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n";
        }
        cudaStatus = cudaMalloc((void**)&dev_output, size * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed!\n";
        }
        cudaStatus = cudaMalloc((void**)&dev_input, size * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed!\n";
        }
        cudaStatus = cudaMalloc((void**)&dev_bias, size * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed!\n";
        }
        cudaStatus = cudaMalloc((void**)&dev_activation, size * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed!\n";
        }
    }
    // Start of Graph Creation
    checkCudaErrors(cudaGraphCreate(&graph, 0));
    // Add memcpy nodes for copying input vectors from host memory to GPU buffers
    memset(&memcpyParams, 0, sizeof(memcpyParams));
    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr((void*)input, size * sizeof(float), size, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(dev_input, size * sizeof(float), size, 1);
    memcpyParams.extent = make_cudaExtent(size * sizeof(float), 1, 1);
    memcpyParams.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaGraphAddMemcpyNode(&memcpyInputNode, graph, NULL, 0, &memcpyParams));
//    cudaMemcpy(dev_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

    memset(&memcpyParams, 0, sizeof(memcpyParams));
    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr((void*)bias, size * sizeof(float), size, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(dev_bias, size * sizeof(float), size, 1);
    memcpyParams.extent = make_cudaExtent(size * sizeof(float), 1, 1);
    memcpyParams.kind = cudaMemcpyHostToDevice;
        
    checkCudaErrors(cudaGraphAddMemcpyNode(&memcpyBiasNode, graph, NULL, 0, &memcpyParams));
    nodeDependencies.push_back(memcpyBiasNode);

    // Add a cudnn node for launching a kernel on the GPU
    ActivationParams act_params { dev_input, dev_activation, size, cudnn };
//    activation(&act_params);
    memset(&cudnnNodeParams, 0, sizeof(cudnnNodeParams));
    cudnnNodeParams.fn = reinterpret_cast<cudaHostFn_t>(activation);
    cudnnNodeParams.userData = &act_params;
    checkCudaErrors(cudaGraphAddHostNode(&cudnnNode, graph, &memcpyInputNode, 1, &cudnnNodeParams));
    nodeDependencies.push_back(cudnnNode);

    // Add a kernel node for launching a kernel on the GPU
    memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
    kernelNodeParams.func = (void*)addKernel;
    kernelNodeParams.gridDim = dim3(blocks, 1, 1);
    kernelNodeParams.blockDim = dim3(threads, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;
    void* kernelArgs[4] = { (void*)&dev_output, (void*)&dev_activation, (void*)&dev_bias, &size };
    kernelNodeParams.kernelParams = kernelArgs;
    kernelNodeParams.extra = NULL;
    checkCudaErrors(cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams));
    nodeDependencies.clear();
    nodeDependencies.push_back(kernelNode);

    // Add memcpy node for copying output vector from GPU buffers to host memory
    memset(&memcpyParams, 0, sizeof(memcpyParams));
    memcpyParams.srcArray = NULL;
    memcpyParams.srcPos = make_cudaPos(0, 0, 0);
    memcpyParams.srcPtr = make_cudaPitchedPtr(dev_output, size * sizeof(float), size, 1);
    memcpyParams.dstArray = NULL;
    memcpyParams.dstPos = make_cudaPos(0, 0, 0);
    memcpyParams.dstPtr = make_cudaPitchedPtr(output, size * sizeof(float), size, 1);
    memcpyParams.extent = make_cudaExtent(size * sizeof(float), 1, 1);
    memcpyParams.kind = cudaMemcpyDeviceToHost;
    checkCudaErrors(cudaGraphAddMemcpyNode(&memcpyOutputNode, graph, nodeDependencies.data(), nodeDependencies.size(), &memcpyParams));
    if (VERBOSE) {
        cudaGraphNode_t* nodes = NULL;
        size_t numNodes = 0;
        checkCudaErrors(cudaGraphGetNodes(graph, nodes, &numNodes));
        std::cout << "Num of nodes in the graph created manually " << numNodes << '\n';
    }
    // Create an executable graph from a graph
    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    // Run the graph
    checkCudaErrors(cudaGraphLaunch(graphExec, streamForGraph));
    checkCudaErrors(cudaStreamSynchronize(streamForGraph));
    // Clean up
    checkCudaErrors(cudaGraphExecDestroy(graphExec));
    checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(cudaStreamDestroy(streamForGraph));
    cudnnDestroy(cudnn);
    cudaFree(dev_output);
    cudaFree(dev_input);
    cudaFree(dev_bias);
    cudaFree(dev_activation);
    return cudaStatus;
}
