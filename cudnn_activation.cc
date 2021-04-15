#include "cudnn_activation.h"
#include "errors.h"
#include <cudnn.h>
#include <stdio.h>
#include <iostream>
#include <cmath>


void activation(ActivationParams* params)
{
    auto size = params->size;
    auto input = params->input;
    auto output = params->output;

    std::cout << "Input size: " << size << '\n';
    cudnnActivationDescriptor_t activDesc;

    checkCUDNN(cudnnCreateActivationDescriptor(&activDesc));

    checkCUDNN(cudnnSetActivationDescriptor(activDesc,
                                            CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN,
                                            0.0));

    cudnnTensorDescriptor_t in_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(in_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1, size,
                                          1,
                                          1));

    cudnnTensorDescriptor_t out_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&out_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(out_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1, size,
                                          1,
                                          1));

    float alpha = 1.0f;
    float beta = 0.0f;
    checkCUDNN(cudnnActivationForward(params->cudnn,
                                      activDesc,
                                      &alpha,
                                      in_desc,
                                      input,
                                      &beta,
                                      out_desc,
                                      output));

    //free cuDNN descriptors
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
}
