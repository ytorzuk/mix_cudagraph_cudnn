#pragma once
#include <cudnn.h>

struct ActivationParams
{
    float* input;
    float* output;
    unsigned int size;
    cudnnHandle_t cudnn;
};

void activation(ActivationParams* params);
