#pragma once

#include <iostream>
#include <cudnn.h>

// This will output the proper CUDA error strings
// in the event that a CUDA host call returns an error
template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    std::cerr << "CUDA error at " << file << ':' << line << " code=" << static_cast<unsigned int>(result);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

//function to print out error message from cuDNN calls
#define checkCUDNN(exp) \
{ \
    cudnnStatus_t status = (exp); \
    if(status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "Error on line " << __FILE__ << ':' << __LINE__ << ": " \
                << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
}
