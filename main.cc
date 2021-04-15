#include <iostream>
#include "graph_cudnn.h"

constexpr int kInputSize = 10;

//float ref[kInputSize] = {0.00000000f,0.46318550f,0.57272375f,0.57411450f,1.24199620f,0.00000000f,0.00000000f,0.00000000f,0.00000000f,0.00000000f};

int main(int argc, const char ** argv)
{
    const float input[kInputSize] = {-0.24822682f,0.46318550f,0.57272375f,0.57411451f,1.24199617f,-1.25960900f,-0.36175220f,-0.58040797f,-0.40433765f,-1.21488504f};;
    const float b[kInputSize] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};
    float output[kInputSize];

    buildAndRunCudaGraph(output, input, b, kInputSize);

    for (int i = 0; i < kInputSize; i++)
        std::cout << "c[" << i << "]=" << output[i] << '\n';

    return 0;
}
