#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "Pixel.h"
#define CHANNELS 3

__global__ void colorConvert(Pixel *pixels, unsigned int width, unsigned int height);
cudaError_t colorConvertWithCuda(Pixel *pixels, unsigned int width, unsigned int height, unsigned int nrChannels);

