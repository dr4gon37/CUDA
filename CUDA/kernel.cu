#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHANNELS 3
__global__ void colorConvert(unsigned char * grayImage,
	unsigned char * rgbImage,
	int width, int height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height) {
		int grayOffset = y * width + x;
		int rgbOffset = grayOffset * CHANNELS;
		unsigned char r = rgbImage[rgbOffset]; 
		unsigned char g = rgbImage[rgbOffset + 2]; 
		unsigned char b = rgbImage[rgbOffset + 3]; 
		grayImage[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
	}
}

cudaError_t colorConvertWithCuda(unsigned char* data, unsigned char*dataGrey, unsigned int width, unsigned int height)
{
	unsigned char* dev_data_i = 0;
	unsigned char* dev_data_o = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_data_i, sizeof(data));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_data_o, sizeof(dataGrey));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_data_i, data, sizeof(data), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_data_o, dataGrey, sizeof(dataGrey), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	dim3 blockSize(16, 16, 1);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);
	colorConvert <<<gridSize, blockSize>>> (data, dataGrey, width, height);

Error:
	cudaFree(dev_data_i);
	cudaFree(dev_data_o);

	return cudaStatus;
}


int main()
{
	int width, height, nrChannels;
	int widthG, heightG, nrChannelsG;
	unsigned char *data = stbi_load("C:\\Users\\bsbar\\OneDrive\\Pulpit\\CUDA\\CUDA\\IMAGES\\batgirl.png", &width, &height, &nrChannels, 0);
	unsigned char *dataGrey = stbi_load("C:\\Users\\bsbar\\OneDrive\\Pulpit\\CUDA\\CUDA\\IMAGES\\batgirl.png", &widthG, &heightG, &nrChannelsG, 0);
	if (data && dataGrey)
	{
		std::cout << "Successed loaded texture" << std::endl;
		colorConvertWithCuda(data, dataGrey, width, height);
		stbi_write_png("C:\\Users\\bsbar\\OneDrive\\Pulpit\\CUDA\\CUDA\\IMAGES\\batgirlGray.png", widthG, heightG, nrChannelsG, dataGrey, width * nrChannels);
	}
	else
	{
		std::cout << "Failed to load texture" << std::endl;
	}
	stbi_image_free(data);
	return EXIT_SUCCESS;
}


