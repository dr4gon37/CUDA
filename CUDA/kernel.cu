#include "kernel.cuh"

__global__ void colorConvert(Pixel *pixels, unsigned int width, unsigned int height) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;
	//if (x < width && y < height) {
		unsigned char r = pixels[offset].r;
		unsigned char g = pixels[offset].g;
		unsigned char b = pixels[offset].b;
		pixels[offset].r = 0.21f*r;
		pixels[offset].g = 0.71f*g;
		pixels[offset].b = 0.07f*b;
	//}
}

cudaError_t colorConvertWithCuda(Pixel *pixels, unsigned int width, unsigned int height, unsigned int nrChannels)
{
	Pixel *pixels_dev;
	cudaError_t cudaStatus;
	size_t size = width * height;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&pixels_dev, size * sizeof(Pixel));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(pixels_dev, pixels, size * sizeof(Pixel), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	dim3 gridSize(width, height);
	colorConvert <<<gridSize, 1>>> (pixels_dev, width, height);
	cudaStatus = cudaMemcpy(pixels, pixels_dev, size * sizeof(Pixel), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy f1ailed!");
		goto Error;
	}

Error:
	cudaFree(pixels_dev);

	return cudaStatus;
}

