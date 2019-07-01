#include "ImageLoader.h"
#include "kernel.cuh"
int main(int argc, char* argv[])
{
	std::string path = "IMAGES\\batgirl.jpg";
	ImageLoader loader;
	loader.loadImage(path.c_str());
	colorConvertWithCuda(loader.getPixels(), loader.getWidth(), loader.getHeight(), loader.getChannels());
	loader.saveImage();
	return 0;
}