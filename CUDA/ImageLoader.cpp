#include "ImageLoader.h"



ImageLoader::ImageLoader()
{
}


ImageLoader::~ImageLoader()
{
	delete pixels;
}

void ImageLoader::loadImage(const char * path)
{
	unsigned char *data = stbi_load(path, &this->width, &this->height, &this->nrChannels, STBI_rgb);
	if (data)
	{
		std::cout << "Successed loaded texture" << std::endl;
		int size = width * height;
		pixels = new Pixel[size];
		for (int i = 0, j = 0; j < width * height; j++, i = i + nrChannels)
		{
			Pixel p;
			p.r = *(data + i);
			p.g = *(data + i + 1);
			p.b = *(data + i + 2);
			pixels[j] = p;
		}
	}
	else
	{
		std::cout << "Failed to load texture" << std::endl;
		stbi_image_free(data);
		return;
	}
	stbi_image_free(data);
}

void ImageLoader::saveImage()
{
	stbi_write_jpg("output.jpg", this->width, this->height, 3, this->pixels, 100);
}

int ImageLoader::getWidth()
{
	return this->width;
}

int ImageLoader::getHeight()
{
	return this->height;
}

int ImageLoader::getChannels()
{
	return this->nrChannels;
}

Pixel * ImageLoader::getPixels()
{
	return this->pixels;
}
