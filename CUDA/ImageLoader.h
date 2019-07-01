#pragma once

#include "Pixel.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <iostream>
class ImageLoader
{

public:
	ImageLoader();
	~ImageLoader();
	void loadImage(const char *path);
	void saveImage();
	int getWidth();
	int getHeight();
	int getChannels();
	Pixel *getPixels();
private:
	Pixel* pixels;
	int width;
	int height;
	int nrChannels;

};

