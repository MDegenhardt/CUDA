#include "BMPLoader.h"

__global__ void kernelMakeBlackAndWhite(RGB* pixVec, RGB* pixVecNew, int imageWidth);

#define NOFTHREADS 128

int main(void)
{

	std::string filename[6];
	filename[0] = "flower.bmp";
	filename[1] = "dimix2.bmp";
	filename[2] = "balloon.bmp";
	filename[3] = "redgreen.bmp";
	filename[4] = "waterdrop.bmp";
	filename[5] = "redgreenSmall.bmp";

	std::string newFilename[6];
	newFilename[0] = "flowerNew";
	newFilename[1] = "dimix2New";
	newFilename[2] = "balloonNew";
	newFilename[3] = "redgreen";
	newFilename[4] = "waterdropNew";
	newFilename[5] = "redgreenSmallNew";

	RGB* readPixels;
	RGB* newPixels;
	RGB* dev_pixels;
	RGB* dev_pixels_new;
	BMPLoader b;
	int idx = 0;

	unsigned int imageSize = b.getImageSize((filename[idx]).c_str());

	//CPU Speicher allokieren
	readPixels = (RGB*)malloc(imageSize);
	newPixels = (RGB*)malloc(imageSize);


	//Bild importieren
	b.loadBMP((filename[idx]).c_str(), readPixels);

	//GPU Speicher allokieren
	cudaMalloc((void**) &dev_pixels, imageSize);
	cudaMalloc((void**) &dev_pixels_new, imageSize);

	//Pixel auf die GPU kopieren
	cudaMemcpy(dev_pixels, readPixels, imageSize, cudaMemcpyHostToDevice);

	int NofBlocks = b.bih.biHeight;
	int NofPixels = b.bih.biHeight*b.bih.biWidth;
	// Kernel launchen mit Anzahl Blocks = Hoehe des Bildes
	kernelMakeBlackAndWhite << <NofBlocks, NOFTHREADS >> >(dev_pixels, dev_pixels_new, NofPixels);

	//Berechnetes Bild zurueckkopieren
	cudaMemcpy(newPixels, dev_pixels_new, imageSize, cudaMemcpyDeviceToHost);

	//GPU Speicher freigeben
	cudaFree(dev_pixels);
	cudaFree(dev_pixels_new);

	//Bild schreiben (von CPU aus)
	b.writeBMP(newPixels, (newFilename[idx]).c_str());

	//CPU Speicher freigeben
	free(readPixels);
	free(newPixels);

	return 0;
}


// Filter fuer schwarz-weiss
__global__ void kernelMakeBlackAndWhite(RGB* pixVec, RGB* pixVecNew, int NofPixels){

	int x = threadIdx.x + blockIdx.x * blockDim.x;

	while (x < NofPixels)
	{
		float sum = pixVec[x].rgbRed + pixVec[x].rgbGreen + pixVec[x].rgbBlue;

		if (sum >= 3.0*255.0 / 2.0) {
			pixVecNew[x].rgbRed = 255;
			pixVecNew[x].rgbGreen = 255;
			pixVecNew[x].rgbBlue = 255;
		}
		else {
			pixVecNew[x].rgbRed = 0;
			pixVecNew[x].rgbGreen = 0;
			pixVecNew[x].rgbBlue = 0;
		}

		x += blockDim.x*gridDim.x;
	
	}

}