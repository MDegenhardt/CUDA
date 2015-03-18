

#include "BMPLoader.h"


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

	std::vector<RGB> pixelsRead;
	BMPLoader b;
	int idx = 0;

	//Bild importieren
	pixelsRead = b.loadBMP((filename[idx]).c_str());

	//Bild bearbeiten (Werte in BMPLoader einstellen)
	std::vector<RGB> pxBW = b.makeBlackAndWhite(pixelsRead);

	//Bild schreiben
	b.writeBMP(pxBW, (newFilename[idx]).c_str());

	return 0;
}
