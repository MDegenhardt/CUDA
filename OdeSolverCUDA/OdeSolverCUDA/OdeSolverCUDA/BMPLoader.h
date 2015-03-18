//
//  BMPLoader.h
//  OdeSolver
//
//  Klasse zum einlesen, bearbeien und schreiben eines BMP-Bildes
//  bearbeiten: schwarz-weiss-Filter, Positionsberechnung der Pixel durch Strahlen im inhomogenen Feld

#ifndef OdeSolver_BMPLoader_h
#define OdeSolver_BMPLoader_h

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <sstream>
#include <vector>


// boost ublas-matrix
//#include <boost/numeric/ublas/matrix.hpp>

//testen ob eine Datei schon existiert
bool file_exists(const std::string& name)
{
	std::ifstream file(name);
	if (!file)    //Datei nicht gefunden -> file ist 0
		return false;    //Datei nicht gefunden
	else         //Datei gefunden -> file ist ungleich 0
		return true;     //Datei gefunden
}


#pragma pack(2)
//size == 3byte
typedef struct
{
	unsigned char  rgbBlue;          /* Blue value */
	unsigned char  rgbGreen;         /* Green value */
	unsigned char  rgbRed;           /* Red value */
} RGB;

//size == 14byte
typedef struct                       /**** BMP file header structure ****/
{
	unsigned short bfType;           /* Magic number for file */
	unsigned int   bfSize;           /* Size of file */
	unsigned short bfReserved1;      /* Reserved */
	unsigned short bfReserved2;      /* ... */
	unsigned int   bfOffBits;        /* Offset to bitmap data */
} BITMAPFILEHEADER;

//size == 40byte
typedef struct                       /**** BMP file info structure ****/
{
	unsigned int   biSize;           /* Size of info header */
	int            biWidth;          /* Width of image */
	int            biHeight;         /* Height of image */
	unsigned short biPlanes;         /* Number of color planes */
	unsigned short biBitCount;       /* Number of bits per pixel */
	unsigned int   biCompression;    /* Type of compression to use */
	unsigned int   biSizeImage;      /* Size of image data */
	int            biXPelsPerMeter;  /* X pixels per meter */
	int            biYPelsPerMeter;  /* Y pixels per meter */
	unsigned int   biClrUsed;        /* Number of colors used */
	unsigned int   biClrImportant;   /* Number of important colors */
} BITMAPINFOHEADER;

class BMPLoader{

private:
	BITMAPFILEHEADER bfh;
	BITMAPINFOHEADER bih;
public:
	std::vector<RGB> pixels;

public:

	// Bild einlesen
	std::vector<RGB> loadBMP(const char* filename) {

		std::ifstream input;
		input.open(filename, std::fstream::in | std::fstream::binary);

		if (!input.is_open()) {
			std::cout << "loadBMP: can not open " << filename << "\n";
			return pixels;
		}

		input.read((char*)&bfh, sizeof(bfh));
		input.read((char*)&bih, sizeof(bih));

		std::cout << "Picture: " << filename << " loaded.\n";
		std::cout << "Width: " << bih.biWidth << ", Height: " << bih.biHeight << ".\n";
		std::cout << "#Pixels: " << bih.biWidth*bih.biHeight << "\n\n";

		RGB pix;
		int pixInLine;

		for (int y = 0; y<bih.biHeight; y++) {
			// wievieltes pixel in einer Zeile (fuer padding)
			pixInLine = 0;

			for (int x = 0; x<bih.biWidth; x++) {

				input.read((char*)&pix, sizeof(pix));
				//                 printf( "PixelR %d: %3d %3d %3d\n", i+1, pix.rgbRed, pix.rgbGreen, pix.rgbBlue );

				pixInLine += sizeof(RGB);
				pixels.push_back(pix);

			}
			// padding
			if (pixInLine % 4 != 0) {
				pixInLine = 4 - (pixInLine % 4);
				input.read((char*)&pix, pixInLine);
			}

		}

		input.close();
		return pixels;

	}

	// Bild schreiben
	void writeBMP(std::vector<RGB> pxVec, const char* flname){

		std::stringstream filename;
		filename << flname << ".bmp";

		//testen, ob Datei schon existiert
		int fileCount = 1;
		while (file_exists(filename.str())) {
			filename.str("");
			filename << flname << "(" << fileCount << ").bmp";
			fileCount++;
		}
		std::string fname = filename.str();
		std::ofstream output(fname);

		output.write((char*)&bfh, sizeof(bfh));
		output.write((char*)&bih, sizeof(bih));

		RGB pix;
		int pixInLine;
		int k = 0;

		for (int y = 0; y<bih.biHeight; y++) {
			// wievieltes pixel in einer Zeile (fuer padding)
			pixInLine = 0;

			for (int x = 0; x<bih.biWidth; x++) {

				pix = pxVec[k];

				output.write((char*)&pix, sizeof(pix));
				printf("PixelWrite %d: %3d %3d %3d\n", k + 1, pix.rgbRed, pix.rgbGreen, pix.rgbBlue);

				pixInLine += sizeof(RGB);
				k++;

			}
			// padding
			if (pixInLine % 4 != 0) {
				pixInLine = 4 - (pixInLine % 4);
				output.write((char*)&pix, pixInLine);
			}

		}


	}

	// Filter fuer schwarz-weiss
	std::vector<RGB> makeBlackAndWhite(std::vector<RGB> pixVec){

		RGB pix, newPix;
		int k = 0;
		double sum = 0;

		for (int j = 0; j<bih.biHeight; j++) {

			for (int i = 0; i<bih.biWidth; i++) {

				pix = pixVec[k];
				sum = pix.rgbRed + pix.rgbGreen + pix.rgbBlue;
				if (sum >= 3.0*255.0 / 2.0) {
					newPix.rgbRed = 255;
					newPix.rgbGreen = 255;
					newPix.rgbBlue = 255;
				}
				else {
					newPix.rgbRed = 0;
					newPix.rgbGreen = 0;
					newPix.rgbBlue = 0;
				}
				pixVec[k] = newPix;
				printf("PixelBW %d: %3d %3d %3d\n", i + 1, pix.rgbRed, pix.rgbGreen, pix.rgbBlue);

				k++;

			}

		}

		return pixVec;

	}




};


#endif
