#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include "import.h"

uint32_t swapEndian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

int* readParameters(char* filename) {
    int* parameters = malloc(3*sizeof(int));
    FILE* f = fopen(filename, "rb");
    assert(parameters != NULL && f != NULL);

    uint32_t magicNumber;
    uint32_t numImages;
    unsigned int height;
    unsigned int width;

    (void) !fread(&magicNumber, sizeof(uint32_t), 1, f);
    (void) !fread(&numImages, sizeof(uint32_t), 1, f);
    (void) !fread(&height, sizeof(unsigned int), 1, f);
    (void) !fread(&width, sizeof(unsigned int), 1, f);

    parameters[0] = (int)swapEndian(numImages);
    parameters[1] = (int)swapEndian(width);
    parameters[2] = (int)swapEndian(height);

    fclose(f);
    return parameters;
}

double** readImage(FILE* f, int width, int height) {
    unsigned char buffer[width * height];
    double** image = malloc(height * sizeof(double*));
    assert(image != NULL);
    (void) !fread(buffer, sizeof(buffer), 1, f);

    for (int i=0; i<height; i++) {
        image[i] = malloc(width * sizeof(double));
        assert(image[i] != NULL);
        for (int j=0; j<width; j++) {
            image[i][j] = buffer[j+i*width] / 255.0;
        }
    }

    return image;
}

double*** readImages(char* filename, int numImages, int width, int height) {
    FILE* f = fopen(filename, "rb");
    double*** images = malloc(numImages * sizeof(double**));
    assert(images != NULL && f != NULL);

    for (int i=0; i<numImages; i++) {
        images[i] = readImage(f, width, height);
    }

    fclose(f);
    return images;
}

int* readLabels(char* filename, int numImages) {
    FILE* f = fopen(filename, "rb");
    int* labels = malloc(numImages * sizeof(int));
    assert(labels != NULL && f != NULL);
    unsigned char buffer[numImages];
    (void) !fread(buffer, sizeof(unsigned char), numImages, f);
    

    for (int i=0; i<numImages; i++) {
        labels[i] = (int)buffer[i];
    }

    fclose(f);
    return labels;
}