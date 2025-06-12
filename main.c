#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <assert.h>

int* readImagesParameters(FILE* f) {
    int* parameters = malloc(4 * sizeof(int));
    uint32_t magicNumber;
    uint32_t numImages;
    uint32_t height;
    uint32_t width;
    fread(&magicNumber, sizeof(uint32_t), 1, f);
    fread(&magicNumber, sizeof(uint32_t), 1, f);
    fread(&height, sizeof(uint32_t), 1, f);
    fread(&width, sizeof(uint32_t), 1, f);
    parameters[0] = __builtin_bswap32(magicNumber);
    parameters[1] = __builtin_bswap32(numImages);
    parameters[2] = __builtin_bswap32(height);
    parameters[3] = __builtin_bswap32(width);
    return parameters;
}

int** readImage(FILE* f, uint32_t height, uint32_t width) {
    unsigned char buffer[width * height];
    int** image = malloc(height * sizeof(int*));
    fread(buffer, sizeof(buffer), 1, f);
    for (int i=0; i<height; i++) {
        int* line = malloc(width * sizeof(int));
        for (int j=0; j<width; j++) {
            line[j] = buffer[j+i*width];
        }
        image[i] = line;
    }
    return image;
}

int*** readImages(char* filename) {
    FILE* f = fopen(filename, "rb");
    assert(!f);
    int* parameters = readImagesParameters(f);
    int*** images = malloc(parameters[1] * sizeof(int**));
    for (int i=0; i<parameters[1]; i++) {
        images[i] = readImage(f, parameters[2], parameters[3]);
    }
    return images;
}

int* readLabelParameters(FILE* f) {
    int* parameters = malloc(2 * sizeof(int));
    uint32_t magicNumber;
    uint32_t numItems;
    fread(&magicNumber, sizeof(uint32_t), 1, f);
    fread(&numItems, sizeof(uint32_t), 1, f);
    parameters[0] = __builtin_bswap32(magicNumber);
    parameters[1] = __builtin_bswap32(numItems);
    return parameters;
}

unsigned int* readLabels(char* filename) {
    FILE* f = fopen(filename, "rb");
    assert(!f);
    int* parameters = readLabelParameters(f);
    unsigned char buffer[parameters[1]];
    fread(buffer, sizeof(unsigned char), parameters[1], f);
    unsigned int* labels = malloc(parameters[1] * sizeof(unsigned int));
    for (int i=0; i<parameters[1]; i++) {
        labels[i] = buffer[i];
    }
    return labels;
}