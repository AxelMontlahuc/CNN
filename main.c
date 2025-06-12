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

typedef struct {
    int numFilters;
    int filterSize;
    double*** filters;
} ConvLayer;

ConvLayer* initConvLayer(int numFilters, int filterSize) {
    ConvLayer* layer = malloc(sizeof(ConvLayer));
    layer->numFilters = numFilters;
    layer->filterSize = filterSize;
    layer->filters = malloc(numFilters * sizeof(double**));
    for (int i=0; i<numFilters; i++) {
        layer->filters[i] = malloc(filterSize * sizeof(double*));
        for (int j=0; j<filterSize; j++) {
            layer->filters[i][j] = malloc(filterSize * sizeof(double));
            for (int k=0; k<filterSize; k++) {
                layer->filters[i][j][k] = (2.0 * rand() / RAND_MAX - 1.0) / 3.0;
            }
        }
    }
    return layer;
}

double**** subImagesExtractionForConvolution(double** image, int height, int width) {
    double**** subImage = malloc((height-2) * sizeof(int***));
    for (int i=0; i<(height-2); i++) {
        subImage[i] = malloc((width-2) * sizeof(int**));
        for (int j=0; j<(width-2); j++) {
            subImage[i][j] = malloc(3 * sizeof(int*));
            for (int k=0; k<3; k++) {
                subImage[i][j][k] = malloc(3 * sizeof(int));
                for (int l=0; l<3; l++) {
                    subImage[i][j][k][l] = image[j+l][i+k];
                }
            }
        }
    }
    return subImage;
}

double convolution(double** filter, double** matrix) {
    double sum = 0;
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            sum += matrix[i][j] * filter[i][j];
        }
    }
    return sum;
}

double*** convolutionForwardPass(ConvLayer* convLayer, double** image, int height, int width) {
    double**** grid= subImagesExtractionForConvolution(image, height, width);
    double*** conv = malloc((height-2) * sizeof(int**));
    for (int i=0; i<(height-2); i++) {
        conv[i] = malloc((width-2) * sizeof(int*));
        for (int j=0; j<(width-1); j++) {
            conv[i][j] = malloc(convLayer->filterSize * sizeof(int));
            for (int k=0; k<convLayer->filterSize; k++) {
                conv[i][j][k] = convolution(convLayer->filters[k], grid[i][j]);
            }
        }
    }
    return conv;
}

double***** subImagesExtractionForPooling(double*** image, int height, int width, int numFilters) {
    double ***** subImage = malloc((height/2) * sizeof(int****));
    for (int i=0; i<(height/2); i++) {
        subImage[i] = malloc((width-2) * sizeof(int***));
        for (int j=0; j<(width/2); j++) {
            subImage[i][j] = malloc((width/2) * sizeof(int**));
            for (int k=0; k<2; k++) {
                subImage[i][j][k] = malloc(2 * sizeof(int*));
                for (int l=0; l<2; l++) {
                    subImage[i][j][k][l] = malloc(numFilters * sizeof(int));
                    for (int m=0; m<numFilters; m++) {
                        subImage[i][j][k][l][m] = image[2*j+l][2*i+k][m];
                    }
                }
            }
        }
    }
    return subImage;
}

double matrixMax(double*** matrix, int height, int width, int cFilter) {
    double max = matrix[0][0][cFilter];
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            if (matrix[i][j][cFilter] > max) max = matrix[i][j][cFilter];
        }
    }
    return max;
}

double*** poolingForwardPass(double*** input, int height, int width, int numFilters) {
    double***** grid = subImagesExtractionForPooling(input, height, width, numFilters);
    double*** output = malloc((height/2) * sizeof(int**));
    for (int i=0; i<(height/2); i++) {
        output[i] = malloc((width/2) * sizeof(int*));
        for (int j=0; j<(width/2); j++) {
            output[i][j] = malloc(numFilters * sizeof(int));
            for (int k=0; k<numFilters; k++) {
                output[i][j][k] = matrixMax(grid[i][j], 2, 2, k);
            }
        }
    }
    return output;
}

typedef struct {
    int size;
    int* weights;
    int* biases;
} DenseLayer;

DenseLayer* initDenseLayer(int size) {
    DenseLayer* layer = malloc(sizeof(DenseLayer));
    layer->size = size;
    double* weights = malloc(size * sizeof(double));
    double* biases = malloc(size * sizeof(double));
    for (int i=0; i<size; i++) {
        weights[i] = 2.0 * rand() / RAND_MAX - 1.0;
        biases[i] = 0;
    }
    return layer;
}

double* denseLayerForwardPass(DenseLayer* denseLayer, double*** input, int height, int width, int numFilters) {
    double* probs = malloc(height * width * numFilters * sizeof(double));
    for (int i=0; i<height*width*numFilters; i++) {
        probs[i] = input[i-(i-(i%numFilters))%width][(i-(i%numFilters))%width][i%numFilters] * denseLayer->weights[i] + denseLayer->biases[i];
    }
    return probs;
}