#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <assert.h>

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

int* readImagesParameters(char* filename) {
    int* tab = (int*)malloc(sizeof(int)*4);
    FILE* f = fopen(filename, "rb");
    assert(f);
    uint32_t magic_number;
    uint32_t number_of_images;
    unsigned int height;
    unsigned int width;
    (void) !fread(&magic_number, sizeof(uint32_t), 1, f);
    magic_number = swap_endian(magic_number);
    assert(magic_number == 2051);
    tab[0] = magic_number;
    (void) !fread(&number_of_images, sizeof(uint32_t), 1, f);
    tab[1] = swap_endian(number_of_images);
    (void) !fread(&height, sizeof(unsigned int), 1, f);
    tab[2] = swap_endian(height);
    (void) !fread(&width, sizeof(unsigned int), 1, f);
    tab[3] = swap_endian(width);
    return tab;
}

int** readImage(FILE* f, uint32_t height, uint32_t width) {
    unsigned char buffer[width * height];
    int** image = malloc(height * sizeof(int*));
    (void) !fread(buffer, sizeof(buffer), 1, f);
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
    assert(f);
    int* parameters = readImagesParameters(filename);
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
    parameters[0] = swap_endian(magicNumber);
    parameters[1] = swap_endian(numItems);
    return parameters;
}

unsigned int* readLabels(char* filename) {
    FILE* f = fopen(filename, "rb");
    assert(f);
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
                layer->filters[i][j][k] = 2.0*((double)rand() / (double)RAND_MAX)-1.0;
                printf("Filter value: %f\n", layer->filters[i][j][k]);
            }
        }
    }
    return layer;
}

double**** subImagesExtractionForConvolution(double** image, int height, int width) {
    double**** subImage = malloc((height-2) * sizeof(double***));
    for (int i=0; i<(height-2); i++) {
        subImage[i] = malloc((width-2) * sizeof(double**));
        for (int j=0; j<(width-2); j++) {
            subImage[i][j] = malloc(3 * sizeof(double*));
            for (int k=0; k<3; k++) {
                subImage[i][j][k] = malloc(3 * sizeof(double));
                for (int l=0; l<3; l++) {
                    subImage[i][j][k][l] = image[i+k][j+l];
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
    double*** conv = malloc((height-2) * sizeof(double**));
    for (int i=0; i<(height-2); i++) {
        conv[i] = malloc((width-2) * sizeof(double*));
        for (int j=0; j<(width-2); j++) {
            conv[i][j] = malloc(convLayer->numFilters * sizeof(double));
            for (int k=0; k<convLayer->numFilters; k++) {
                conv[i][j][k] = convolution(convLayer->filters[k], grid[i][j]);
                // printf("Convolution: %f\n", conv[i][j][k]);
            }
        }
    }
    return conv;
}

double***** subImagesExtractionForPooling(double*** image, int height, int width, int numFilters) {
    double ***** subImage = malloc(((height-2)/2) * sizeof(double****));
    for (int i=0; i<((height-2)/2); i++) {
        subImage[i] = malloc(((width-2)/2) * sizeof(double***));
        for (int j=0; j<((width-2)/2); j++) {
            subImage[i][j] = malloc(2 * sizeof(double**));
            for (int k=0; k<2; k++) {
                subImage[i][j][k] = malloc(2 * sizeof(double*));
                for (int l=0; l<2; l++) {
                    subImage[i][j][k][l] = malloc(numFilters * sizeof(double));
                    for (int m=0; m<numFilters; m++) {
                        subImage[i][j][k][l][m] = image[2*i+k][2*j+l][m];
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
    // printf("Max: %f\n", max);
    return max;
}

double*** poolingForwardPass(double*** input, int height, int width, int numFilters) {
    double***** grid = subImagesExtractionForPooling(input, height, width, numFilters);
    double*** output = malloc(((height-2)/2) * sizeof(double**));
    for (int i=0; i<((height-2)/2); i++) {
        output[i] = malloc(((width-2)/2) * sizeof(double*));
        for (int j=0; j<((width-2)/2); j++) {
            output[i][j] = malloc(numFilters * sizeof(double));
            for (int k=0; k<numFilters; k++) {
                output[i][j][k] = matrixMax(grid[i][j], 2, 2, k);
            }
        }
    }
    return output;
}

typedef struct {
    int size;
    double** weights;
    double* biases;
} DenseLayer;

DenseLayer* initDenseLayer(int size, int height, int width, int numFilters) {
    DenseLayer* layer = malloc(sizeof(DenseLayer));
    layer->size = size;
    layer->biases = malloc(size * sizeof(double));
    layer->weights = malloc(size * sizeof(double*));
    for (int i=0; i<size; i++) {
        layer->weights[i] = malloc(height*width*numFilters * sizeof(double));
        for (int j=0; j<height*width*numFilters; j++) {
            layer->weights[i][j] = 2.0*((double)rand() / (double)RAND_MAX)-1.0;
            // printf("Weights: %f\n", layer->weights[i][j]);
        }
        layer->biases[i] = 0;
    }
    return layer;
}

/*double* denseLayerForwardPass(DenseLayer* denseLayer, double*** input, int height, int width, int numFilters) {
    double* probs = malloc(height * width * numFilters * sizeof(double));
    for (int i=0; i<height*width*numFilters; i++) {
        probs[i] = input[i-(i-(i%numFilters))%width][(i-(i%numFilters))%width][i%numFilters] * denseLayer->weights[i] + denseLayer->biases[i];
    }
    return probs;
}*/

double* calculations(double* tab, double** weights, double* biases, int height, int width) {
    double* output = malloc(width * sizeof(double));
    for (int i=0; i<width; i++) {
        double sum = 0;
        for (int j=0; j<height; j++) {
            sum += tab[j] * weights[i][j];
        }
        output[i] = sum + biases[i];
        // printf("Ouput[%d]: %f\n", i, output[i]);
    }
    return output;
}

double* denseLayerForwardPass(DenseLayer* denseLayer, double*** input, int height, int width, int numFilters) {
    double* flatInput = malloc(height * width * numFilters * sizeof(double));
    for (int i=0; i<height*width*numFilters; i++) {
        int z = i % numFilters; 
        int x = (i / numFilters) % width;
        int y = i / (width * numFilters);
        flatInput[i] = input[y][x][z];
    }
    double* probs = calculations(flatInput, denseLayer->weights, denseLayer->biases, height*width*numFilters, 10);
    free(flatInput);
    double expSum = 0;
    for (int i=0; i<10; i++) {
        expSum += exp(probs[i]);
    }
    for (int i=0; i<10; i++) {
        probs[i] = (exp(probs[i]) / expSum);
        // printf("Probs[%d]: %f\n", i, probs[i]);
    }
    return probs;
}

double** sanitizeImage(int** image, int height, int width) {
    double** output = malloc(height * sizeof(double*));
    for (int i=0; i<height; i++) {
        output[i] = malloc(width * sizeof(double));
        for (int j=0; j<height; j++) {
            output[i][j] = output[i][j] / 255 - 0.5;
        }
    }
    return output;
}

int maxArrayIndex(double* array, int size) {
    double max = array[0];
    int index = 0;
    for (int i=0; i<size; i++) {
        if (array[i] > max) {
            max = array[i];
            index = i;
        }
    }
    return index;
}

double* forwardPass(ConvLayer* convLayer, DenseLayer* denseLayer, int** image, int height, int width) {
    double** sanitizedImage = sanitizeImage(image, height, width);
    double*** output = convolutionForwardPass(convLayer,  sanitizedImage, height, width);
    output = poolingForwardPass(output, height, width, convLayer->numFilters);
    double* probs = denseLayerForwardPass(denseLayer, output, ((height-2)/2), ((width-2)/2), convLayer->numFilters);
    return probs;
}

double getLoss(double* probs, int label) {
    return -log(probs[label]);
}

int getAccuracy(double* probs, int label) {
    if (maxArrayIndex(probs, 10) == label) return 1;
    else return 0;
}

void epoch() {
    int*** testImages = readImages("./MNIST/train-images.idx3-ubyte");
    int* testLabels = readLabels("./MNIST/train-labels.idx1-ubyte");
    int* parameters = readImagesParameters("./MNIST/train-images.idx3-ubyte");
    ConvLayer* convLayer = initConvLayer(8, 3);
    DenseLayer* denseLayer = initDenseLayer(10, 13, 13, 8);
    printf("CNN Initialized. \n");
    printf("Magic number: %d\n", parameters[0]);
    printf("Number of images: %d\n", parameters[1]);
    printf("Heigt: %d\n", parameters[2]);
    printf("Width: %d\n", parameters[3]);
    double loss = 0;
    int numCorrect = 0;
    double* probs = malloc(10 * sizeof(double));
    for (int i=0; i<parameters[1]; i++) {
        probs = forwardPass(convLayer, denseLayer, testImages[i], parameters[2], parameters[3]);
        loss = getLoss(probs, testLabels[i]);
        numCorrect += getAccuracy(probs, testLabels[i]);
        if (i%100 == 99) {
            printf("[Step %d] Past 100 steps : Average Loss: %f | Accuracy: %d\n", i+1, loss/100, numCorrect);
            numCorrect = 0;
        }
    }
}

int main() {
    srand(time(NULL));
    epoch();
    return 0;
}