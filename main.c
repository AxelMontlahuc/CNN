#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "lib/import.h"
#include "lib/convolution.h"
#include "lib/pooling.h"
#include "lib/dense.h"
#include "lib/output.h"
#include "lib/backprop.h"


double* forward(ConvLayer* convLayer, DenseLayer* denseLayer, double** image, int width, int height, int divisor) {
    double** convolutedImage = convolutionForward(convLayer, image, width, height, divisor);
    double* pooledImage = poolingForward(convolutedImage, (width-(divisor-1))/2, (height-(divisor-1))/2, convLayer->numFilters);
    double* totals = denseForward(denseLayer, pooledImage, (width-(divisor-1))/2, (height-(divisor-1))/2, convLayer->numFilters);
    double* probs = softmax(totals, denseLayer->size);

    for (int i = 0; i < convLayer->numFilters; i++) {
        free(convolutedImage[i]);
    }
    free(convolutedImage);
    free(pooledImage);
    free(totals);
    return probs;
}

void train(ConvLayer* convLayer, DenseLayer* denseLayer, int epoch, double learningRate) {
    char* imagesPath = "./MNIST/train-images.idx3-ubyte";
    char* labelsPath = "./MNIST/train-labels.idx1-ubyte";
    int* parameters = readParameters(imagesPath);
    double*** testImages = readImages(imagesPath);
    int* testLabels = readLabels(labelsPath);

    printf("Number of images: %d\n", parameters[0]);
    printf("Heigt: %d\n", parameters[1]);
    printf("Width: %d\n", parameters[2]);
    
    double* probs = malloc(denseLayer->size * sizeof(double));
    for (int j=0; j<epoch; j++) {
        double l = 0;
        int correct = 0;
        for (int i=0; i<parameters[0]; i++) {
            probs = backpropagation(convLayer, denseLayer, testImages[i], parameters[1], parameters[2], convLayer->filterSize, testLabels[i], learningRate);
            l += loss(probs, testLabels[i]);
            correct += accuracy(probs, testLabels[i], denseLayer->size);
            if (i%1000 == 999) {
                printf("[Epoch %d][Step %d] Past 1000 steps : Average Loss: %f | Accuracy: %d%%\n", j+1, i+1, l/1000, correct/10);
                l = 0;
                correct = 0;
            }
        }
    }

    free(probs);
    free(testImages);
    free(testLabels);
    free(parameters);
    printf("Training completed.\n\n");
}

void test(ConvLayer* convLayer, DenseLayer* denseLayer) {
    char* imagesPath = "./MNIST/t10k-images.idx3-ubyte";
    char* labelsPath = "./MNIST/t10k-labels.idx1-ubyte";
    int* parameters = readParameters(imagesPath);
    double*** testImages = readImages(imagesPath);
    int* testLabels = readLabels(labelsPath);

    printf("Testing CNN on %d images...\n", parameters[0]);
    
    double* probs = malloc(denseLayer->size * sizeof(double));
    double l = 0;
    int correct = 0;
    for (int i=0; i<parameters[0]; i++) {
        probs = forward(convLayer, denseLayer, testImages[i], parameters[1], parameters[2], convLayer->filterSize);
        l += loss(probs, testLabels[i]);
        correct += accuracy(probs, testLabels[i], denseLayer->size);
    }
    printf("\n|----------------------------------------|\n| Average Loss: %f | Accuracy: %d%% |\n|----------------------------------------|\n\n", l/parameters[0], correct*100/parameters[0]);

    free(probs);
    free(testImages);
    free(testLabels);
    free(parameters);
    printf("Testing completed.\n");
}

int main() {
    srand(time(NULL));

    ConvLayer* convLayer = initConvLayer(8, 3);
    DenseLayer* denseLayer = initDenseLayer(10, 13, 13, 8);
    printf("CNN Initialized. \n");

    train(convLayer, denseLayer, 1, 0.005);
    test(convLayer, denseLayer);

    freeConvLayer(convLayer);
    freeDenseLayer(denseLayer);
    return 0;
}