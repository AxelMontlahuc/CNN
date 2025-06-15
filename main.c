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

void epoch() {
    char* imagesPath = "./MNIST/train-images.idx3-ubyte";
    char* labelsPath = "./MNIST/train-labels.idx1-ubyte";
    int* parameters = readParameters(imagesPath);
    double*** testImages = readImages(imagesPath);
    int* testLabels = readLabels(labelsPath);

    ConvLayer* convLayer = initConvLayer(8, 3);
    DenseLayer* denseLayer = initDenseLayer(10, 13, 13, 8);

    printf("CNN Initialized. \n");
    printf("Number of images: %d\n", parameters[0]);
    printf("Heigt: %d\n", parameters[1]);
    printf("Width: %d\n", parameters[2]);
    
    double l = 0;
    int correct = 0;
    double* probs = NULL;
    for (int i=0; i<10000; i++) {
        // printf("Label: %d | ", testLabels[i]);
        //probs = forward(convLayer, denseLayer, testImages[i], parameters[1], parameters[2], 3);
        probs = backpropagation(convLayer, denseLayer, testImages[i], parameters[1], parameters[2], 3, testLabels[i], 0.05);
        l += loss(probs, testLabels[i]);
        correct += accuracy(probs, testLabels[i], denseLayer->size);
        if (i%1000 == 999) {
            printf("[Step %d] Past 100 steps : Average Loss: %f | Accuracy: %d%%\n", i+1, l/1000, correct/10);
            l = 0;
            correct = 0;
            for (int j=0; j<denseLayer->size; j++) {
                printf("Prob[%d]: %f | ", j, probs[j]);
            }
            printf("\n");
        }
        free(probs);
        probs = NULL;
    }

    free(probs);
    freeDenseLayer(denseLayer);
    freeConvLayer(convLayer);
    free(testImages);
    free(testLabels);
    free(parameters);
    printf("Epoch completed.\n");
}

int main() {
    srand(time(NULL));
    epoch();
    return 0;
}