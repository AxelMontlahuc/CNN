#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "dense.h"

DenseLayer* initDenseLayer(int size, int width, int height, int numFilters) {
    DenseLayer* layer = malloc(sizeof(DenseLayer));
    assert(layer != NULL);

    layer->size = size;
    layer->biases = calloc(size, sizeof(double));
    assert(layer->biases != NULL);
    layer->weights = malloc(size * sizeof(double*));
    assert(layer->weights != NULL);

    for (int i=0; i<size; i++) {
        layer->weights[i] = malloc(width * height * numFilters * sizeof(double));
        assert(layer->weights[i] != NULL);

        for (int j=0; j<width*height*numFilters; j++) {
            double xavierInit = (2.0 * ((double)rand() / (double)RAND_MAX) - 1.0) * sqrt(6.0 / ((double)width * (double)height * (double)numFilters + 10.0));
            layer->weights[i][j] = xavierInit;
        }
    }
    return layer;
}

void freeDenseLayer(DenseLayer* layer) {
    for (int i=0; i<layer->size; i++) {
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer->biases);
    free(layer);
}

double* denseForward(DenseLayer* denseLayer, double** input, int width, int height, int numFilters) {
    double* flatInput = malloc(width * height * numFilters * sizeof(double));
    assert(flatInput != NULL);

    for (int i=0; i<numFilters; i++) {
        for (int j=0; j<width*height; j++) {
            flatInput[i*width*height+j] = input[j][i];
        }
    }

    double* output = malloc(denseLayer->size * sizeof(double));
    for (int i=0; i<denseLayer->size; i++) {
        output[i] = 0.0;
        for (int j=0; j<width*height*numFilters; j++) {
            output[i] += flatInput[j] * denseLayer->weights[i][j];
        }
        output[i] += denseLayer->biases[i];
    }

    free(flatInput);
    return output;
}