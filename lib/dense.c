#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "dense.h"

double denseBoxMuller() {
    static int hasSpare = 0;
    static double spare;
    
    if (hasSpare) {
        hasSpare = 0;
        return spare;
    }

    hasSpare = 1;
    double u, v, s;
    do {
        u = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        v = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return u * s;
}

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
            double heInit = denseBoxMuller() * sqrt(2.0 / ((double)width * (double)height * (double)numFilters));
            layer->weights[i][j] = heInit;
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

double* denseForward(DenseLayer* denseLayer, double* input, int width, int height, int numFilters) {
    double* output = malloc(denseLayer->size * sizeof(double));
    for (int i=0; i<denseLayer->size; i++) {
        output[i] = 0.0;
        for (int j=0; j<width*height*numFilters; j++) {
            output[i] += input[j] * denseLayer->weights[i][j];
        }
        output[i] += denseLayer->biases[i];
    }

    return output;
}