#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "convolution.h"

ConvLayer* initConvLayer(int numFilters, int filterSize) {
    ConvLayer* layer = malloc(sizeof(ConvLayer));
    assert(layer != NULL);

    layer->numFilters = numFilters;
    layer->filterSize = filterSize;
    layer->filters = malloc(numFilters * sizeof(double**));
    assert(layer->filters != NULL);

    for (int i=0; i<numFilters; i++) {
        layer->filters[i] = malloc(filterSize * sizeof(double*));
        assert(layer->filters[i] != NULL);
        for (int j=0; j<filterSize; j++) {
            layer->filters[i][j] = malloc(filterSize * sizeof(double));
            for (int k=0; k<filterSize; k++) {
                double xavierInit = (2.0 * ((double)rand() / (double)RAND_MAX) - 1.0) * sqrt(6.0 / ((double)filterSize * (double)filterSize * ((double)numFilters + 1.0)));
                layer->filters[i][j][k] = xavierInit;
            }
        }
    }
    return layer;
}

void freeConvLayer(ConvLayer* layer) {
    for (int i=0; i<layer->numFilters; i++) {
        for (int j=0; j<layer->filterSize; j++) {
            free(layer->filters[i][j]);
        }
        free(layer->filters[i]);
    }
    free(layer->filters);
    free(layer);
}

double** convolutionGrid(double** image, int width, int height, int divisor) {
    double** grid = malloc((width - (divisor-1)) * (height - (divisor-1)) * sizeof(double*));
    assert(grid != NULL);

    for (int i=0; i<(width - (divisor-1)); i++) {
        for (int j=0; j<(height - (divisor-1)); j++) {
            double* cell = malloc(divisor * divisor * sizeof(double));
            assert(cell != NULL);

            for (int k=0; k<divisor; k++) {
                for (int l=0; l<divisor; l++) {
                    cell[k * divisor + l] = image[i + k][j + l];
                }
            }

            grid[i * (height - (divisor-1)) + j] = cell;
            free(cell);
        }
    }
    return grid;
}

double convolution(double** filter, double* cell) {
    double sum = 0.0;
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            sum += cell[i * 3 + j] * filter[i][j];
        }
    }
    return sum;
}

double** convolutionForward(ConvLayer* convLayer, double** image, int width, int height, int divisor) {
    double** grid = convolutionGrid(image, width, height, divisor);
    double** output = malloc((width - (divisor-1)) * (height - (divisor-1)) * sizeof(double*));
    assert(output != NULL);

    for (int i=0; i<(width - (divisor-1)); i++) {
        for (int j=0; j<(height - (divisor-1)); j++) {
            output[i * (height - (divisor-1)) + j] = malloc(convLayer->numFilters * sizeof(double));
            assert(output[i * (height - (divisor-1)) + j] != NULL);
            for (int k=0; k<convLayer->numFilters; k++) {
                output[i * (height - (divisor-1)) + j][k] = convolution(convLayer->filters[k], grid[i * (height - (divisor-1)) + j]);
            }
        }
    }
    return output;
}