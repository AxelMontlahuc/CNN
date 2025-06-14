#ifndef DENSE_H
#define DENSE_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

typedef struct {
    int size;
    double* biases;
    double** weights;
} DenseLayer;

DenseLayer* initDenseLayer(int size, int width, int height, int numFilters);
void freeDenseLayer(DenseLayer* layer);
double* denseForward(DenseLayer* denseLayer, double** input, int width, int height, int numFilters);

#endif