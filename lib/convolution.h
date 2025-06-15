#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

typedef struct {
    int numFilters;
    int filterSize;
    double*** filters;
} ConvLayer;

ConvLayer* initConvLayer(int numFilters, int filterSize);
void freeConvLayer(ConvLayer* layer);
double** convolutionForward(ConvLayer* convLayer, double** image, int width, int height, int divisor);

#endif