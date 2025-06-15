#ifndef BACKPROP_OLD_H
#define BACKPROP_OLD_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "import.h"
#include "convolution.h"
#include "pooling.h"
#include "dense.h"
#include "output.h"

double* backpropagation(ConvLayer* convLayer, DenseLayer* denseLayer, double** image, int width, int height, int divisor, int label, double learningRate);

#endif