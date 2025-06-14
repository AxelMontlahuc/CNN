#ifndef POOLING_H
#define POOLING_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

double tabMax(double* tab, int size);
double** poolingForward(double** input, int width, int height, int numFilters);

#endif